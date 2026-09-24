"""End-to-end clearing cycle against an in-memory fake off-chain DB
(guide §9 step 5: mock DB responses, run full cycle, assert trades)."""

import httpx
import pytest
from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse

from src.clearing import run_clearing
from src.config import Config
from src.contract import ContractError, MockContractClient
from src.offchain_db import OffchainDBClient

MARKET = "0x" + "aa" * 32
COMMUNITY = "communityid_1"
SLOT = 900


class FakeOffchainDB:
    """Implements the OffchainDBClient interface used by run_clearing."""

    def __init__(self, orders=None):
        self.orders = {o["order_id"]: o for o in (orders or [])}
        self.trades = []
        self.order_patches = []

    async def close(self):
        pass

    async def get_orders(self, market_id, start_time=None, end_time=None):
        return [o for o in self.orders.values()
                if o["market_id"] == market_id
                and (start_time is None or o["time_slot"] >= start_time)
                and (end_time is None or o["time_slot"] <= end_time)]

    async def get_trades(self, market_id):
        return [t for t in self.trades if t["market_id"] == market_id]

    async def post_trades(self, trades):
        self.trades.extend(trades)
        return trades

    async def update_order(self, order_id, *, status=None, energy=None):
        self.order_patches.append((order_id, status, energy))
        order = self.orders[order_id]
        if status is not None:
            order["status"] = status
        return order


def order(idx, order_type, name, energy, *, status="Open", time_slot=SLOT):
    rate = 28.5 if order_type == "Bid" else 8.0
    return {"order_id": f"0x{idx:064x}", "order_type": order_type,
            "status": status, "created_by": name,
            "area_uuid": f"area_{name.lower().replace(' ', '_')}",
            "market_id": MARKET, "time_slot": time_slot,
            "creation_time": 100 + idx, "energy": energy,
            "energy_rate": rate, "requirements": None}


# A fixed test input, not a configuration value. These tests check the
# clearing arithmetic against a worked example, so the band that example was
# computed on is part of the test rather than something the tests read from
# the config. That keeps them valid across every change to the community
# parameters -- K_upper on 17.09.2026 (D-77), theta and B when the
# calibration lands (T-19). That the *configured* band is loaded and applied
# is a separate question, and the second slot of
# `evaluation/harness/smoke.py` is what answers it.
GOLDEN_SIGMOID = {"k_upper": 28.5, "k_lower": 8.0,
                  "theta": 1.0, "steepness": 2.5}


def trigger(**overrides):
    payload = {"market_id": MARKET, "community_uuid": COMMUNITY,
               "time_slot": SLOT,
               "sigmoid_params": dict(GOLDEN_SIGMOID)}
    payload.update(overrides)
    return payload


@pytest.fixture()
def cfg():
    return Config()


# Guide example numbers: supply 12.5 kWh vs demand 10 kWh -> ratio 1.25,
# price ~15.1472 ct/kWh, demand-limited round.
def demand_limited_orders():
    return [
        order(1, "Offer", "PV A", 5.0),
        order(2, "Offer", "PV B", 3.5),
        order(3, "Offer", "Battery", 4.0),
        order(4, "Bid", "Household 1", 4.5),
        order(5, "Bid", "Household 2", 3.0),
        order(6, "Bid", "Bakery", 2.5),
    ]


@pytest.mark.anyio
async def test_demand_limited_clearing(cfg):
    db = FakeOffchainDB(demand_limited_orders())
    chain = MockContractClient()

    result = await run_clearing(trigger(), cfg, db, chain)

    assert result["status"] == "cleared"
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(15.1472, abs=1e-3)
    assert result["ratio"] == pytest.approx(1.25)
    assert result["traded_quantity_kwh"] == pytest.approx(10.0)
    assert result["round_type"] == "DEMAND_LIMITED"
    assert result["num_trades"] == 6  # one per participant

    # consumers fully filled, producers pro-rata at 10/12.5 = 80%
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)
    for consumer in result["allocations"]["consumers"]:
        assert consumer["fill_rate"] == pytest.approx(1.0)

    pv_a = next(p for p in result["allocations"]["producers"]
                if p["name"] == "PV A")
    assert pv_a["allocated_kwh"] == pytest.approx(4.0)
    assert pv_a["value_ct"] == pytest.approx(4.0 * 15.1472, abs=1e-2)

    # allocations conserve energy: sum(producer) == sum(consumer) == traded
    total_sold = sum(p["allocated_kwh"]
                     for p in result["allocations"]["producers"])
    total_bought = sum(c["allocated_kwh"]
                       for c in result["allocations"]["consumers"])
    assert total_sold == pytest.approx(10.0, abs=1e-6)
    assert total_bought == pytest.approx(10.0, abs=1e-6)

    # trades persisted; producers carry residual_offer (partial fill)
    assert len(db.trades) == 6
    seller_trades = [t for t in db.trades if t["buyer"].startswith("AMM_POOL")]
    assert all(t["residual_offer"] is not None for t in seller_trades)
    buyer_trades = [t for t in db.trades if t["seller"].startswith("AMM_POOL")]
    assert all(t["residual_bid"] is None for t in buyer_trades)

    # every matched order marked Executed
    assert all(o["status"] == "Executed" for o in db.orders.values())

    # on-chain anchor was recorded before trades were written
    assert chain.records[MARKET]["clearing_price"] == 151472
    assert result["tx_hash"] == chain.records[MARKET]["tx_hash"]
    assert all(t["parameters"]["amm_tx_hash"] == result["tx_hash"]
               for t in db.trades)


@pytest.mark.anyio
async def test_supply_limited_clearing(cfg):
    db = FakeOffchainDB([
        order(1, "Offer", "PV A", 4.0),
        order(2, "Bid", "Household 1", 5.0),
        order(3, "Bid", "Household 2", 3.0),
    ])
    result = await run_clearing(trigger(), cfg, db, MockContractClient())

    assert result["round_type"] == "SUPPLY_LIMITED"
    assert result["ratio"] == pytest.approx(0.5)
    assert result["traded_quantity_kwh"] == pytest.approx(4.0)
    # scarce supply -> price above band center
    assert result["clearing_price_ct_per_kwh"] > (28.5 + 8.0) / 2
    for consumer in result["allocations"]["consumers"]:
        assert consumer["fill_rate"] == pytest.approx(0.5)


@pytest.mark.anyio
async def test_no_trade_when_one_side_is_empty(cfg):
    db = FakeOffchainDB([
        order(1, "Bid", "Household 1", 5.0),
        order(2, "Bid", "Household 2", 3.0),
    ])
    chain = MockContractClient()
    result = await run_clearing(trigger(), cfg, db, chain)

    assert result["status"] == "no_trade"
    assert result["num_orders_expired"] == 2
    assert db.trades == []
    assert chain.records == {}  # nothing anchored on-chain
    assert all(o["status"] == "Expired" for o in db.orders.values())


@pytest.mark.anyio
async def test_non_open_orders_are_excluded(cfg):
    db = FakeOffchainDB([
        order(1, "Offer", "PV A", 5.0),
        order(2, "Offer", "PV ghost", 99.0, status="Expired"),
        order(3, "Bid", "Household 1", 5.0),
        order(4, "Bid", "Ghost bid", 99.0, status="Executed"),
    ])
    result = await run_clearing(trigger(), cfg, db, MockContractClient())
    assert result["total_supply_kwh"] == pytest.approx(5.0)
    assert result["total_demand_kwh"] == pytest.approx(5.0)
    assert result["num_trades"] == 2


@pytest.mark.anyio
async def test_orders_outside_delivery_window_are_excluded(cfg):
    db = FakeOffchainDB([
        order(1, "Offer", "PV A", 5.0),
        order(2, "Bid", "Household 1", 5.0),
        order(3, "Bid", "Next slot", 99.0, time_slot=SLOT + 901),
    ])
    result = await run_clearing(trigger(), cfg, db, MockContractClient())
    assert result["total_demand_kwh"] == pytest.approx(5.0)


@pytest.mark.anyio
async def test_next_slot_orders_are_not_pulled_in(cfg):
    """Regression: the order window must be exclusive at the upper end.

    Both orders share the same market_id here — otherwise the market filter
    hides the off-by-one and the test would pass without testing anything.
    """
    orders = demand_limited_orders() + [
        order(99, "Offer", "Next Slot PV", 100.0, time_slot=SLOT + 900)]
    result = await run_clearing(trigger(), cfg, FakeOffchainDB(orders),
                                MockContractClient())

    # 12.5 kWh supply from the current slot only — not 112.5.
    assert result["total_supply_kwh"] == pytest.approx(12.5)
    assert result["num_trades"] == 6


@pytest.mark.anyio
async def test_clearing_is_idempotent(cfg):
    db = FakeOffchainDB(demand_limited_orders())
    chain = MockContractClient()

    first = await run_clearing(trigger(), cfg, db, chain)
    second = await run_clearing(trigger(), cfg, db, chain)

    assert second["status"] == "already_cleared"
    assert len(db.trades) == 6  # not duplicated
    assert second["clearing_price_ct_per_kwh"] == pytest.approx(
        first["clearing_price_ct_per_kwh"], abs=1e-4)
    assert second["tx_hash"] == first["tx_hash"]
    # allocation summary reconstructed from stored trades
    producers = {p["name"]: p for p in second["allocations"]["producers"]}
    assert producers["PV A"]["allocated_kwh"] == pytest.approx(4.0)
    assert producers["PV A"]["requested_kwh"] == pytest.approx(5.0)

    # Provenance: the re-trigger computed nothing, it read the price back out
    # of the stored trades. A campaign grouping by `price_source` would
    # otherwise count this as a fresh clearing.
    assert first["price_source"] == "computed"
    assert second["price_source"] == "stored"


@pytest.mark.anyio
async def test_both_paths_return_the_same_summary_shape(cfg):
    """Fresh run and idempotent re-trigger must be interchangeable for any
    consumer of the API (demo UI, simulation harness)."""
    db = FakeOffchainDB(demand_limited_orders())
    chain = MockContractClient()

    first = await run_clearing(trigger(), cfg, db, chain)
    second = await run_clearing(trigger(), cfg, db, chain)

    shared = set(first) & set(second)
    assert {"preferences", "allocations", "trades", "sigmoid_params"} <= shared
    assert first["preferences"].keys() == second["preferences"].keys()
    assert (first["preferences"]["multipliers"].keys()
            == second["preferences"]["multipliers"].keys())
    assert first["preferences"] == second["preferences"]
    for side in ("producers", "consumers"):
        assert (first["allocations"][side][0].keys()
                == second["allocations"][side][0].keys())


@pytest.mark.anyio
async def test_summary_reports_the_preference_block(cfg):
    result = await run_clearing(trigger(), cfg,
                                FakeOffchainDB(demand_limited_orders()),
                                MockContractClient())
    preferences = result["preferences"]
    assert preferences["order"] == "preferences_first"
    assert preferences["mutual_pairs"] == []
    assert preferences["multipliers"]["mode"] == "multiplicative"
    assert preferences["multipliers"]["sides"] == "seller"
    # every allocation row carries the two new preference columns
    for row in (result["allocations"]["producers"]
                + result["allocations"]["consumers"]):
        assert row["preference_matched"] is False
        assert row["final_energy_rate"] == pytest.approx(
            result["clearing_price_ct_per_kwh"])


@pytest.mark.anyio
async def test_trigger_sigmoid_param_overrides(cfg):
    db = FakeOffchainDB([
        order(1, "Offer", "PV A", 5.0),
        order(2, "Bid", "Household 1", 5.0),
    ])
    result = await run_clearing(
        trigger(sigmoid_params={"k_upper": 40.0, "k_lower": 10.0,
                                "theta": 1.0, "steepness": 2.5}),
        cfg, db, MockContractClient())
    # ratio 1.0 == theta -> band center of the OVERRIDDEN band
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(25.0)
    assert result["sigmoid_params"]["k_upper"] == 40.0


# ------------------------------------------- off-chain DB without PATCH

def db_without_patch(orders: list[dict], status_code: int) -> FastAPI:
    """A DB that serves orders and trades but rejects status updates.

    The production off-chain storage registers no PATCH route, so this is
    what the Clearing Node meets when `OFFCHAIN_DB_URL` points at the real
    service.
    """
    app = FastAPI()
    app.state.trades = []
    by_id = {o["order_id"]: o for o in orders}

    @app.get("/orders")
    async def get_orders(market_id: str = Query(...),
                         start_time: int | None = Query(default=None),
                         end_time: int | None = Query(default=None)) -> list:
        return [o for o in by_id.values()
                if o["market_id"] == market_id
                and (start_time is None or o["time_slot"] >= start_time)
                and (end_time is None or o["time_slot"] <= end_time)]

    @app.get("/trades")
    async def get_trades(market_id: str | None = Query(default=None)) -> list:
        return [t for t in app.state.trades if t["market_id"] == market_id]

    @app.post("/trades-normalized", status_code=201)
    async def post_trades(payload: list = Body(...)) -> list:
        app.state.trades.extend(payload)
        return payload

    @app.patch("/orders/{order_id}")
    async def patch_order(order_id: str):
        return JSONResponse(status_code=status_code,
                            content={"detail": "no such route"})

    return app


@pytest.mark.parametrize("status_code", [404, 405])
@pytest.mark.anyio
async def test_clearing_survives_a_db_without_a_patch_route(cfg, status_code):
    app = db_without_patch(demand_limited_orders(), status_code)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                               base_url="http://offchain-db")
    db = OffchainDBClient("http://offchain-db", client=client)

    result = await run_clearing(trigger(), cfg, db, MockContractClient())

    # Identical to the run against a DB that accepts the PATCH: only the
    # order-status side effect is missing.
    assert result["status"] == "cleared"
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(15.147225)
    assert result["traded_quantity_kwh"] == pytest.approx(10.0)
    assert result["num_trades"] == 6
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)
    assert len(app.state.trades) == 6

    await db.close()
    await client.aclose()


# --------------------------------------------- recovery from a lost anchor

@pytest.mark.anyio
async def test_recovers_when_the_anchor_outlived_the_trade_write_back(cfg):
    """`clearMarket` succeeded, `post_trades` did not.

    The slot then holds an anchor but no trades, so a re-trigger passes the
    idempotency check, clears again, and the contract reverts. Before the
    recovery that meant a 502 for that slot forever.
    """
    chain = MockContractClient()
    db = FakeOffchainDB(demand_limited_orders())

    first = await run_clearing(trigger(), cfg, db, chain)
    assert first["recovered_from_anchor"] is False

    # the write-back is lost; the on-chain anchor survives
    db.trades = []
    for order_ in db.orders.values():
        order_["status"] = "Open"

    recovered = await run_clearing(trigger(), cfg, db, chain)

    assert recovered["status"] == "cleared"
    assert recovered["recovered_from_anchor"] is True
    # the original anchor, not a second one
    assert recovered["tx_hash"] == first["tx_hash"]
    assert len(chain.records) == 1
    assert recovered["clearing_price_ct_per_kwh"] == pytest.approx(15.147225)
    # and the write-back this time succeeded
    assert len(db.trades) == 6
    assert all(t["parameters"]["amm_tx_hash"] == first["tx_hash"]
               for t in db.trades)


@pytest.mark.anyio
async def test_other_contract_reverts_still_fail_the_run(cfg):
    """A bounds-check failure or an unauthorised caller must fail loudly."""

    class RejectingChain(MockContractClient):
        async def clear_market(self, **kwargs):
            raise ContractError("AMMBA: clearing price out of bounds")

    db = FakeOffchainDB(demand_limited_orders())
    with pytest.raises(ContractError, match="out of bounds"):
        await run_clearing(trigger(), cfg, db, RejectingChain())
    # nothing was written on the way out
    assert db.trades == []


@pytest.mark.anyio
async def test_a_duplicate_revert_without_an_anchor_still_raises(cfg):
    """Nothing to recover from means the original error stands."""

    class AmnesiacChain(MockContractClient):
        async def clear_market(self, **kwargs):
            raise ContractError("AMMBA: market already cleared: x")

        async def get_clearing_result(self, market_id):
            return None

    db = FakeOffchainDB(demand_limited_orders())
    with pytest.raises(ContractError, match="already cleared"):
        await run_clearing(trigger(), cfg, db, AmnesiacChain())


# ----------------------------------------- issue #26: the idempotency filter

@pytest.mark.anyio
async def test_a_foreign_market_trade_does_not_suppress_the_clearing(cfg):
    """The production GSY API ignores the `market_id` query parameter.

    `get_trades(market_id)` therefore returns foreign trades against a real
    DB, and one with a matching `time_slot` would be read as "already
    cleared" — a genuine clearing skipped in silence. The mock filters
    server-side, so the leak only appears with a DB that does not.
    """

    class UnfilteredDB(FakeOffchainDB):
        """Mirrors routes/trades.rs: the parameter is accepted and ignored."""

        async def get_trades(self, market_id):
            return self.foreign + [t for t in self.trades
                                   if t["market_id"] == market_id]

    db = UnfilteredDB(demand_limited_orders())
    db.foreign = [{"trade_uuid": "0x" + "cc" * 32,
                   "market_id": "0x" + "ee" * 32,   # another market
                   "time_slot": SLOT,               # same slot
                   "parameters": {"energy_rate": 99.0}}]

    result = await run_clearing(trigger(), cfg, db, MockContractClient())

    assert result["status"] == "cleared"
    assert result["num_trades"] == 6
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(15.147225)


# ------------------------------- D-63 / #28: what a recovery has to report

class _AnchorAtChain(MockContractClient):
    """A chain that reports "already cleared" and hands back a stored anchor.

    `price` is what the anchor holds, `tx_hash` what its log read yields —
    None models a node that prunes logs or refuses `fromBlock=0`.
    """

    def __init__(self, price, tx_hash="0x" + "dd" * 32):
        super().__init__()
        self._price = price
        self._tx_hash = tx_hash

    async def clear_market(self, **kwargs):
        raise ContractError("AMMBA: market already cleared: x")

    async def get_clearing_result(self, market_id):
        return {"clearing_price": self._price, "tx_hash": self._tx_hash}


@pytest.mark.anyio
async def test_a_normal_run_reports_a_computed_price_source(cfg):
    # The keys exist unconditionally, so a 672-slot campaign can group by them.
    result = await run_clearing(trigger(), cfg,
                                FakeOffchainDB(demand_limited_orders()),
                                MockContractClient())

    assert result["price_source"] == "computed"
    assert result["anchored_price_ct"] is None
    assert result["recomputed_price_ct"] is None
    assert result["anchor_tx_hash_recovered"] is True


@pytest.mark.anyio
async def test_a_diverging_anchor_is_reported_not_only_logged(cfg):
    """D-63: the anchor wins, and the divergence leaves the log.

    In a recovery `clearMarket` reverts on the duplicate market *before* its
    bounds check runs, so the recomputed price is the one no contract has
    verified — the anchor is correctly kept. A 672-slot campaign cannot count
    log lines, so both prices travel in the response.
    """
    db = FakeOffchainDB(demand_limited_orders())
    result = await run_clearing(trigger(), cfg, db, _AnchorAtChain(14.0))

    assert result["status"] == "cleared"
    # behaviour is unchanged: the anchored value settles the round
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(14.0)
    assert result["price_source"] == "anchor"
    assert result["anchored_price_ct"] == pytest.approx(14.0)
    assert result["recomputed_price_ct"] == pytest.approx(15.147225)
    assert result["recovered_from_anchor"] is True
    # the trades settle on the anchored price too
    assert all(t["parameters"]["energy_rate"] == pytest.approx(14.0)
               for t in db.trades)


@pytest.mark.anyio
async def test_an_agreeing_anchor_reports_both_prices_without_switching(cfg):
    # Same on-chain integer: nothing diverged, so the recomputed value stays
    # the settlement reference and `price_source` says so.
    result = await run_clearing(trigger(), cfg,
                                FakeOffchainDB(demand_limited_orders()),
                                _AnchorAtChain(15.1472))

    assert result["recovered_from_anchor"] is True
    assert result["price_source"] == "computed"
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(15.147225)
    assert result["anchored_price_ct"] == pytest.approx(15.1472)
    assert result["recomputed_price_ct"] == pytest.approx(15.147225)


@pytest.mark.anyio
async def test_an_unrecoverable_anchor_hash_is_flagged_not_hidden(cfg):
    """Issue #28: a trade written with an empty `amm_tx_hash`.

    The contract stores the clearing result but not the hash of the
    transaction that wrote it, so on recovery the hash comes only from the
    `MarketCleared` log. A node that prunes logs yields nothing and the
    record loses its audit link — the recovery is still valid, so the run
    must not fail, but it must not look like an ordinary success either.
    """
    db = FakeOffchainDB(demand_limited_orders())
    result = await run_clearing(trigger(), cfg, db,
                                _AnchorAtChain(15.1472, tx_hash=None))

    assert result["status"] == "cleared"        # the recovery completed
    assert result["tx_hash"] is None
    assert result["anchor_tx_hash_recovered"] is False
    assert len(db.trades) == 6


def test_configuration_yaml_ships_a_usable_sigmoid_band():
    """The configured band is loaded and is a band, whatever its values are.

    Deliberately not an assertion on 40.0 / 8.0: pinning the numbers here
    would put this test back in the way of every parameter change, which is
    the thing `GOLDEN_SIGMOID` exists to avoid. What must hold is that
    `configuration.yaml` reaches `resolve_community` at all and describes a
    band a price can sit in -- the silent failure this guards against is a
    parameter update that edits the file but never takes effect.
    """
    from src.config import load_config, resolve_community

    community = resolve_community(load_config(), "any-unknown-community")
    assert community.k_lower < community.k_upper
    assert community.k_lower > 0
    assert community.steepness > 0
