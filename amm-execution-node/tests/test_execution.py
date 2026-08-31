"""Execution cycle tests against an in-memory fake off-chain DB."""

import pytest

from src.config import Config
from src.execution import (compute_previous_timeslot, run_execution,
                           run_polling_cycle)
from src.sigmoid import sigmoid_price

MARKET = "0x" + "aa" * 32
COMMUNITY = "communityid_1"
POOL = f"AMM_POOL_{COMMUNITY}"
SLOT = 900
SIGMOID = {"k_upper": 28.5, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}


def _parameters(traded, price, total_supply, total_demand):
    return {"selected_energy": traded, "energy_rate": price,
            "trade_uuid": None, "amm_tx_hash": "0x" + "ab" * 32,
            **SIGMOID, "total_supply_kwh": total_supply,
            "total_demand_kwh": total_demand, "pool_id": POOL,
            "preference_matched": False}


def seller_trade(uuid, name, area, traded, requested, price, supply, demand):
    residual = requested - traded
    return {
        "trade_uuid": uuid, "status": "Settled", "buyer": POOL,
        "seller": name, "market_id": MARKET, "time_slot": SLOT,
        "offer": {"seller": name,
                  "offer_component": {"area_uuid": area, "energy": traded,
                                      "energy_rate": price}},
        "bid": {"buyer": POOL,
                "bid_component": {"area_uuid": POOL, "energy": traded,
                                  "energy_rate": price}},
        "residual_offer": ({"energy": residual, "energy_rate": 8.0}
                           if residual > 1e-9 else None),
        "residual_bid": None,
        "parameters": {**_parameters(traded, price, supply, demand),
                       "trade_uuid": uuid},
    }


def buyer_trade(uuid, name, area, traded, requested, price, supply, demand):
    residual = requested - traded
    return {
        "trade_uuid": uuid, "status": "Settled", "buyer": name,
        "seller": POOL, "market_id": MARKET, "time_slot": SLOT,
        "bid": {"buyer": name,
                "bid_component": {"area_uuid": area, "energy": traded,
                                  "energy_rate": price}},
        "offer": {"seller": POOL,
                  "offer_component": {"area_uuid": POOL, "energy": traded,
                                      "energy_rate": price}},
        "residual_bid": ({"energy": residual, "energy_rate": 28.5}
                         if residual > 1e-9 else None),
        "residual_offer": None,
        "parameters": {**_parameters(traded, price, supply, demand),
                       "trade_uuid": uuid},
    }


class FakeOffchainDB:
    def __init__(self, trades=None, measurements=None, markets=None,
                 forecasts=None):
        self.trades = {t["trade_uuid"]: t for t in (trades or [])}
        # (area_uuid, time_slot) -> kWh, on both channels
        self.measurements = measurements or {}
        self.forecasts = forecasts or {}
        self.markets = markets or []
        self.trade_patches = []

    async def close(self):
        pass

    async def get_trades(self, market_id):
        return [t for t in self.trades.values()
                if t["market_id"] == market_id]

    @staticmethod
    def _rows(channel, community_uuid, time_slot, time_slot_sec, area_uuid):
        # Mirrors the real client: an inclusive [t, t + Δ - 1] window.
        end_time = time_slot + time_slot_sec - 1
        return [{"community_uuid": community_uuid, "area_uuid": area,
                 "time_slot": slot, "energy_kwh": kwh}
                for (area, slot), kwh in channel.items()
                if (area_uuid is None or area == area_uuid)
                and time_slot <= slot <= end_time]

    async def get_measurements(self, community_uuid, time_slot, time_slot_sec,
                               area_uuid=None):
        return self._rows(self.measurements, community_uuid, time_slot,
                          time_slot_sec, area_uuid)

    async def get_forecasts(self, community_uuid, time_slot, time_slot_sec,
                            area_uuid=None):
        return self._rows(self.forecasts, community_uuid, time_slot,
                          time_slot_sec, area_uuid)

    async def get_community_markets(self, community_uuid):
        return self.markets

    async def update_trade(self, trade_uuid, *, status=None, parameters=None):
        self.trade_patches.append((trade_uuid, status, parameters))
        trade = self.trades[trade_uuid]
        if status:
            trade["status"] = status
        if parameters:
            merged = dict(trade.get("parameters") or {})
            merged.update(parameters)
            trade["parameters"] = merged
        return trade


def trigger():
    return {"market_id": MARKET, "community_uuid": COMMUNITY,
            "time_slot": SLOT}


@pytest.fixture()
def cfg():
    return Config(penalty_gamma=1.1, penalty_eta_kwh=0.0)


# ---------------------------------------------------------- supply-limited

def supply_limited_db(measurements, forecasts=None):
    # supply 8 < demand 10, price ~20.76, traded quantity 8;
    # sellers fully filled, buyers filled at 80%.
    price = sigmoid_price(0.8, **SIGMOID)
    trades = [
        seller_trade("t-pva", "PV A", "area_pva", 5.0, 5.0, price, 8.0, 10.0),
        seller_trade("t-pvb", "PV B", "area_pvb", 3.0, 3.0, price, 8.0, 10.0),
        buyer_trade("t-h1", "House 1", "area_h1", 4.8, 6.0, price, 8.0, 10.0),
        buyer_trade("t-h2", "House 2", "area_h2", 3.2, 4.0, price, 8.0, 10.0),
    ]
    return FakeOffchainDB(trades, measurements, forecasts=forecasts), price


@pytest.mark.anyio
async def test_supply_limited_round(cfg):
    db, price = supply_limited_db({
        ("area_pva", SLOT): 4.0,   # under-delivered by 1 kWh -> shortfall
        ("area_pvb", SLOT): 5.0,   # could deliver 5, traded 3 -> withheld 2
        ("area_h1", SLOT): 4.8,    # consumed exactly as allocated
        ("area_h2", SLOT): 10.0,   # over-consumed, but round is supply-limited
    })
    result = await run_execution(trigger(), cfg, db)

    assert result["status"] == "executed"
    assert result["round_type"] == "SUPPLY_LIMITED"
    assert result["clearing_price_ct_per_kwh"] == pytest.approx(price, abs=1e-6)
    rows = {r["name"]: r for r in result["results"]}
    assert len(rows) == 4

    # PV A: shortfall penalty 1 kWh * 1.1 * 28.5
    assert rows["PV A"]["shortfall_kwh"] == pytest.approx(1.0)
    assert rows["PV A"]["shortfall_penalty_ct"] == pytest.approx(31.35)
    assert rows["PV A"]["externality_penalty_ct"] == 0.0

    # PV B: withheld 2 kWh -> counterfactual ratio 1.0, price 18.25
    assert rows["PV B"]["shortfall_penalty_ct"] == 0.0
    assert rows["PV B"]["externality_kwh"] == pytest.approx(2.0)
    assert rows["PV B"]["counterfactual_price_ct_per_kwh"] == pytest.approx(18.25)
    assert rows["PV B"]["externality_penalty_ct"] == pytest.approx(
        (price - 18.25) * 8.0, abs=1e-4)
    assert rows["PV B"]["total_penalty_ct"] == pytest.approx(
        rows["PV B"]["externality_penalty_ct"])

    # buyers: no buyer externality in supply-limited rounds (guide §5.3)
    assert rows["House 1"]["total_penalty_ct"] == 0.0
    assert rows["House 2"]["total_penalty_ct"] == 0.0

    # Without a forecast channel every row falls back to the meter, which is
    # what produced the figures asserted above.
    assert {r["deliverable_source"] for r in result["results"]} == {"meter"}
    assert rows["PV B"]["deliverable_kwh"] == pytest.approx(5.0)

    # penalties written back into trade parameters, trades marked Executed
    assert len(db.trade_patches) == 4
    assert db.trades["t-pva"]["status"] == "Executed"
    assert db.trades["t-pva"]["parameters"]["shortfall_penalty_ct"] == \
        pytest.approx(31.35)
    # original clearing parameters survive the merge
    assert db.trades["t-pva"]["parameters"]["amm_tx_hash"] == "0x" + "ab" * 32


@pytest.mark.anyio
async def test_forecast_makes_a_successful_withholder_visible(cfg):
    """The point of the second channel (issue #13).

    PV B trades 3.0 kWh and delivers exactly 3.0, so the meter shows no
    deviation at all: measured == traded, and with a single channel
    `W_sell = max(0, deliverable - traded - eta)` is zero by construction.
    The forecast says the area could have delivered 5.0, which is what the
    externality penalty is meant to catch.
    """
    db, price = supply_limited_db(
        {("area_pva", SLOT): 5.0, ("area_pvb", SLOT): 3.0,
         ("area_h1", SLOT): 4.8, ("area_h2", SLOT): 3.2},
        forecasts={("area_pvb", SLOT): 5.0})
    result = await run_execution(trigger(), cfg, db)

    rows = {r["name"]: r for r in result["results"]}
    pv_b = rows["PV B"]
    assert pv_b["deliverable_source"] == "forecast"
    assert pv_b["deliverable_kwh"] == pytest.approx(5.0)
    # delivered as traded -> no shortfall, but 2 kWh withheld
    assert pv_b["shortfall_penalty_ct"] == 0.0
    assert pv_b["externality_kwh"] == pytest.approx(2.0)
    assert pv_b["externality_penalty_ct"] == pytest.approx(
        (price - 18.25) * 8.0, abs=1e-4)
    assert pv_b["externality_penalty_ct"] > 0.0

    # the provenance travels with the penalty write-back
    assert db.trades["t-pvb"]["parameters"]["deliverable_source"] == "forecast"
    assert db.trades["t-pvb"]["parameters"]["deliverable_kwh"] ==         pytest.approx(5.0)


@pytest.mark.anyio
async def test_forecast_coverage_is_per_area(cfg):
    # Only PV B carries a forecast; PV A falls back to its meter reading in
    # the same slot, independently.
    db, _ = supply_limited_db(
        {("area_pva", SLOT): 5.0, ("area_pvb", SLOT): 3.0,
         ("area_h1", SLOT): 4.8, ("area_h2", SLOT): 3.2},
        forecasts={("area_pvb", SLOT): 5.0})
    result = await run_execution(trigger(), cfg, db)

    rows = {r["name"]: r for r in result["results"]}
    assert rows["PV B"]["deliverable_source"] == "forecast"
    assert rows["PV A"]["deliverable_source"] == "meter"
    assert rows["PV A"]["deliverable_kwh"] == pytest.approx(5.0)
    # PV A traded 5.0 and delivered 5.0 -> nothing to penalise either way
    assert rows["PV A"]["total_penalty_ct"] == 0.0


@pytest.mark.anyio
async def test_forecast_from_another_slot_is_ignored(cfg):
    # The forecast window is the slot's own; the next slot's row must not
    # leak in and manufacture a withholding penalty.
    db, _ = supply_limited_db(
        {("area_pvb", SLOT): 3.0},
        forecasts={("area_pvb", SLOT + 900): 5.0})
    result = await run_execution(trigger(), cfg, db)

    pv_b = {r["name"]: r for r in result["results"]}["PV B"]
    assert pv_b["deliverable_source"] == "meter"
    assert pv_b["externality_penalty_ct"] == 0.0


# ---------------------------------------------------------- demand-limited

def demand_limited_db(measurements):
    # guide example: supply 12.5 > demand 10, buyers fully filled,
    # producers at 80%.
    price = sigmoid_price(1.25, **SIGMOID)
    trades = [
        seller_trade("t-pva", "PV A", "area_pva", 4.0, 5.0, price, 12.5, 10.0),
        seller_trade("t-bat", "Battery", "area_bat", 6.0, 7.5, price, 12.5, 10.0),
        buyer_trade("t-h1", "House 1", "area_h1", 7.5, 7.5, price, 12.5, 10.0),
        buyer_trade("t-bak", "Bakery", "area_bak", 2.5, 2.5, price, 12.5, 10.0),
    ]
    return FakeOffchainDB(trades, measurements), price


@pytest.mark.anyio
async def test_demand_limited_round(cfg):
    db, price = demand_limited_db({
        ("area_pva", SLOT): 4.0,
        ("area_bat", SLOT): 6.0,
        ("area_h1", SLOT): 7.5,
        ("area_bak", SLOT): 4.5,   # consumed 2 kWh more than reported
    })
    result = await run_execution(trigger(), cfg, db)

    assert result["round_type"] == "DEMAND_LIMITED"
    rows = {r["name"]: r for r in result["results"]}

    # Bakery: underreported 2 kWh -> counterfactual demand 12
    bakery = rows["Bakery"]
    assert bakery["externality_kwh"] == pytest.approx(2.0)
    p_cf = sigmoid_price(12.5 / 12.0, **SIGMOID)
    assert bakery["counterfactual_price_ct_per_kwh"] == pytest.approx(
        p_cf, abs=1e-4)
    assert bakery["externality_penalty_ct"] == pytest.approx(
        (p_cf - price) * 10.0, abs=1e-3)

    # honest participants unpenalized; sellers withholding does not apply
    # in demand-limited rounds
    for name in ("PV A", "Battery", "House 1"):
        assert rows[name]["total_penalty_ct"] == 0.0


@pytest.mark.anyio
async def test_missing_measurement_assumes_delivery(cfg):
    db, _ = demand_limited_db({})  # no measurements at all
    result = await run_execution(trigger(), cfg, db)
    assert all(r["total_penalty_ct"] == 0.0 for r in result["results"])
    assert all(r["measurement_found"] is False for r in result["results"])
    # actual falls back to traded
    rows = {r["name"]: r for r in result["results"]}
    assert rows["PV A"]["actual_kwh"] == pytest.approx(4.0)


@pytest.mark.anyio
async def test_no_trades_yields_empty_result(cfg):
    result = await run_execution(trigger(), cfg, FakeOffchainDB())
    assert result["status"] == "no_trades"
    assert result["results"] == []


# ------------------------------------------------------------------ timing

def test_compute_previous_timeslot():
    # offset 120 min, 900s slots: cutoff = now - 7200, floored to slot start
    assert compute_previous_timeslot(900, -120, now=100000) == 92700
    # exactly on a boundary
    assert compute_previous_timeslot(900, -120, now=900 * 10 + 7200) == 9000
    # sign of the offset must not matter
    assert compute_previous_timeslot(900, 120, now=100000) == \
        compute_previous_timeslot(900, -120, now=100000)


@pytest.mark.anyio
async def test_polling_cycle_executes_due_markets(cfg, monkeypatch):
    db, _ = demand_limited_db({})
    db.markets = [{"market_id": MARKET, "community_uuid": COMMUNITY,
                   "time_slot": SLOT},
                  {"market_id": "0x" + "bb" * 32,
                   "community_uuid": COMMUNITY, "time_slot": SLOT + 900}]
    cfg = Config(poll_communities=(COMMUNITY,))
    monkeypatch.setattr("src.execution.compute_previous_timeslot",
                        lambda *a, **kw: SLOT)
    summaries = await run_polling_cycle(cfg, db)
    assert len(summaries) == 1
    assert summaries[0]["market_id"] == MARKET
