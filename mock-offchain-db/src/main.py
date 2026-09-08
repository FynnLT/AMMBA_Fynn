"""Mock GSY-DEX Off-Chain Database.

In-memory stand-in for the GSY-DEX off-chain storage service, which is not
deployed yet. The route set follows the production service
(gsy-decentralized-exchange main @ aa99ea2,
gsy-offchain-storage/src/startup.rs) so that the Clearing Node and Execution
Node are developed against the real interface and can be pointed at the
production DB later with zero changes to their core logic (only
`OFFCHAIN_DB_URL`).

Mock-only conveniences, each clearly marked:

* ``POST /market`` generates ``market_id`` (blake2b-256 of "Spot" + the
  big-endian delivery timestamp) when the caller does not provide one — in
  production the Market Orchestrator owns this id.
* ``PATCH /orders/{order_id}`` — the production service registers no PATCH
  route; there, order status is driven by the blockchain event listener. The
  clients treat a 404/405 on it as a no-op, so this stays a pure convenience.
* ``PATCH /trades/{trade_uuid}`` — TODO(confirm-with-supervisor): penalty
  output schema is unresolved. The PoC extends the trade ``parameters``
  field through this endpoint.
* ``POST /reset`` — wipes the store (used by tests and demos).
"""

import logging
from typing import Union

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.store import InMemoryStore

logger = logging.getLogger("mock-offchain-db")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Guide §2 order lifecycle.
VALID_ORDER_STATUSES = ("Open", "Executed", "Expired", "Deleted")
VALID_ORDER_TYPES = ("Bid", "Offer")
# `attributes`/`requirements` are optional in the offchain-primitives order
# schema and stored opaquely; only the energy type is validated, because a
# typo there silently changes the clearing economics. `preferred_partner` is
# deliberately NOT validated against existing areas: the DB has no
# cross-order view, and the Clearing Node degrades unmatchable partners
# gracefully.
VALID_ENERGY_TYPES = ("green", "grey")


def _as_list(payload: Union[list, dict]) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]


def create_app(store: InMemoryStore | None = None) -> FastAPI:
    store = store or InMemoryStore()

    app = FastAPI(
        title="Mock GSY-DEX Off-Chain Database",
        description=__doc__,
        version="0.1.0",
    )
    app.state.store = store

    # The demo UI is served from another origin (localhost:3000) and calls
    # this service directly from the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- health

    @app.get("/health_check")
    async def health_check() -> dict:
        return {"status": "ok", "service": "mock-offchain-db"}

    # ------------------------------------------------------------ markets

    @app.get("/market")
    async def get_market(market_id: str = Query(...)) -> dict:
        market = store.markets.get(market_id)
        if market is None:
            raise HTTPException(status_code=404,
                                detail=f"market {market_id} not found")
        return market

    @app.get("/community-market")
    async def get_community_markets(community_uuid: str = Query(...)) -> list:
        return store.markets_for_community(community_uuid)

    @app.post("/market", status_code=201)
    async def post_market(market: dict = Body(...)) -> dict:
        if market.get("time_slot") is None:
            raise HTTPException(status_code=400, detail="time_slot is required")
        created = store.upsert_market(market)
        logger.info("market %s (community=%s, time_slot=%s)",
                    created["market_id"], created.get("community_uuid"),
                    created.get("time_slot"))
        return created

    # ------------------------------------------------------------- orders

    @app.get("/orders")
    async def get_orders(market_id: str | None = Query(default=None),
                         start_time: int | None = Query(default=None),
                         end_time: int | None = Query(default=None)) -> list:
        return store.query_orders(market_id, start_time, end_time)

    @app.post("/orders")
    async def post_orders_scale() -> None:
        # The real endpoint accepts SCALE-encoded binary payloads from the
        # GSY node; the mock only supports the JSON variant.
        raise HTTPException(
            status_code=501,
            detail="SCALE-encoded /orders is not implemented in the mock; "
                   "use POST /orders-normalized",
        )

    @app.post("/orders-normalized", status_code=201)
    async def post_orders_normalized(
            payload: Union[list, dict] = Body(...)) -> list:
        orders = _as_list(payload)
        created = []
        for order in orders:
            if order.get("order_type") not in VALID_ORDER_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail="order_type must be 'Bid' or 'Offer'")
            if not order.get("market_id"):
                raise HTTPException(status_code=400,
                                    detail="market_id is required")
            energy = order.get("energy")
            if not isinstance(energy, (int, float)) or energy <= 0:
                raise HTTPException(status_code=400,
                                    detail="energy must be a positive number")
            attributes = order.get("attributes")
            if isinstance(attributes, dict) and "energy_type" in attributes:
                if attributes["energy_type"] not in VALID_ENERGY_TYPES:
                    raise HTTPException(
                        status_code=400,
                        detail="attributes.energy_type must be "
                               "'green' or 'grey'")
            created.append(store.add_order(order))
        logger.info("stored %d order(s)", len(created))
        return created

    @app.patch("/orders/{order_id}")
    async def patch_order(order_id: str, patch: dict = Body(...)) -> dict:
        # MOCK-ONLY endpoint: the production service registers no PATCH route
        # for orders (gsy-offchain-storage/src/startup.rs @ aa99ea2), where
        # the blockchain event listener drives order status instead.
        order = store.orders.get(order_id)
        if order is None:
            raise HTTPException(status_code=404,
                                detail=f"order {order_id} not found")
        if "status" in patch:
            if patch["status"] not in VALID_ORDER_STATUSES:
                raise HTTPException(status_code=400,
                                    detail=f"invalid status {patch['status']!r}")
            order["status"] = patch["status"]
        if "energy" in patch:
            order["energy"] = patch["energy"]
        return order

    # ------------------------------------------------------------- trades

    @app.get("/trades")
    async def get_trades(market_id: str | None = Query(default=None)) -> list:
        return store.query_trades(market_id)

    @app.post("/trades-normalized", status_code=201)
    async def post_trades_normalized(
            payload: Union[list, dict] = Body(...)) -> list:
        trades = _as_list(payload)
        created = []
        for trade in trades:
            if not trade.get("market_id"):
                raise HTTPException(status_code=400,
                                    detail="market_id is required")
            created.append(store.add_trade(trade))
        logger.info("stored %d trade(s)", len(created))
        return created

    @app.patch("/trades/{trade_uuid}")
    async def patch_trade(trade_uuid: str, patch: dict = Body(...)) -> dict:
        # MOCK-ONLY endpoint, like PATCH /orders above.
        # TODO(confirm-with-supervisor): penalty output schema unresolved —
        # the PoC extends the trade `parameters` field in place.
        trade = store.trades.get(trade_uuid)
        if trade is None:
            raise HTTPException(status_code=404,
                                detail=f"trade {trade_uuid} not found")
        if "status" in patch:
            trade["status"] = patch["status"]
        if "parameters" in patch:
            merged = dict(trade.get("parameters") or {})
            merged.update(patch["parameters"] or {})
            trade["parameters"] = merged
        return trade

    # ------------------------------------------- measurements / forecasts

    @app.get("/measurements")
    async def get_measurements(
            area_uuid: str | None = Query(default=None),
            start_time: int | None = Query(default=None),
            end_time: int | None = Query(default=None),
            community_uuid: str | None = Query(default=None)) -> list:
        # `start_time`/`end_time` are inclusive on both ends, exactly as on
        # /orders, so one slot is asked for as [t, t + Δ - 1].
        # `community_uuid` is a MOCK-ONLY convenience filter: the production
        # route keys on area and time window alone.
        return store.query_measurements(community_uuid, area_uuid,
                                        start_time, end_time)

    @app.post("/measurements", status_code=201)
    async def post_measurements(payload: Union[list, dict] = Body(...)) -> list:
        measurements = _as_list(payload)
        created = []
        for m in measurements:
            for field in ("community_uuid", "area_uuid", "time_slot"):
                if m.get(field) is None:
                    raise HTTPException(status_code=400,
                                        detail=f"{field} is required")
            if not isinstance(m.get("energy_kwh"), (int, float)):
                raise HTTPException(status_code=400,
                                    detail="energy_kwh must be a number")
            # Upsert keyed by (community, area, slot) so demo re-runs with
            # corrected meter values overwrite instead of duplicating.
            created.append(store.upsert_measurement(m))
        logger.info("stored %d measurement(s)", len(created))
        return created

    @app.get("/forecasts")
    async def get_forecasts(
            area_uuid: str | None = Query(default=None),
            start_time: int | None = Query(default=None),
            end_time: int | None = Query(default=None),
            community_uuid: str | None = Query(default=None)) -> list:
        # Same filter semantics as /measurements above, so the Execution Node
        # queries both channels identically.
        return store.query_forecasts(community_uuid, area_uuid,
                                     start_time, end_time)

    @app.post("/forecasts", status_code=201)
    async def post_forecasts(payload: Union[list, dict] = Body(...)) -> list:
        """A *forecast*, not a meter reading.

        In production this channel carries `ForecastSchema`
        (gsy-decentralized-exchange main @ aa99ea2,
        offchain-primitives/src/db_api_schema/profiles.rs): what an area
        expected to deliver or consume in a slot. The mock provides it
        because the evaluation has to separate what *was* delivered from what
        *could have been* delivered — with only a meter reading the two are
        equal by construction and a successful withholder is invisible.
        """
        # SIGN CONVENTION: GSY's own E2E fixtures use consumption positive and
        # generation negative (`energy_kwh: -8.0` for a producer). This
        # artifact keeps kWh positive on both channels, as /measurements
        # already does and as the penalty arithmetic assumes. The divergence
        # is named as a limitation, not reconciled here.
        forecasts = _as_list(payload)
        created = []
        for f in forecasts:
            # DELIBERATE DIVERGENCE (issue #27): GSY's `ForecastSchema` carries
            # `community_uuid` but does not enforce it; the mock does. The
            # Execution Node queries this channel with a community filter, so a
            # row stored without one is present and invisible — and the failure
            # is not an error but the silent fallback `deliverable = actual`,
            # which makes withholding undetectable while the run looks normal.
            # A rejected write is worse than nothing only if nothing were the
            # alternative; here the alternative is a row no query can return.
            for field in ("community_uuid", "area_uuid", "time_slot"):
                if f.get(field) is None:
                    raise HTTPException(status_code=400,
                                        detail=f"{field} is required")
            if not isinstance(f.get("energy_kwh"), (int, float)):
                raise HTTPException(status_code=400,
                                    detail="energy_kwh must be a number")
            created.append(store.upsert_forecast(f))
        logger.info("stored %d forecast(s)", len(created))
        return created

    # --------------------------------------------------------- mock admin

    @app.post("/reset")
    async def reset() -> dict:
        # MOCK-ONLY: wipe all collections (tests / demo restarts).
        store.reset()
        logger.info("store reset")
        return {"status": "reset"}

    @app.get("/")
    async def root() -> dict:
        return {
            "service": "mock-offchain-db",
            "description": "In-memory mock of the GSY-DEX off-chain database",
            "collections": {
                "markets": len(store.markets),
                "orders": len(store.orders),
                "trades": len(store.trades),
                "measurements": len(store.measurements),
                "forecasts": len(store.forecasts),
            },
        }

    return app


app = create_app()
