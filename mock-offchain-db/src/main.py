"""Mock GSY-DEX Off-Chain Database.

In-memory stand-in for the GSY-DEX off-chain storage service, which does not
exist yet. It implements the REST endpoints documented in the AMMBA
implementation guide (Section 2) so that the Clearing Node and Execution Node
are developed against the real interface and can be pointed at the production
DB later with zero changes to their core logic (only `OFFCHAIN_DB_URL`).

Mock-only conveniences, each clearly marked:

* ``POST /market`` generates ``market_id`` (blake2b-256 of "spot" +
  delivery timestamp bytes) when the caller does not provide one — in
  production the Market Orchestrator owns this id.
* ``PATCH /orders/{order_id}`` — TODO(confirm-with-supervisor): unknown
  whether the production orderbook exposes this (guide §7.2). In GSY-DEX,
  status updates are driven by the blockchain event listener instead.
* ``PATCH /trades/{trade_uuid}`` — TODO(confirm-with-supervisor): penalty
  output schema is unresolved (guide §7.5). The PoC extends the trade
  ``parameters`` field through this endpoint.
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
            created.append(store.add_order(order))
        logger.info("stored %d order(s)", len(created))
        return created

    @app.patch("/orders/{order_id}")
    async def patch_order(order_id: str, patch: dict = Body(...)) -> dict:
        # MOCK-ONLY endpoint.
        # TODO(confirm-with-supervisor): does a PATCH /orders/{id} endpoint
        # exist in the production orderbook service? (guide §7.2)
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
        # MOCK-ONLY endpoint.
        # TODO(confirm-with-supervisor): penalty output schema unresolved
        # (guide §7.5) — PoC extends the trade `parameters` field in place.
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

    # ------------------------------------------------------- measurements

    @app.get("/asset_measurements")
    async def get_asset_measurements(
            community_uuid: str = Query(...),
            area_uuid: str | None = Query(default=None),
            time_slot: int | None = Query(default=None)) -> list:
        # `area_uuid` is optional in the mock (returns the whole community
        # when omitted) — the documented production API requires it.
        return store.query_measurements(community_uuid, area_uuid, time_slot)

    @app.post("/asset_measurements", status_code=201)
    async def post_asset_measurements(
            payload: Union[list, dict] = Body(...)) -> list:
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
                "asset_measurements": len(store.measurements),
            },
        }

    return app


app = create_app()
