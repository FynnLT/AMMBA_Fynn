"""Transport behaviour of the off-chain DB client.

The execution tests use a fake DB object, so the real client's own contract —
a caller may inject its transport, and only a self-created one is closed —
has no coverage there.
"""

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.offchain_db import OffchainDBClient, OffchainDBError

BASE_URL = "http://offchain-db"
MARKET = "0x" + "aa" * 32


def stub_app() -> FastAPI:
    """Minimal stand-in for the mock DB.

    Binding the transport to `mock-offchain-db`'s own FastAPI app would be the
    realistic wiring, but every service ships its code as a top-level `src`
    package, so importing it here resolves to *this* service's `src`. The
    injection contract is transport-level, and this app exercises it as well.
    """
    app = FastAPI()
    app.state.params = {}

    @app.get("/health_check")
    async def health_check() -> dict:
        return {"status": "ok", "service": "stub"}

    @app.get("/measurements")
    async def measurements(request: Request) -> list:
        app.state.params["measurements"] = dict(request.query_params)
        return []

    @app.get("/forecasts")
    async def forecasts(request: Request) -> list:
        app.state.params["forecasts"] = dict(request.query_params)
        return []

    @app.get("/trades")
    async def trades() -> list:
        # A DB whose server-side market_id filter does nothing, as the
        # production one does not (its filter line is commented out).
        return [{"trade_uuid": "t-mine", "market_id": MARKET},
                {"trade_uuid": "t-foreign", "market_id": "0x" + "cc" * 32}]

    return app


def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url=BASE_URL)


@pytest.mark.anyio
async def test_request_goes_through_the_injected_client():
    client = asgi_client(stub_app())
    db = OffchainDBClient(BASE_URL, client=client)
    # Answered by the ASGI app in-process; no port is opened.
    assert (await db.health_check())["service"] == "stub"
    await client.aclose()


@pytest.mark.anyio
async def test_close_leaves_an_injected_client_open():
    # The harness shares one ASGI-backed client across both services, so
    # either service's lifespan teardown must leave the other one usable.
    client = asgi_client(stub_app())
    db = OffchainDBClient(BASE_URL, client=client)
    await db.close()
    assert client.is_closed is False
    await client.aclose()


@pytest.mark.anyio
async def test_close_closes_a_self_created_client():
    db = OffchainDBClient(BASE_URL)
    await db.close()
    assert db._client.is_closed is True


# ------------------------------------------------------ GSY API alignment

@pytest.mark.anyio
async def test_slot_window_is_sent_on_both_channels():
    """One delivery slot is asked for as the inclusive bounds [t, t + Δ - 1].

    `time_slot + time_slot_sec` is already the next slot, so asking up to it
    would pull that slot's rows in — the same off-by-one the order window
    fixed. Asserted on the query string, not on the (empty) result.
    """
    app = stub_app()
    client = asgi_client(app)
    db = OffchainDBClient(BASE_URL, client=client)

    await db.get_measurements("communityid_1", 1787580000, 900)
    await db.get_forecasts("communityid_1", 1787580000, 900)

    window = {"community_uuid": "communityid_1",
              "start_time": "1787580000", "end_time": "1787580899"}
    assert app.state.params["measurements"] == window
    assert app.state.params["forecasts"] == window
    await client.aclose()


@pytest.mark.anyio
async def test_foreign_trades_are_dropped_client_side():
    client = asgi_client(stub_app())
    db = OffchainDBClient(BASE_URL, client=client)
    trades = await db.get_trades(MARKET)
    assert [t["trade_uuid"] for t in trades] == ["t-mine"]
    await client.aclose()


@pytest.mark.parametrize("status_code", [404, 405])
@pytest.mark.anyio
async def test_patch_without_a_route_is_a_no_op(status_code):
    # The production API registers no PATCH route; a missing status update
    # must never abort a run.
    app = stub_app()

    @app.patch("/trades/{trade_uuid}")
    async def patch_trade(trade_uuid: str):
        return JSONResponse(status_code=status_code, content={"detail": "nope"})

    client = asgi_client(app)
    db = OffchainDBClient(BASE_URL, client=client)
    assert await db.update_trade("t-1", status="Executed") is None
    await client.aclose()


@pytest.mark.anyio
async def test_other_patch_failures_still_raise():
    app = stub_app()

    @app.patch("/trades/{trade_uuid}")
    async def patch_trade(trade_uuid: str):
        return JSONResponse(status_code=500, content={"detail": "boom"})

    client = asgi_client(app)
    db = OffchainDBClient(BASE_URL, client=client)
    with pytest.raises(OffchainDBError):
        await db.update_trade("t-1", status="Executed")
    await client.aclose()
