"""Transport behaviour of the off-chain DB client.

The clearing tests use a fake DB object, so the real client's own contract —
a caller may inject its transport, and only a self-created one is closed —
has no coverage there.
"""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.offchain_db import OffchainDBClient, OffchainDBError

BASE_URL = "http://offchain-db"


def stub_app() -> FastAPI:
    """Minimal stand-in for the mock DB.

    Binding the transport to `mock-offchain-db`'s own FastAPI app would be the
    realistic wiring, but every service ships its code as a top-level `src`
    package, so importing it here resolves to *this* service's `src`. The
    injection contract is transport-level, and this app exercises it as well.
    """
    app = FastAPI()

    @app.get("/health_check")
    async def health_check() -> dict:
        return {"status": "ok", "service": "stub"}

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

@pytest.mark.parametrize("status_code", [404, 405])
@pytest.mark.anyio
async def test_patch_without_a_route_is_a_no_op(status_code):
    """`PATCH /orders/{id}` exists only on the mock.

    The production off-chain storage registers no PATCH route, so a DB
    answering 404 or 405 must leave the clearing run intact rather than turn
    a status update into a 502.
    """
    app = stub_app()

    @app.patch("/orders/{order_id}")
    async def patch_order(order_id: str):
        return JSONResponse(status_code=status_code, content={"detail": "nope"})

    client = asgi_client(app)
    db = OffchainDBClient(BASE_URL, client=client)
    assert await db.update_order("0x01", status="Executed") is None
    await client.aclose()


@pytest.mark.anyio
async def test_other_patch_failures_still_raise():
    app = stub_app()

    @app.patch("/orders/{order_id}")
    async def patch_order(order_id: str):
        return JSONResponse(status_code=500, content={"detail": "boom"})

    client = asgi_client(app)
    db = OffchainDBClient(BASE_URL, client=client)
    with pytest.raises(OffchainDBError):
        await db.update_order("0x01", status="Executed")
    await client.aclose()
