"""HTTP layer tests for the Execution Node service."""

from fastapi.testclient import TestClient

from src.config import Config
from src.main import create_app
from src.offchain_db import OffchainDBError

from .test_execution import (COMMUNITY, MARKET, SLOT, FakeOffchainDB,
                             demand_limited_db)


def make_client(db) -> TestClient:
    return TestClient(create_app(cfg=Config(), db=db))


def test_health():
    client = make_client(FakeOffchainDB())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["polling_enabled"] is False


def test_trigger_execution():
    db, _ = demand_limited_db({("area_bak", SLOT): 4.5})
    client = make_client(db)
    resp = client.post("/trigger-execution", json={
        "market_id": MARKET, "community_uuid": COMMUNITY, "time_slot": SLOT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "executed"
    assert body["round_type"] == "DEMAND_LIMITED"
    assert len(body["results"]) == 4


def test_trigger_validates_payload():
    client = make_client(FakeOffchainDB())
    assert client.post("/trigger-execution",
                       json={"market_id": "m"}).status_code == 422


def test_offchain_db_failure_maps_to_502():
    class BrokenDB(FakeOffchainDB):
        async def get_trades(self, market_id):
            raise OffchainDBError("boom")

    client = make_client(BrokenDB())
    resp = client.post("/trigger-execution", json={
        "market_id": MARKET, "community_uuid": COMMUNITY, "time_slot": SLOT})
    assert resp.status_code == 502
