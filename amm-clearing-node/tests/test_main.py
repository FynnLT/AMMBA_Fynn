"""HTTP layer tests for the Clearing Node service."""

from fastapi.testclient import TestClient

from src.config import Config
from src.contract import MockContractClient
from src.main import create_app
from src.offchain_db import OffchainDBError

from .test_clearing import FakeOffchainDB, demand_limited_orders, MARKET


def make_client(db) -> TestClient:
    app = create_app(cfg=Config(), db=db, chain=MockContractClient())
    return TestClient(app)


def test_health():
    client = make_client(FakeOffchainDB())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["blockchain_mode"] == "mock"


def test_trigger_clearing_returns_full_result():
    client = make_client(FakeOffchainDB(demand_limited_orders()))
    resp = client.post("/trigger-clearing", json={
        "market_id": MARKET,
        "community_uuid": "communityid_1",
        "time_slot": 900,
        "community_name": "Community 1",
        "sigmoid_params": {"k_upper": 28.5, "k_lower": 8.0,
                           "theta": 1.0, "steepness": 2.5},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cleared"
    assert body["num_trades"] == 6
    assert body["tx_hash"].startswith("0x")


def test_trigger_validates_payload():
    client = make_client(FakeOffchainDB())
    resp = client.post("/trigger-clearing", json={"market_id": "m1"})
    assert resp.status_code == 422  # missing community_uuid / time_slot


def test_offchain_db_failure_maps_to_502():
    class BrokenDB(FakeOffchainDB):
        async def get_trades(self, market_id):
            raise OffchainDBError("connection refused")

    client = make_client(BrokenDB())
    resp = client.post("/trigger-clearing", json={
        "market_id": MARKET, "community_uuid": "c1", "time_slot": 900})
    assert resp.status_code == 502
    assert "connection refused" in resp.json()["detail"]
