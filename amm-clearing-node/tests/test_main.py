"""HTTP layer tests for the Clearing Node service."""

from fastapi.testclient import TestClient

import pytest

from src.config import CommunityParams, Config
from src.contract import ContractError, MockContractClient
from src.main import create_app
from src.offchain_db import OffchainDBError
from src.preferences import AllocationError

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


def test_trigger_accepts_preference_params():
    client = make_client(FakeOffchainDB(demand_limited_orders()))
    resp = client.post("/trigger-clearing", json={
        "market_id": MARKET, "community_uuid": "communityid_1",
        "time_slot": 900,
        "preference_params": {"order": "pro_rata_first", "mode": "additive"},
    })
    assert resp.status_code == 200
    preferences = resp.json()["preferences"]
    assert preferences["order"] == "pro_rata_first"
    assert preferences["multipliers"]["mode"] == "additive"


def test_unusable_preference_params_map_to_400():
    client = make_client(FakeOffchainDB(demand_limited_orders()))
    resp = client.post("/trigger-clearing", json={
        "market_id": MARKET, "community_uuid": "communityid_1",
        "time_slot": 900, "preference_params": {"mode": "exponential"},
    })
    assert resp.status_code == 400
    assert "exponential" in resp.json()["detail"]


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


class RecordingChainClient(MockContractClient):
    """Mock client that records every parameter verification it was asked
    for, so the startup check can be asserted on."""

    def __init__(self, error: ContractError | None = None) -> None:
        super().__init__()
        self.verified: list[tuple[str, CommunityParams]] = []
        self._error = error

    async def verify_community_params(self, community_uuid, local):
        self.verified.append((community_uuid, local))
        if self._error is not None:
            raise self._error


def test_startup_aborts_when_on_chain_params_differ():
    """A parameter mismatch must stop the service from coming up — no silent
    operation with parameters other than the contract's."""
    chain = RecordingChainClient(
        ContractError("on-chain community parameters for communityid_1 "
                      "differ from the local configuration"))
    app = create_app(cfg=Config(communities={"communityid_1": CommunityParams()}),
                     db=FakeOffchainDB(), chain=chain)

    with pytest.raises(ContractError, match="communityid_1"):
        with TestClient(app):
            pass


def test_startup_verifies_every_configured_community_once():
    communities = {"communityid_1": CommunityParams(theta=1.0),
                   "communityid_2": CommunityParams(theta=1.5)}
    chain = RecordingChainClient()
    app = create_app(cfg=Config(communities=communities),
                     db=FakeOffchainDB(), chain=chain)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert [uuid for uuid, _ in chain.verified] == list(communities)
    assert dict(chain.verified) == communities


def test_startup_without_configured_communities_verifies_nothing():
    """`Config()` has no communities — the default mock-mode startup path."""
    chain = RecordingChainClient()
    app = create_app(cfg=Config(), db=FakeOffchainDB(), chain=chain)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert chain.verified == []


def test_allocation_invariant_answers_a_defined_error(monkeypatch):
    """The conservation invariant must not escape as an unhandled error."""
    def broken(*_args, **_kwargs):
        raise AllocationError("allocation must balance: bids=4.0 offers=10.0")

    monkeypatch.setattr("src.clearing.apply_preference_allocation", broken)
    client = make_client(FakeOffchainDB(demand_limited_orders()))

    resp = client.post("/trigger-clearing", json={
        "market_id": MARKET, "community_uuid": "communityid_1",
        "time_slot": 900})

    assert resp.status_code == 500
    assert "allocation must balance" in resp.json()["detail"]
