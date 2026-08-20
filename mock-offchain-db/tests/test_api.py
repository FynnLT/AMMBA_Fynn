import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.store import InMemoryStore, generate_market_id


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(InMemoryStore()))


def test_health_check(client):
    resp = client.get("/health_check")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_market_id_is_generated_when_missing(client):
    resp = client.post("/market", json={
        "community_uuid": "communityid_1",
        "community_name": "Community 1",
        "time_slot": 1750000500,
    })
    assert resp.status_code == 201
    market = resp.json()
    # blake2b-256 of "spot" + LE u64 timestamp, 0x-prefixed
    assert market["market_id"] == generate_market_id(1750000500)
    assert market["market_id"].startswith("0x")
    assert len(market["market_id"]) == 66

    got = client.get("/market", params={"market_id": market["market_id"]})
    assert got.status_code == 200
    assert got.json()["community_uuid"] == "communityid_1"


def test_market_create_is_idempotent(client):
    body = {"community_uuid": "c1", "time_slot": 1000}
    first = client.post("/market", json=body).json()
    second = client.post("/market", json=dict(body)).json()
    assert first["market_id"] == second["market_id"]

    listed = client.get("/community-market",
                        params={"community_uuid": "c1"}).json()
    assert len(listed) == 1


def test_unknown_market_returns_404(client):
    assert client.get("/market", params={"market_id": "nope"}).status_code == 404


def test_orders_roundtrip_and_time_filter(client):
    orders = [
        {"order_type": "Bid", "created_by": "A", "market_id": "m1",
         "area_uuid": "a1", "time_slot": 900, "energy": 1.5, "energy_rate": 28.5},
        {"order_type": "Offer", "created_by": "B", "market_id": "m1",
         "area_uuid": "a2", "time_slot": 900, "energy": 2.0, "energy_rate": 8.0},
        {"order_type": "Bid", "created_by": "C", "market_id": "m1",
         "area_uuid": "a3", "time_slot": 1800, "energy": 1.0, "energy_rate": 28.5},
    ]
    resp = client.post("/orders-normalized", json=orders)
    assert resp.status_code == 201
    created = resp.json()
    assert all(o["order_id"].startswith("0x") for o in created)
    assert all(o["status"] == "Open" for o in created)

    in_window = client.get("/orders", params={
        "market_id": "m1", "start_time": 900, "end_time": 1799}).json()
    assert len(in_window) == 2
    assert {o["created_by"] for o in in_window} == {"A", "B"}


def test_identical_orders_get_distinct_ids(client):
    order = {"order_type": "Bid", "created_by": "A", "market_id": "m1",
             "area_uuid": "a1", "time_slot": 900, "energy": 1.0,
             "energy_rate": 28.5, "creation_time": 5}
    created = client.post("/orders-normalized",
                          json=[order, dict(order)]).json()
    assert created[0]["order_id"] != created[1]["order_id"]


def test_order_validation(client):
    bad_type = {"order_type": "Buy", "market_id": "m1", "energy": 1.0}
    assert client.post("/orders-normalized", json=bad_type).status_code == 400
    bad_energy = {"order_type": "Bid", "market_id": "m1", "energy": 0}
    assert client.post("/orders-normalized", json=bad_energy).status_code == 400


def test_preference_fields_are_stored_opaquely(client):
    order = {"order_type": "Offer", "created_by": "PV A", "market_id": "m1",
             "area_uuid": "area-pv-a", "time_slot": 900, "energy": 5.0,
             "energy_rate": 8.0,
             "attributes": {"energy_type": "grey"},
             "requirements": {"preferred_partner": "area-house-1"}}
    created = client.post("/orders-normalized", json=order).json()[0]
    assert created["attributes"] == {"energy_type": "grey"}
    assert created["requirements"] == {"preferred_partner": "area-house-1"}

    stored = client.get("/orders", params={"market_id": "m1"}).json()[0]
    assert stored["requirements"]["preferred_partner"] == "area-house-1"


def test_energy_type_is_validated_but_preferred_partner_is_not(client):
    base = {"order_type": "Offer", "created_by": "PV A", "market_id": "m1",
            "area_uuid": "area-pv-a", "time_slot": 900, "energy": 5.0,
            "energy_rate": 8.0}
    bad = dict(base, attributes={"energy_type": "Green"})
    assert client.post("/orders-normalized", json=bad).status_code == 400

    # An unknown partner area is accepted: the DB has no cross-order view and
    # the Clearing Node degrades unmatchable partners gracefully.
    unknown = dict(base, requirements={"preferred_partner": "area-nowhere"})
    assert client.post("/orders-normalized", json=unknown).status_code == 201
    # …and so is an order without any preference data at all.
    assert client.post("/orders-normalized", json=base).status_code == 201


def test_scale_orders_endpoint_is_not_implemented(client):
    assert client.post("/orders", json={}).status_code == 501


def test_patch_order_status(client):
    created = client.post("/orders-normalized", json={
        "order_type": "Bid", "created_by": "A", "market_id": "m1",
        "area_uuid": "a1", "time_slot": 900, "energy": 1.0,
        "energy_rate": 28.5}).json()[0]

    resp = client.patch(f"/orders/{created['order_id']}",
                        json={"status": "Executed"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "Executed"

    assert client.patch(f"/orders/{created['order_id']}",
                        json={"status": "Nonsense"}).status_code == 400
    assert client.patch("/orders/missing",
                        json={"status": "Executed"}).status_code == 404


def test_trades_roundtrip_and_patch(client):
    trade = {"trade_uuid": "t-1", "market_id": "m1", "status": "Settled",
             "buyer": "A", "seller": "POOL", "time_slot": 900,
             "parameters": {"selected_energy": 1.0}}
    created = client.post("/trades-normalized", json=[trade]).json()
    assert created[0]["_id"].startswith("0x")

    listed = client.get("/trades", params={"market_id": "m1"}).json()
    assert len(listed) == 1

    patched = client.patch("/trades/t-1", json={
        "status": "Executed",
        "parameters": {"penalty_ct": 12.5}}).json()
    assert patched["status"] == "Executed"
    # parameters are merged, not replaced
    assert patched["parameters"] == {"selected_energy": 1.0, "penalty_ct": 12.5}


def test_measurements_upsert_and_filter(client):
    m = {"community_uuid": "c1", "area_uuid": "a1", "time_slot": 900,
         "energy_kwh": 3.0}
    client.post("/asset_measurements", json=m)
    # same key again -> overwrite, not duplicate
    client.post("/asset_measurements", json={**m, "energy_kwh": 4.5})
    client.post("/asset_measurements", json={**m, "area_uuid": "a2",
                                             "energy_kwh": 1.0})

    all_for_community = client.get(
        "/asset_measurements", params={"community_uuid": "c1"}).json()
    assert len(all_for_community) == 2

    only_a1 = client.get("/asset_measurements", params={
        "community_uuid": "c1", "area_uuid": "a1"}).json()
    assert len(only_a1) == 1
    assert only_a1[0]["energy_kwh"] == 4.5


def test_reset(client):
    client.post("/market", json={"community_uuid": "c1", "time_slot": 900})
    client.post("/reset")
    assert client.get("/community-market",
                      params={"community_uuid": "c1"}).json() == []
