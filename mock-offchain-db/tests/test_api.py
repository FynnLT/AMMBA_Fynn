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
    # blake2b-256 of "Spot" + BE u64 timestamp, 0x-prefixed
    assert market["market_id"] == generate_market_id(1750000500)
    assert market["market_id"].startswith("0x")
    assert len(market["market_id"]) == 66

    got = client.get("/market", params={"market_id": market["market_id"]})
    assert got.status_code == 200
    assert got.json()["community_uuid"] == "communityid_1"


def test_market_id_matches_the_gsy_preimage():
    """Pinned against the Market Orchestrator's own derivation.

    `blake2b(32, "Spot" ++ delivery_timestamp.to_be_bytes())`
    (gsy-decentralized-exchange main @ aa99ea2,
    gsy-market-orchestrator/src/orchestrator.rs:85). A drifting preimage
    produces an id of the right shape and length, so only a fixed vector
    catches it.
    """
    assert generate_market_id(1787580000) == (
        "0x7d53efb681b8f8b96bbe2dfe257a886fe2193cefdc6181c48722c11723ab4749")


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
    client.post("/measurements", json=m)
    # same key again -> overwrite, not duplicate
    client.post("/measurements", json={**m, "energy_kwh": 4.5})
    client.post("/measurements", json={**m, "area_uuid": "a2",
                                       "energy_kwh": 1.0})

    all_for_community = client.get(
        "/measurements", params={"community_uuid": "c1"}).json()
    assert len(all_for_community) == 2

    only_a1 = client.get("/measurements", params={
        "community_uuid": "c1", "area_uuid": "a1"}).json()
    assert len(only_a1) == 1
    assert only_a1[0]["energy_kwh"] == 4.5


def slot_rows(area: str) -> list[dict]:
    """One row per slot boundary around the 900 s slot starting at 900."""
    return [{"community_uuid": "c1", "area_uuid": area, "time_slot": slot,
             "energy_kwh": float(slot)}
            for slot in (899, 900, 1799, 1800)]


@pytest.mark.parametrize("route", ["/measurements", "/forecasts"])
def test_slot_window_is_inclusive_on_both_ends(client, route):
    # The Execution Node asks for [t, t + Δ - 1]; the row at t + Δ belongs to
    # the next slot and must stay out, exactly as on /orders.
    for row in slot_rows("a1"):
        assert client.post(route, json=row).status_code == 201

    in_window = client.get(route, params={
        "community_uuid": "c1", "start_time": 900, "end_time": 1799}).json()
    assert [r["time_slot"] for r in in_window] == [900, 1799]


def test_forecast_roundtrip_and_validation(client):
    forecast = {"community_uuid": "c1", "area_uuid": "area-pv-a",
                "time_slot": 900, "energy_kwh": 5.0, "confidence": 0.8}
    resp = client.post("/forecasts", json=forecast)
    assert resp.status_code == 201

    stored = client.get("/forecasts", params={
        "community_uuid": "c1", "area_uuid": "area-pv-a"}).json()
    assert len(stored) == 1
    # Positive kWh for a producer: the artifact keeps its own sign convention
    # rather than GSY's (generation negative).
    assert stored[0]["energy_kwh"] == 5.0
    assert stored[0]["confidence"] == 0.8

    # same key again -> upsert, like the measurement channel
    client.post("/forecasts", json={**forecast, "energy_kwh": 6.0})
    assert client.get("/forecasts", params={
        "community_uuid": "c1"}).json()[0]["energy_kwh"] == 6.0

    bad = dict(forecast, energy_kwh="lots")
    assert client.post("/forecasts", json=bad).status_code == 400
    del bad["area_uuid"]
    assert client.post("/forecasts", json=bad).status_code == 400


def test_forecasts_and_measurements_are_separate_channels(client):
    row = {"community_uuid": "c1", "area_uuid": "a1", "time_slot": 900}
    client.post("/measurements", json={**row, "energy_kwh": 3.0})
    client.post("/forecasts", json={**row, "energy_kwh": 5.0})

    assert client.get("/measurements", params={
        "community_uuid": "c1"}).json()[0]["energy_kwh"] == 3.0
    assert client.get("/forecasts", params={
        "community_uuid": "c1"}).json()[0]["energy_kwh"] == 5.0


def test_reset(client):
    client.post("/market", json={"community_uuid": "c1", "time_slot": 900})
    client.post("/reset")
    assert client.get("/community-market",
                      params={"community_uuid": "c1"}).json() == []


def test_forecast_without_community_uuid_is_rejected(client):
    """Issue #27: a row no query can return is worse than a rejected write.

    The Execution Node queries this channel with a community filter, so a
    forecast stored without a `community_uuid` is present and invisible — and
    the failure is not an error but the silent fallback `deliverable =
    actual`: `W_sell = 0` for every seller, withholding undetectable, and the
    run comes back empty while looking entirely normal.
    """
    forecast = {"area_uuid": "area-pv-a", "time_slot": 900, "energy_kwh": 5.0}

    resp = client.post("/forecasts", json=forecast)
    assert resp.status_code == 400
    assert "community_uuid" in resp.json()["detail"]

    # nothing was stored on the way out
    assert client.get("/forecasts").json() == []


def test_forecast_with_community_uuid_is_returned_by_a_filtered_get(client):
    forecast = {"community_uuid": "c1", "area_uuid": "area-pv-a",
                "time_slot": 900, "energy_kwh": 5.0}
    assert client.post("/forecasts", json=forecast).status_code == 201

    stored = client.get("/forecasts", params={"community_uuid": "c1"}).json()
    assert len(stored) == 1
    assert stored[0]["area_uuid"] == "area-pv-a"
    assert stored[0]["energy_kwh"] == 5.0
