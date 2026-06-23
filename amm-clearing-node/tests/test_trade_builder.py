import hashlib
import json

from src.config import CommunityParams
from src.trade_builder import (blake2b_hash, build_all_trades,
                               build_buyer_trade, build_seller_trade)

COMMUNITY = CommunityParams(k_upper=28.5, k_lower=8.0, theta=1.0, steepness=2.5)
TX_HASH = "0x" + "ab" * 32
POOL = "AMM_POOL_communityid_1"


def _bid(energy=4.0, allocated=4.0):
    return {
        "order_id": "0x" + "11" * 32, "order_type": "Bid", "status": "Open",
        "created_by": "Household 1", "area_uuid": "area_h1",
        "market_id": "m1", "time_slot": 900, "creation_time": 100,
        "energy": energy, "energy_rate": 28.5, "allocated_energy": allocated,
    }


def _offer(energy=5.0, allocated=4.0):
    return {
        "order_id": "0x" + "22" * 32, "order_type": "Offer", "status": "Open",
        "created_by": "Rooftop PV", "area_uuid": "area_pv",
        "market_id": "m1", "time_slot": 900, "creation_time": 100,
        "energy": energy, "energy_rate": 8.0, "allocated_energy": allocated,
    }


def _common():
    return dict(market_id="m1", time_slot=900, clearing_price=15.1472,
                tx_hash=TX_HASH, pool_id=POOL, community=COMMUNITY,
                total_supply_kwh=12.5, total_demand_kwh=10.0)


def test_blake2b_hash_matches_reference_implementation():
    data = {"b": 2, "a": 1}
    expected = "0x" + hashlib.blake2b(
        json.dumps(data, sort_keys=True).encode(), digest_size=32).hexdigest()
    assert blake2b_hash(data) == expected
    assert len(blake2b_hash(data)) == 2 + 64
    # key order must not matter (sorted-keys serialisation)
    assert blake2b_hash({"a": 1, "b": 2}) == expected


def test_buyer_trade_structure():
    trade = build_buyer_trade(_bid(), **_common())
    assert trade["buyer"] == "Household 1"
    assert trade["seller"] == POOL
    assert trade["status"] == "Settled"
    assert trade["bid_hash"] == "0x" + "11" * 32          # original order hash
    assert trade["offer_hash"] == TX_HASH                  # EVM anchor
    assert trade["bid"]["bid_component"]["energy"] == 4.0
    assert trade["bid"]["bid_component"]["energy_rate"] == 15.1472
    assert trade["offer"]["offer_component"]["area_uuid"] == POOL
    assert trade["_id"].startswith("0x") and len(trade["_id"]) == 66
    assert trade["residual_bid"] is None                   # fully filled
    assert trade["residual_offer"] is None


def test_seller_trade_structure_with_residual():
    trade = build_seller_trade(_offer(energy=5.0, allocated=4.0), **_common())
    assert trade["buyer"] == POOL
    assert trade["seller"] == "Rooftop PV"
    assert trade["offer_hash"] == "0x" + "22" * 32
    assert trade["bid_hash"] == TX_HASH
    # partial fill -> residual at the ORIGINAL order rate
    assert trade["residual_offer"] == {"energy": 1.0, "energy_rate": 8.0}
    assert trade["residual_bid"] is None


def test_partial_buyer_fill_sets_residual_bid():
    trade = build_buyer_trade(_bid(energy=4.0, allocated=3.2), **_common())
    assert trade["residual_bid"]["energy"] == 0.8
    assert trade["residual_bid"]["energy_rate"] == 28.5


def test_float_noise_does_not_create_residual():
    trade = build_buyer_trade(_bid(energy=4.0, allocated=4.0 - 1e-12),
                              **_common())
    assert trade["residual_bid"] is None


def test_parameters_make_execution_node_self_contained():
    trade = build_seller_trade(_offer(), **_common())
    params = trade["parameters"]
    assert params["selected_energy"] == 4.0
    assert params["energy_rate"] == 15.1472
    assert params["amm_tx_hash"] == TX_HASH
    # guide §5.4: sigmoid params + aggregates needed for counterfactuals
    assert params["theta"] == 1.0
    assert params["steepness"] == 2.5
    assert params["k_upper"] == 28.5
    assert params["k_lower"] == 8.0
    assert params["total_supply_kwh"] == 12.5
    assert params["total_demand_kwh"] == 10.0
    assert params["pool_id"] == POOL
    assert params["preference_matched"] is False
    assert params["trade_uuid"] == trade["trade_uuid"]


def test_id_is_hash_of_trade_contents():
    trade = build_buyer_trade(_bid(), **_common())
    body = {k: v for k, v in trade.items() if k != "_id"}
    assert trade["_id"] == blake2b_hash(body)


def test_one_trade_per_participant():
    bids = [_bid(), {**_bid(), "order_id": "0x" + "33" * 32,
                     "created_by": "Household 2"}]
    offers = [_offer()]
    trades = build_all_trades(bids, offers, **_common())
    assert len(trades) == 3  # len(bids) + len(offers)
    uuids = {t["trade_uuid"] for t in trades}
    assert len(uuids) == 3   # each trade has its own uuid
