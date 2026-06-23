"""Trade object construction and blake2b hashing (guide §4.4 step 7).

In the AMM every buyer trades with the community pool and every seller trades
with the pool, so a cleared market produces len(bids) + len(offers) trade
objects (one per participant):

    Trade A:  buyer  -> pool   (one per bid)
    Trade B:  pool   -> seller (one per offer)
"""

import hashlib
import json
import time
from uuid import uuid4

# 1e-9 kWh: float-noise threshold below which a residual is "fully matched"
RESIDUAL_EPSILON = 1e-9

# Energies/prices in trade objects are rounded to 6 decimals — comfortably
# finer than the on-chain resolution of 1/NODE_FLOAT_SCALING_FACTOR (1e-4).
_ROUND = 6


def blake2b_hash(data: dict) -> str:
    """blake2b-256 hex hash of a JSON-serialised dict, sorted keys
    (matches the GSY-DEX order/trade id convention)."""
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return "0x" + hashlib.blake2b(encoded, digest_size=32).hexdigest()


def _residual(total_energy: float, allocated: float, energy_rate: float) -> dict | None:
    # TODO(confirm-with-supervisor): partial-fill residual policy (guide §7.8)
    # — keep order Open with reduced energy, or Executed + re-post remainder?
    # The PoC records the residual on the trade object only.
    remainder = total_energy - allocated
    if remainder > RESIDUAL_EPSILON:
        return {"energy": round(remainder, _ROUND), "energy_rate": energy_rate}
    return None


def _parameters(allocated: float, clearing_price: float, trade_uuid: str,
                tx_hash: str, community, pool_id: str,
                total_supply_kwh: float, total_demand_kwh: float) -> dict:
    return {
        "selected_energy": round(allocated, _ROUND),
        "energy_rate": round(clearing_price, _ROUND),
        "trade_uuid": trade_uuid,
        "amm_tx_hash": tx_hash,
        "theta": community.theta,
        "steepness": community.steepness,
        # k_upper/k_lower are not in the guide §2 trade schema but §5.4
        # requires them in `parameters` so the Execution Node can run
        # counterfactual price calculations from trade objects alone.
        "k_upper": community.k_upper,
        "k_lower": community.k_lower,
        "total_supply_kwh": round(total_supply_kwh, _ROUND),
        "total_demand_kwh": round(total_demand_kwh, _ROUND),
        # Lets the Execution Node identify the pool side of each trade
        # without configuration.
        "pool_id": pool_id,
        # TODO(phase2): set when preference matching is implemented.
        "preference_matched": False,
    }


def build_buyer_trade(bid: dict, *, market_id: str, time_slot: int,
                      clearing_price: float, tx_hash: str, pool_id: str,
                      community, total_supply_kwh: float,
                      total_demand_kwh: float) -> dict:
    """Trade A: buyer -> pool."""
    trade_uuid = str(uuid4())
    now = int(time.time())
    allocated = round(bid["allocated_energy"], _ROUND)
    price = round(clearing_price, _ROUND)

    trade = {
        "trade_uuid": trade_uuid,
        "status": "Settled",
        "buyer": bid["created_by"],
        "seller": pool_id,
        "market_id": market_id,
        "time_slot": time_slot,
        "creation_time": now,
        "bid": {
            "buyer": bid["created_by"],
            "nonce": bid.get("nonce", 1),
            "bid_component": {
                "area_uuid": bid["area_uuid"],
                "market_id": bid["market_id"],
                "time_slot": bid["time_slot"],
                "creation_time": bid["creation_time"],
                "energy": allocated,
                "energy_rate": price,
            },
        },
        # order_id IS the blake2b hash of the original order in GSY-DEX
        "bid_hash": bid["order_id"],
        "offer": {
            "seller": pool_id,
            "offer_component": {
                "area_uuid": pool_id,
                "market_id": market_id,
                "time_slot": time_slot,
                "creation_time": now,
                "energy": allocated,
                "energy_rate": price,
            },
        },
        # EVM contract tx hash anchors the pool side of the trade
        "offer_hash": tx_hash,
        "residual_bid": _residual(bid["energy"], allocated, bid["energy_rate"]),
        "residual_offer": None,
        "parameters": _parameters(allocated, price, trade_uuid, tx_hash,
                                  community, pool_id, total_supply_kwh,
                                  total_demand_kwh),
    }
    trade["_id"] = blake2b_hash(trade)
    return trade


def build_seller_trade(offer: dict, *, market_id: str, time_slot: int,
                       clearing_price: float, tx_hash: str, pool_id: str,
                       community, total_supply_kwh: float,
                       total_demand_kwh: float) -> dict:
    """Trade B: pool -> seller."""
    trade_uuid = str(uuid4())
    now = int(time.time())
    allocated = round(offer["allocated_energy"], _ROUND)
    price = round(clearing_price, _ROUND)

    trade = {
        "trade_uuid": trade_uuid,
        "status": "Settled",
        "buyer": pool_id,
        "seller": offer["created_by"],
        "market_id": market_id,
        "time_slot": time_slot,
        "creation_time": now,
        "offer": {
            "seller": offer["created_by"],
            "offer_component": {
                "area_uuid": offer["area_uuid"],
                "market_id": offer["market_id"],
                "time_slot": offer["time_slot"],
                "creation_time": offer["creation_time"],
                "energy": allocated,
                "energy_rate": price,
            },
        },
        "offer_hash": offer["order_id"],
        "bid": {
            "buyer": pool_id,
            "nonce": 1,
            "bid_component": {
                "area_uuid": pool_id,
                "market_id": market_id,
                "time_slot": time_slot,
                "creation_time": now,
                "energy": allocated,
                "energy_rate": price,
            },
        },
        "bid_hash": tx_hash,
        "residual_offer": _residual(offer["energy"], allocated,
                                    offer["energy_rate"]),
        "residual_bid": None,
        "parameters": _parameters(allocated, price, trade_uuid, tx_hash,
                                  community, pool_id, total_supply_kwh,
                                  total_demand_kwh),
    }
    trade["_id"] = blake2b_hash(trade)
    return trade


def build_all_trades(bids: list[dict], offers: list[dict], *, market_id: str,
                     time_slot: int, clearing_price: float, tx_hash: str,
                     pool_id: str, community, total_supply_kwh: float,
                     total_demand_kwh: float) -> list[dict]:
    common = dict(market_id=market_id, time_slot=time_slot,
                  clearing_price=clearing_price, tx_hash=tx_hash,
                  pool_id=pool_id, community=community,
                  total_supply_kwh=total_supply_kwh,
                  total_demand_kwh=total_demand_kwh)
    trades = [build_buyer_trade(bid, **common) for bid in bids]
    trades += [build_seller_trade(offer, **common) for offer in offers]
    return trades
