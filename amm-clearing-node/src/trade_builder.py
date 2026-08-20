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


def _parameters(order: dict, allocated: float, clearing_price: float,
                trade_uuid: str, tx_hash: str, community, pool_id: str,
                total_supply_kwh: float, total_demand_kwh: float,
                is_participant_seller: bool,
                preference_meta: dict | None) -> dict:
    """Trade `parameters` (guide §5.4: the Execution Node reconstructs the
    clearing context from these alone).

    The preference fields sit *next to* `energy_rate`, never on top of it:
    `energy_rate` remains the uniform clearing price the Execution Node uses
    for its counterfactual penalty calculations, `final_energy_rate` carries
    the energy-type multiplier. `preferences` repeats the run-level settings
    so the idempotent re-trigger path can rebuild the clearing summary from
    stored trades alone.
    """
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
        "preference_matched": bool(order.get("preference_matched", False)),
        # Buyers are served from the uniform pool, so their trade carries the
        # round's mix rather than a single source.
        "energy_type": (order.get("energy_type", "green")
                        if is_participant_seller else "mixed"),
        # Overwritten by apply_energy_type_multipliers; the uniform price is
        # the correct value whenever no multiplier applies.
        "final_energy_rate": round(clearing_price, _ROUND),
        "multiplier_applied": 0.0,
        "preferences": {
            **(preference_meta or {}),
            "partner_area": order.get("preference_partner_area"),
            "pair_kwh": round(float(order.get("preference_pair_kwh", 0.0)),
                              _ROUND),
        },
    }


def _component(area_uuid: str, market_id: str, time_slot: int,
               creation_time: int, energy: float, energy_rate: float) -> dict:
    return {
        "area_uuid": area_uuid,
        "market_id": market_id,
        "time_slot": time_slot,
        "creation_time": creation_time,
        "energy": energy,
        "energy_rate": energy_rate,
    }


def _build_trade(order: dict, *, order_side: str, market_id: str,
                 time_slot: int, clearing_price: float, tx_hash: str,
                 pool_id: str, community, total_supply_kwh: float,
                 total_demand_kwh: float,
                 preference_meta: dict | None = None) -> dict:
    """Build one trade between a participant's order and the pool.

    `order_side` is the participant's side: "bid" (Trade A: buyer -> pool)
    or "offer" (Trade B: pool -> seller). The pool always takes the opposite
    side, anchored by the EVM tx hash instead of an order hash.
    """
    trade_uuid = str(uuid4())
    now = int(time.time())
    allocated = round(order["allocated_energy"], _ROUND)
    price = round(clearing_price, _ROUND)

    participant_component = _component(
        order["area_uuid"], order["market_id"], order["time_slot"],
        order["creation_time"], allocated, price)
    pool_component = _component(
        pool_id, market_id, time_slot, now, allocated, price)
    residual = _residual(order["energy"], allocated, order["energy_rate"])

    if order_side == "bid":
        buyer, seller = order["created_by"], pool_id
        bid = {"buyer": order["created_by"], "nonce": order.get("nonce", 1),
               "bid_component": participant_component}
        offer = {"seller": pool_id, "offer_component": pool_component}
        # order_id IS the blake2b hash of the original order in GSY-DEX;
        # the EVM contract tx hash anchors the pool side of the trade.
        bid_hash, offer_hash = order["order_id"], tx_hash
        residual_bid, residual_offer = residual, None
    else:
        buyer, seller = pool_id, order["created_by"]
        offer = {"seller": order["created_by"],
                 "offer_component": participant_component}
        bid = {"buyer": pool_id, "nonce": 1, "bid_component": pool_component}
        offer_hash, bid_hash = order["order_id"], tx_hash
        residual_offer, residual_bid = residual, None

    trade = {
        "trade_uuid": trade_uuid,
        "status": "Settled",
        "buyer": buyer,
        "seller": seller,
        "market_id": market_id,
        "time_slot": time_slot,
        "creation_time": now,
        "bid": bid,
        "bid_hash": bid_hash,
        "offer": offer,
        "offer_hash": offer_hash,
        "residual_bid": residual_bid,
        "residual_offer": residual_offer,
        "parameters": _parameters(order, allocated, price, trade_uuid, tx_hash,
                                  community, pool_id, total_supply_kwh,
                                  total_demand_kwh,
                                  is_participant_seller=(order_side == "offer"),
                                  preference_meta=preference_meta),
    }
    return rehash_trade(trade)


def rehash_trade(trade: dict) -> dict:
    """(Re-)derive `_id` from the trade contents.

    The energy-type multipliers extend `parameters` after the trade object is
    built (guide §4.4 step 6.3 is explicitly ex post), so the id has to be
    refreshed afterwards to stay the hash of what is actually stored."""
    trade["_id"] = blake2b_hash({k: v for k, v in trade.items() if k != "_id"})
    return trade


def build_buyer_trade(bid: dict, **kwargs) -> dict:
    """Trade A: buyer -> pool."""
    return _build_trade(bid, order_side="bid", **kwargs)


def build_seller_trade(offer: dict, **kwargs) -> dict:
    """Trade B: pool -> seller."""
    return _build_trade(offer, order_side="offer", **kwargs)


def build_all_trades(bids: list[dict], offers: list[dict], *, market_id: str,
                     time_slot: int, clearing_price: float, tx_hash: str,
                     pool_id: str, community, total_supply_kwh: float,
                     total_demand_kwh: float,
                     preference_meta: dict | None = None) -> list[dict]:
    common = dict(market_id=market_id, time_slot=time_slot,
                  clearing_price=clearing_price, tx_hash=tx_hash,
                  pool_id=pool_id, community=community,
                  total_supply_kwh=total_supply_kwh,
                  total_demand_kwh=total_demand_kwh,
                  preference_meta=preference_meta)
    trades = [build_buyer_trade(bid, **common) for bid in bids]
    trades += [build_seller_trade(offer, **common) for offer in offers]
    return trades
