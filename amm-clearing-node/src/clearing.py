"""Core AMMBA clearing algorithm (guide §4.4).

One call of :func:`run_clearing` executes a full market clearing cycle:
fetch open orders, aggregate, compute the sigmoid clearing price, anchor the
result on-chain, allocate pro-rata, write trade objects to the off-chain DB
and update order statuses.
"""

import asyncio
import logging

from src.config import Config, resolve_community
from src.contract import BaseContractClient
from src.offchain_db import OffchainDBClient
from src.preferences import (apply_energy_type_multipliers,
                             apply_preference_allocation)
from src.sigmoid import clamp_price, sigmoid_price
from src.trade_builder import build_all_trades

logger = logging.getLogger("amm-clearing-node.clearing")

SUPPLY_LIMITED = "SUPPLY_LIMITED"
DEMAND_LIMITED = "DEMAND_LIMITED"
BALANCED = "BALANCED"

# Same tolerance as the Execution Node's determine_round_type: pro-rata
# allocation introduces float noise, so exact comparison would misclassify
# effectively balanced markets.
_EPSILON = 1e-9

_SIGMOID_PARAM_KEYS = ("k_upper", "k_lower", "theta", "steepness")


def round_type(total_supply_kwh: float, total_demand_kwh: float) -> str:
    if total_supply_kwh < total_demand_kwh - _EPSILON:
        return SUPPLY_LIMITED
    if total_supply_kwh > total_demand_kwh + _EPSILON:
        return DEMAND_LIMITED
    return BALANCED


def _allocation_summary(orders: list[dict], clearing_price: float) -> list[dict]:
    summary = []
    for order in orders:
        requested = order["energy"]
        allocated = order["allocated_energy"]
        summary.append({
            "name": order.get("created_by", "?"),
            "area_uuid": order.get("area_uuid"),
            "order_id": order.get("order_id"),
            "requested_kwh": round(requested, 6),
            "allocated_kwh": round(allocated, 6),
            "fill_rate": round(allocated / requested, 6) if requested else 0.0,
            "value_ct": round(allocated * clearing_price, 4),
        })
    return summary


def _clearing_summary(*, market_id: str, community_uuid: str, time_slot: int,
                      clearing_price: float, total_supply_kwh: float,
                      total_demand_kwh: float, sigmoid_params: dict,
                      tx_hash: str | None, pool_id: str, producers: list[dict],
                      consumers: list[dict], trades: list[dict],
                      **extra) -> dict:
    """Common shape of a clearing result (fresh run and idempotent re-trigger
    build the same summary; `extra` carries the path-specific keys)."""
    return {
        "market_id": market_id,
        "community_uuid": community_uuid,
        "time_slot": time_slot,
        "clearing_price_ct_per_kwh": clearing_price,
        "ratio": (round(total_supply_kwh / total_demand_kwh, 6)
                  if total_demand_kwh else None),
        "total_supply_kwh": round(total_supply_kwh, 6),
        "total_demand_kwh": round(total_demand_kwh, 6),
        "traded_quantity_kwh": round(min(total_supply_kwh, total_demand_kwh), 6),
        "round_type": round_type(total_supply_kwh, total_demand_kwh),
        "sigmoid_params": sigmoid_params,
        "tx_hash": tx_hash,
        "pool_id": pool_id,
        "allocations": {"producers": producers, "consumers": consumers},
        "num_trades": len(trades),
        "trades": trades,
        **extra,
    }


def _summary_from_existing_trades(trades: list[dict], market_id: str,
                                  community_uuid: str, time_slot: int) -> dict:
    """Rebuild a clearing summary from already-stored trades (idempotent
    re-trigger, guide §4.7)."""
    params = trades[0].get("parameters") or {}
    pool_id = params.get("pool_id", "")
    clearing_price = params.get("energy_rate", 0.0)

    def entry(trade: dict, component_key: str, counterparty_key: str) -> dict:
        component = (trade.get(component_key) or {}).get(
            f"{component_key}_component") or {}
        residual_key = ("residual_bid" if component_key == "bid"
                        else "residual_offer")
        residual = (trade.get(residual_key) or {}).get("energy", 0.0)
        allocated = (trade.get("parameters") or {}).get("selected_energy", 0.0)
        requested = allocated + residual
        return {
            "name": trade.get(counterparty_key, "?"),
            "area_uuid": component.get("area_uuid"),
            "order_id": trade.get(f"{component_key}_hash"),
            "requested_kwh": round(requested, 6),
            "allocated_kwh": round(allocated, 6),
            "fill_rate": round(allocated / requested, 6) if requested else 0.0,
            "value_ct": round(allocated * clearing_price, 4),
        }

    consumers = [entry(t, "bid", "buyer") for t in trades
                 if t.get("seller") == pool_id]
    producers = [entry(t, "offer", "seller") for t in trades
                 if t.get("buyer") == pool_id]
    return _clearing_summary(
        market_id=market_id, community_uuid=community_uuid,
        time_slot=time_slot, clearing_price=clearing_price,
        total_supply_kwh=params.get("total_supply_kwh", 0.0),
        total_demand_kwh=params.get("total_demand_kwh", 0.0),
        sigmoid_params={key: params.get(key) for key in _SIGMOID_PARAM_KEYS},
        tx_hash=params.get("amm_tx_hash"), pool_id=pool_id,
        producers=producers, consumers=consumers, trades=trades,
        status="already_cleared",
        message="trades already exist for this market_id + time_slot; "
                "clearing is idempotent and was not re-run",
    )


async def _expire_one_sided_market(db: OffchainDBClient, open_orders: list[dict],
                                   market_id: str, community_uuid: str,
                                   time_slot: int, total_supply_kwh: float,
                                   total_demand_kwh: float) -> dict:
    """One-sided market: no trade for this slot, expire everything
    (guide §4.4 step 2)."""
    await asyncio.gather(*(db.update_order(o["order_id"], status="Expired")
                           for o in open_orders))
    logger.info("no trade for market %s (one side empty); %d orders expired",
                market_id, len(open_orders))
    return {
        "status": "no_trade",
        "message": "supply or demand is zero — all open orders expired",
        "market_id": market_id,
        "community_uuid": community_uuid,
        "time_slot": time_slot,
        "total_supply_kwh": round(total_supply_kwh, 6),
        "total_demand_kwh": round(total_demand_kwh, 6),
        "num_orders_expired": len(open_orders),
        "trades": [],
    }


def _allocate_pro_rata(bids: list[dict], offers: list[dict],
                       total_supply_kwh: float,
                       total_demand_kwh: float) -> None:
    """Traded quantity + per-order `allocated_energy` (guide §4.4 step 5)."""
    traded_quantity = min(total_supply_kwh, total_demand_kwh)
    for offer in offers:
        offer["allocated_energy"] = (offer["energy"] / total_supply_kwh) * traded_quantity
    for bid in bids:
        bid["allocated_energy"] = (bid["energy"] / total_demand_kwh) * traded_quantity


async def run_clearing(trigger: dict, cfg: Config, db: OffchainDBClient,
                       chain: BaseContractClient) -> dict:
    market_id = trigger["market_id"]
    community_uuid = trigger["community_uuid"]
    time_slot = int(trigger["time_slot"])
    community = resolve_community(cfg, community_uuid,
                                  trigger.get("sigmoid_params"))
    pool_id = community.pool_for(community_uuid)

    logger.info("clearing triggered: market=%s community=%s slot=%s",
                market_id, community_uuid, time_slot)

    # ---- Idempotency check (guide §4.7) --------------------------------
    existing = [t for t in await db.get_trades(market_id)
                if t.get("time_slot") == time_slot]
    if existing:
        logger.warning("market %s already cleared (%d trades) — skipping",
                       market_id, len(existing))
        return _summary_from_existing_trades(existing, market_id,
                                             community_uuid, time_slot)

    # ---- Step 1: fetch open orders for the delivery window -------------
    orders = await db.get_orders(market_id, start_time=time_slot,
                                 end_time=time_slot + cfg.time_slot_sec)
    # Executed/Expired/Deleted orders are already settled or cancelled.
    open_orders = [o for o in orders if o.get("status") == "Open"]
    bids = [o for o in open_orders if o.get("order_type") == "Bid"]
    offers = [o for o in open_orders if o.get("order_type") == "Offer"]

    # ---- Step 2: aggregate ---------------------------------------------
    total_supply_kwh = sum(o["energy"] for o in offers)
    total_demand_kwh = sum(b["energy"] for b in bids)
    logger.info("aggregated: %d offers (%.4f kWh supply), %d bids (%.4f kWh demand)",
                len(offers), total_supply_kwh, len(bids), total_demand_kwh)

    if total_supply_kwh <= 0 or total_demand_kwh <= 0:
        return await _expire_one_sided_market(
            db, open_orders, market_id, community_uuid, time_slot,
            total_supply_kwh, total_demand_kwh)

    # ---- Step 3: clearing price ----------------------------------------
    ratio = total_supply_kwh / total_demand_kwh
    raw_price = sigmoid_price(ratio, community.k_upper, community.k_lower,
                              community.theta, community.steepness)
    clearing_price = clamp_price(raw_price, community.k_lower, community.k_upper)
    if clearing_price != raw_price:
        logger.warning("clearing price %.6f outside [%s, %s] — clamped to %.6f",
                       raw_price, community.k_lower, community.k_upper,
                       clearing_price)
    logger.info("ratio=%.6f -> clearing price %.6f ct/kWh", ratio, clearing_price)

    # ---- Step 4: record on-chain (must succeed before trades are written,
    #      guide §4.7) ----------------------------------------------------
    tx_hash = await chain.clear_market(
        market_id=market_id, community_uuid=community_uuid,
        time_slot=time_slot, total_supply_kwh=total_supply_kwh,
        total_demand_kwh=total_demand_kwh, clearing_price=clearing_price)

    # ---- Step 6 (Phase 2 stub): preference allocation ------------------
    # TODO(phase2): preference matching — runs before pro-rata once GSY DEX
    # embeds `requirements`/`attributes` in orders (guide §4.4 step 6).
    bids, offers, _preferred_pairs = apply_preference_allocation(bids, offers)

    # ---- Step 5: traded quantity + pro-rata allocations -----------------
    _allocate_pro_rata(bids, offers, total_supply_kwh, total_demand_kwh)

    # ---- Step 7: build trade objects ------------------------------------
    trades = build_all_trades(
        bids, offers, market_id=market_id, time_slot=time_slot,
        clearing_price=clearing_price, tx_hash=tx_hash, pool_id=pool_id,
        community=community, total_supply_kwh=total_supply_kwh,
        total_demand_kwh=total_demand_kwh)
    # TODO(phase2): ex-post energy-type multipliers (guide §4.4 step 6.3)
    trades = apply_energy_type_multipliers(trades)

    # ---- Step 8: persist trades, then update order statuses -------------
    await db.post_trades(trades)
    # TODO(confirm-with-supervisor): residual policy for partial fills
    # (guide §7.8). PoC marks every matched order Executed; the residual
    # amount is recorded on the trade object (residual_bid/-_offer).
    await asyncio.gather(*(db.update_order(o["order_id"], status="Executed")
                           for o in bids + offers))

    logger.info("market %s cleared: %d trades written, tx %s",
                market_id, len(trades), tx_hash)

    return _clearing_summary(
        market_id=market_id, community_uuid=community_uuid,
        time_slot=time_slot, clearing_price=round(clearing_price, 6),
        total_supply_kwh=total_supply_kwh, total_demand_kwh=total_demand_kwh,
        sigmoid_params={key: getattr(community, key)
                        for key in _SIGMOID_PARAM_KEYS},
        tx_hash=tx_hash, pool_id=pool_id,
        producers=_allocation_summary(offers, clearing_price),
        consumers=_allocation_summary(bids, clearing_price),
        trades=trades,
        status="cleared",
        community_name=trigger.get("community_name"),
        blockchain_mode=chain.mode,
    )
