"""Core AMMBA clearing algorithm (guide §4.4).

One call of :func:`run_clearing` executes a full market clearing cycle:
fetch open orders, aggregate, compute the sigmoid clearing price, anchor the
result on-chain, allocate (preferred pairs first, then pro-rata — see
:mod:`src.preferences`), write trade objects to the off-chain DB and update
order statuses.
"""

import asyncio
import logging

from src.config import Config, resolve_community, resolve_preferences
from src.contract import BaseContractClient, ContractError
from src.offchain_db import OffchainDBClient
from src.preferences import (GREY, MIXED, MultiplierResult,
                             apply_energy_type_multipliers,
                             apply_preference_allocation, bonus_scale)
from src.sigmoid import clamp_price, sigmoid_price, to_node_int
from src.trade_builder import build_all_trades, rehash_trade

logger = logging.getLogger("amm-clearing-node.clearing")

SUPPLY_LIMITED = "SUPPLY_LIMITED"
DEMAND_LIMITED = "DEMAND_LIMITED"
BALANCED = "BALANCED"

# Same tolerance as the Execution Node's determine_round_type: pro-rata
# allocation introduces float noise, so exact comparison would misclassify
# effectively balanced markets.
_EPSILON = 1e-9

_SIGMOID_PARAM_KEYS = ("k_upper", "k_lower", "theta", "steepness")

# The contract's revert when a market already carries an anchor. Matching on
# the reason string is what the chain offers: web3 surfaces it as text, and
# MockContractClient mirrors the same wording.
_ALREADY_CLEARED = "already cleared"


def round_type(total_supply_kwh: float, total_demand_kwh: float) -> str:
    if total_supply_kwh < total_demand_kwh - _EPSILON:
        return SUPPLY_LIMITED
    if total_supply_kwh > total_demand_kwh + _EPSILON:
        return DEMAND_LIMITED
    return BALANCED


def _allocation_summary(orders: list[dict], clearing_price: float, *,
                        multipliers: MultiplierResult,
                        is_seller_side: bool) -> list[dict]:
    """Per-order allocation rows.

    `value_ct` stays at the uniform clearing price (the settlement
    reference); `final_energy_rate` reports what the participant actually
    receives/pays after the energy-type multiplier."""
    summary = []
    for order in orders:
        requested = order["energy"]
        allocated = order["allocated_energy"]
        energy_type = (order.get("energy_type", "green") if is_seller_side
                       else MIXED)
        if not is_seller_side:
            final_rate = multipliers.buyer_final_ct_per_kwh
        elif energy_type == GREY:
            final_rate = multipliers.grey_final_ct_per_kwh
        else:
            final_rate = multipliers.green_final_ct_per_kwh
        summary.append({
            "name": order.get("created_by", "?"),
            "area_uuid": order.get("area_uuid"),
            "order_id": order.get("order_id"),
            "requested_kwh": round(requested, 6),
            "allocated_kwh": round(allocated, 6),
            "fill_rate": round(allocated / requested, 6) if requested else 0.0,
            "value_ct": round(allocated * clearing_price, 4),
            "energy_type": energy_type,
            "preference_matched": bool(order.get("preference_matched", False)),
            "final_energy_rate": round(final_rate, 6),
        })
    return summary


def _clearing_summary(*, market_id: str, community_uuid: str, time_slot: int,
                      clearing_price: float, total_supply_kwh: float,
                      total_demand_kwh: float, sigmoid_params: dict,
                      tx_hash: str | None, pool_id: str, producers: list[dict],
                      consumers: list[dict], trades: list[dict],
                      preferences: dict, **extra) -> dict:
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
        "preferences": preferences,
        "num_trades": len(trades),
        "trades": trades,
        **extra,
    }


def _preferences_from_trades(trades: list[dict], pool_id: str,
                             clearing_price: float) -> dict:
    """Rebuild the `preferences` summary block from stored trades.

    The fresh and the idempotent path must return the same shape, so every
    input this needs is written into trade `parameters` by the trade builder
    (run-level settings + per-trade pair/energy-type data)."""
    meta = ((trades[0].get("parameters") or {}).get("preferences") or {})
    seller_trades = [t for t in trades if t.get("buyer") == pool_id]
    buyer_trades = [t for t in trades if t.get("seller") == pool_id]

    def params(trade: dict) -> dict:
        return trade.get("parameters") or {}

    def allocated(trade: dict) -> float:
        return float(params(trade).get("selected_energy", 0.0))

    def final_rate(trade: dict) -> float:
        return float(params(trade).get("final_energy_rate", clearing_price))

    def first_rate(candidates: list[dict]) -> float:
        return final_rate(candidates[0]) if candidates else clearing_price

    green = [t for t in seller_trades if params(t).get("energy_type") != GREY]
    grey = [t for t in seller_trades if params(t).get("energy_type") == GREY]
    green_alloc = sum(allocated(t) for t in green)
    grey_alloc = sum(allocated(t) for t in grey)
    green_final = first_rate(green)
    grey_final = first_rate(grey)

    active = bool(meta.get("enabled") and meta.get("multipliers_enabled"))
    bonus_requested = green_alloc * clearing_price * float(
        meta.get("green_multiplier", 0.0) if active else 0.0)
    bonus_paid = green_alloc * (green_final - clearing_price)
    buyers_pay = sum(allocated(t) * final_rate(t) for t in buyer_trades)
    sellers_receive = sum(allocated(t) * final_rate(t) for t in seller_trades)

    multipliers = MultiplierResult(
        enabled=active,
        mode=meta.get("mode", "multiplicative"),
        sides=meta.get("sides", "seller"),
        green_alloc_kwh=green_alloc, grey_alloc_kwh=grey_alloc,
        levy_collected_ct=grey_alloc * (clearing_price - grey_final),
        bonus_requested_ct=bonus_requested, bonus_paid_ct=bonus_paid,
        scale=bonus_scale(bonus_paid, bonus_requested),
        green_final_ct_per_kwh=green_final, grey_final_ct_per_kwh=grey_final,
        buyer_final_ct_per_kwh=first_rate(buyer_trades),
        buyers_pay_ct=buyers_pay, sellers_receive_ct=sellers_receive,
        pool_surplus_ct=buyers_pay - sellers_receive)

    pairs = []
    for trade in seller_trades:
        pair = params(trade).get("preferences") or {}
        if not params(trade).get("preference_matched"):
            continue
        component = (trade.get("offer") or {}).get("offer_component") or {}
        pairs.append({"bid_area": pair.get("partner_area"),
                      "offer_area": component.get("area_uuid"),
                      "energy_kwh": round(float(pair.get("pair_kwh", 0.0)), 6)})
    pairs.sort(key=lambda p: (p["bid_area"] or "", p["offer_area"] or ""))

    return {
        "enabled": bool(meta.get("enabled", False)),
        "order": meta.get("order", "preferences_first"),
        "mutual_pairs": pairs,
        "pairs_rationed": bool(meta.get("pairs_rationed", False)),
        "multipliers": multipliers.as_dict(),
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
        trade_params = trade.get("parameters") or {}
        allocated = trade_params.get("selected_energy", 0.0)
        requested = allocated + residual
        return {
            "name": trade.get(counterparty_key, "?"),
            "area_uuid": component.get("area_uuid"),
            "order_id": trade.get(f"{component_key}_hash"),
            "requested_kwh": round(requested, 6),
            "allocated_kwh": round(allocated, 6),
            "fill_rate": round(allocated / requested, 6) if requested else 0.0,
            "value_ct": round(allocated * clearing_price, 4),
            "energy_type": trade_params.get("energy_type", "green"),
            "preference_matched": bool(
                trade_params.get("preference_matched", False)),
            "final_energy_rate": round(
                trade_params.get("final_energy_rate", clearing_price), 6),
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
        preferences=_preferences_from_trades(trades, pool_id, clearing_price),
        status="already_cleared",
        recovered_from_anchor=False,
        message="trades already exist for this market_id + time_slot; "
                "clearing is idempotent and was not re-run",
    )


async def _anchor_or_recover(chain: BaseContractClient, *, market_id: str,
                             community_uuid: str, time_slot: int,
                             total_supply_kwh: float, total_demand_kwh: float,
                             clearing_price: float) -> tuple[str, float, bool]:
    """Anchor the clearing on-chain, or continue from an anchor that exists.

    The failure this recovers from: `clearMarket` succeeded but `post_trades`
    then failed, leaving the slot with an anchor and no trades. A re-trigger
    finds no trades, clears again, and the contract reverts "already cleared"
    — so without this the endpoint answers 502 for that slot forever and it
    takes manual intervention to settle. Reading the stored anchor lets the
    trade write-back be retried against the original result.

    Only that one revert is recovered from. A bounds-check failure or an
    unauthorised caller still propagates and fails the run loudly.

    Returns `(tx_hash, clearing_price, recovered)`.
    """
    try:
        tx_hash = await chain.clear_market(
            market_id=market_id, community_uuid=community_uuid,
            time_slot=time_slot, total_supply_kwh=total_supply_kwh,
            total_demand_kwh=total_demand_kwh, clearing_price=clearing_price)
        return tx_hash, clearing_price, False
    except ContractError as exc:
        if _ALREADY_CLEARED not in str(exc):
            raise
        anchor = await chain.get_clearing_result(market_id)
        if anchor is None:
            # The chain rejected the clearing as a duplicate but has no result
            # to offer: nothing to recover from, so the original error stands.
            raise

    # Compared in the anchor's own precision, not as floats: the contract
    # stores the price scaled by 10,000, so reading it back yields 15.1472
    # where the clearing computed 15.147225. Both anchor the same on-chain
    # integer, and the recomputed value is the one every downstream stage
    # rebuilds its counterfactuals from — keeping it is what makes a
    # recovered slot settle identically to a first-time one.
    anchored_price = anchor["clearing_price"]
    price = clearing_price
    if to_node_int(anchored_price) != to_node_int(clearing_price):
        # A real divergence: the order book changed between the anchor and
        # this retry. The committed anchor wins, at the precision it holds.
        logger.warning("market %s is anchored at %.6f ct/kWh but this run "
                       "computed %.6f — settling on the anchored price",
                       market_id, anchored_price, clearing_price)
        price = anchored_price
    logger.warning("market %s already anchored on-chain (tx %s) but carries "
                   "no trades — recovering the anchor and retrying the trade "
                   "write-back", market_id, anchor["tx_hash"])
    return anchor["tx_hash"], price, True


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


async def run_clearing(trigger: dict, cfg: Config, db: OffchainDBClient,
                       chain: BaseContractClient) -> dict:
    market_id = trigger["market_id"]
    community_uuid = trigger["community_uuid"]
    time_slot = int(trigger["time_slot"])
    community = resolve_community(cfg, community_uuid,
                                  trigger.get("sigmoid_params"))
    preferences = resolve_preferences(cfg, trigger.get("preference_params"))
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
    # `end_time` is inclusive in the off-chain DB filter, and
    # `time_slot + time_slot_sec` is already the *next* slot — subtract one
    # second so a slot never pulls in the following slot's orders.
    orders = await db.get_orders(market_id, start_time=time_slot,
                                 end_time=time_slot + cfg.time_slot_sec - 1)
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
    # Round once, here: this is the settlement reference written into every
    # trade object, so the multiplier economics must be derived from exactly
    # the same number the idempotent re-trigger path reads back.
    clearing_price = round(clearing_price, 6)
    logger.info("ratio=%.6f -> clearing price %.6f ct/kWh", ratio, clearing_price)

    # ---- Step 4: record on-chain (must succeed before trades are written,
    #      guide §4.7) ----------------------------------------------------
    tx_hash, clearing_price, recovered = await _anchor_or_recover(
        chain, market_id=market_id, community_uuid=community_uuid,
        time_slot=time_slot, total_supply_kwh=total_supply_kwh,
        total_demand_kwh=total_demand_kwh, clearing_price=clearing_price)

    # ---- Steps 5+6: allocation (preferred pairs + pro-rata residual) ----
    # `apply_preference_allocation` owns the whole allocation so a single
    # function carries the balance invariant (guide §4.4 steps 5 and 6).
    traded_quantity = min(total_supply_kwh, total_demand_kwh)
    allocation = apply_preference_allocation(
        bids, offers, traded_quantity=traded_quantity,
        total_supply_kwh=total_supply_kwh, total_demand_kwh=total_demand_kwh,
        cfg=preferences)
    bids, offers = allocation.bids, allocation.offers

    # ---- Step 7: build trade objects ------------------------------------
    preference_meta = {**preferences.as_dict(),
                       "pairs_rationed": allocation.pairs_rationed}
    trades = build_all_trades(
        bids, offers, market_id=market_id, time_slot=time_slot,
        clearing_price=clearing_price, tx_hash=tx_hash, pool_id=pool_id,
        community=community, total_supply_kwh=total_supply_kwh,
        total_demand_kwh=total_demand_kwh, preference_meta=preference_meta)

    # ---- Step 6.3: ex-post energy-type multipliers ----------------------
    # Extends trade `parameters` (never overwriting `energy_rate`), so the
    # trade ids have to be refreshed afterwards.
    multipliers = apply_energy_type_multipliers(
        trades, clearing_price=clearing_price, pool_id=pool_id,
        cfg=preferences)
    trades = [rehash_trade(trade) for trade in trades]

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
        time_slot=time_slot, clearing_price=clearing_price,
        total_supply_kwh=total_supply_kwh, total_demand_kwh=total_demand_kwh,
        sigmoid_params={key: getattr(community, key)
                        for key in _SIGMOID_PARAM_KEYS},
        tx_hash=tx_hash, pool_id=pool_id,
        producers=_allocation_summary(offers, clearing_price,
                                      multipliers=multipliers,
                                      is_seller_side=True),
        consumers=_allocation_summary(bids, clearing_price,
                                      multipliers=multipliers,
                                      is_seller_side=False),
        trades=trades,
        preferences={**allocation.as_dict(preferences),
                     "multipliers": multipliers.as_dict()},
        status="cleared",
        # Explicit, so a caller can tell a first clearing from one that
        # continued an anchor left behind by a failed write-back.
        recovered_from_anchor=recovered,
        community_name=trigger.get("community_name"),
        blockchain_mode=chain.mode,
    )
