"""User preference allocation — Phase 2 stub.

TODO(phase2): preference matching. The `requirements` / `attributes` fields
are not yet provided by GSY DEX (guide §2, §7.3); orders must be tolerated
without them (treated as None). Once available, implement (guide §4.4 step 6):

1. Preferred trading partner priority: identify mutual pairs (buyer A listed
   seller B AND seller B listed buyer A) and allocate their quantities first
   at the clearing price, up to their demanded/offered volumes.
2. Residual pool: distribute remaining unfilled volume pro-rata to all
   remaining participants.
3. Energy type multipliers (fetched from the off-chain DB, submitted by the
   community manager), applied ex-post per trade:
       green:  final_price = clearing_price * (1 + green_multiplier)
       grey:   final_price = clearing_price * (1 - grey_levy), capped at levy_cap
   with dynamic subsidy scaling if grey revenue is insufficient.
   TODO(phase2): multiplier storage + submission endpoint (guide §7.4).
"""

import logging

logger = logging.getLogger("amm-clearing-node.preferences")


def apply_preference_allocation(bids: list[dict],
                                offers: list[dict]) -> tuple[list[dict], list[dict], list]:
    """No-op stub: returns the order books unchanged and no preferred pairs.

    Reads `requirements` defensively so the call site already tolerates both
    present and absent preference data.
    """
    for order in bids + offers:
        if order.get("requirements") is not None:
            logger.info(
                "order %s carries `requirements` but preference matching is "
                "not implemented yet (Phase 2) — ignored",
                order.get("order_id"))
    preferred_pairs: list = []
    return bids, offers, preferred_pairs


def apply_energy_type_multipliers(trades: list[dict]) -> list[dict]:
    """No-op stub for ex-post energy-type price multipliers (Phase 2)."""
    return trades
