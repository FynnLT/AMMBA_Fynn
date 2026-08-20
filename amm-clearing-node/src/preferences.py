"""User preference mechanisms (guide §4.4 step 6, §6).

Two independent mechanisms live here:

1. **Preferred trading partner priority** — an order may nominate *exactly
   one* preferred counterparty. Where the nomination is mutual (buyer A
   names seller B *and* seller B names buyer A) the pair is served first at
   the clearing price; everything left over is split pro-rata over the
   remaining volumes. `apply_preference_allocation` owns the *whole*
   allocation, pairs and residual alike, so a single function carries the
   balance invariant of §3.5 (see `_check_balance`).

2. **Energy-type multipliers** — a green bonus funded by a grey levy,
   applied ex post to the trade objects (`apply_energy_type_multipliers`).

**Partners are identified by `area_uuid`, never by `created_by`.**
`created_by` is a free-text display name in the demo UI and is not
guaranteed to be unique; `area_uuid` is the canonical participant identity
used everywhere else in the GSY-DEX schema (order components, trade
components, asset measurements).

Both `requirements` and `attributes` are optional in the GSY-DEX
`offchain-primitives` order schema, and both are parsed defensively here: a
missing key, `null`, a wrong type or an `area_uuid` that is not a
counterparty in this market all degrade to "no preference" / "green" with a
`logger.warning`. Malformed order data must never abort a clearing run.

The module is pure computation: it receives order/trade lists plus a
`PreferenceConfig` and returns data — no HTTP, no DB access.
"""

import logging
from dataclasses import dataclass

from src.config import PreferenceConfig

logger = logging.getLogger("amm-clearing-node.preferences")

# Same tolerance as clearing.round_type: allocation arithmetic introduces
# float noise, so exact comparisons would misfire.
EPSILON = 1e-9

# Energies/prices are rounded like everywhere else in the trade objects.
_ROUND = 6

# Above this a pool surplus is a real property of the formulation rather than
# the rounding dust that settling rates at `_ROUND` decimals always leaves.
_SURPLUS_LOG_THRESHOLD_CT = 1e-4

GREEN = "green"
GREY = "grey"
#: Buyers are served from the uniform pool, so their trades carry no single
#: energy source — they are settled against the round's green/grey mix.
MIXED = "mixed"

ENERGY_TYPES = (GREEN, GREY)


# --------------------------------------------------------- defensive parsing

def parse_energy_type(order: dict) -> str:
    """`attributes.energy_type` of an order, defaulting to "green".

    Absent `attributes` is the documented backward-compatible case and stays
    silent; anything present but unusable is warned about and degrades to
    "green" (the neutral choice: it attracts no levy).
    """
    attributes = order.get("attributes")
    if attributes is None:
        return GREEN
    if not isinstance(attributes, dict):
        logger.warning("order %s has non-object `attributes` (%r) — treated "
                       "as %s", order.get("order_id"), attributes, GREEN)
        return GREEN
    if "energy_type" not in attributes:
        return GREEN
    energy_type = attributes["energy_type"]
    if energy_type in ENERGY_TYPES:
        return energy_type
    logger.warning("order %s has unknown attributes.energy_type %r (expected "
                   "one of %s) — treated as %s", order.get("order_id"),
                   energy_type, list(ENERGY_TYPES), GREEN)
    return GREEN


def parse_preferred_partner(order: dict,
                            counterparty_areas: set[str]) -> str | None:
    """`requirements.preferred_partner` of an order, as an `area_uuid`.

    Returns None when no usable preference is expressed. `counterparty_areas`
    are the `area_uuid`s on the *opposite* side of this market: a partner
    that is unknown (or on the same side) cannot be matched, which is worth a
    warning because it is almost always a configuration mistake.
    """
    requirements = order.get("requirements")
    if requirements is None:
        return None
    if not isinstance(requirements, dict):
        logger.warning("order %s has non-object `requirements` (%r) — no "
                       "preference", order.get("order_id"), requirements)
        return None
    partner = requirements.get("preferred_partner")
    if partner is None:
        return None  # explicit "no preference"
    if not isinstance(partner, str) or not partner:
        logger.warning("order %s has non-string requirements."
                       "preferred_partner (%r) — no preference",
                       order.get("order_id"), partner)
        return None
    if partner not in counterparty_areas:
        logger.warning("order %s prefers area %r, which is not a counterparty "
                       "in this market — no preference",
                       order.get("order_id"), partner)
        return None
    return partner


# ------------------------------------------------------------------ results

@dataclass(frozen=True)
class MutualPair:
    """One mutually preferred (bid, offer) pair and the volume it was served
    preferentially. `energy_kwh` is 0.0 under `order == "pro_rata_first"`,
    where pairs are detected and reported but get no priority volume."""

    bid_area: str
    offer_area: str
    energy_kwh: float

    def as_dict(self) -> dict:
        return {"bid_area": self.bid_area, "offer_area": self.offer_area,
                "energy_kwh": round(self.energy_kwh, _ROUND)}


@dataclass(frozen=True)
class PreferenceResult:
    """Outcome of the allocation step: the mutated order books, the matched
    pairs and the per-pair quantities (carried by `MutualPair.energy_kwh`)."""

    bids: list[dict]
    offers: list[dict]
    pairs: list[MutualPair]
    pairs_rationed: bool = False

    @property
    def preferential_kwh(self) -> float:
        return sum(pair.energy_kwh for pair in self.pairs)

    def as_dict(self, cfg: PreferenceConfig) -> dict:
        return {
            "enabled": cfg.enabled,
            "order": cfg.order,
            "mutual_pairs": [pair.as_dict() for pair in self.pairs],
            "pairs_rationed": self.pairs_rationed,
        }


@dataclass(frozen=True)
class MultiplierResult:
    """Round-level economics of the energy-type multipliers.

    `pool_surplus_ct` is deliberately reported rather than absorbed: in the
    multiplicative formulation an over-collecting levy leaves the pool with
    the difference (buyers pay `p`, sellers receive less in total). That
    zero-sum violation is a property of the formulation and an evaluation
    result, not a bug to be patched away.
    """

    enabled: bool
    mode: str
    sides: str
    green_alloc_kwh: float
    grey_alloc_kwh: float
    levy_collected_ct: float
    bonus_requested_ct: float
    bonus_paid_ct: float
    scale: float
    green_final_ct_per_kwh: float
    grey_final_ct_per_kwh: float
    buyer_final_ct_per_kwh: float
    buyers_pay_ct: float
    sellers_receive_ct: float
    pool_surplus_ct: float

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "sides": self.sides,
            "green_alloc_kwh": round(self.green_alloc_kwh, _ROUND),
            "grey_alloc_kwh": round(self.grey_alloc_kwh, _ROUND),
            "levy_collected_ct": round(self.levy_collected_ct, _ROUND),
            "bonus_requested_ct": round(self.bonus_requested_ct, _ROUND),
            "bonus_paid_ct": round(self.bonus_paid_ct, _ROUND),
            "scale": round(self.scale, _ROUND),
            "green_final_ct_per_kwh": round(self.green_final_ct_per_kwh, _ROUND),
            "grey_final_ct_per_kwh": round(self.grey_final_ct_per_kwh, _ROUND),
            "buyer_final_ct_per_kwh": round(self.buyer_final_ct_per_kwh, _ROUND),
            "buyers_pay_ct": round(self.buyers_pay_ct, _ROUND),
            "sellers_receive_ct": round(self.sellers_receive_ct, _ROUND),
            "pool_surplus_ct": round(self.pool_surplus_ct, _ROUND),
        }


# ------------------------------------------------------------- partner match

def _mutual_pairs(bids: list[dict], offers: list[dict]) -> list[tuple[int, int]]:
    """Indices of mutually preferred (bid, offer) pairs.

    With one preferred partner per order every *order* belongs to at most one
    pair, so no matching optimisation is needed. An `area_uuid` may still post
    several orders, so candidates are sorted by
    `(bid_area, offer_area, bid_order_id, offer_order_id)` and consumed
    greedily — that keeps the result deterministic regardless of the order in
    which the off-chain DB returned the book.
    """
    bid_areas = {b.get("area_uuid") for b in bids}
    offer_areas = {o.get("area_uuid") for o in offers}
    bid_partner = [parse_preferred_partner(b, offer_areas) for b in bids]
    offer_partner = [parse_preferred_partner(o, bid_areas) for o in offers]

    candidates = []
    for bid_idx, bid in enumerate(bids):
        partner = bid_partner[bid_idx]
        if partner is None:
            continue
        for offer_idx, offer in enumerate(offers):
            if offer.get("area_uuid") != partner:
                continue
            if offer_partner[offer_idx] != bid.get("area_uuid"):
                continue
            candidates.append((str(bid.get("area_uuid")),
                               str(offer.get("area_uuid")),
                               str(bid.get("order_id") or ""),
                               str(offer.get("order_id") or ""),
                               bid_idx, offer_idx))
    candidates.sort()

    used_bids: set[int] = set()
    used_offers: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for *_sort_key, bid_idx, offer_idx in candidates:
        if bid_idx in used_bids or offer_idx in used_offers:
            continue
        used_bids.add(bid_idx)
        used_offers.add(offer_idx)
        pairs.append((bid_idx, offer_idx))
    return pairs


def _distribute_residual(orders: list[dict], residual: float) -> None:
    """Split `residual` over the orders' still-unfilled volumes, pro rata."""
    remainders = [order["energy"] - order["allocated_energy"]
                  for order in orders]
    rest = sum(remainders)
    if rest <= EPSILON:
        # Everything was already served by the pairs; then the residual must
        # be ~0 as well, otherwise the traded quantity was inconsistent.
        if residual > EPSILON:
            logger.error("residual %.9f kWh cannot be distributed: no "
                         "unfilled volume left", residual)
        assert residual <= EPSILON, "residual without remaining volume"
        return
    for order, remainder in zip(orders, remainders):
        order["allocated_energy"] += remainder / rest * residual


def _clamp_allocations(orders: list[dict]) -> None:
    """Clip float noise into `0 <= allocated_energy <= energy` (§3.5)."""
    for order in orders:
        order["allocated_energy"] = min(max(order["allocated_energy"], 0.0),
                                        order["energy"])


def _check_balance(bids: list[dict], offers: list[dict],
                   traded_quantity: float) -> None:
    """§3.5 invariant, asserted in code and not only in the tests."""
    tolerance = 1e-9 * max(1.0, traded_quantity)
    bought = sum(b["allocated_energy"] for b in bids)
    sold = sum(o["allocated_energy"] for o in offers)
    ok = (abs(bought - traded_quantity) <= tolerance
          and abs(sold - traded_quantity) <= tolerance)
    if not ok:
        logger.error("allocation invariant violated: bids=%.9f offers=%.9f "
                     "traded=%.9f", bought, sold, traded_quantity)
    assert ok, (f"allocation must balance: bids={bought} offers={sold} "
                f"traded={traded_quantity}")


def apply_preference_allocation(
        bids: list[dict], offers: list[dict], *, traded_quantity: float,
        total_supply_kwh: float, total_demand_kwh: float,
        cfg: PreferenceConfig) -> PreferenceResult:
    """Allocate the traded quantity over the order books (guide §4.4 steps 5+6).

    This is the *only* place `allocated_energy` is written. With
    `order == "preferences_first"` mutually preferred pairs are served first
    (rationed proportionally if they collectively demand more than the market
    clears), and the rest is split pro-rata over the remaining volumes; with
    `order == "pro_rata_first"` the split is purely pro-rata and the pairs
    only set `preference_matched` — that is the pre-Phase-2 baseline the
    evaluation compares against.

    Every matched order is flagged (`preference_matched`, plus the partner
    area and pair volume) so the trade builder can carry the routing
    information into `parameters`.
    """
    expected = min(total_supply_kwh, total_demand_kwh)
    if abs(traded_quantity - expected) > 1e-9 * max(1.0, expected):
        logger.error("traded_quantity %.9f does not match min(supply=%.9f, "
                     "demand=%.9f)", traded_quantity, total_supply_kwh,
                     total_demand_kwh)

    for order in bids + offers:
        order["allocated_energy"] = 0.0
        order["energy_type"] = parse_energy_type(order)
        order["preference_matched"] = False
        order["preference_partner_area"] = None
        order["preference_pair_kwh"] = 0.0

    pairs = _mutual_pairs(bids, offers) if cfg.enabled else []

    # Step 2/3: pair quantities, rationed if they oversubscribe the market.
    # NOTE: with exactly one preferred partner per order the rationing branch
    # cannot actually fire — every order is in at most one pair, so
    # Σ min(b, o) ≤ Σ b ≤ demand and ≤ Σ o ≤ supply, hence ≤ Q. It stays as
    # cheap insurance for the day the "one partner" rule is relaxed (multiple
    # partners / bipartite matching are out of scope), and
    # test_pairs_can_never_oversubscribe_the_cleared_quantity pins the
    # property that makes it dead code today.
    quantities = [min(bids[b]["energy"], offers[o]["energy"]) for b, o in pairs]
    pairs_rationed = False
    if cfg.order == "preferences_first":
        requested = sum(quantities)
        if requested > traded_quantity + EPSILON:
            factor = traded_quantity / requested
            quantities = [q * factor for q in quantities]
            pairs_rationed = True
            logger.info("preferred pairs oversubscribed (%.6f kWh requested, "
                        "%.6f kWh clearing) — rationed by %.6f",
                        requested, traded_quantity, factor)
    else:
        # pro_rata_first: pairs are reported for routing but get no priority.
        quantities = [0.0] * len(pairs)

    # Step 4: assign. Each pair reduces both sides by the same amount, so the
    # market balance is preserved by construction.
    matched = []
    for (bid_idx, offer_idx), quantity in zip(pairs, quantities):
        bid, offer = bids[bid_idx], offers[offer_idx]
        bid["allocated_energy"] += quantity
        offer["allocated_energy"] += quantity
        for order, partner_area in ((bid, offer.get("area_uuid")),
                                    (offer, bid.get("area_uuid"))):
            order["preference_matched"] = True
            order["preference_partner_area"] = partner_area
            order["preference_pair_kwh"] = quantity
        matched.append(MutualPair(bid_area=bid.get("area_uuid"),
                                  offer_area=offer.get("area_uuid"),
                                  energy_kwh=quantity))

    # Step 5: residual pro-rata over what is still unfilled on either side.
    residual = traded_quantity - sum(quantities)
    _distribute_residual(bids, residual)
    _distribute_residual(offers, residual)

    _clamp_allocations(bids)
    _clamp_allocations(offers)
    _check_balance(bids, offers, traded_quantity)

    if matched:
        logger.info("%d mutually preferred pair(s), %.6f kWh allocated "
                    "preferentially (%s)", len(matched), sum(quantities),
                    cfg.order)
    return PreferenceResult(bids=bids, offers=offers, pairs=matched,
                            pairs_rationed=pairs_rationed)


# ----------------------------------------------------- energy-type multipliers

def _multiplicative_rates(green_alloc: float, grey_alloc: float, price: float,
                          cfg: PreferenceConfig) -> tuple[float, float]:
    """Guide formulation. Returns (green_final, grey_final).

    The grey levy is the parameter; the green bonus is scaled down when the
    levy does not fully fund it. When the levy *over*-collects, the scaling
    stays at 1 and the difference remains with the pool — see
    `MultiplierResult`.
    """
    levy_eff = min(cfg.grey_levy, cfg.levy_cap)
    levy_collected = grey_alloc * price * levy_eff
    bonus_requested = green_alloc * price * cfg.green_multiplier
    if bonus_requested > EPSILON:
        scale = min(1.0, levy_collected / bonus_requested)
        if grey_alloc <= EPSILON:
            logger.warning("green bonus configured but no grey volume in this "
                           "round — no funding source, bonus is 0")
    else:
        scale = 1.0
    return price * (1 + cfg.green_multiplier * scale), price * (1 - levy_eff)


def _additive_rates(green_alloc: float, grey_alloc: float, price: float,
                    cfg: PreferenceConfig) -> tuple[float, float]:
    """InfoPaper formulation. Returns (green_final, grey_final).

    Bonus-driven and zero-sum by construction: the green bonus per kWh is the
    parameter and the grey levy per kWh follows from funding it completely.
    A levy above `levy_cap` is capped, and the bonus is scaled down
    proportionally so the zero-sum property survives the cap.

    TODO(verify-against-infopaper): this formulation is reconstructed from the
    worked example (48.75 / 51.25 / 60), not quoted from the paper. Verify
    before relying on it for evaluation results.
    """
    bonus_per_kwh = price * cfg.green_multiplier
    if grey_alloc <= EPSILON:
        if bonus_per_kwh > EPSILON and green_alloc > EPSILON:
            logger.warning("green bonus configured but no grey volume in this "
                           "round — no funding source, bonus is 0")
        return price, price
    if green_alloc <= EPSILON:
        return price, price

    levy_per_kwh = bonus_per_kwh * green_alloc / grey_alloc
    levy_cap_ct = price * cfg.levy_cap
    if levy_per_kwh > levy_cap_ct:
        logger.info("additive levy %.6f ct/kWh exceeds the cap %.6f ct/kWh — "
                    "capped, green bonus scaled down accordingly",
                    levy_per_kwh, levy_cap_ct)
        bonus_per_kwh *= levy_cap_ct / levy_per_kwh
        levy_per_kwh = levy_cap_ct
    return price + bonus_per_kwh, price - levy_per_kwh


def _is_pool_side(name: str, pool_id: str) -> bool:
    return bool(pool_id) and name == pool_id


def bonus_scale(bonus_paid_ct: float, bonus_requested_ct: float) -> float:
    """Fraction of the requested green bonus that was actually paid.

    Shared by the fresh run and the idempotent re-trigger path so both report
    the same number; clamped to [0, 1] because neither formulation pays out
    more than was requested."""
    if abs(bonus_requested_ct) <= EPSILON:
        return 1.0
    return min(1.0, max(0.0, bonus_paid_ct / bonus_requested_ct))


def apply_energy_type_multipliers(trades: list[dict], *, clearing_price: float,
                                  pool_id: str,
                                  cfg: PreferenceConfig) -> MultiplierResult:
    """Ex-post green bonus / grey levy on the trade objects (guide §4.4 6.3).

    **The uniform clearing price is never overwritten.** The Execution Node
    reads `parameters.energy_rate` as *the* clearing price for every
    counterfactual penalty calculation (guide §5.4), so this function only
    *adds* `final_energy_rate`, `energy_type` and `multiplier_applied`
    alongside it. The uniform price stays the settlement reference and the
    multiplier is a redistribution layer on top of it.
    TODO(confirm-with-supervisor): B-03 — whether the settlement rate the
    production system records is the uniform or the adjusted one.

    `sides == "seller"` adjusts only the seller trades (buyers pay `p`).
    `sides == "both"` also passes the pool's net position through to the
    buyers: since the pool is uniform, every buyer is served the same
    green/grey mix, so the buyer rate is that mix's volume-weighted final
    rate — which makes the round zero-sum even in multiplicative mode.
    TODO(confirm-with-supervisor): B-04.

    The trades are mutated in place; the round-level economics are returned.
    """
    price = clearing_price
    seller_trades = [t for t in trades
                     if _is_pool_side(t.get("buyer", ""), pool_id)]
    buyer_trades = [t for t in trades
                    if _is_pool_side(t.get("seller", ""), pool_id)]

    def allocated(trade: dict) -> float:
        return float((trade.get("parameters") or {}).get("selected_energy", 0.0))

    def energy_type(trade: dict) -> str:
        return (trade.get("parameters") or {}).get("energy_type", GREEN)

    green_alloc = sum(allocated(t) for t in seller_trades
                      if energy_type(t) != GREY)
    grey_alloc = sum(allocated(t) for t in seller_trades
                     if energy_type(t) == GREY)

    active = cfg.enabled and cfg.multipliers_enabled
    if not active:
        green_final = grey_final = price
    elif cfg.mode == "additive":
        green_final, grey_final = _additive_rates(
            green_alloc, grey_alloc, price, cfg)
    else:
        green_final, grey_final = _multiplicative_rates(
            green_alloc, grey_alloc, price, cfg)

    # A side with no volume has no rate: reporting the untouched clearing
    # price there keeps the block meaningful and lets the idempotent
    # re-trigger path — which can only read rates back off existing trades —
    # rebuild exactly the same numbers.
    if green_alloc <= EPSILON:
        green_final = price
    if grey_alloc <= EPSILON:
        grey_final = price

    # Round the rates before deriving the round economics: these are the
    # values written into the trade objects, so the re-trigger path, which
    # can only read them back rounded, must see the same sums.
    green_final = round(green_final, _ROUND)
    grey_final = round(grey_final, _ROUND)

    levy_collected = grey_alloc * (price - grey_final)
    bonus_requested = green_alloc * price * (cfg.green_multiplier
                                             if active else 0.0)
    bonus_paid = green_alloc * (green_final - price)
    sellers_receive = green_alloc * green_final + grey_alloc * grey_final

    traded = green_alloc + grey_alloc
    if active and cfg.sides == "both" and traded > EPSILON:
        buyer_final = round(sellers_receive / traded, _ROUND)
    else:
        buyer_final = price

    for trade in seller_trades:
        final = grey_final if energy_type(trade) == GREY else green_final
        _write_rate(trade, final, price)
    for trade in buyer_trades:
        _write_rate(trade, buyer_final, price)

    buyers_pay = sum(allocated(t) * float(t["parameters"]["final_energy_rate"])
                     for t in buyer_trades)
    result = MultiplierResult(
        enabled=active, mode=cfg.mode, sides=cfg.sides,
        green_alloc_kwh=green_alloc, grey_alloc_kwh=grey_alloc,
        levy_collected_ct=levy_collected, bonus_requested_ct=bonus_requested,
        bonus_paid_ct=bonus_paid,
        # Derived, not carried out of the rate functions: the same formula
        # then works for both modes and for the reconstruction path. Neither
        # formulation can pay out more than was requested, so the ratio is
        # clamped — without it, rounding the rates to 6 decimals would report
        # a "scale" a hair above 1.
        scale=bonus_scale(bonus_paid, bonus_requested),
        green_final_ct_per_kwh=green_final, grey_final_ct_per_kwh=grey_final,
        buyer_final_ct_per_kwh=buyer_final, buyers_pay_ct=buyers_pay,
        sellers_receive_ct=sellers_receive,
        pool_surplus_ct=buyers_pay - sellers_receive)

    # Threshold, not EPSILON: settling rates at 6 decimals leaves sub-0.0001 ct
    # dust on the pool either way, which is rounding rather than economics.
    if result.pool_surplus_ct > _SURPLUS_LOG_THRESHOLD_CT:
        logger.info("levy over-collects: pool retains %.6f ct (a property of "
                    "the %s formulation, reported not absorbed)",
                    result.pool_surplus_ct, cfg.mode)
    return result


def _write_rate(trade: dict, final_rate: float, price: float) -> None:
    parameters = trade.setdefault("parameters", {})
    parameters["final_energy_rate"] = round(final_rate, _ROUND)
    parameters["multiplier_applied"] = round(
        final_rate / price - 1.0, _ROUND) if price else 0.0
