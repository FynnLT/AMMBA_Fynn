"""Post-delivery penalty formulas (guide §5.3).

Three penalty types from the simulation specification:

* Seller shortfall   Φ_t,j = K_sho * max(0, traded - delivered - η),
                     K_sho = γ * K_upper (γ > 1)
* Seller externality (supply-limited rounds): a seller who withheld supply
  raised the price for all buyers — VCG-style counterfactual.
* Buyer externality  (demand-limited rounds): a buyer who underreported
  demand lowered the price for all sellers.

`actual_delivered` and the buyer's `actual_demand` are the meter reading;
`actual_deliverable` comes from the forecast channel (`GET /forecasts`) and
falls back to the meter reading where a slot carries no forecast.
"""

import logging

from src.sigmoid import sigmoid_price

logger = logging.getLogger("amm-execution-node.penalties")

SUPPLY_LIMITED = "SUPPLY_LIMITED"
DEMAND_LIMITED = "DEMAND_LIMITED"
BALANCED = "BALANCED"

_EPSILON = 1e-9


def determine_round_type(total_supply_kwh: float, total_demand_kwh: float) -> str:
    if total_supply_kwh < total_demand_kwh - _EPSILON:
        return SUPPLY_LIMITED
    if total_supply_kwh > total_demand_kwh + _EPSILON:
        return DEMAND_LIMITED
    return BALANCED


def seller_shortfall_penalty(energy_traded: float, actual_delivered: float,
                             k_upper: float, gamma: float, eta: float) -> dict:
    """Φ_t,j = K_sho * max(0, energy_traded - actual_delivered - η)."""
    k_sho = gamma * k_upper
    shortfall = max(0.0, energy_traded - actual_delivered - eta)
    return {
        "shortfall_kwh": round(shortfall, 6),
        "penalty_ct": round(k_sho * shortfall, 6),
        "k_sho_ct_per_kwh": round(k_sho, 6),
    }


def seller_externality_penalty(energy_traded: float, actual_deliverable: float,
                               eta: float, total_supply_kwh: float,
                               total_demand_kwh: float,
                               traded_quantity_kwh: float,
                               clearing_price: float, sigmoid: dict) -> dict | None:
    """Supply-limited rounds: W_sell = max(0, deliverable - traded - η).

    Counterfactual: had the withheld energy been offered, supply would have
    been higher and the clearing price lower for everyone.
    """
    withheld = max(0.0, actual_deliverable - energy_traded - eta)
    if withheld <= _EPSILON:
        return None
    counterfactual_supply = total_supply_kwh + withheld
    counterfactual_ratio = counterfactual_supply / total_demand_kwh
    p_counterfactual = sigmoid_price(
        counterfactual_ratio, sigmoid["k_upper"], sigmoid["k_lower"],
        sigmoid["theta"], sigmoid["steepness"])
    # TODO(confirm-with-supervisor): the spec multiplies by the round's
    # traded_quantity (market-wide volume), not the agent's own volume —
    # confirm this VCG interpretation.
    penalty = max(0.0, (clearing_price - p_counterfactual) * traded_quantity_kwh)
    return {
        "withheld_kwh": round(withheld, 6),
        "counterfactual_ratio": round(counterfactual_ratio, 6),
        "counterfactual_price_ct_per_kwh": round(p_counterfactual, 6),
        "penalty_ct": round(penalty, 6),
    }


def buyer_externality_penalty(reported_demand: float, actual_demand: float,
                              total_supply_kwh: float, total_demand_kwh: float,
                              traded_quantity_kwh: float,
                              clearing_price: float, sigmoid: dict) -> dict | None:
    """Demand-limited rounds: W_buy = max(0, actual_demand - reported_demand).

    Counterfactual: had the true demand been reported, demand would have been
    higher and the clearing price higher for all sellers. (No η tolerance for
    buyers in the spec.)
    """
    underreported = max(0.0, actual_demand - reported_demand)
    if underreported <= _EPSILON:
        return None
    counterfactual_demand = total_demand_kwh + underreported
    counterfactual_ratio = total_supply_kwh / counterfactual_demand
    p_counterfactual = sigmoid_price(
        counterfactual_ratio, sigmoid["k_upper"], sigmoid["k_lower"],
        sigmoid["theta"], sigmoid["steepness"])
    penalty = max(0.0, (p_counterfactual - clearing_price) * traded_quantity_kwh)
    return {
        "underreported_kwh": round(underreported, 6),
        "counterfactual_ratio": round(counterfactual_ratio, 6),
        "counterfactual_price_ct_per_kwh": round(p_counterfactual, 6),
        "penalty_ct": round(penalty, 6),
    }


def redistribution(rows: list[dict], clearing_price: float,
                   traded_quantity_kwh: float) -> dict:
    """Proportional compensation of the price damage — computed, never paid.

    Budget balance is a property of the rule, not an assumption:
    summed over one market side, Σ_i (p − p_cf)·q_i = (p − p_cf)·Q_t is
    exactly the externality penalty. `budget_balance_ct` is reported so a
    round where it does *not* hold — several simultaneous deviators, whose
    counterfactuals are not additive — is visible in the output instead of
    being averaged away.

    The harmed set is the deviators' *counterparty* side: a seller who
    withholds raises the price, so the buyers are harmed and the remaining
    sellers profit; a buyer who underreports depresses it, so the sellers are
    harmed. Deviators themselves are excluded (D-43), but only on the
    externality axis — a seller carrying a `shortfall_penalty_ct` is still
    compensated (D-61): a shortfall does not manipulate the price and is
    already penalised on its own axis.

    `harmed_side_kwh` is the invariant that guards the harmed set itself:
    every trade has the community pool on one side, so each market side trades
    `traded_quantity_kwh` in total and the sum over the harmed role must match
    it. With no deviator there is no harmed side, and the field is 0.0.

    No transfer is performed anywhere. D-25 keeps the payout mechanics out of
    the artifact; D-42 moves only the arithmetic into the node.
    """
    empty = {"rule": "proportional", "computed_only": True,
             "harmed_side": None, "harmed_side_kwh": 0.0,
             "penalty_pool_ct": 0.0,
             "compensated_ct": 0.0, "budget_balance_ct": 0.0,
             "excluded_deviators": [], "rows": []}

    deviators = [r for r in rows if r.get("externality_penalty_ct", 0.0) > 0.0]
    if not deviators:
        return empty
    penalty_pool = sum(r["externality_penalty_ct"] for r in deviators)

    # The harmed side is the opposite of the deviators' own side. Externality
    # penalties are applied on one market side per round, so the deviators
    # agree on their role; the first one names the side.
    deviator_role = deviators[0]["role"]
    harmed_side = "buyer" if deviator_role == "seller" else "seller"
    excluded = [r.get("area_uuid") for r in deviators]

    # Each market side trades Q_t in total (the pool is the counterparty of
    # every trade), so this sum must equal `traded_quantity_kwh`. A result
    # near 2·Q_t means the harmed set was taken across both sides — the defect
    # this function was rewritten to avoid, and one that `budget_balance_ct`
    # cannot reveal, because the pool is distributed in full either way.
    # Summed over the whole role, before the deviator filter: the invariant is
    # about the market side, not about who is eligible for compensation.
    harmed_side_kwh = sum(float(r.get("traded_kwh", 0.0)) for r in rows
                          if r.get("role") == harmed_side)
    if (abs(harmed_side_kwh - traded_quantity_kwh)
            > 1e-6 * max(1.0, traded_quantity_kwh)):
        # Reported and logged, never raised: a violated invariant belongs in
        # the output and not in an AssertionError that answers HTTP 500
        # instead of a defined state (issue #14).
        logger.warning("harmed side (%s) trades %.6f kWh but the round's "
                       "traded quantity is %.6f — the harmed set may span "
                       "both market sides", harmed_side, harmed_side_kwh,
                       traded_quantity_kwh)

    harmed = [r for r in rows
              if r.get("role") == harmed_side
              and abs(float(r.get("traded_kwh", 0.0))) > _EPSILON
              and r.get("externality_penalty_ct", 0.0) <= 0.0]

    # Damage attribution: each deviator moved the price by |p − p_cf|, and a
    # harmed participant carries that per-kWh damage on his own volume.
    damage_per_kwh = 0.0
    for deviator in deviators:
        p_cf = deviator.get("counterfactual_price_ct_per_kwh")
        if p_cf is None:
            continue
        damage_per_kwh += abs(clearing_price - float(p_cf))

    damages = [(r, damage_per_kwh * float(r["traded_kwh"])) for r in harmed]
    total_damage = sum(damage for _row, damage in damages)
    if total_damage <= _EPSILON:
        return {**empty, "harmed_side": harmed_side,
                "harmed_side_kwh": round(harmed_side_kwh, 6),
                "penalty_pool_ct": round(penalty_pool, 6),
                "budget_balance_ct": round(penalty_pool, 6),
                "excluded_deviators": excluded}

    out_rows = []
    compensated = 0.0
    for row, damage in damages:
        compensation = penalty_pool * damage / total_damage
        compensated += compensation
        out_rows.append({"area_uuid": row.get("area_uuid"),
                         "damage_ct": round(damage, 6),
                         "compensation_ct": round(compensation, 6)})

    return {
        "rule": "proportional",
        "computed_only": True,
        "harmed_side": harmed_side,
        "harmed_side_kwh": round(harmed_side_kwh, 6),
        "penalty_pool_ct": round(penalty_pool, 6),
        "compensated_ct": round(compensated, 6),
        "budget_balance_ct": round(penalty_pool - compensated, 6),
        "excluded_deviators": excluded,
        "rows": out_rows,
    }
