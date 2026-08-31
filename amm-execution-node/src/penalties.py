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

from src.sigmoid import sigmoid_price

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
