"""The replay engine (A1): one slot, one participant's report changed.

A replay evaluates a recorded slot under a different report of **one**
participant, every other report held fixed -- the numerical check of
Chapter 3.4.1, in the energy-based model of Chapter 4:

    x = E / D_hat,  p = f(x),  Q = min(E, D_hat)
    s_j = E_j / E * Q   (sellers, pro rata)
    q_i = D_hat_i / D_hat * Q   (buyers, pro rata)

Seller j with deliverable surplus A, sold quantity s and settlement rate r:

    u = (r - K_lower) * s - Phi - phi
    Phi = seller_shortfall_penalty(s, delivered = min(s, A), ...)
    phi = seller_externality_penalty(s, deliverable = A, ...)  (supply-limited)

Buyer i with actual demand D, report D_hat_i, received quantity q:

    u = (K_upper - r) * min(q, D) - r * [q - D]^+ - phi
    phi = buyer_externality_penalty(reported = D_hat_i, actual = D, ...)
          (demand-limited)

The deadband is `eta_relative * s` for sellers and absent for buyers. The
regime after a deviation comes from `determine_round_type` on the deviated
aggregates; a deviation that switches it is computed all the same and
flagged, because Propositions 1 and 2 hold only within the regime.

**The artifact's functions are the reference.** `replay` calls
`sigmoid_price`, `determine_round_type` and the three penalty functions of
`amm-execution-node/src/`, loaded through `stack.EXE`. `replay_vec` is the
same arithmetic on numpy arrays, for the DR2 sweep; check R3 holds it to the
artifact functions, and nothing reads it without that check having passed.

In the energy-based model neither Q nor the allocation depends on the price,
which is what makes a band sensitivity a pure repricing of the same book.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HARNESS = Path(__file__).resolve().parent.parent / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import stack  # noqa: E402  (the path above has to be in place first)

_SIGMOID = stack.EXE["sigmoid"]
_PENALTIES = stack.EXE["penalties"]

sigmoid_price = _SIGMOID.sigmoid_price
determine_round_type = _PENALTIES.determine_round_type
seller_shortfall_penalty = _PENALTIES.seller_shortfall_penalty
seller_externality_penalty = _PENALTIES.seller_externality_penalty
buyer_externality_penalty = _PENALTIES.buyer_externality_penalty

SUPPLY_LIMITED = _PENALTIES.SUPPLY_LIMITED
DEMAND_LIMITED = _PENALTIES.DEMAND_LIMITED
BALANCED = _PENALTIES.BALANCED

#: The node's own tolerances, read rather than restated: the round-type band
#: and the "withheld <= eps -> no externality" cut share one constant there.
EPSILON = _PENALTIES._EPSILON
EXP_ARG_LIMIT = _SIGMOID._EXP_ARG_LIMIT
#: Penalties are reported at six decimals (`round(..., 6)` in penalties.py).
PENALTY_DECIMALS = 6

#: Integer codes for the vectorised regime, same order as a ratio axis.
SL, BAL, DL = -1, 0, 1
REGIME_CODE = {SUPPLY_LIMITED: SL, BALANCED: BAL, DEMAND_LIMITED: DL}
REGIME_NAME = {code: name for name, code in REGIME_CODE.items()}

SELLER, BUYER = "seller", "buyer"


@dataclass(frozen=True)
class Band:
    k_upper: float
    k_lower: float
    theta: float
    steepness: float

    @classmethod
    def from_dict(cls, sigmoid: dict) -> "Band":
        return cls(float(sigmoid["k_upper"]), float(sigmoid["k_lower"]),
                   float(sigmoid["theta"]), float(sigmoid["steepness"]))

    def as_sigmoid(self) -> dict:
        """The dict shape the penalty functions take."""
        return {"k_upper": self.k_upper, "k_lower": self.k_lower,
                "theta": self.theta, "steepness": self.steepness}

    def price(self, ratio: float) -> float:
        return sigmoid_price(ratio, self.k_upper, self.k_lower, self.theta,
                             self.steepness)

    def ratio_at_price(self, price: float) -> float:
        """The ratio at which f(x) = price, inside the open band."""
        span = self.k_upper - self.k_lower
        share = (self.k_upper - price) / span        # = 1 / (1 + e^a)
        return self.theta - np.log(1.0 / share - 1.0) / self.steepness


def price_derivative(ratio, band: Band):
    """f'(x), analytic: -(K_u - K_l) * B * e^a / (1 + e^a)^2, a = -B(x - theta).

    Zero where the artifact's sigmoid saturates (|a| > the exp limit), which
    is where it returns K_upper / K_lower flat.
    """
    ratio = np.asarray(ratio, dtype=float)
    arg = -band.steepness * (ratio - band.theta)
    e = np.exp(np.clip(arg, -EXP_ARG_LIMIT, EXP_ARG_LIMIT))
    slope = -(band.k_upper - band.k_lower) * band.steepness * e / (1.0 + e) ** 2
    slope = np.where(np.abs(arg) > EXP_ARG_LIMIT, 0.0, slope)
    if band.k_upper <= band.k_lower:
        slope = np.zeros_like(slope)
    return slope if slope.ndim else float(slope)


def margin_elasticity(ratio, band: Band, side: str):
    """epsilon_t: the elasticity of the side's own margin in the ratio.

    Seller: |f'(x)| * x / (p - K_lower), the definition the worked example of
    Section 4.3.3 evaluates to 2.036 at x = 1.25. Buyer: the same against the
    buyer's margin, |f'(x)| * x / (K_upper - p) -- the reading under which
    the long-side condition takes the same form on both sides.
    """
    ratio = np.asarray(ratio, dtype=float)
    price = sigmoid_vec(ratio, band.k_upper, band.k_lower, band.theta,
                        band.steepness)
    margin = (price - band.k_lower) if side == SELLER else (band.k_upper - price)
    value = -price_derivative(ratio, band) * ratio / margin
    return value if np.ndim(value) else float(value)


# ---------------------------------------------------------- scalar replay

@dataclass(frozen=True)
class SlotState:
    """The recorded aggregates of one slot and the parameters it ran at."""
    supply: float
    demand: float
    band: Band
    gamma: float
    eta_relative: float


@dataclass(frozen=True)
class Outcome:
    supply: float
    demand: float
    ratio: float
    price: float
    traded: float
    allocation: float
    round_type: str
    shortfall_ct: float
    externality_ct: float
    #: Unrounded penalty > 0: where a penalty is active, before the node's
    #: six-decimal rounding can make a tiny one read as zero.
    penalty_active: bool
    #: Utility without the two penalties (the "gross" side of Table 4.2).
    gross_utility: float

    @property
    def penalty_ct(self) -> float:
        return self.shortfall_ct + self.externality_ct

    @property
    def utility(self) -> float:
        return self.gross_utility - self.penalty_ct


def seller_gross_utility(rate, sold, k_lower):
    """(r - K_lower) * s: unsold energy earns the feed-in tariff, i.e. zero
    margin."""
    return (rate - k_lower) * sold


def buyer_gross_utility(rate, received, demand, k_upper):
    """(K_upper - r) * min(q, D) - r * [q - D]^+.

    Demand that is not received is bought at K_upper (saving zero); energy
    received beyond D is paid for and worth nothing (Table 4.2, "payment for
    unneeded energy").
    """
    used = np.minimum(received, demand)
    unneeded = np.maximum(received - demand, 0.0)
    return (k_upper - rate) * used - rate * unneeded


def replay(state: SlotState, side: str, own_report: float, new_report: float,
           truth: float) -> Outcome:
    """The slot with this participant's `own_report` replaced by
    `new_report`, everyone else fixed. `truth` is A (seller) or D (buyer).

    Scalar, and built from the artifact's own functions throughout.
    """
    band = state.band
    supply, demand = state.supply, state.demand
    if side == SELLER:
        supply = supply - own_report + new_report
    else:
        demand = demand - own_report + new_report
    ratio = supply / demand
    price = band.price(ratio)
    traded = min(supply, demand)
    round_type = determine_round_type(supply, demand)
    own_total = supply if side == SELLER else demand
    allocation = new_report / own_total * traded if own_total > 0 else 0.0

    shortfall_ct = externality_ct = 0.0
    active = False
    if side == SELLER:
        eta = state.eta_relative * allocation
        shortfall = seller_shortfall_penalty(
            allocation, min(allocation, truth), band.k_upper, state.gamma, eta)
        shortfall_ct = shortfall["penalty_ct"]
        active = allocation - min(allocation, truth) - eta > 0.0
        if round_type == SUPPLY_LIMITED:
            ext = seller_externality_penalty(
                allocation, truth, eta, supply, demand, traded, price,
                band.as_sigmoid())
            if ext is not None:
                externality_ct = ext["penalty_ct"]
                raw = (price - ext["counterfactual_price_ct_per_kwh"]) * traded
                active = active or raw > 0.0
        gross = float(seller_gross_utility(price, allocation, band.k_lower))
    else:
        if round_type == DEMAND_LIMITED:
            ext = buyer_externality_penalty(
                new_report, truth, supply, demand, traded, price,
                band.as_sigmoid())
            if ext is not None:
                externality_ct = ext["penalty_ct"]
                raw = (ext["counterfactual_price_ct_per_kwh"] - price) * traded
                active = raw > 0.0
        gross = float(buyer_gross_utility(price, allocation, truth,
                                          band.k_upper))
    return Outcome(supply=supply, demand=demand, ratio=ratio, price=price,
                   traded=traded, allocation=allocation, round_type=round_type,
                   shortfall_ct=shortfall_ct, externality_ct=externality_ct,
                   penalty_active=bool(active), gross_utility=gross)


# ------------------------------------------------------ vectorised replay

def sigmoid_vec(ratio, k_upper, k_lower, theta, steepness):
    """`sigmoid_price` on arrays, including its saturation cut-offs."""
    arg = -steepness * (ratio - theta)
    e = np.exp(np.clip(arg, -EXP_ARG_LIMIT, EXP_ARG_LIMIT))
    price = k_upper - (k_upper - k_lower) / (1.0 + e)
    price = np.where(arg > EXP_ARG_LIMIT, k_upper, price)
    price = np.where(arg < -EXP_ARG_LIMIT, k_lower, price)
    return np.where(k_upper <= k_lower, k_lower, price)


def regime_vec(supply, demand):
    return np.where(supply < demand - EPSILON, SL,
                    np.where(supply > demand + EPSILON, DL, BAL))


def replay_vec(*, is_seller, supply, demand, own_report, new_report, truth,
               k_upper, k_lower, theta, steepness, gamma, eta_relative):
    """`replay` on broadcastable arrays. Returns a dict of arrays.

    Rounds the penalties exactly where the artifact does, so check R3 can
    hold the two to 1e-9; `penalty_raw` is the unrounded sum, and is what
    `penalty_active` and the DR2 threshold search read.
    """
    is_seller = np.asarray(is_seller, dtype=bool)
    delta = new_report - own_report
    supply = np.where(is_seller, supply + delta, supply)
    demand = np.where(is_seller, demand, demand + delta)
    ratio = supply / demand
    price = sigmoid_vec(ratio, k_upper, k_lower, theta, steepness)
    traded = np.minimum(supply, demand)
    regime = regime_vec(supply, demand)
    own_total = np.where(is_seller, supply, demand)
    with np.errstate(divide="ignore", invalid="ignore"):
        allocation = np.where(own_total > 0, new_report / own_total * traded,
                              0.0)

    # Seller: shortfall against min(s, A), externality against A.
    eta = eta_relative * allocation
    shortfall_raw = np.maximum(
        0.0, allocation - np.minimum(allocation, truth) - eta)
    shortfall_raw = np.where(is_seller, shortfall_raw, 0.0)
    shortfall_ct = np.round(gamma * k_upper * shortfall_raw, PENALTY_DECIMALS)
    shortfall_pen_raw = gamma * k_upper * shortfall_raw

    withheld = np.maximum(0.0, truth - allocation - eta)
    seller_ext = is_seller & (regime == SL) & (withheld > EPSILON)
    p_cf_seller = sigmoid_vec((supply + withheld) / demand, k_upper, k_lower,
                              theta, steepness)
    seller_ext_raw = np.where(seller_ext,
                              np.maximum(0.0, (price - p_cf_seller) * traded),
                              0.0)

    underreported = np.maximum(0.0, truth - new_report)
    buyer_ext = (~is_seller) & (regime == DL) & (underreported > EPSILON)
    p_cf_buyer = sigmoid_vec(supply / (demand + underreported), k_upper,
                             k_lower, theta, steepness)
    buyer_ext_raw = np.where(buyer_ext,
                             np.maximum(0.0, (p_cf_buyer - price) * traded),
                             0.0)

    externality_raw = seller_ext_raw + buyer_ext_raw
    externality_ct = np.round(externality_raw, PENALTY_DECIMALS)

    gross = np.where(
        is_seller, seller_gross_utility(price, allocation, k_lower),
        buyer_gross_utility(price, allocation, truth, k_upper))
    penalty = shortfall_ct + externality_ct
    return {"supply": supply, "demand": demand, "ratio": ratio,
            "price": price, "traded": traded, "regime": regime,
            "allocation": allocation, "shortfall_ct": shortfall_ct,
            "externality_ct": externality_ct, "penalty_ct": penalty,
            "penalty_raw": shortfall_pen_raw + externality_raw,
            "gross_utility": gross, "utility": gross - penalty}
