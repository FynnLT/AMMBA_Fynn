"""The calibration objective for theta and B: Saber et al. (2026) eq. (8).

The value `fit.fit` minimises, in the form it takes in this artifact:

    minimise   -mean_t[ min(K_up - p_t, p_t - K_lo) ]
               + alpha * ( (p(r_max) - K_lo) + (K_up - p(r_min)) )

`p_t` is the sigmoid price at the supply/demand ratio the Clearing Node
reported for slot `t`; `r_min` and `r_max` are the smallest and the largest
ratio in the fitted sample.

**The reduction, and the condition it holds under.** Section 5.1 has to repeat
this, so it is written out here rather than left in the paper. Equations (6)
and (7) define consumer utility and producer profit *per kWh*, not per slot,
and the quantity cancels out of both:

* eq. (7), producer profit, reduces to `p - K_lo` exactly. The producer's
  alternative to selling into the community is the feed-in tariff, so what it
  gains per kWh is the clearing price above that tariff, whatever it sold.

* eq. (6), consumer utility, reduces to `K_up - p` -- but only as long as no
  consumer receives more energy than it demanded. A consumer handed surplus it
  never bid for does not value that surplus at the retail price it avoided,
  because it avoided no retail purchase with it, and the per-kWh form stops
  collapsing.

That over-supply case cannot arise in this artifact, and the argument does not
go by round type. `run_clearing` sets
`traded_quantity = min(total_supply_kwh, total_demand_kwh)` and
`apply_preference_allocation` splits exactly that quantity over the order books
in shares bounded by each order's own energy. Every buyer therefore ends the
round with `E_rec <= D`: exactly its demand in a demand-limited round, less
than its demand in a supply-limited one, more than its demand in neither. It
is the cap that carries the reduction, not which side of the market happened
to bind.

**This is a reduction under a named condition, not a simplification of
convenience.** If the allocation is ever changed so that a consumer can be
allocated more than it bid, eq. (6) no longer reduces to `K_up - p` and this
module is wrong rather than merely imprecise. The condition is a property of
`apply_preference_allocation`, and it is that function, not this one, that
would have to be re-read before the objective is trusted again.

**What the two terms do.** The first rewards a price that keeps its distance
from both bounds, which on its own is degenerate: a flat curve parked at the
band midpoint maximises it, and the fit answers with a steepness of zero. The
penalty is what makes the fit non-trivial -- it charges for the curve failing
to reach K_lo at the most supply-rich slot observed and K_up at the most
supply-poor one, so alpha buys span at the cost of margin. `alpha` is
therefore not a nuisance parameter to be tuned away but the axis the sweep in
`fit.sweep` reports along (D-74).
"""
import csv
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import NamedTuple


def sigmoid_price_unclamped(ratio: float, k_upper: float, k_lower: float,
                            theta: float, steepness: float) -> float:
    """The sigmoid clearing price, with nothing clamped to the band.

        price = K_up - (K_up - K_lo) / (1 + exp(-B * (ratio - theta)))

    **This is a deliberate third copy of a formula the repository already
    carries twice in Python** (`amm-clearing-node/src/sigmoid.py`,
    `amm-execution-node/src/sigmoid.py`; `ui/app.js` holds a fourth in
    JavaScript). The copy is not laziness about importing one of them: the two
    service copies return exactly `k_upper` or `k_lower` once the exponent
    saturates, and the penalty term of the objective measures precisely how
    close the curve gets to each bound. A price that has been snapped onto the
    bound reports a penalty of zero for a curve that never actually reached it,
    which is the one signal the fit depends on. Clamping is correct in a
    settlement path and destructive in a calibration one.

    The agreement inside the band is not left to trust:
    `tests/test_calibration.py` pins this function against the Clearing Node's
    `sigmoid_price` to 1e-9 across the ratios the calibration data occupies.

    The node's `k_upper <= k_lower` guard is deliberately not mirrored. A
    degenerate band has no calibration meaning, and silently answering
    `k_lower` would turn a configuration error into a plausible-looking fit.
    """
    # Written as a share of the band and split on the sign of the exponent, so
    # neither branch can overflow: the service copies need an explicit +/-700
    # cut-off precisely because the single-expression form does.
    x = steepness * (ratio - theta)
    if x >= 0.0:
        share = 1.0 / (1.0 + exp(-x))
    else:
        scaled = exp(x)
        share = scaled / (1.0 + scaled)
    return k_upper - (k_upper - k_lower) * share


@dataclass(frozen=True)
class Slot:
    """One cleared slot, as the Clearing Node reported it.

    `ratio` is recomputed from the two totals rather than read from the CSV's
    own `ratio` column. Both come from the same clearing response, and they
    differ only by that column's six-decimal rounding -- but the totals are
    what the node divided, so dividing them again cannot drift away from the
    artifact the way recomputing the ratio from the scenario would
    (`aggregates.py` says why that distinction matters).

    The band travels with the slot instead of being passed alongside it: a
    calibration that mixed two bands and averaged over them would be a
    silently wrong number rather than an error, and this makes the band
    explicit at every use.
    """
    total_supply_kwh: float
    total_demand_kwh: float
    k_upper: float
    k_lower: float

    @property
    def ratio(self) -> float:
        return self.total_supply_kwh / self.total_demand_kwh


class LoadedSlots(NamedTuple):
    """What `load_slots` returns: the usable slots, and what it discarded."""
    slots: list
    dropped: int

    @property
    def total(self) -> int:
        return len(self.slots) + self.dropped


def _number(raw):
    """A CSV cell as a float, or None when the column carried no value."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return float(text)


def load_slots(csv_path, k_upper: float, k_lower: float) -> LoadedSlots:
    """Read an `aggregates.write_slot_csv` file into fittable slots.

    **Slots that did not trade are dropped, and the count is returned rather
    than logged away.** A slot with no ratio, or with nothing offered, or with
    nothing demanded, is not a cheap observation of a market at an extreme
    price -- it is a slot on which utility and profit are undefined, because
    nobody bought and nobody sold. Saber et al. fit over traded slots for that
    reason. Keeping them would not merely add noise: a no-trade slot has no
    ratio at all, and inventing one (0, or infinity) would drag `r_min` or
    `r_max` to the edge of the grid and let the penalty term fit a bound that
    was never observed.

    On this artifact's profile weeks the drop is large -- roughly half the
    slots, because a residential community has no PV at night and the node
    reports a no-trade slot rather than a zero-price one. That is why the
    count comes back: a caller that fits 336 of 672 slots should say so in
    5.1, and a caller that suddenly fits 60 has a data problem, not a fit.
    """
    kept, dropped = [], 0
    with open(Path(csv_path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ratio = _number(row.get("ratio"))
            supply = _number(row.get("total_supply_kwh"))
            demand = _number(row.get("total_demand_kwh"))
            if ratio is None or not supply or not demand:
                dropped += 1
                continue
            kept.append(Slot(total_supply_kwh=supply, total_demand_kwh=demand,
                             k_upper=k_upper, k_lower=k_lower))
    return LoadedSlots(kept, dropped)


def band(slots) -> tuple:
    """The (k_upper, k_lower) the slots share.

    Raises when they do not. Averaging an objective over two bands produces a
    number, and the number is meaningless; this is the cheapest place to
    refuse.
    """
    if not slots:
        raise ValueError("no slots: the band is undefined")
    bands = {(s.k_upper, s.k_lower) for s in slots}
    if len(bands) != 1:
        raise ValueError(f"slots carry {len(bands)} different bands: "
                         f"{sorted(bands)}")
    return bands.pop()


def objective(slots, theta: float, steepness: float, alpha: float) -> float:
    """Eq. (8) for one parameter pair. Lower is better.

    Returns `-mean_t[min(K_up - p_t, p_t - K_lo)] + alpha * penalty`, with the
    penalty evaluated at the extreme ratios of `slots` themselves rather than
    at the grid bounds: the term is about what the curve does where this
    community actually traded, not where it could in principle be asked.
    """
    if not slots:
        raise ValueError("cannot evaluate the objective on zero slots")

    total_margin = 0.0
    lowest = highest = slots[0]
    lowest_ratio = highest_ratio = slots[0].ratio
    for slot in slots:
        ratio = slot.ratio
        price = sigmoid_price_unclamped(ratio, slot.k_upper, slot.k_lower,
                                        theta, steepness)
        total_margin += min(slot.k_upper - price, price - slot.k_lower)
        if ratio < lowest_ratio:
            lowest, lowest_ratio = slot, ratio
        if ratio > highest_ratio:
            highest, highest_ratio = slot, ratio

    at_max = sigmoid_price_unclamped(highest_ratio, highest.k_upper,
                                     highest.k_lower, theta, steepness)
    at_min = sigmoid_price_unclamped(lowest_ratio, lowest.k_upper,
                                     lowest.k_lower, theta, steepness)
    penalty = (at_max - highest.k_lower) + (lowest.k_upper - at_min)
    return -(total_margin / len(slots)) + alpha * penalty
