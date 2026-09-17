"""Block 2: who deviates, by how much, and through which channel (D-72).

Three facts about the artifact fix the whole design, and every one of them is
read off the execution node rather than assumed:

1. **A withholder is invisible from meter data** (issue #13,
   `amm-execution-node/src/execution.py:167-176`): `deliverable = actual`
   unless a forecast exists, so `W_sell = max(0, deliverable - traded - eta)`
   is identically zero on meter data alone. The seller externality is only
   measurable through the *forecast* channel, and the forecast is what the
   seller could have delivered -- which for a withholder is **more than it
   traded**, not the same. See `SELLER_ARM` below.

2. **The buyer externality is `max(0, actual - reported)`**
   (`penalties.py:90`), and `reported_kwh` is the original bid volume
   (allocation + residual, `execution.py:59-67`), i.e. `requested_kwh`. A
   buyer deviates by consuming *more* than it bid. D-72 first said "buyers
   under-consume"; on this artifact that produces no penalty at all, because
   buyers have no shortfall term and a negative externality clamps to zero.

3. **Both externalities are round-typed**: the seller's fires only in
   SUPPLY_LIMITED rounds and the buyer's only in DEMAND_LIMITED ones
   (`execution.py:194-206`). A deviator in the wrong round type deviates and
   is not penalised, which is a result and not a defect -- nothing here
   filters on the round type.

The accidental layer sits under all three arms: every allocated area delivers
`allocated * (1 + eps)`, `eps ~ N(0, sigma)`. **[D-80]** sigma = 0.05 against
`eta_relative = 0.10`, so roughly 95 % of accidental deviations fall inside
the deadband -- "noise floor two sigma below the deadband", chosen and stated
as chosen in 5.1, not sourced. The better alternative is to derive sigma from
a PV forecast-error reference, which is a literature search away.
"""
import random
from dataclasses import dataclass, field

NONE = "none"
SELLER_ARM = "sellers_withhold"
BUYER_ARM = "buyers_underreport"
ARMS = (NONE, SELLER_ARM, BUYER_ARM)

#: D-80. The accidental noise floor, two sigma below `eta_relative = 0.10`.
DEFAULT_SIGMA = 0.05

#: Clip at four sigma: a draw far enough into the tail to flip the sign of a
#: delivery is noise in the generator, not in the market.
_CLIP_SIGMA = 4.0


class DeviationError(ValueError):
    """A deviation plan that cannot be carried out as specified."""


@dataclass(frozen=True)
class DeviationPlan:
    """Everything the run's deviation depends on, fixed before slot 0.

    `deviators` are named once for the whole run rather than redrawn per
    slot, for the same reason the preference pairs are (D-81): a coalition
    that changes membership every quarter hour is not a coalition.
    """
    arm: str = NONE
    share: float = 0.0
    k: int = 0
    sigma: float = DEFAULT_SIGMA
    seed: int = 20260919
    deviators: tuple = ()
    #: Slots in which a named deviator held no allocation at all. Counted,
    #: not silently skipped: at `k = 10` on a week where most households net
    #: to zero in most slots, the realised coalition is much smaller than the
    #: nominal one, and 5.3 has to be able to say so.
    absent: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"arm": self.arm, "share": self.share, "k": self.k,
                "sigma": self.sigma, "seed": self.seed,
                "deviators": list(self.deviators),
                "slots_with_absent_deviator": len(self.absent)}


def plan_deviations(spec, players, batteries=()) -> DeviationPlan:
    """The run's named deviators, drawn from `spec.deviation`.

    **Households only** (**[D-82]**): not from the battery areas, because the
    battery charging rule is already a declared modelling assumption (D-75).
    A battery as the deviator would make the coalition axis a statement about
    that rule rather than about the penalty mechanism. `batteries` is taken
    only so the exclusion is explicit at the call site rather than implicit
    in what `players` happens to contain.
    """
    config = dict(getattr(spec, "deviation", None) or {})
    arm = config.get("arm", NONE)
    if arm not in ARMS:
        raise DeviationError(f"unknown deviation arm {arm!r}; known: "
                             f"{list(ARMS)}")
    sigma = float(config.get("sigma", DEFAULT_SIGMA))
    if sigma < 0:
        raise DeviationError(f"sigma must be >= 0, got {sigma}")
    share = float(config.get("share", 0.0))
    k = int(config.get("k", 0))
    seed = int(config.get("seed", getattr(spec, "deviation_seed", 20260919)))

    from scenario import area_uuid_for
    households = [area_uuid_for(p) for p in players]
    battery_areas = {b.area_uuid for b in batteries}
    households = [a for a in households if a not in battery_areas]

    deviators = ()
    if arm != NONE and k > 0:
        if k > len(households):
            raise DeviationError(
                f"asked for {k} deviators but the community has "
                f"{len(households)} households")
        deviators = tuple(sorted(random.Random(seed).sample(households, k)))
    return DeviationPlan(arm=arm, share=share, k=k, sigma=sigma, seed=seed,
                         deviators=deviators)


def _noise(rng, sigma: float) -> float:
    """Clipped Gaussian, and exactly 0.0 at `sigma = 0`.

    Returning 0.0 rather than drawing from a degenerate Gaussian keeps the
    noise-free cell bit-exact against the allocation, and keeps it
    independent of how many times the RNG happened to be advanced before it.
    """
    if sigma <= 0.0:
        return 0.0
    return max(-_CLIP_SIGMA * sigma,
               min(_CLIP_SIGMA * sigma, rng.gauss(0.0, sigma)))


def apply(plan: DeviationPlan, clearing: dict, slot: int) -> tuple:
    """`(measurements, forecasts)` for one cleared slot.

    Called between clearing and execution, because every delivered quantity
    here is relative to what the slot *allocated* -- which only the clearing
    response knows.

    Per-slot RNG (`Random(seed ^ slot)`) rather than one stream over the
    week: a slot then reproduces on its own, so a single deviating slot can
    be replayed without walking the 671 before it.

    The accidental layer applies to **every** area in **every** arm,
    including `arm = "none"`; sellers additionally carry an honest forecast
    equal to their allocation, so `deliverable == traded` and the seller
    externality is silent unless someone withholds. Buyers carry no forecast:
    the buyer externality reads the meter against the bid and never touches
    the forecast channel.

    **At `sigma = 0` every allocated area still gets a measurement**, equal
    to its allocation to the response's own six decimals. That is deliberate
    and is what `b2_noise_off` rests on (D-84): the execution node falls back
    to "no measurement -> delivered == traded" for an area it finds no meter
    reading for, and a reference cell that was clean only because the harness
    posted nothing would be measuring that fallback instead of the mechanism.
    The write path is exercised in every arm, at every sigma.

    The two named arms:

    * `sellers_withhold` -- the named seller's **forecast** is
      `allocated * (1 + share)`: it could have delivered that much and
      offered only `allocated`, so `W_sell = share * allocated` above the
      deadband. Its meter reads exactly `allocated`, which is the point of
      issue #13: the withholding is invisible on the meter and visible only
      on the forecast channel. Noise is off for the named seller -- the
      strategy is the signal.
    * `buyers_underreport` -- the named buyer's **meter** reads
      `requested * (1 + share)` against a bid of `requested`, so
      `W_buy = share * requested`. No forecast, on either side.
    """
    allocations = clearing.get("allocations") or {}
    producers = allocations.get("producers") or []
    consumers = allocations.get("consumers") or []
    rng = random.Random(plan.seed ^ int(slot))

    named = set(plan.deviators)
    present = set()
    measurements, forecasts = {}, {}

    for entry in producers:
        area = entry.get("area_uuid")
        allocated = float(entry.get("allocated_kwh") or 0.0)
        if area in named:
            present.add(area)
        if plan.arm == SELLER_ARM and area in named:
            # No noise: the strategy is the signal. The meter reads exactly
            # what was traded, and only the forecast channel carries the
            # withholding (issue #13).
            measurements[area] = round(allocated, 6)
            forecasts[area] = round(allocated * (1.0 + plan.share), 6)
            continue
        measurements[area] = round(
            max(0.0, allocated * (1.0 + _noise(rng, plan.sigma))), 6)
        # The honest deliverable: what this seller could have delivered is
        # what it traded, so `W_sell` is zero for everyone who is not
        # withholding, whatever the meter happened to read.
        forecasts[area] = round(allocated, 6)

    for entry in consumers:
        area = entry.get("area_uuid")
        allocated = float(entry.get("allocated_kwh") or 0.0)
        requested = float(entry.get("requested_kwh") or 0.0)
        if area in named:
            present.add(area)
        if plan.arm == BUYER_ARM and area in named:
            # Against the *bid*, not the allocation: `reported_kwh` in the
            # execution node is allocation + residual (`execution.py:59-67`),
            # which is `requested_kwh`.
            measurements[area] = round(requested * (1.0 + plan.share), 6)
            continue
        measurements[area] = round(
            max(0.0, allocated * (1.0 + _noise(rng, plan.sigma))), 6)

    # A named deviator with no allocation in this slot deviates nothing.
    if named and not named <= present:
        plan.absent.append(int(slot))
    return measurements, forecasts
