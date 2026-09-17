"""Section 3: the deviation layer and per-slot measurements (D-72, D-80, D-82).

On 16e696a there was no execution path at all: `RunSpec.execute` defaulted to
False and no cell set it, `run_sequence` forwarded `**kw` unchanged so a
`measurements=` dict would have been the same dict in all 672 slots, and
nothing generated a deviation. Every test here fails on that state.
"""
import random

import pytest

import deviations
import profiles
import runner
import runs
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    return stack.Stack()


def _clearing(sellers, buyers):
    """A cleared response carrying only what `deviations.apply` reads."""
    return {"status": "cleared", "allocations": {
        "producers": [{"area_uuid": a, "requested_kwh": req,
                       "allocated_kwh": alloc}
                      for a, req, alloc in sellers],
        "consumers": [{"area_uuid": a, "requested_kwh": req,
                       "allocated_kwh": alloc}
                      for a, req, alloc in buyers]}}


SELLERS = [("player-001", 5.0, 4.0), ("player-002", 3.0, 2.4),
           ("player-003", 2.0, 1.6)]
BUYERS = [("player-010", 4.0, 4.0), ("player-011", 6.0, 6.0)]


def _plan(**kw):
    base = {"arm": deviations.NONE, "share": 0.0, "k": 0, "sigma": 0.05,
            "seed": 20260919, "deviators": ()}
    base.update(kw)
    return deviations.DeviationPlan(**base)


# ------------------------------------------------------------ determinism

def test_the_same_seed_and_slot_reproduce_the_slot_on_its_own():
    """Per-slot RNG, so a single deviating slot can be replayed without
    walking the 671 before it."""
    plan = _plan()
    clearing = _clearing(SELLERS, BUYERS)

    first = deviations.apply(plan, clearing, BASE_SLOT)
    again = deviations.apply(_plan(), clearing, BASE_SLOT)
    assert first == again

    other_slot = deviations.apply(_plan(), clearing, BASE_SLOT + 900)
    assert other_slot[0] != first[0], "the noise must move between slots"

    other_seed = deviations.apply(_plan(seed=20260920), clearing, BASE_SLOT)
    assert other_seed[0] != first[0]

    # The accidental layer covers every allocated area in every arm, and the
    # honest seller's forecast is its allocation.
    measurements, forecasts = first
    assert set(measurements) == {a for a, _r, _al in SELLERS + BUYERS}
    assert forecasts == {a: alloc for a, _r, alloc in SELLERS}


def test_zero_sigma_delivers_exactly_the_allocation():
    measurements, forecasts = deviations.apply(
        _plan(sigma=0.0), _clearing(SELLERS, BUYERS), BASE_SLOT)
    for area, _requested, allocated in SELLERS + BUYERS:
        assert measurements[area] == allocated
    for area, _requested, allocated in SELLERS:
        assert forecasts[area] == allocated


# --------------------------------------------------------------- the arms

def test_the_seller_arm_withholds_through_the_forecast_channel():
    """Issue #13: `W_sell = max(0, deliverable - traded - eta)` and
    `deliverable` is the forecast. A withholder's meter reads exactly what it
    traded -- that is the whole point -- and only the forecast carries the
    withholding."""
    plan = _plan(arm=deviations.SELLER_ARM, share=0.25, k=1,
                 deviators=("player-002",))
    measurements, forecasts = deviations.apply(
        plan, _clearing(SELLERS, BUYERS), BASE_SLOT)

    assert forecasts["player-002"] == pytest.approx(2.4 * 1.25)
    assert measurements["player-002"] == pytest.approx(2.4)   # noise off

    for area, _requested, allocated in SELLERS:
        if area == "player-002":
            continue
        assert forecasts[area] == allocated
        assert measurements[area] == pytest.approx(allocated, rel=4 * 0.05)

    assert "player-010" not in forecasts and "player-011" not in forecasts


def test_the_buyer_arm_over_consumes_against_its_own_bid():
    """D-72, corrected 17.09.: the paper's buyer case is under-*reporting*
    demand, i.e. actual above reported. Under-consuming produces no penalty
    at all on this artifact -- buyers have no shortfall term and a negative
    externality clamps to zero (`penalties.py:90`)."""
    plan = _plan(arm=deviations.BUYER_ARM, share=0.25, k=1,
                 deviators=("player-010",))
    measurements, forecasts = deviations.apply(
        plan, _clearing(SELLERS, [("player-010", 8.0, 4.0)] + BUYERS[1:]),
        BASE_SLOT)

    # Against the *bid* (requested_kwh), which is what the execution node
    # reads back as `reported_kwh` -- not against the allocation.
    assert measurements["player-010"] == pytest.approx(8.0 * 1.25)
    assert not any(a.startswith("player-01") for a in forecasts)


def test_an_absent_deviator_deviates_nothing_and_is_counted():
    plan = _plan(arm=deviations.SELLER_ARM, share=0.25, k=1,
                 deviators=("player-099",))
    measurements, _forecasts = deviations.apply(
        plan, _clearing(SELLERS, BUYERS), BASE_SLOT)
    assert "player-099" not in measurements
    assert plan.absent == [BASE_SLOT]


# ---------------------------------------------------------------- the plan

def test_the_plan_draws_households_only_and_never_a_battery():
    """D-82. A battery as the deviator would make the coalition axis a
    statement about the D-75 charging rule instead of about the penalty."""
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    batteries = profiles.make_battery_areas(community.bess, players,
                                            participation=1.0, seed=7)
    assert batteries, "the exclusion is only meaningful with batteries present"

    spec = runs.RunSpec(run_id="d", seed=0, execute=True,
                        deviation={"arm": "sellers_withhold", "share": 0.25,
                                   "k": 10, "sigma": 0.05, "seed": 20260919})
    plan = deviations.plan_deviations(spec, players, batteries)
    assert len(plan.deviators) == 10
    assert all(a.startswith("player-") for a in plan.deviators)
    battery_areas = {b.area_uuid for b in batteries}
    assert not battery_areas & set(plan.deviators)

    # Deterministic in the deviation seed, and only in it.
    again = deviations.plan_deviations(spec, players, batteries)
    assert again.deviators == plan.deviators


def test_an_unknown_arm_or_an_impossible_coalition_is_refused():
    spec = runs.RunSpec(run_id="d", seed=0, execute=True,
                        deviation={"arm": "sellers_underdeliver", "k": 1})
    with pytest.raises(deviations.DeviationError):
        deviations.plan_deviations(spec, list(range(10)))

    spec = runs.RunSpec(run_id="d", seed=0, execute=True,
                        deviation={"arm": "sellers_withhold", "k": 99})
    with pytest.raises(deviations.DeviationError):
        deviations.plan_deviations(spec, list(range(10)))


def test_passing_both_measurements_and_deviate_is_refused():
    """Two ways of setting the same channel."""
    import anyio

    async def go():
        await runner.run_slot(None, {"producers": [], "consumers": []},
                              community="c", slot=BASE_SLOT,
                              measurements={"a": 1.0},
                              deviate=lambda clearing, slot: ({}, {}))

    with pytest.raises(ValueError, match="not both"):
        anyio.run(go)


# ------------------------------------------------------------- end to end

@pytest.mark.anyio
async def test_the_deviation_reaches_the_penalty_on_a_real_slot(st):
    """The end-to-end check: a withholding seller in a SUPPLY_LIMITED slot
    produces a non-empty redistribution pool, and `arm = "none"` at sigma = 0
    produces none."""
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    batteries = profiles.make_battery_areas(community.bess, players,
                                            participation=0.25, seed=7)
    week = profiles.extend_to_week(community.day_kwh(players), seed=4242,
                                   days=1)

    # A supply-limited slot with several sellers, so the harmed set is not
    # empty either.
    index, scen = next(
        (i, s) for i in range(week.n_slots)
        for s in [scenario.make_scenario_from_profiles(week, i, batteries)]
        if 0 < s["posted_supply_kwh"] < s["posted_demand_kwh"]
        and len(s["producers"]) >= 3 and len(s["consumers"]) >= 3)

    seller = sorted(p["area_uuid"] for p in scen["producers"])[0]
    plan = deviations.DeviationPlan(arm=deviations.SELLER_ARM, share=0.25,
                                    k=1, sigma=0.0, deviators=(seller,))
    quiet = deviations.DeviationPlan(arm=deviations.NONE, sigma=0.0)

    try:
        withheld = await runner.run_slot(
            st, scen, community="s3-withhold", slot=BASE_SLOT,
            market_id=runner.market_id_for("s3-withhold", BASE_SLOT),
            preferences={"enabled": False, "multipliers_enabled": False},
            execute=True, gamma=1.1, eta_relative=0.10,
            deviate=lambda clearing, slot: deviations.apply(plan, clearing,
                                                            slot))
        honest = await runner.run_slot(
            st, scen, community="s3-honest", slot=BASE_SLOT,
            market_id=runner.market_id_for("s3-honest", BASE_SLOT),
            preferences={"enabled": False, "multipliers_enabled": False},
            execute=True, gamma=1.1, eta_relative=0.10,
            deviate=lambda clearing, slot: deviations.apply(quiet, clearing,
                                                            slot))
    finally:
        await st.close()

    assert withheld["execution"]["round_type"] == "SUPPLY_LIMITED"
    assert withheld["execution"]["redistribution"]["penalty_pool_ct"] > 0
    assert withheld["execution"]["redistribution"]["excluded_deviators"] == \
        [seller]
    assert withheld["execution"]["redistribution"]["harmed_side"] == "buyer"

    assert honest["execution"]["round_type"] == "SUPPLY_LIMITED"
    assert honest["execution"]["redistribution"]["penalty_pool_ct"] == 0.0
    assert honest["execution"]["total_penalties_ct"] == 0.0

    # And the deviator is the only one carrying an externality penalty.
    charged = [r["area_uuid"] for r in withheld["execution"]["results"]
               if r["externality_penalty_ct"] > 0]
    assert charged == [seller]


# ---------------------------------------------------- the noise-free cell

def test_sigma_zero_posts_the_allocation_for_every_area():
    """D-84. `b2_noise_off` has to be clean because the harness wrote the
    allocation, not because it wrote nothing.

    The execution node falls back to "no measurement -> delivered == traded"
    for an area it finds no meter reading for (`execution.py`), so a
    reference cell that skipped the write would be measuring that fallback
    instead of the mechanism. Every allocated area gets a row, at every
    sigma.
    """
    clearing = _clearing(SELLERS, BUYERS)
    measurements, forecasts = deviations.apply(
        _plan(sigma=0.0, arm=deviations.NONE), clearing, BASE_SLOT)

    assert set(measurements) == {a for a, _r, _al in SELLERS + BUYERS}
    for area, _requested, allocated in SELLERS + BUYERS:
        assert measurements[area] == allocated, area
    assert set(forecasts) == {a for a, _r, _al in SELLERS}
    for area, _requested, allocated in SELLERS:
        assert forecasts[area] == allocated, area

    # And it is exact rather than "within a tolerance": at sigma = 0 the
    # noise is 0.0 by construction, not a draw from a degenerate Gaussian.
    assert deviations._noise(random.Random(1), 0.0) == 0.0
    again = deviations.apply(_plan(sigma=0.0, arm=deviations.NONE),
                             clearing, BASE_SLOT + 900)
    assert again == (measurements, forecasts), "no slot-to-slot variation"
