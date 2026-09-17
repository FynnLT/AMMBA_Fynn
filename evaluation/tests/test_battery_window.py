"""Section 7: the battery discharge window as a run parameter (D-83).

The finding this closes: `profiles.EVENING_PERIODS` is 18:15-22:00 and PV is
non-zero only to 17:45, so on the reference window no slot ever holds a green
*and* a grey offer. `bonus_paid_ct` is then 0 in every slot at every
`green_multiplier`, `crossover_mult` is None throughout, and the two
multiplier formulations cannot be told apart -- which is the comparison D-38
names. The first test here is that measurement turned into a guard.
"""
import json

import pytest

import aggregates
import campaign
import profiles
import runner
import runs
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    return stack.Stack()


@pytest.fixture(scope="module")
def week():
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    return community, players, profiles.extend_to_week(
        community.day_kwh(players), seed=4242, days=7)


def _overlap_slots(community, players, week, evening_periods):
    """Slots whose posted book carries both a green and a grey offer.

    Counted on the posted book rather than on the allocation: pro-rata fills
    every order in a cleared slot, so a slot with both kinds posted is a slot
    with both kinds allocated. The response-level fact is checked separately
    below, on one slot, through the real stack.
    """
    batteries = profiles.make_battery_areas(
        community.bess, players, participation=0.25, seed=7,
        evening_periods=evening_periods)
    overlap = []
    for index in range(week.n_slots):
        scen = scenario.make_scenario_from_profiles(week, index, batteries)
        green = sum(p["energy"] for p in scen["producers"]
                    if p.get("energy_type") == "green")
        grey = sum(p["energy"] for p in scen["producers"]
                   if p.get("energy_type") == "grey")
        if green > 0 and grey > 0 and scen["posted_demand_kwh"] > 0:
            overlap.append(index)
    return overlap


def test_the_reference_window_never_overlaps_pv_and_the_new_one_does(week):
    """**The measurement the D-83 decision rests on.**

    Periods are labelled by their end: the reference window is 73..88
    (18:15-22:00) and PV runs to the period ending 17:45. The overlap window
    is 61..76 (15:00-19:00), so 61..71 sit inside the PV day.
    """
    community, players, profile_week = week

    reference = _overlap_slots(community, players, profile_week,
                               profiles.EVENING_PERIODS)
    assert reference == [], (
        "the reference window is expected to carry no slot with both a green "
        "and a grey offer; the D-83 decision exists because of that")

    overlap = _overlap_slots(community, players, profile_week,
                             campaign.OVERLAP_EVENING_PERIODS)
    assert len(overlap) > 0
    # At least one per day of the week, which is what the group needs to be
    # readable per day rather than as a single lucky slot.
    assert len(overlap) >= profile_week.n_slots // 96, len(overlap)

    # The window really is the only thing that moved: the reference window
    # still discharges, it just never does so while the sun is up.
    assert profiles.EVENING_PERIODS == tuple(range(73, 89))
    assert campaign.OVERLAP_EVENING_PERIODS == tuple(range(61, 77))

    # The *delivery* window the group is named after: a period is labelled by
    # its end (61 ends 15:15), but it delivers from its start, so periods
    # 61..76 deliver 15:00-19:00.
    def window(periods):
        return (profiles.period_window_sec(min(periods))[0],
                profiles.period_window_sec(max(periods))[1])

    start, end = window(campaign.OVERLAP_EVENING_PERIODS)
    assert (start // 3600, end // 3600) == (15, 19)
    assert window(profiles.EVENING_PERIODS)[0] // 3600 == 18


@pytest.mark.anyio
async def test_an_overlap_slot_reports_both_allocations_and_a_crossover(st, week):
    """The scenario-level count above, tied to the response fields the CSV
    actually carries."""
    community, players, profile_week = week
    index = _overlap_slots(community, players, profile_week,
                           campaign.OVERLAP_EVENING_PERIODS)[0]
    batteries = profiles.make_battery_areas(
        community.bess, players, participation=0.25, seed=7,
        evening_periods=campaign.OVERLAP_EVENING_PERIODS)
    scen = scenario.make_scenario_from_profiles(profile_week, index, batteries)

    try:
        record = await runner.run_slot(
            st, scen, community="s7-overlap", slot=BASE_SLOT,
            market_id=runner.market_id_for("s7-overlap", BASE_SLOT),
            preferences=campaign.prefs(
                multipliers_enabled=True, mode="multiplicative",
                green_multiplier=campaign.GREEN_MULTIPLIER,
                grey_levy=campaign.GREY_LEVY, levy_cap=campaign.LEVY_CAP),
            execute=False)
    finally:
        await st.close()

    row = aggregates.slot_row(record)
    assert row["green_alloc_kwh"] > 0
    assert row["grey_alloc_kwh"] > 0
    assert row["levy_collected_ct"] > 0
    # Both of these are None on every slot of the reference window.
    assert row["crossover_mult"] is not None
    assert row["bonus_requested_ct"] > 0


def test_the_window_is_a_run_parameter_and_reaches_the_manifest():
    """D-83. `profiles.py` is unchanged: the window is passed in, not edited
    in place, so every existing cell keeps the D-75 reference."""
    default = runs.RunSpec(run_id="d", seed=0)
    assert tuple(default.evening_periods) == profiles.EVENING_PERIODS
    assert "evening_periods" in default.config()

    moved = runs.RunSpec(run_id="m", seed=0,
                         evening_periods=campaign.OVERLAP_EVENING_PERIODS)
    community, players, _week, batteries, _prefs = runs.build_week(
        runs.RunSpec(run_id="m", seed=0, days=1, n_players=30,
                     evening_periods=campaign.OVERLAP_EVENING_PERIODS))
    assert batteries, "the variant still builds battery areas"
    # Period 61 is index 60; the reference window is empty there.
    assert any(b.discharge_kwh[60] > 0 for b in batteries)

    reference = runs.build_week(
        runs.RunSpec(run_id="r", seed=0, days=1, n_players=30))[3]
    assert all(b.discharge_kwh[60] == 0 for b in reference)
    assert moved.config()["evening_periods"] == campaign.OVERLAP_EVENING_PERIODS


def test_the_overlap_cells_differ_in_nothing_but_their_preferences():
    """A three-cell group read against its own baseline: if anything else
    moved between them, the delta would not be the formulation."""
    by_cell = {spec.cell: spec for spec in campaign.cells(1) if spec.seed == 0}
    group = [by_cell[f"mult_overlap_{name}"]
             for name in ("off", "multiplicative", "additive")]

    for spec in group:
        assert tuple(spec.evening_periods) == campaign.OVERLAP_EVENING_PERIODS
        assert spec.named_share == 0.50
        assert spec.mutual_share == campaign.REFERENCE_MUTUAL
        assert spec.participation == campaign.REFERENCE_PARTICIPATION

    def without_preferences(spec):
        # json, because a config carries dicts (`sigmoid`, `deviation`) and a
        # tuple of items is not hashable through them.
        return json.dumps({k: v for k, v in spec.config().items()
                           if k not in ("preferences", "cell")},
                          sort_keys=True, default=str)

    assert len({without_preferences(s) for s in group}) == 1

    off, multiplicative, additive = group
    assert off.preferences["multipliers_enabled"] is False
    assert multiplicative.preferences["mode"] == "multiplicative"
    assert additive.preferences["mode"] == "additive"
    for spec in (multiplicative, additive):
        assert spec.preferences["multipliers_enabled"] is True
        assert spec.preferences["green_multiplier"] == 0.02
    assert {k: v for k, v in multiplicative.preferences.items() if k != "mode"} \
        == {k: v for k, v in additive.preferences.items() if k != "mode"}


def test_the_reference_multiplier_cells_keep_the_declared_window():
    """D-75 stands for every existing cell: the overlap variant is an extra
    group, not a change to the reference."""
    for spec in campaign.cells(1):
        if spec.cell.startswith("mult_overlap_"):
            continue
        assert tuple(spec.evening_periods) == profiles.EVENING_PERIODS, spec.cell
    for spec in campaign.cells(2) + campaign.calibration_cells():
        assert tuple(spec.evening_periods) == profiles.EVENING_PERIODS, spec.cell
