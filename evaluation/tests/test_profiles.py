"""Section 2: the Faia et al. workbook as market profiles.

Every test here fixes a conversion or an assumption that is silent when wrong:
reading kW as kWh inflates every quantity by four, an off-by-one in the period
mapping shifts the whole day by 15 minutes, and a week that is one day
repeated seven times looks exactly like a week until the calibration fits it.
"""
import pytest

import profiles
import scenario

WORKBOOK = profiles.DEFAULT_WORKBOOK


@pytest.fixture(scope="module")
def community():
    if not WORKBOOK.is_file():
        pytest.skip(f"workbook not present at {WORKBOOK}")
    return profiles.load_community(WORKBOOK)


# ------------------------------------------------------------- conversions

def test_kw_to_kwh_on_a_known_cell(community):
    """Load!B2 is player 1, period 1: 1.556922571 kW over 900 s."""
    assert community.load_kw[1][0] == pytest.approx(1.556922571)
    assert profiles.kw_to_kwh(community.load_kw[1][0]) == pytest.approx(
        1.556922571 * 0.25)
    day = community.day_kwh([1])
    assert day.load_kwh[1][0] == pytest.approx(0.38923064275)
    # The factor is a quarter, not one: the whole campaign hangs off it.
    assert profiles.HOURS_PER_PERIOD == 0.25


def test_period_to_slot_mapping():
    """The workbook labels a period by its *end*; the artifact's time_slot is
    the *start* of the delivery window."""
    assert profiles.period_label(33) == "08:15"
    assert profiles.period_label(76) == "19:00"
    assert profiles.period_window_sec(33) == (28800, 29700)   # 08:00 -> 08:15
    assert profiles.period_window_sec(76) == (67500, 68400)   # 18:45 -> 19:00

    midnight = 1_757_000_000 // 86400 * 86400
    assert profiles.period_to_slot(33, midnight) == midnight + 28800
    assert profiles.period_to_slot(76, midnight) == midnight + 67500
    assert profiles.period_to_slot(1, midnight, day=1) == midnight + 86400

    with pytest.raises(profiles.ProfileError):
        profiles.period_window_sec(0)
    with pytest.raises(profiles.ProfileError):
        profiles.period_window_sec(97)


def test_non_numeric_cells_fail_loudly():
    """Cells come back as strings; a silent nan would reach an order book."""
    assert profiles._number("1.25", "x") == 1.25
    assert profiles._number("'0.5'", "x") == 0.5
    for bad in (None, "", "n/a", "Capacity"):
        with pytest.raises(profiles.ProfileError):
            profiles._number(bad, "x")


# -------------------------------------------------------- what the file says

def test_the_workbook_has_no_grey_selling_side(community):
    """Measured on 16.09.2026, and the reason D-75 exists. If this changes,
    the battery-area model is answering a question the data no longer asks."""
    flags = community.flags
    assert len(flags) == 250
    assert sum(1 for f in flags.values() if f.pv) == 200
    assert sum(1 for f in flags.values() if f.ess) == 150
    assert sum(1 for f in flags.values() if f.pv and f.ess) == 150
    assert sum(1 for f in flags.values() if f.ess and not f.pv) == 0

    lit = [p for p in range(1, 97)
           if any(community.pv_kw[pid][p - 1] > 0 for pid in community.players)]
    assert (min(lit), max(lit)) == (32, 71)
    assert len(lit) == 40, "56 of 96 periods carry no PV at all"


def test_select_players_is_reproducible_from_the_seed_alone(community):
    """D-70. The manifest records the seed and the ids; either has to be
    enough to rebuild the other."""
    first = profiles.select_players(community.flags, n=100, seed=20260916)
    again = profiles.select_players(community.flags, n=100, seed=20260916)
    assert first == again == sorted(first)
    assert len(set(first)) == 100
    assert profiles.select_players(community.flags, n=100, seed=1) != first

    # Insertion order of the flags dict must not enter the draw.
    shuffled = dict(reversed(list(community.flags.items())))
    assert profiles.select_players(shuffled, n=100, seed=20260916) == first

    with pytest.raises(profiles.ProfileError):
        profiles.select_players(community.flags, n=251, seed=0)


# -------------------------------------------------------------- the netting

def test_netting_posts_exactly_one_order_per_player_per_slot(community):
    """D-61. A player is on one side or on neither, never on both and never
    twice."""
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    day = community.day_kwh(players)

    for index in (0, 40, 60, 95):
        scen = scenario.make_scenario_from_profiles(day, index)
        areas = [o["area_uuid"] for o in scen["producers"] + scen["consumers"]]
        assert len(areas) == len(set(areas))
        assert set(areas) <= {f"player-{p:03d}" for p in players}

        for player in players:
            load = day.load_kwh[player][index]
            pv = day.pv_kwh[player][index]
            posted = [o for o in scen["producers"] + scen["consumers"]
                      if o["area_uuid"] == f"player-{player:03d}"]
            net = load - pv
            if abs(net) <= scenario.NET_EPSILON:
                assert posted == []
            else:
                assert len(posted) == 1
                assert posted[0]["energy"] == pytest.approx(abs(net), abs=1e-6)

        # The gross quantities survive the netting, so 5.1 can state what was
        # netted away rather than infer it.
        assert scen["consumption_kwh"] == pytest.approx(
            sum(day.load_kwh[p][index] for p in players), abs=1e-5)
        assert scen["generation_kwh"] == pytest.approx(
            sum(day.pv_kwh[p][index] for p in players), abs=1e-5)
        assert scen["net_kwh"] == pytest.approx(
            scen["consumption_kwh"] - scen["generation_kwh"], abs=1e-5)


def test_a_double_posting_raises_rather_than_being_netted_quietly():
    """The artifact enforces nothing here, so the generator has to."""
    broken = {"producers": [{"area_uuid": "player-001", "name": "a",
                             "energy": 1.0}],
              "consumers": [{"area_uuid": "player-001", "name": "a",
                             "energy": 2.0}],
              "slot_index": 3}
    with pytest.raises(scenario.ScenarioError, match="player-001"):
        scenario._assert_one_order_per_area(broken)


# ------------------------------------------------------------- the batteries

def test_a_battery_never_exceeds_its_capacity_or_its_power_limit(community):
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    areas = profiles.make_battery_areas(community.bess, players, seed=7)
    assert areas, "the reference community should contain storage units"

    for area in areas:
        spec = area.spec
        assert max(area.discharge_kwh) <= spec.discharge_kwh_per_period + 1e-9
        assert max(area.grid_charge_kwh) <= spec.charge_kwh_per_period + 1e-9
        assert sum(area.discharge_kwh) <= spec.capacity_kwh + 1e-9

        # Walk the state of charge: it stays inside [0, capacity] all day.
        stored = 0.0
        for index in range(profiles.PERIODS_PER_DAY):
            stored += area.grid_charge_kwh[index] * spec.efficiency
            stored -= area.discharge_kwh[index]
            assert -1e-9 <= stored <= spec.capacity_kwh + 1e-9

        # Grey, because it was bought from the grid. PV output never is.
        assert area.energy_type == "grey"
        assert area.rule == "night_charge_evening_discharge"


def test_batteries_sell_when_there_is_no_pv(community):
    """The point of D-75: 56 of 96 periods have no supply without them."""
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    areas = profiles.make_battery_areas(community.bess, players, seed=7)
    evening = profiles.EVENING_PERIODS[0] - 1
    assert sum(a.slot_energy(evening) for a in areas) > 0
    lit_period = 50                                   # inside the PV window
    assert sum(a.slot_energy(lit_period) for a in areas) == 0


def test_participation_is_the_green_share_axis(community):
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    full = profiles.make_battery_areas(community.bess, players,
                                       participation=1.0, seed=7)
    half = profiles.make_battery_areas(community.bess, players,
                                       participation=0.5, seed=7)
    none = profiles.make_battery_areas(community.bess, players,
                                       participation=0.0, seed=7)
    assert len(half) == round(0.5 * len(full))
    assert none == []
    assert half == profiles.make_battery_areas(community.bess, players,
                                               participation=0.5, seed=7)
    with pytest.raises(profiles.ProfileError):
        profiles.make_battery_areas(community.bess, players, participation=1.5)
    with pytest.raises(profiles.ProfileError):
        profiles.make_battery_areas(community.bess, players, rule="charge_pv")


# ----------------------------------------------------------------- the week

@pytest.fixture(scope="module")
def week(community):
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    return profiles.extend_to_week(community.day_kwh(players), seed=4242)


def test_the_week_has_672_slots_and_is_not_one_day_seven_times(week):
    assert week.days == 7
    assert week.n_slots == 672
    assert all(len(v) == 672 for v in week.load_kwh.values())
    assert all(len(v) == 672 for v in week.pv_kwh.values())

    daily = [sum(sum(week.load_kwh[p][d * 96:(d + 1) * 96])
                 for p in week.players) for d in range(7)]
    assert len(set(round(x, 6) for x in daily)) == 7, "days must differ"
    assert all(v >= 0 for series in week.load_kwh.values() for v in series)
    assert all(v >= 0 for series in week.pv_kwh.values() for v in series)
    assert week.seed == 4242
    assert week.sigma_load > 0 and week.sigma_pv > 0


def test_the_week_regenerates_from_its_seed(community):
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    day = community.day_kwh(players)
    first = profiles.extend_to_week(day, seed=4242)
    again = profiles.extend_to_week(day, seed=4242)
    other = profiles.extend_to_week(day, seed=4243)
    assert first.load_kwh == again.load_kwh
    assert first.pv_kwh == again.pv_kwh
    assert first.load_kwh != other.load_kwh


def test_the_week_has_a_ratio_span_above_a_quarter(community, week):
    """A fixed-ratio sequence makes the calibration look fitted with nothing
    reporting an error, so the span is measured, not assumed."""
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    batteries = profiles.make_battery_areas(community.bess, players, seed=7)

    ratios = []
    for index in range(week.n_slots):
        scen = scenario.make_scenario_from_profiles(week, index, batteries)
        if scen["posted_supply_kwh"] > 0 and scen["posted_demand_kwh"] > 0:
            ratios.append(scen["posted_supply_kwh"] / scen["posted_demand_kwh"])

    assert ratios, "the week must contain two-sided slots"
    assert max(ratios) - min(ratios) > 0.25
