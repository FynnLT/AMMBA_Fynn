"""Section 1: preferences on the profile path (D-71, D-81).

The first block-1 campaign produced byte-identical slot CSVs for
`preferences_first` and `pro_rata_first` because `make_scenario_from_profiles`
set no `preferred_partner` at all -- the pairing block of `make_scenario` is
the synthetic generator's and was never on the profile path. Every test here
fails on that state.
"""
import pytest

import profiles
import scenario


PLAYERS = list(range(100))


def _areas(players=PLAYERS):
    return [scenario.area_uuid_for(p) for p in players]


class _Week:
    """A hand-built profile week: `players`, `load_kwh`, `pv_kwh` per slot.

    Same duck type `make_scenario_from_profiles` takes from
    `profiles.ProfileWeek`, small enough that the netting outcome of each slot
    is written down rather than computed.
    """

    def __init__(self, load, pv):
        self.players = list(load)
        self.load_kwh = load
        self.pv_kwh = pv
        self.n_slots = len(next(iter(load.values())))


# --------------------------------------------------------------- the draw

def test_no_named_share_reproduces_todays_book_exactly():
    """`named_share = 0` is the first pass, and has to stay reproducible.

    Dict equality on both sides of the book: not "no pairs were posted" but
    "no order differs in any field", which is what makes the second pass
    comparable to the runs already on disk.
    """
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=40, seed=20260916)
    week = profiles.extend_to_week(community.day_kwh(players), seed=4242,
                                   days=1)
    batteries = profiles.make_battery_areas(community.bess, players,
                                            participation=0.25, seed=7)

    assert scenario.draw_preferences(players, named_share=0.0,
                                     mutual_share=0.5, seed=1) == {}

    for index in (0, 40, 80, week.n_slots - 1):
        plain = scenario.make_scenario_from_profiles(week, index, batteries)
        drawn = scenario.make_scenario_from_profiles(week, index, batteries,
                                                     preferences={})
        assert drawn["producers"] == plain["producers"], index
        assert drawn["consumers"] == plain["consumers"], index
        assert drawn["pairs_posted"] == 0
        assert drawn["pairs_unpostable"] == 0
        assert drawn["pairs_mutual"] == 0
        assert not any("preferred_partner" in order
                       for order in plain["producers"] + plain["consumers"])


def test_mutual_share_is_the_realised_reciprocity_rate():
    """`mutual_share = 1` means every named household's partner names back,
    `mutual_share = 0` means none does -- realised, not in expectation."""
    everyone = scenario.draw_preferences(PLAYERS, named_share=0.5,
                                         mutual_share=1.0, seed=7)
    assert len(everyone) == 50
    for area, partner in everyone.items():
        assert everyone.get(partner) == area, (area, partner)

    nobody = scenario.draw_preferences(PLAYERS, named_share=0.5,
                                       mutual_share=0.0, seed=7)
    assert len(nobody) == 50
    for area, partner in nobody.items():
        assert nobody.get(partner) != area, (area, partner)

    # A household names exactly one partner, never itself, and never a
    # non-player -- a battery area cannot appear on either side (D-75).
    areas = set(_areas())
    for area, partner in everyone.items():
        assert area in areas and partner in areas and area != partner


def test_the_draw_is_deterministic_in_its_own_seed():
    """`random.Random(seed)`, never the module-level RNG: a campaign worker
    shares that one with everything else it imports."""
    first = scenario.draw_preferences(PLAYERS, named_share=0.4,
                                      mutual_share=0.5, seed=20260918)
    again = scenario.draw_preferences(PLAYERS, named_share=0.4,
                                      mutual_share=0.5, seed=20260918)
    other = scenario.draw_preferences(PLAYERS, named_share=0.4,
                                      mutual_share=0.5, seed=20260919)
    assert first == again
    assert first != other
    assert len(first) == len(other) == 40


# ------------------------------------------------------- posting the pairs

def test_a_partner_that_nets_to_zero_makes_the_pair_unpostable():
    """D-81. The pairs are held over the week, so a slot in which the partner
    has nothing to post is a counted outcome and not a redraw.

    Two slots, three players. In slot 0 player 0 buys and player 1 sells, so
    the pair is postable. In slot 1 player 1's load and PV cancel exactly, it
    posts nothing, and player 0's preference has no counterparty.
    """
    load = {0: [2.0, 2.0], 1: [1.0, 1.0], 2: [3.0, 3.0]}
    pv = {0: [0.0, 0.0], 1: [4.0, 1.0], 2: [0.0, 0.0]}
    week = _Week(load, pv)
    preferences = {"player-000": "player-001"}

    first = scenario.make_scenario_from_profiles(week, 0,
                                                 preferences=preferences)
    assert first["pairs_posted"] == 1
    assert first["pairs_unpostable"] == 0
    assert first["pairs_mutual"] == 0
    buyer = [c for c in first["consumers"] if c["area_uuid"] == "player-000"]
    assert buyer[0]["preferred_partner"] == "player-001"

    second = scenario.make_scenario_from_profiles(week, 1,
                                                  preferences=preferences)
    assert second["pairs_unpostable"] == 1
    assert second["pairs_posted"] == 0
    assert not any(order.get("preferred_partner") == "player-001"
                   for order in second["producers"] + second["consumers"])
    # Player 1 netted to zero, so it is not in the book at all.
    assert not any(order["area_uuid"] == "player-001"
                   for order in second["producers"] + second["consumers"])


def test_a_partner_on_the_same_side_yields_no_pair_either():
    """The clearing node drops a `preferred_partner` that is not a
    counterparty in the market, so posting one would trade a countable result
    for a warning in a log nobody reads over 672 slots."""
    load = {0: [2.0], 1: [2.0]}
    pv = {0: [0.0], 1: [0.0]}          # both net positive -> both are buyers
    scen = scenario.make_scenario_from_profiles(
        _Week(load, pv), 0, preferences={"player-000": "player-001",
                                         "player-001": "player-000"})
    assert scen["pairs_posted"] == 0
    assert scen["pairs_unpostable"] == 2
    assert not any("preferred_partner" in order
                   for order in scen["producers"] + scen["consumers"])


def test_a_mutual_pair_is_counted_once_and_posted_on_both_sides():
    load = {0: [2.0], 1: [1.0]}
    pv = {0: [0.0], 1: [4.0]}
    scen = scenario.make_scenario_from_profiles(
        _Week(load, pv), 0, preferences={"player-000": "player-001",
                                         "player-001": "player-000"})
    assert scen["pairs_posted"] == 2
    assert scen["pairs_mutual"] == 1
    assert scen["pairs_unpostable"] == 0
    assert scen["consumers"][0]["preferred_partner"] == "player-001"
    assert scen["producers"][0]["preferred_partner"] == "player-000"


@pytest.mark.parametrize("named_share, mutual_share",
                         [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_a_share_outside_the_unit_interval_is_refused(named_share,
                                                      mutual_share):
    with pytest.raises(scenario.ScenarioError):
        scenario.draw_preferences(PLAYERS, named_share=named_share,
                                  mutual_share=mutual_share, seed=1)
