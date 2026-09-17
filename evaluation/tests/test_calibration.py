"""T-19 calibration: the four properties the fit is only usable if it has.

Each of these stands for a way the calibration could return a number that
looks like a result and is not.
"""
import csv

import pytest

import fit
import objective as calib
import stack

# The band the campaign and the calibration both run in (D-77).
K_UPPER, K_LOWER = 40.0, 8.0

# The ratios a profile week actually occupies, with room either side. The
# agreement test has to cover where the data is, not a neighbourhood of theta.
RATIOS = [0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def configured_params():
    """theta and B as `configuration.yaml` currently ships them."""
    config_module = stack.CLR["config"]
    community = config_module.resolve_community(config_module.load_config(),
                                                "communityid_1")
    return community


def test_unclamped_matches_the_clearing_node_inside_the_band():
    """The calibration's own copy of the sigmoid is the same curve.

    `objective.sigmoid_price_unclamped` exists because the settlement copies
    snap to the band and the penalty term needs the unsnapped value. That is
    only a defensible reason as long as the two agree everywhere the snapping
    does not apply -- otherwise the calibration is fitting a different curve
    from the one the artifact prices with, and every fitted theta is wrong in
    a way no other test would show.
    """
    node_sigmoid = stack.CLR["sigmoid"].sigmoid_price
    community = configured_params()

    for ratio in RATIOS:
        mine = calib.sigmoid_price_unclamped(
            ratio, K_UPPER, K_LOWER, community.theta, community.steepness)
        theirs = node_sigmoid(ratio, K_UPPER, K_LOWER,
                              community.theta, community.steepness)
        assert mine == pytest.approx(theirs, abs=1e-9), f"ratio {ratio}"
        assert K_LOWER < mine < K_UPPER


def slots_at(ratios, k_upper=K_UPPER, k_lower=K_LOWER):
    """Slots at given ratios, demand fixed at 10 kWh."""
    return [calib.Slot(total_supply_kwh=10.0 * r, total_demand_kwh=10.0,
                       k_upper=k_upper, k_lower=k_lower) for r in ratios]


def test_alpha_zero_collapses_the_fit_onto_a_flat_curve():
    """The degeneracy the penalty term exists to rule out.

    Without the penalty the objective is maximised by a price that stays as
    far from both bounds as it can, everywhere -- which is a horizontal line
    through the band midpoint, i.e. a steepness of zero and a clearing price
    that ignores the supply/demand ratio completely. That is not a fitting
    artefact to be worked around; it is the property that makes alpha part of
    the model, and D-74 reports the sweep because of it. If this test ever
    fails, the penalty term has stopped being load-bearing and the sweep
    stops meaning anything.
    """
    slots = slots_at([0.4, 0.7, 1.0, 1.4, 2.0, 2.6])
    result = fit.fit(slots, alpha=0.0, coarse=40, fine=20)

    assert result.steepness < 0.05
    midpoint = (K_UPPER + K_LOWER) / 2
    for ratio in (0.4, 1.0, 2.6):
        price = calib.sigmoid_price_unclamped(ratio, K_UPPER, K_LOWER,
                                              result.theta, result.steepness)
        assert price == pytest.approx(midpoint, abs=0.05)


def test_a_slot_that_did_not_trade_is_dropped(tmp_path):
    """No-trade slots carry no ratio, and a fit over them is meaningless.

    The harness deliberately keeps them in the CSV so the round-type census
    sees the empty half of the week (`aggregates.slot_row`). The calibration
    is the consumer that must not.
    """
    path = tmp_path / "slots.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "slot", "total_supply_kwh", "total_demand_kwh", "ratio",
            "clearing_price_ct", "traded_kwh", "n_producers", "n_consumers"))
        writer.writeheader()
        # cleared
        writer.writerow({"slot": 0, "total_supply_kwh": 12.5,
                         "total_demand_kwh": 10.0, "ratio": 1.25,
                         "clearing_price_ct": 15.147274, "traded_kwh": 10.0,
                         "n_producers": 4, "n_consumers": 6})
        # night: nothing generated, so the node reported no trade at all
        writer.writerow({"slot": 900, "total_supply_kwh": 0.0,
                         "total_demand_kwh": 8.0, "ratio": "",
                         "clearing_price_ct": "", "traded_kwh": "",
                         "n_producers": 0, "n_consumers": 6})
        # nobody demanded anything: also undefined, and the ratio column is
        # empty because the node divides by demand
        writer.writerow({"slot": 1800, "total_supply_kwh": 4.0,
                         "total_demand_kwh": 0.0, "ratio": "",
                         "clearing_price_ct": "", "traded_kwh": "",
                         "n_producers": 3, "n_consumers": 0})

    loaded = calib.load_slots(path, k_upper=K_UPPER, k_lower=K_LOWER)

    assert loaded.dropped == 2
    assert loaded.total == 3
    assert [s.ratio for s in loaded.slots] == [1.25]


def test_stability_is_deterministic_for_a_fixed_seed_list():
    """The spread across weeks has to reproduce like the fit itself.

    `stability` is what says whether the fitted pair is a property of the
    objective or of the two slots that happen to anchor the penalty term. A
    spread that came out differently on each invocation could not settle that
    question either way.
    """
    # One distinct week per seed, so a seed that was silently ignored would
    # show up as an identical row rather than passing unnoticed.
    weeks = {
        1: slots_at([0.3, 0.6, 0.9, 1.2, 1.8, 2.5]),
        2: slots_at([0.2, 0.5, 1.0, 1.4, 2.1, 2.8]),
        3: slots_at([0.4, 0.8, 1.1, 1.5, 1.9, 3.1]),
    }
    seeds = (1, 2, 3)

    first = fit.stability(seeds, alphas=(0.25, 0.5), coarse=30, fine=15,
                          slots_for_seed=weeks.__getitem__)
    second = fit.stability(seeds, alphas=(0.25, 0.5), coarse=30, fine=15,
                           slots_for_seed=weeks.__getitem__)

    assert first == second
    assert [row["seed"] for row in first["rows"]] == [1, 2, 3, 1, 2, 3]
    # Each week has its own extremes, so the r_min/r_max columns must differ
    assert first["extremes"]["1"] != first["extremes"]["2"]


def test_the_fit_is_deterministic():
    """`config + seed + SHA` has to reproduce the fitted pair.

    This is the property the grid was chosen over a solver for, so it is
    tested rather than assumed.
    """
    slots = slots_at([0.3, 0.6, 0.9, 1.1, 1.6, 2.2, 2.9])

    first = fit.fit(slots, alpha=0.5, coarse=40, fine=20)
    second = fit.fit(slots, alpha=0.5, coarse=40, fine=20)

    assert first == second
    assert first.theta == second.theta
    assert first.steepness == second.steepness
    assert first.objective == second.objective
