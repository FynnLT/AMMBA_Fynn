"""Section 5: run isolation, five seeds per cell, one parameter convention."""
import ast
import json
from pathlib import Path

import pytest

import campaign
import runs

HARNESS = Path(__file__).resolve().parent.parent / "harness"


def test_five_seeds_per_cell_recorded_individually():
    """D-59. Not "seeds 0-4": one deviating cell has to be re-runnable."""
    block1 = campaign.cells(1)
    block2 = campaign.cells(2)
    # Written out rather than derived from the grid under test: a count
    # computed from `cells()` itself cannot notice the grid losing a cell.
    #
    # Block 1: the baseline, two orders x three densities, two reciprocity
    # sensitivities, two multiplier formulations, the three-cell D-83 overlap
    # group and the five-point green-share axis --
    # 1 + 6 + 2 + 2 + 3 + 5 = 19 cells, 95 runs.
    assert len(block1) == 95
    # Block 2: two noise references (D-84), two arms x (three shares + three
    # coalition sizes) -- 2 + 12 = 14 cells, 70 runs.
    assert len(block2) == 70

    for specs, n_cells, n_runs in ((block1, 19, 95), (block2, 14, 70)):
        by_cell = {}
        for spec in specs:
            by_cell.setdefault(spec.cell, []).append(spec.seed)
        assert len(by_cell) == n_cells
        for cell, seeds in by_cell.items():
            assert seeds == list(campaign.SEEDS), cell
        assert all(spec.cell_seeds == campaign.SEEDS for spec in specs)
        # Every replicate gets its own week, which is the only stochastic
        # input a profile run has.
        assert len({s.dataset_extension_seed for s in specs}) == \
            len(campaign.SEEDS)
        assert len({s.run_id for s in specs}) == n_runs

    # The groups are disjoint and `cells()` is their union plus calibration.
    # The sweep's cells are named `b2_*` because they are block 2's
    # configuration with one parameter moved, but they are their own group
    # and must not collide with the cells they are read against.
    sweep = campaign.cells("sweep")
    assert not {s.cell for s in block1} & {s.cell for s in block2}
    assert not {s.cell for s in sweep} & ({s.cell for s in block1}
                                          | {s.cell for s in block2})
    assert len(campaign.cells()) == 95 + 70 + 60 + 6


def test_block_two_executes_and_block_one_does_not():
    """D-72. `RunSpec.execute` defaulted to False and no cell set it, so no
    campaign run has ever executed. Block 2 is the group that does."""
    for spec in campaign.cells(2):
        assert spec.execute is True, spec.cell
        assert isinstance(spec.deviation, dict), spec.cell
        assert spec.deviation["arm"] in campaign.deviations.ARMS
        # `b2_noise_off` is the one cell that runs without noise (D-84); every
        # other cell, strategic or not, carries the campaign sigma.
        assert spec.deviation["sigma"] == (
            0.0 if spec.cell == "b2_noise_off" else campaign.SIGMA), spec.cell
        assert spec.deviation["seed"] == campaign.DEVIATION_SEED
        assert spec.eta_relative == campaign.ETA_RELATIVE
        # Read off the execution node's own configuration, never restated.
        assert spec.gamma == campaign.execution_gamma()
        # Block 2 measures the penalty layer, not the preference one.
        assert spec.preferences == campaign.BASELINE_PREFERENCES
        assert spec.named_share == 0.0

    for spec in campaign.cells(1):
        assert spec.execute is False, spec.cell
        assert spec.deviation is None, spec.cell


def test_the_deviation_axes_each_vary_one_thing():
    """`_s10/_s25/_s50` moves the share at k = 1; `_k02/_k05/_k10` moves the
    coalition at the reference share. A cell that moved both would not be a
    point on either axis."""
    by_cell = {spec.cell: spec for spec in campaign.cells(2)}

    for arm, prefix in (("sellers_withhold", "b2_sell"),
                        ("buyers_underreport", "b2_buy")):
        shares = [by_cell[f"{prefix}_s{campaign.pct(s):02d}"]
                  for s in campaign.DEVIATION_SHARES]
        assert [s.deviation["share"] for s in shares] == \
            list(campaign.DEVIATION_SHARES)
        assert {s.deviation["k"] for s in shares} == {1}
        assert {s.deviation["arm"] for s in shares} == {arm}

        coalitions = [by_cell[f"{prefix}_k{k:02d}"]
                      for k in campaign.COALITION_SIZES]
        assert [s.deviation["k"] for s in coalitions] == \
            list(campaign.COALITION_SIZES)
        assert {s.deviation["share"] for s in coalitions} == \
            {campaign.REFERENCE_SHARE}
        assert {s.deviation["arm"] for s in coalitions} == {arm}

    noise = by_cell["b2_noise_only"]
    assert noise.deviation["arm"] == "none"
    assert noise.deviation["k"] == 0
    assert noise.deviation["sigma"] == campaign.SIGMA


def test_the_two_noise_references_differ_only_in_sigma():
    """D-84. The buyer externality carries no eta deadband -- that is the
    mechanism's own specification -- so `b2_noise_only` has a buyer-side
    penalty floor that the named deviator's signal is read against.
    `b2_noise_off` is what separates the two, and it is only a separation if
    nothing else moves between them."""
    by_cell = {spec.cell: spec for spec in campaign.cells(2) if spec.seed == 0}
    on, off = by_cell["b2_noise_only"], by_cell["b2_noise_off"]

    assert on.deviation["sigma"] == campaign.SIGMA > 0
    assert off.deviation["sigma"] == 0.0
    assert {k: v for k, v in on.deviation.items() if k != "sigma"} == \
        {k: v for k, v in off.deviation.items() if k != "sigma"}
    assert off.deviation["arm"] == "none" and off.deviation["k"] == 0

    # The noise stays on in every other arm: the floor is a finding about the
    # mechanism, not something to configure away.
    strategic = [s for s in campaign.cells(2)
                 if s.deviation["arm"] != "none"]
    assert strategic and all(s.deviation["sigma"] == campaign.SIGMA
                             for s in strategic)

    def without(spec, *keys):
        return {k: v for k, v in spec.config().items() if k not in keys}

    assert without(on, "deviation", "cell") == without(off, "deviation", "cell")


# ------------------------------------------------- the gamma/eta sweep

def test_the_sweep_is_twelve_cells_at_five_seeds():
    """D-26/B-10. The parameter axis block 2 does not carry: all 70 of its
    runs sat at one point of the (gamma, eta) plane."""
    specs = campaign.cells("sweep")
    # Written out rather than derived from the grid under test, as above: a
    # count computed from `sweep_grid()` itself cannot notice it losing a
    # cell. Four eta points, four gamma points on the shortfall arm, three on
    # the withholding arm, one share control -- 12 cells, 60 runs.
    assert len(specs) == 60

    by_cell = {}
    for spec in specs:
        by_cell.setdefault(spec.cell, []).append(spec.seed)
    assert len(by_cell) == 12
    assert sorted(by_cell) == [
        "b2_eta000", "b2_eta005", "b2_eta015", "b2_eta025",
        "b2_sell_s20",
        "b2_sell_s25_g150", "b2_sell_s25_g200", "b2_sell_s25_g300",
        "b2_short_g110", "b2_short_g150", "b2_short_g200", "b2_short_g300"]
    for cell, seeds in by_cell.items():
        assert seeds == list(campaign.SEEDS), cell
    assert len({s.run_id for s in specs}) == 60

    assert campaign.parse_block(["campaign.py", "--block", "sweep"]) == "sweep"
    # And the reference point is deliberately absent: `b2_sell_s25` at
    # `eta_relative = 0.10` *is* it, and `write_manifest` appends, so a
    # re-run would add a second entry to the manifest 5.3 is written against.
    assert campaign.ETA_RELATIVE not in campaign.SWEEP_ETAS


def test_each_sweep_axis_varies_exactly_one_parameter():
    """An axis is only an axis if one thing moves along it. Both of these
    are read against `b2_sell_s25`, which is the point they omit."""
    specs = campaign.cells("sweep")
    for spec in specs:
        assert spec.execute is True, spec.cell
        assert isinstance(spec.deviation, dict), spec.cell
        assert spec.deviation["arm"] in campaign.deviations.ARMS, spec.cell
        # Explicit, never left to the service's configuration: a sweep CSV
        # whose own parameter is not in its manifest cannot be read at all.
        assert spec.eta_relative is not None, spec.cell
        assert spec.gamma is not None, spec.cell
        assert spec.deviation["sigma"] == campaign.SIGMA, spec.cell
        assert spec.deviation["seed"] == campaign.DEVIATION_SEED, spec.cell
        assert spec.deviation["k"] == 1, spec.cell
        assert spec.preferences == campaign.BASELINE_PREFERENCES, spec.cell
        assert spec.participation == campaign.REFERENCE_PARTICIPATION
        assert spec.sigmoid == campaign.CALIBRATED_SIGMOID, spec.cell

    by_cell = {spec.cell: spec for spec in specs if spec.seed == 0}

    def without(spec, *keys):
        data = spec.config()
        for key in ("cell", "seed", "dataset_extension_seed") + keys:
            data.pop(key, None)
        return json.dumps(data, sort_keys=True)

    # The eta axis: `eta_relative` moves, on the withholding arm, at the
    # configured gamma. Nothing else.
    eta_cells = [by_cell[f"b2_eta{campaign.pct(e):03d}"] for e in campaign.SWEEP_ETAS]
    assert [c.eta_relative for c in eta_cells] == list(campaign.SWEEP_ETAS)
    assert len({without(c, "eta_relative") for c in eta_cells}) == 1
    assert {c.deviation["arm"] for c in eta_cells} == {"sellers_withhold"}
    assert {c.gamma for c in eta_cells} == {campaign.execution_gamma()}

    # The gamma axis: `gamma` moves, on the under-delivery arm, which is the
    # only arm the shortfall term -- and therefore gamma -- is visible in.
    gamma_cells = [by_cell[f"b2_short_g{campaign.pct(g):03d}"]
                   for g in campaign.SWEEP_GAMMAS]
    assert [c.gamma for c in gamma_cells] == list(campaign.SWEEP_GAMMAS)
    assert len({without(c, "gamma") for c in gamma_cells}) == 1
    assert {c.deviation["arm"] for c in gamma_cells} == {"sellers_underdeliver"}
    assert {c.eta_relative for c in gamma_cells} == {campaign.ETA_RELATIVE}


def test_the_sweep_controls_move_one_thing_off_the_block_two_cell():
    """`b2_sell_s25_g300` is the far end of the measured demonstration that
    gamma does not touch withholding: it is `b2_sell_s25` at gamma 3.0 and is
    expected to be identical to it in every penalty column. `b2_sell_s20`
    fills the gap in block 2's share axis at the reference eta."""
    reference = next(s for s in campaign.cells(2)
                     if s.cell == "b2_sell_s25" and s.seed == 0)
    by_cell = {s.cell: s for s in campaign.cells("sweep") if s.seed == 0}

    def without(spec, *keys):
        data = spec.config()
        for key in ("cell",) + keys:
            data.pop(key, None)
        return data

    at_gamma_three = by_cell["b2_sell_s25_g300"]
    assert at_gamma_three.gamma == 3.0
    assert reference.gamma == campaign.execution_gamma() != 3.0
    assert without(at_gamma_three, "gamma") == without(reference, "gamma")

    at_share_twenty = by_cell["b2_sell_s20"]
    assert at_share_twenty.deviation["share"] == 0.20
    assert reference.deviation["share"] == 0.25
    assert without(at_share_twenty, "deviation") == \
        without(reference, "deviation")
    assert {k: v for k, v in at_share_twenty.deviation.items() if k != "share"} \
        == {k: v for k, v in reference.deviation.items() if k != "share"}


def test_the_withholding_gamma_group_is_three_cells_differing_only_in_gamma():
    """Table 5.7 reports the honest-seller shortfall at four gammas on the
    withholding arm and only two of them were run; 1.5 and 2.0 were derived
    from the exact linearity of `Phi`. A derived value in a results table
    reads as data, so the group is now measured end to end.

    The three cells here plus `b2_sell_s25` are those four points. They share
    an arm, a share, an eta, a deviation seed and a sigma, so the only thing
    that moves along the group is gamma.
    """
    grid = campaign.sweep_grid()
    assert len(grid) == 12

    names = ["b2_sell_s25_g150", "b2_sell_s25_g200", "b2_sell_s25_g300"]
    assert set(names) <= set(grid)

    # The gammas are the non-default points of `SWEEP_GAMMAS`, read off the
    # axis rather than restated: "default" is `execution_gamma()`, and the
    # default point at the reference eta *is* `b2_sell_s25`.
    default = campaign.execution_gamma()
    expected = [g for g in campaign.SWEEP_GAMMAS if g != default]
    assert expected == [1.5, 2.0, 3.0]
    assert [grid[n]["gamma"] for n in names] == expected

    for name in names:
        cell = grid[name]
        assert cell["deviation"]["arm"] == campaign.deviations.SELLER_ARM
        assert cell["deviation"]["share"] == campaign.REFERENCE_SHARE
        assert cell["eta_relative"] == campaign.ETA_RELATIVE

    def without_gamma(cell):
        return json.dumps({k: v for k, v in cell.items() if k != "gamma"},
                          sort_keys=True)

    assert len({without_gamma(grid[n]) for n in names}) == 1


def test_no_sweep_cell_that_already_has_runs_lost_its_name():
    """The ten cells of the 18.09. sweep have manifests under `out/` and 5.3
    is written against them. A renamed cell would orphan those runs and
    `--resume` would re-run it, so the old names are asserted explicitly
    rather than as a count -- a count cannot tell a rename from an addition.
    """
    grid = campaign.sweep_grid()
    already_run = [
        "b2_eta000", "b2_eta005", "b2_eta015", "b2_eta025",
        "b2_sell_s20", "b2_sell_s25_g300",
        "b2_short_g110", "b2_short_g150", "b2_short_g200", "b2_short_g300"]
    missing = sorted(set(already_run) - set(grid))
    assert not missing, missing
    # And the only names that are new are the two this task adds.
    assert sorted(set(grid) - set(already_run)) == ["b2_sell_s25_g150",
                                                    "b2_sell_s25_g200"]


def test_the_density_cells_differ_in_nothing_but_the_two_shares():
    """D-71. The order axis is only an axis if the books it compares are
    populated the same way."""
    by_cell = {}
    for spec in campaign.cells(1):
        by_cell.setdefault(spec.cell, []).append(spec)

    density_cells = [c for c in by_cell
                     if c.startswith(("prefs_first_d", "pro_rata_first_d"))]
    assert len(density_cells) == 8        # 2 orders x 3 densities + 2 mutual

    def fingerprint(spec):
        data = spec.config()
        for key in ("named_share", "mutual_share", "preferences", "seed",
                    "cell", "dataset_extension_seed"):
            data.pop(key)
        return json.dumps(data, sort_keys=True)

    assert len({fingerprint(by_cell[c][0]) for c in density_cells}) == 1, \
        "the density cells differ in more than the two shares"

    # Same three densities on both orders, at the reference reciprocity.
    for prefix in ("prefs_first", "pro_rata_first"):
        cells = [by_cell[f"{prefix}_d{campaign.pct(d):03d}"][0]
                 for d in campaign.DENSITIES]
        assert [c.named_share for c in cells] == list(campaign.DENSITIES)
        assert {c.mutual_share for c in cells} == {campaign.REFERENCE_MUTUAL}

    # And the two orders differ in nothing but `order`.
    left = by_cell["prefs_first_d050"][0].preferences
    right = by_cell["pro_rata_first_d050"][0].preferences
    assert left["order"] == "preferences_first"
    assert right["order"] == "pro_rata_first"
    assert {k: v for k, v in left.items() if k != "order"} == \
        {k: v for k, v in right.items() if k != "order"}

    # The reciprocity sensitivity moves `mutual_share` at the same density.
    sensitivity = [by_cell[f"prefs_first_d050_m{campaign.pct(m):03d}"][0]
                   for m in campaign.MUTUAL_SENSITIVITY]
    assert [s.mutual_share for s in sensitivity] == \
        list(campaign.MUTUAL_SENSITIVITY)
    assert {s.named_share for s in sensitivity} == {0.50}


def test_the_two_multiplier_cells_differ_only_in_mode():
    """D-46. The two formulations differ at every measured parameter point;
    the comparison is only a comparison if nothing else moves between them."""
    by_cell = {spec.cell: spec for spec in campaign.cells(1)}
    multiplicative = by_cell["multipliers_on"]
    additive = by_cell["multipliers_on_additive"]

    assert multiplicative.preferences["mode"] == "multiplicative"
    assert additive.preferences["mode"] == "additive"
    assert {k: v for k, v in multiplicative.preferences.items() if k != "mode"} \
        == {k: v for k, v in additive.preferences.items() if k != "mode"}

    def without(spec, *keys):
        return {k: v for k, v in spec.config().items() if k not in keys}

    # `cell` is the name, which is the one thing that must differ.
    assert without(multiplicative, "preferences", "cell") == \
        without(additive, "preferences", "cell")

    # D-79: G = 0.02, below the crossover of the campaign mix, not 0.03.
    for spec in (multiplicative, additive):
        assert spec.preferences["green_multiplier"] == 0.02
        assert spec.preferences["grey_levy"] == 0.10
        assert spec.preferences["levy_cap"] == 0.20
        assert spec.preferences["sides"] == "seller"
        assert spec.preferences["multipliers_enabled"] is True


# ---------------------------------------------------------- section 6

def test_the_calibration_cells_differ_in_nothing_but_the_extension_seed():
    """Code To-Do 4.6. The six weeks T-19 fitted on, from one command."""
    import fit

    specs = campaign.calibration_cells()
    assert len(specs) == 6
    assert [s.run_id for s in specs] == \
        [f"calib-{seed}" for seed in fit.DEFAULT_STABILITY_SEEDS] + ["oos-4242"]
    assert [s.dataset_extension_seed for s in specs] == \
        list(fit.DEFAULT_STABILITY_SEEDS) + [4242]

    def fingerprint(spec):
        data = spec.config()
        data.pop("dataset_extension_seed")
        return json.dumps(data, sort_keys=True)

    assert len({fingerprint(s) for s in specs}) == 1
    assert all(s.cell == "calibration" for s in specs)
    assert all(s.cell_seeds == (0,) and s.seed == 0 for s in specs)
    assert all(s.execute is False and s.sigmoid is None for s in specs)
    assert all(s.participation == campaign.REFERENCE_PARTICIPATION
               for s in specs)
    assert all(s.preferences == campaign.PREF_OFF for s in specs)


def test_the_calibration_seeds_are_disjoint_from_the_campaign_weeks():
    """A calibration week that is also a campaign replicate would fit the
    sigmoid on the data it is then evaluated against."""
    import fit

    campaign_weeks = {campaign.EXTENSION_SEED_BASE + s for s in campaign.SEEDS}
    assert not set(fit.DEFAULT_STABILITY_SEEDS) & campaign_weeks


def test_the_calibration_cells_match_the_manifest_they_reproduce():
    """`out/calib-20260917/manifest.json` is the specification for
    `calibration_cells`, so it is read rather than paraphrased."""
    manifest_path = (Path(__file__).resolve().parent.parent / "out"
                     / "calib-20260917" / "manifest.json")
    if not manifest_path.exists():                     # pragma: no cover
        pytest.skip("the fitted week's manifest is not in this checkout")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))[0]

    spec = next(s for s in campaign.calibration_cells()
                if s.run_id == recorded["run_id"])
    config = spec.config()
    # Only the fields the recorded manifest carries: `RunSpec` has gained
    # `named_share`, `mutual_share`, `preference_seed`, `deviation` and
    # `deviation_seed` since, and they sit at their defaults here.
    for key, value in recorded["config"].items():
        # JSON has no tuples: `cell_seeds` comes back as a list.
        recorded_value = tuple(value) if isinstance(value, list) else value
        assert config[key] == recorded_value, key
    for key in ("named_share", "deviation", "deviation_seed"):
        assert key not in recorded["config"]
    assert config["named_share"] == 0.0
    assert config["deviation"] is None


def test_the_block_argument_selects_a_group():
    assert campaign.parse_block(["campaign.py"]) is None
    assert campaign.parse_block(["campaign.py", "--block", "1"]) == 1
    assert campaign.parse_block(["campaign.py", "--block", "2"]) == 2
    assert campaign.parse_block(["campaign.py", "--calibration"]) == "calibration"
    assert campaign.parse_block(["campaign.py", "--block", "calibration"]) == \
        "calibration"
    with pytest.raises(SystemExit):
        campaign.parse_block(["campaign.py", "--block", "3"])


def test_the_out_argument_redirects_every_run(tmp_path):
    """`--calibration` into the live `out/` would append to the manifests of
    the five weeks T-19 fitted on: `write_manifest` appends, it does not
    replace. The redirect is what makes the command safe to run at all."""
    assert campaign.parse_out(["campaign.py"]) is None
    assert campaign.parse_out(["campaign.py", "--out", "somewhere"]) == \
        "somewhere"

    specs = campaign.calibration_cells()
    assert all(spec.out_dir is None for spec in specs)

    moved = campaign.redirect(specs, str(tmp_path))
    assert [Path(spec.out_dir) for spec in moved] == \
        [tmp_path / spec.run_id for spec in specs]
    # Recorded in the manifest, because it is set on the spec rather than
    # passed beside it.
    assert all(spec.config()["out_dir"] == spec.out_dir for spec in moved)
    # And everything else is untouched.
    for before, after in zip(specs, moved):
        assert before.config() | {"out_dir": after.out_dir} == after.config()

    assert campaign.redirect(specs, None) is specs


def test_the_green_share_axis_varies_participation_and_nothing_else():
    """D-76. The axis is only an axis if one thing moves along it.

    The four mechanism cells stay at the reference participation, so a change
    measured along this axis cannot be confused with one measured across the
    preference variants.
    """
    specs = campaign.cells(1)
    axis = [s for s in specs if s.cell.startswith("green_share_")]
    assert len(axis) == len(campaign.GREEN_SHARES) * len(campaign.SEEDS)

    by_cell = {}
    for spec in axis:
        by_cell.setdefault(spec.cell, []).append(spec)
    assert sorted(by_cell) == ["green_share_000", "green_share_025",
                               "green_share_050", "green_share_075",
                               "green_share_100"]
    assert {run.participation for cell in by_cell.values() for run in cell} == \
        set(campaign.GREEN_SHARES)
    # Nothing but participation moves: same preference set as the baseline
    assert all(s.preferences == campaign.BASELINE_PREFERENCES for s in axis)

    mechanism = [s for s in specs if not s.cell.startswith("green_share_")]
    assert all(s.participation == campaign.REFERENCE_PARTICIPATION
               for s in mechanism)


def test_the_delta_baseline_is_named_once():
    """Named in the manifest, never re-derived per table."""
    specs = campaign.cells()
    assert all(spec.baseline == runs.BASELINE for spec in specs)
    baseline = [s for s in specs if s.cell == "baseline_pro_rata"]
    assert len(baseline) == len(campaign.SEEDS)
    # The baseline is produced by the same code path as everything it is
    # subtracted from: preferences disabled, nothing else special.
    assert all(s.preferences == {"enabled": False,
                                 "multipliers_enabled": False}
               for s in baseline)


def test_the_spawn_start_method_is_used():
    """Under fork a child would inherit the parent's already-rewritten
    sys.modules and the three service packages would be half-loaded."""
    source = (HARNESS / "campaign.py").read_text(encoding="utf-8")
    assert 'get_context("spawn")' in source
    assert "get_context(\"fork\")" not in source


def test_campaign_runs_do_not_set_preferences_through_the_stack():
    """One convention only: preferences and sigmoid parameters go through the
    trigger, which run_clearing resolves. Two ways of setting the same thing
    is how a manifest stops describing the run it names."""
    for name in ("campaign.py", "runs.py", "runner.py"):
        tree = ast.parse((HARNESS / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "clear"):
                continue
            keywords = {kw.arg for kw in node.keywords}
            assert "preferences" not in keywords, (
                f"{name}: Stack.clear(preferences=...) in a campaign path; "
                f"use trigger['preference_params']")


def test_the_trigger_carries_both_parameter_blocks():
    """Where the one convention actually lands."""
    source = (HARNESS / "runner.py").read_text(encoding="utf-8")
    assert '"sigmoid_params"' in source
    assert '"preference_params"' in source


# One day at 20 players rather than a week at 100, so the property is checked
# on every test run instead of being deselected into never running. The full
# campaign scale is measured in the section's acceptance run, not here.
ISOLATION_RUN = {"days": 1, "n_players": 20}


def test_parallel_and_sequential_produce_identical_results(tmp_path):
    """Four runs, both paths: parallel is faster and the results are the same.

    Same results is the load-bearing half. Each worker builds its own `Stack`
    and therefore its own empty store, so there is no shared order book for
    two runs to collide in -- which is what makes the process boundary a
    substitute for reset choreography and slot offsets.
    """
    import time

    def specs(prefix):
        return [campaign.spec_for("isolation", seed,
                                  out_dir=str(tmp_path / f"{prefix}-{seed}"),
                                  **ISOLATION_RUN)
                for seed in range(4)]

    started = time.perf_counter()
    sequential = campaign.run_sequential(specs("seq"), sha="0" * 40)
    sequential_sec = time.perf_counter() - started

    started = time.perf_counter()
    parallel = campaign.run_parallel(specs("par"), processes=4, sha="0" * 40)
    parallel_sec = time.perf_counter() - started

    assert parallel_sec < sequential_sec, (
        f"parallel {parallel_sec:.1f}s vs sequential {sequential_sec:.1f}s")

    for seq, par in zip(sequential, parallel):
        assert seq["run_id"] == par["run_id"]
        assert seq["checks"] == par["checks"]
        assert (Path(seq["csv"]).read_text(encoding="utf-8")
                == Path(par["csv"]).read_text(encoding="utf-8"))

    # And the manifests agree on everything except when they were written.
    for seq, par in zip(sequential, parallel):
        a, b = dict(seq["manifest"]), dict(par["manifest"])
        for entry in (a, b):
            entry.pop("timestamp_utc")
            entry.pop("wall_sec")
            entry["config"].pop("out_dir")
        assert a == b


def test_cell_names_survive_float_error():
    assert campaign.pct(2.3) == 230
    assert campaign.pct(0.29) == 29
    assert campaign.pct(2.01) != campaign.pct(2.0)