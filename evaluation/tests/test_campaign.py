"""Section 5: run isolation, five seeds per cell, one parameter convention."""
import ast
import json
from pathlib import Path

import campaign
import runs

HARNESS = Path(__file__).resolve().parent.parent / "harness"


def test_five_seeds_per_cell_recorded_individually():
    """D-59. Not "seeds 0-4": one deviating cell has to be re-runnable."""
    specs = campaign.cells()
    # Four mechanism cells plus the five-point green-share axis (D-76), all at
    # five seeds. Written out rather than derived from `cells()` itself: a
    # count computed from the grid under test cannot notice the grid losing a
    # cell.
    assert len(specs) == 45
    by_cell = {}
    for spec in specs:
        by_cell.setdefault(spec.cell, []).append(spec.seed)
    assert len(by_cell) == 9
    for cell, seeds in by_cell.items():
        assert seeds == list(campaign.SEEDS), cell
    assert all(spec.cell_seeds == campaign.SEEDS for spec in specs)
    # Every replicate gets its own week, which is the only stochastic input a
    # profile run has.
    extension_seeds = {s.dataset_extension_seed for s in specs}
    assert len(extension_seeds) == len(campaign.SEEDS)
    assert len({s.run_id for s in specs}) == 45


def test_the_green_share_axis_varies_participation_and_nothing_else():
    """D-76. The axis is only an axis if one thing moves along it.

    The four mechanism cells stay at the reference participation, so a change
    measured along this axis cannot be confused with one measured across the
    preference variants.
    """
    specs = campaign.cells()
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
