"""DR2-DR5 over the recorded campaign runs (Chapter 3.4.3 -> Chapter 5).

    python dr_analysis.py [--runs DIR ...] [--dest DIR]
                          [--full-cells CELL ...] [--compare-to DIR]

`--runs` defaults to `evaluation/out`; `--dest` to
`evaluation/out/analysis/<YYYYMMDD>-<short sha>`, with `-dirty` appended
when the working tree has uncommitted changes.

`--full-cells` names the block-1 cells the DR2 check of the full artifact
(`dr2_full.py`) replays, all seeds each; the default is `dr2_full.CELLS`,
and a subset is for development runs. `--compare-to` compares the tables
this script wrote before that stage existed byte for byte with the files of
the same name in DIR, after writing, and prints the result.

**Read-only on run data.** The script never writes into a run directory,
never appends to a manifest and never re-runs anything; everything it
writes goes into `--dest`, and it refuses to start into a `--dest` that
already holds a `manifest.json` -- the same rule as `ncurve.py`.

One pass over the runs, each run's two CSVs loaded once. R1-R3 and the
single-deviator coverage check must pass before any DR table is written;
otherwise only `checks.json` is written and the script exits 1 naming the
failure. The other assertions (A6, the band reproduction, the census
against the manifests) are written with the tables and make the exit code 1
when one fails.

The DR2 check of the full artifact runs after the existing DR2 stage. Its
own hard checks (a missing cell or seed, the run preconditions, R4, R5, C1)
stop that stage only: the `dr2_full_*` tables are then not written, the
other tables are, and the exit code is 1.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import checks  # noqa: E402
import coalitions  # noqa: E402
import columns  # noqa: E402
import discovery  # noqa: E402
import dr2  # noqa: E402
import dr2_full  # noqa: E402
import dr3  # noqa: E402
import dr4  # noqa: E402
import dr5  # noqa: E402
import replay as R  # noqa: E402

EVALUATION_DIR = ANALYSIS_DIR.parent
REPO = EVALUATION_DIR.parent
DEFAULT_RUNS = EVALUATION_DIR / "out"
#: The honest reference cell: pro rata, sigma = 0, no deviator (D-84).
REFERENCE_CELL = "b2_noise_off"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args],
                                   text=True).strip()


def script_state() -> tuple:
    """(SHA, dirty) of the checkout this script runs from."""
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, dirty


def default_dest() -> Path:
    sha, dirty = script_state()
    name = f"{datetime.now():%Y%m%d}-{(sha or 'nogit')[:7]}"
    if dirty:
        name += "-dirty"
    return DEFAULT_RUNS / "analysis" / name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="DR2-DR5 analysis over recorded campaign runs. Reads "
                    "run directories, writes only into --dest.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help=f"run roots (default {DEFAULT_RUNS})")
    parser.add_argument("--dest", default=None,
                        help="output directory (default out/analysis/"
                             "<date>-<sha>[-dirty])")
    parser.add_argument("--full-cells", nargs="+", default=None,
                        metavar="CELL",
                        help="block-1 cells the DR2 check of the full "
                             "artifact replays (default: "
                             f"{' '.join(dr2_full.CELLS)})")
    parser.add_argument("--compare-to", default=None, metavar="DIR",
                        help="after writing, compare the tables that "
                             "predate the full-artifact stage byte for byte "
                             "with the files of the same name in DIR")
    return parser.parse_args(argv)


# ----------------------------------------------------------------- census

def _distinct(values) -> str:
    values = np.asarray(values, dtype=float)
    values = sorted(set(values[~np.isnan(values)].tolist()))
    return " ".join(f"{v:g}" for v in values)


def census_row(data) -> tuple:
    """One census row, and whether it agrees with the run's own records."""
    run, slots = data.run, data.slots
    price = slots.floats("clearing_price_ct")
    cleared = ~np.isnan(price)
    regime = R.regime_vec(slots.floats("total_supply_kwh"),
                          slots.floats("total_demand_kwh"))
    counts = {R.SUPPLY_LIMITED: int((cleared & (regime == R.SL)).sum()),
              R.DEMAND_LIMITED: int((cleared & (regime == R.DL)).sum()),
              R.BALANCED: int((cleared & (regime == R.BAL)).sum()),
              "no_trade": int((~cleared).sum())}
    recorded = ((run.entry.get("checks") or {}).get("round_type_census"))
    manifest_agrees = recorded is None or all(
        int(recorded.get(key, 0)) == value for key, value in counts.items())
    exec_agrees = None
    if run.executed:
        rte = slots.strings("round_type_exec")
        executed = rte != ""
        names = np.array([R.REGIME_NAME[int(c)] for c in regime], dtype=object)
        exec_agrees = bool((rte[executed] == names[executed]).all())
    band = R.Band.from_dict(run.sigmoid)
    row = {"cell": run.cell, "seed": run.seed, "run_id": run.run_id,
           "path": str(run.path), "repo_sha": run.repo_sha,
           "n_manifest_entries": run.n_entries,
           "sigmoid_source": run.sigmoid_source,
           "k_upper": band.k_upper, "k_lower": band.k_lower,
           "theta": band.theta, "steepness": band.steepness,
           "executed": run.executed, "arm": run.arm,
           "param_source": data.param_source,
           "n_slots": slots.n, "n_cleared": int(cleared.sum()),
           "n_supply_limited": counts[R.SUPPLY_LIMITED],
           "n_demand_limited": counts[R.DEMAND_LIMITED],
           "n_balanced": counts[R.BALANCED], "n_no_trade": counts["no_trade"],
           "manifest_census_agrees": manifest_agrees,
           "exec_round_type_agrees": exec_agrees,
           **{name: (_distinct(slots.floats(name)) if slots.has(name)
                     else None)
              for name in ("gamma_eff", "eta_relative_eff",
                           "k_sho_ct_per_kwh")}}
    return row, manifest_agrees and exec_agrees is not False


# ---------------------------------------------------------------- output

def write_table(dest: Path, name: str, rows: list) -> None:
    fields = columns.names(name)
    undocumented = sorted({key for row in rows for key in row} - set(fields))
    if undocumented:
        raise RuntimeError(f"{name}: columns {undocumented} are not in "
                           f"columns.py; document them before writing them")
    with open(dest / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def compare_tables(dest: Path, other: Path, names) -> dict:
    """name -> `identical`, or where `dest/<name>.csv` first parts from
    `other/<name>.csv`. A regression guard, not a check: a difference is
    printed, and it is for the reader to say whether it is one."""
    out = {}
    for name in names:
        theirs = other / f"{name}.csv"
        if not theirs.exists():
            out[name] = f"missing in {other}"
            continue
        mine_bytes = (dest / f"{name}.csv").read_bytes()
        theirs_bytes = theirs.read_bytes()
        if mine_bytes == theirs_bytes:
            out[name] = "identical"
            continue
        mine, their = mine_bytes.splitlines(), theirs_bytes.splitlines()
        for number, (a, b) in enumerate(zip(mine, their), 1):
            if a != b:
                out[name] = (f"line {number}: "
                             f"{a.decode('utf-8', 'replace')[:300]!r} != "
                             f"{b.decode('utf-8', 'replace')[:300]!r}")
                break
        else:
            out[name] = (f"line {min(len(mine), len(their)) + 1}: "
                         f"{len(mine)} lines here, {len(their)} there"
                         if len(mine) != len(their)
                         else "same lines, different line endings")
    return out


def _json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (set, tuple)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default),
                    encoding="utf-8")


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    roots = [Path(r) for r in (args.runs or [DEFAULT_RUNS])]
    dest = Path(args.dest) if args.dest else default_dest()
    if (dest / "manifest.json").exists():
        print(f"refusing to start: {dest / 'manifest.json'} already exists; "
              f"pass --dest with a fresh directory", file=sys.stderr)
        return 2
    try:
        found = discovery.discover(roots, exclude=[dest])
    except discovery.DiscoveryError as exc:
        print(f"discovery failed: {exc}", file=sys.stderr)
        return 2
    if not found.runs:
        print(f"no runs found under {[str(r) for r in roots]}",
              file=sys.stderr)
        return 2
    print(f"{len(found.runs)} runs, {len(found.skipped)} directories skipped")
    for directory in found.multi_entry:
        print(f"  re-run appended, last manifest entry used: {directory}")
    full_cells = tuple(args.full_cells or dr2_full.CELLS)
    missing_full = dr2_full.missing_runs(found.runs, full_cells)
    if missing_full:
        # Said now, not only after DR2: the other tables are still written.
        print(f"DR2 full artifact: missing {', '.join(missing_full)}; no "
              f"dr2_full table will be written (--full-cells restricts the "
              f"set)", file=sys.stderr)

    census, census_ok = [], []
    r2 = checks.R2()
    acc3, acc4, acc5 = dr3.DR3(), dr4.DR4(), dr5.DR5()
    acc6 = coalitions.Coalitions()
    named, reference, incomplete, full_runs = [], [], [], []
    param_source = {}
    for number, run in enumerate(found.runs, 1):
        try:
            data = discovery.RunData.load(run)
        except discovery.MissingColumns as exc:
            incomplete.append(str(exc))
            continue
        param_source[run.run_id] = data.param_source
        row, ok = census_row(data)
        census.append(row)
        census_ok.append((run.run_id, ok))
        r2.add(data)
        acc3.add(data)
        acc4.add(data)
        acc5.add(data)
        acc6.add(data)
        named.extend(dr2.named_deviators(data))
        if run.cell == REFERENCE_CELL:
            reference.append(data)
        if run.cell in full_cells:
            full_runs.append(dr2_full.extract(data))
        if number % 20 == 0:
            print(f"  read {number}/{len(found.runs)} runs "
                  f"({time.perf_counter() - started:.0f}s)")

    if incomplete:
        # Named in full and nothing written: a table over the runs that
        # happen to be complete would read as a table over all of them.
        print(f"{len(incomplete)} run(s) lack columns the analysis reads; "
              f"re-run them with the current harness or pass --runs without "
              f"them:", file=sys.stderr)
        for line in incomplete:
            print(f"  {line}", file=sys.stderr)
        return 2

    pop = dr2.build_population(reference) if reference else {}
    results = {"R1": checks.r1(pop), "R2": r2.result(),
               "R3": checks.r3(pop),
               "dr4_single_deviator_coverage": acc4.single_deviator_check()}
    dest.mkdir(parents=True, exist_ok=True)
    hard = [name for name, result in results.items() if not result["passed"]]
    if hard:
        write_json(dest / "checks.json", {"checks": results, "stopped": hard})
        for name in hard:
            print(f"CHECK FAILED: {name}: {results[name]}", file=sys.stderr)
        print("no DR table written", file=sys.stderr)
        return 1

    print(f"R1-R3 passed ({time.perf_counter() - started:.0f}s); DR2 sweep")
    cases, curves, prop2 = dr2.analyse_cases(pop)
    conditions = dr2.conditions(pop)
    band_table, band_ok = dr2.band_sensitivity(pop, cases, conditions)
    print(f"  DR2 done ({time.perf_counter() - started:.0f}s)")
    full = dr2_full.run(full_runs, reference, cases, full_cells,
                        err=lambda text: print(text, file=sys.stderr))
    print(f"  DR2 full artifact done ({full.wall_sec:.0f}s of "
          f"{time.perf_counter() - started:.0f}s)")

    results["dr2_band_calibrated_reproduces_main"] = {"passed": band_ok}
    results["census_agrees_with_manifests"] = {
        "passed": all(ok for _run, ok in census_ok),
        "disagreeing": [run for run, ok in census_ok if not ok]}
    results["dr5_pair_kwh_matches_slot_csv"] = {
        "passed": acc5.pair_kwh_max_dev <= 1e-5
        and acc5.partner_rows_missing == 0,
        "tolerance": 1e-5, "max_deviation_kwh": acc5.pair_kwh_max_dev,
        "served_first_rows_without_partner_row": acc5.partner_rows_missing}
    results.update(acc6.assertions())
    results.update(full.checks)

    coalition_rows = acc6.table()
    notes = coalition_notes(coalition_rows)
    findings = {
        "proposition2_condition": dr2.proposition2_condition(pop),
        "proposition2_violations": prop2,
        "census_cleared_slots": sorted({r["n_cleared"] for r in census}),
        "slots_with_more_than_one_externality_deviator": acc4.n_multi,
        "same_from_slot_csv_n_deviators": acc4.n_multi_column,
        "executed_slots": acc4.n_executed_slots,
        "vault_figure_for_comparison": 13212,
        "dr2_full": full.findings,
    }

    tables = {
        "census": census, "dr2_cases": cases, "dr2_curves": curves,
        "dr2_conditions": conditions, "dr2_band_sensitivity": band_table,
        "dr2_named": named, "dr3_ir": acc3.ir_table(),
        "dr3_settlement": acc3.settlement_table(),
        "dr3_eq48": acc3.eq48_table(), "dr4_layers": acc4.layers_table(),
        "dr4_coverage": acc4.coverage_table(),
        "dr5_fairness": acc5.fairness_table(),
        "dr5_preferences": acc5.preferences_table(),
        "coalitions": coalition_rows,
    }
    preexisting = list(tables)
    if full.tables is not None:
        tables.update(full.tables)
    for name, rows in tables.items():
        write_table(dest, name, rows)
    (dest / "columns.md").write_text(columns.markdown(notes),
                                     encoding="utf-8")
    write_json(dest / "checks.json", {"checks": results,
                                      "findings": findings})
    comparison = None
    if args.compare_to:
        comparison = compare_tables(dest, Path(args.compare_to), preexisting)
        print(f"--compare-to {args.compare_to}:")
        for name, verdict in comparison.items():
            print(f"  {name}.csv: {verdict}")

    sha, dirty = script_state()
    write_json(dest / "manifest.json", {
        "script": "evaluation/analysis/dr_analysis.py",
        "script_sha": sha, "dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "runs_roots": [str(r) for r in roots], "dest": str(dest),
        "runs": [{"run_id": r.run_id, "cell": r.cell, "seed": r.seed,
                  "repo_sha": r.repo_sha, "sigmoid_source": r.sigmoid_source,
                  "n_manifest_entries": r.n_entries,
                  "param_source": param_source.get(r.run_id),
                  "path": str(r.path)}
                 for r in found.runs],
        "multi_entry_manifests": found.multi_entry,
        "skipped": [{"path": p, "reason": why} for p, why in found.skipped],
        "checks_passed": {name: bool(result["passed"])
                          for name, result in results.items()},
        "full_cells": list(full_cells),
        "dr2_full_penalty_parameters": full.params,
        "dr2_full_wall_sec": full.wall_sec,
        "compare_to": ({"dir": str(args.compare_to), "tables": comparison}
                       if comparison is not None else None),
        "wall_sec": round(time.perf_counter() - started, 1)})

    print_headlines(findings, tables)
    failed = [name for name, result in results.items() if not result["passed"]]
    print(f"wrote {len(tables)} tables to {dest} in "
          f"{time.perf_counter() - started:.0f}s")
    for name in failed:
        print(f"CHECK FAILED: {name}", file=sys.stderr)
    return 1 if failed else 0


def print_headlines(findings, tables) -> None:
    """The conditions the task asks to see printed, with the run values."""
    for cond in findings["proposition2_condition"]:
        print(f"Proposition 2 condition gamma (1 - eta_rel) K_upper > K_upper "
              f"- K_lower: {cond['gamma']} * (1 - {cond['eta_relative']}) * "
              f"{cond['k_upper']} = {cond['lhs']:.4f} > {cond['rhs']:.4f}: "
              f"{cond['holds']}")
    for row in findings["proposition2_violations"]:
        print(f"Proposition 2 violations, {row['case']}: "
              f"{row['n_violations']} of {row['n']}")
    for row in tables["dr3_eq48"]:
        if row["part"] == "exposure":
            print(f"Eq. (4.8) exposure, {row['cell']}, lambda = {row['levy']}: "
                  f"p < {row['price_threshold_ct']:.4f} ct above x > "
                  f"{row['ratio_threshold']:.2f}; {row['n_below']} of "
                  f"{row['n']} grey slots")
    print(f"cleared slots per run: {findings['census_cleared_slots']}")
    print(f"slots with more than one externality deviator: "
          f"{findings['slots_with_more_than_one_externality_deviator']} "
          f"(vault: {findings['vault_figure_for_comparison']})")


def coalition_notes(rows) -> list:
    """The per-kWh penalty along k, stated where it rises anywhere: that is
    composition (larger coalitions add members with other quantities), not
    a coalition effect -- A6 shows every member's own penalty is unchanged
    across the cells."""
    notes = []
    for arm in coalitions.ARMS:
        series = [(r["k"], r["penalty_per_kwh_ct"]) for r in rows
                  if r["arm"] == arm and r["seed"] == "all"
                  and r["penalty_per_kwh_ct"] is not None]
        series.sort()
        if len(series) > 1 and any(b[1] > a[1] for a, b in zip(series,
                                                               series[1:])):
            text = ", ".join(f"k={k}: {v:.4f}" for k, v in series)
            notes.append(
                f"coalitions ({arm}): the penalty per penalised kWh is not "
                f"constant in k and rises between some sizes ({text} "
                f"ct/kWh). That is composition -- the larger coalitions add "
                f"members with other quantities and slots -- and not a "
                f"coalition effect: each member's own penalty is identical "
                f"in every cell it appears in (checks.json, "
                f"a6_{arm}_member_penalty_identical).")
    return notes


if __name__ == "__main__":
    sys.exit(main())
