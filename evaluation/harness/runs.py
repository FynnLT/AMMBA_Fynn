"""One run: build the stack, walk the week, write the CSVs and the manifest.

This is the unit a campaign parallelises over. Everything a run needs is in
its `RunSpec`, and everything it produces is identified by its manifest entry
-- a figure is cited by run id, never by a file path.

The pre-flight checks here exist because each of them stands for a campaign
that produced numbers and had to be thrown away:

* a sequence whose supply/demand ratio never moves fits a sigmoid that
  describes nothing (`check_ratio_spread`);
* a census with only one round type means half of Chapter 5 has no data
  behind it;
* a measurement channel that silently rejects writes returns null penalties
  that read like findings (`assert_measurement_round_trip`).
"""
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import aggregates
import profiles
import runner
import scenario
import stack

HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"

# The manifest fields without which a figure cannot be traced back to the run
# that produced it. Checked on write, not on read: a manifest that is missing
# one of these is useless later and cheap to reject now.
REQUIRED_MANIFEST_FIELDS = ("config", "seed", "repo_sha",
                            "dataset_extension_seed", "player_ids")

# The delta baseline, named once. Every table that reports a delta reports it
# against this and does not re-derive its own.
BASELINE = "pro_rata_preferences_disabled_at_calibrated_theta"

DAY_START = 1_757_000_000 // 86400 * 86400

# The SHA-256 recorded in evaluation/data/PROVENANCE.md. Carried into every
# manifest entry so a figure names the data it came from, not just the code.
DATASET_SHA256 = ("f96ef125d1c0f49cf3288b17598c166062570c7baa6c4d2d25ac"
                  "545813531c3a")


class PreflightError(RuntimeError):
    """A run was not allowed to start, or its output was not usable."""


def repo_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(stack.REPO), "rev-parse", "HEAD"], text=True).strip()


@dataclass(frozen=True)
class RunSpec:
    """Everything one run needs, and nothing it can derive from the clock.

    `cell_seeds` carries the five seeds of the cell this run belongs to,
    individually rather than as a range, so one deviating cell can be re-run
    on its own without guessing which seeds it held (D-59).
    """
    run_id: str
    seed: int
    cell_seeds: tuple = (0, 1, 2, 3, 4)
    cell: str = "default"
    dataset_extension_seed: int = 4242
    player_seed: int = 20260916
    battery_seed: int = 7
    n_players: int = 100
    days: int = 7
    participation: float = 1.0
    preferences: dict = None
    sigmoid: dict = None
    gamma: float = None
    eta_relative: float = None
    baseline: str = BASELINE
    day_start: int = DAY_START
    execute: bool = False
    workbook: str = None
    out_dir: str = None
    min_ratio_spread: float = 0.25

    def config(self) -> dict:
        """The run's parameters as they go into the manifest: everything the
        run depends on, nothing it produces."""
        data = asdict(self)
        data.pop("run_id")
        return data


# ----------------------------------------------------------- the guards

def check_ratio_spread(records, minimum: float = 0.25) -> float:
    """Raise when the supply/demand ratio barely moves over a sequence.

    `scenario.make_scenario` scales generation so the ratio hits `sd_ratio`
    exactly. A synthetic sequence at a fixed `sd_ratio` therefore yields the
    same ratio in all 672 slots, the sigmoid fit converges on a single point,
    and the result comes back looking fitted with nothing reporting an error.
    The calibration needs the ratio to vary; this is where that is enforced.
    """
    values = aggregates.ratios(records)
    if not values:
        raise PreflightError(
            "no slot cleared, so the supply/demand ratio has no values at all")
    span = max(values) - min(values)
    if span < minimum:
        raise PreflightError(
            f"supply/demand ratio span {span:.6f} is below the required "
            f"{minimum} (min {min(values):.6f}, max {max(values):.6f} over "
            f"{len(values)} cleared slots) -- a fit against this measures "
            f"nothing")
    return span


def check_round_type_census(records) -> dict:
    """Both round types have to occur, or half of Chapter 5 has no data."""
    census = aggregates.round_type_census(records)
    empty = [name for name in (aggregates.SUPPLY_LIMITED,
                               aggregates.DEMAND_LIMITED)
             if census.get(name, 0) == 0]
    if empty:
        raise PreflightError(
            f"round-type census is empty on {', '.join(empty)}: {census}")
    return census


async def assert_measurement_round_trip(st, community: str, slot: int) -> None:
    """Post a measurement, read it back, fail if it is absent.

    On 25.08. `POST /asset_measurements` had been renamed in the same commit.
    The harness took the 404, two blocks returned null results, and the nulls
    read like findings. A status code is not enough -- the write has to be
    visible on the read path the Execution Node actually uses.
    """
    area = "preflight-area"
    probe = 0.123456
    await st.post("/measurements", [{"community_uuid": community,
                                     "area_uuid": area, "time_slot": slot,
                                     "energy_kwh": probe}])
    stored = await st.get("/measurements", community_uuid=community,
                          start_time=slot, end_time=slot + runner.SLOT_SEC - 1)
    found = [row for row in stored if row.get("area_uuid") == area]
    if not found:
        raise PreflightError(
            f"measurement written for {area} @ {slot} is not readable back "
            f"(read {len(stored)} rows for community {community}); the "
            f"measurement channel is not storing what the harness posts")
    if abs(float(found[0]["energy_kwh"]) - probe) > 1e-9:
        raise PreflightError(
            f"measurement read back as {found[0]['energy_kwh']!r}, "
            f"posted {probe}")


def preflight(records, *, minimum: float = 0.25) -> dict:
    """The checks a run's output must pass before it counts as a result."""
    span = check_ratio_spread(records, minimum=minimum)
    census = check_round_type_census(records)
    return {"ratio_span": round(span, 6), "round_type_census": census,
            "n_slots": len(records)}


# ---------------------------------------------------------- the manifest

def write_manifest(path, run_id: str, *, config: dict, seed: int,
                   dataset_extension_seed: int, player_ids, cell_seeds,
                   output_files=(), sha: str = None, **extra) -> dict:
    """Append one provenance record per run to `manifest.json`.

    A figure that cannot be traced back to `config + seed + SHA` is not
    evidence, so every run writes one of these, and the required fields are
    checked here rather than discovered missing six weeks later.

    `cell_seeds` is written out one by one, not as "seeds 0-4": a cell that
    deviates has to be re-runnable alone.
    """
    entry = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": dict(config),
        "seed": seed,
        "repo_sha": sha if sha is not None else repo_sha(),
        "dataset_extension_seed": dataset_extension_seed,
        "player_ids": list(player_ids),
        "cell_seeds": [int(s) for s in cell_seeds],
        "baseline": config.get("baseline", BASELINE),
        "dataset_sha256": DATASET_SHA256,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "output_files": [str(Path(f).name) for f in output_files],
    }
    entry.update(extra)

    missing = [f for f in REQUIRED_MANIFEST_FIELDS if entry.get(f) is None]
    if missing:
        raise PreflightError(
            f"manifest entry for {run_id} is missing {missing}; without them "
            f"the run cannot be reproduced and its figures cannot be cited")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if path.exists():
        try:
            runs = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
        if not isinstance(runs, list):
            runs = [runs]
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return entry


# --------------------------------------------------------------- the run

def build_week(spec: RunSpec):
    """The profiles side of a run: community, draw, week, battery areas."""
    community = profiles.load_community(
        spec.workbook or profiles.DEFAULT_WORKBOOK)
    players = profiles.select_players(community.flags, n=spec.n_players,
                                      seed=spec.player_seed)
    week = profiles.extend_to_week(community.day_kwh(players),
                                   seed=spec.dataset_extension_seed,
                                   days=spec.days)
    batteries = profiles.make_battery_areas(
        community.bess, players, participation=spec.participation,
        seed=spec.battery_seed)
    return community, players, week, batteries


def run_areas(players, batteries) -> list:
    """The community area list, built once per run.

    Every player is a member for the whole run even in slots where netting
    leaves it with nothing to post: membership is a property of the run, not
    of the slot.
    """
    return ([{"area_uuid": f"player-{p:03d}", "name": f"Player {p:03d}",
              "area_type": "Load"} for p in players] +
            [{"area_uuid": b.area_uuid, "name": b.name, "area_type": "PV"}
             for b in batteries])


async def execute_run(spec: RunSpec, *, out_dir=None, sha: str = None) -> dict:
    """Build the stack, run the sequence, write the CSVs, write the manifest,
    run the pre-flight checks. One process's worth of work."""
    started = time.perf_counter()
    out_dir = Path(out_dir or spec.out_dir or (OUT / spec.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    community, players, week, batteries = build_week(spec)
    scenarios = [scenario.make_scenario_from_profiles(week, index, batteries)
                 for index in range(week.n_slots)]
    slots = profiles.slot_times(week, spec.day_start)

    st = stack.Stack()
    try:
        await assert_measurement_round_trip(st, f"preflight-{spec.run_id}",
                                            spec.day_start)
        records = await runner.run_sequence(
            st, spec.run_id, slots, lambda index, slot: scenarios[index],
            areas=run_areas(players, batteries),
            preferences=spec.preferences, sigmoid=spec.sigmoid,
            gamma=spec.gamma, eta_relative=spec.eta_relative,
            execute=spec.execute)
    finally:
        await st.close()

    checks = preflight(records, minimum=spec.min_ratio_spread)
    csv_path = aggregates.write_slot_csv(out_dir / f"{spec.run_id}_slots.csv",
                                         records)
    # One manifest per run directory, not one shared file. A campaign runs one
    # process per run, and `write_manifest` is a read-modify-write: two workers
    # appending to a single manifest.json would lose entries silently, which is
    # the one failure this file exists to prevent.
    manifest = write_manifest(
        out_dir / "manifest.json", spec.run_id, config=spec.config(),
        seed=spec.seed, dataset_extension_seed=spec.dataset_extension_seed,
        player_ids=players, cell_seeds=spec.cell_seeds,
        output_files=[csv_path], sha=sha,
        cell=spec.cell,
        community_composition=profiles.community_composition(community.flags,
                                                             players),
        n_battery_areas=len(batteries),
        battery_rule=(batteries[0].rule if batteries else None),
        sigma_load=round(week.sigma_load, 6),
        sigma_pv=round(week.sigma_pv, 6),
        checks=checks,
        wall_sec=round(time.perf_counter() - started, 3))
    return {"run_id": spec.run_id, "records": records, "checks": checks,
            "manifest": manifest, "csv": str(csv_path)}


def run_one(spec: RunSpec, *, sha: str = None) -> dict:
    """Synchronous entry point, so a worker process can call it directly.

    The records themselves stay in the worker: they are large, and everything
    a campaign needs downstream is in the CSV and the manifest.
    """
    import asyncio

    result = asyncio.run(execute_run(spec, sha=sha))
    return {"run_id": result["run_id"], "checks": result["checks"],
            "manifest": result["manifest"], "csv": result["csv"]}
