"""The campaign: one process per run, five seeds per cell. Writes into out/.

Run isolation is by process, not by bookkeeping. `stack.py` rewrites
`sys.modules` at import time to load the three `src` packages side by side,
which is not safe concurrently in one interpreter but is perfectly safe once
per process -- so every run gets its own interpreter, its own `Stack` and
therefore its own empty store. That removes the `market_id` collision question
entirely: no reset choreography between runs, no slot offsets to keep two runs
out of each other's order books.

The start method is **spawn**, not fork: it is the only one Windows has, and
under fork a child would inherit the parent's already-rewritten `sys.modules`
and the three packages would be half-loaded rather than freshly loaded.

**Parameter passing has one convention.** Preferences and sigmoid parameters
go through the trigger -- `trigger["preference_params"]` and
`trigger["sigmoid_params"]`, which `run_clearing` resolves at
`amm-clearing-node/src/clearing.py:371-372`. `Stack.clear(preferences=...)`
is not used in campaign runs: two ways of setting the same thing is how a
manifest stops describing the run it names.

Paths are derived from this file's own location so the harness runs on
Windows as well as on the Linux sandbox the pilot used.
"""
import asyncio, csv, functools, itertools, json, logging, multiprocessing
import platform, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
import runner, runs, scenario, stack
logging.disable(logging.WARNING)

# Every run_slot call names its community and its slot. The slot is the
# position of the case in its block's parameter grid, so it is a function of
# the parameters and not of call order -- re-running one cell lands in the
# same slot it landed in before. The market id carries the community, because
# the mock derives its own id from the slot alone and two blocks would
# otherwise share an order book at the same grid position.
BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


def slot_at(index: int) -> int:
    return BASE_SLOT + index * runner.SLOT_SEC


async def run_case(st, scen, *, community, index, **kw):
    slot = slot_at(index)
    return await runner.run_slot(
        st, scen, community=community, slot=slot,
        market_id=runner.market_id_for(community, slot), **kw)

HARNESS_DIR = Path(__file__).resolve().parent
REPO = stack.REPO                      # the pinned clone next to the harness
OUT = HARNESS_DIR.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "manifest.json"


@functools.lru_cache(maxsize=1)
def repo_sha() -> str:
    """Resolved on demand, not at import.

    Under spawn every worker re-imports this module, and an import-time
    `git rev-parse` would be one subprocess per worker. The parent resolves it
    once and hands it down.
    """
    return runs.repo_sha()

PREF_OFF = {"enabled": False, "multipliers_enabled": False}
def prefs(**kw):
    base = {"enabled": True, "order": "preferences_first",
            "multipliers_enabled": False, "mode": "multiplicative",
            "sides": "seller", "green_multiplier": 0.10, "grey_levy": 0.10,
            "levy_cap": 0.20}
    base.update(kw); return base

def write(name, rows):
    if not rows: return
    keys = list(rows[0].keys())
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"  -> {name}  ({len(rows)} rows)")


def write_manifest(run_id, *, params, seeds, wall_sec, output_files, **extra):
    """Append one provenance record per run to out/manifest.json.

    A figure that cannot be traced back to `config + seed + SHA` is not usable
    as evidence, so every run this harness performs writes one of these.
    """
    entry = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_sha": repo_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "params": params,
        "seeds": seeds,
        "wall_sec": round(wall_sec, 3),
        "output_files": [str(Path(f).name) for f in output_files],
    }
    entry.update(extra)
    entries = []
    if MANIFEST.exists():
        try:
            entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
        if not isinstance(entries, list):
            entries = [entries]
    entries.append(entry)
    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"  -> manifest.json  (+{run_id})")
    return entry


async def block_golden(st, rows):
    """A: reproduce the implementation-guide reference numbers."""
    r = await run_case(st, scenario.GUIDE, community="A-golden", index=0,
                       preferences=PREF_OFF, execute=False)
    c = r["clearing"]
    rows.append({"case": "baseline_pro_rata", "price": c["clearing_price_ct_per_kwh"],
                 "ratio": c["ratio"], "round": c["round_type"],
                 **{p["name"]: p["fill_rate"] for p in c["allocations"]["producers"]}})
    r2 = await run_case(st, scenario.GUIDE, community="A-golden", index=1,
                        preferences=prefs(multipliers_enabled=True), execute=False)
    c2 = r2["clearing"]
    m = c2["preferences"]["multipliers"]
    rows.append({"case": "preferences_first_mult", "price": c2["clearing_price_ct_per_kwh"],
                 "ratio": c2["ratio"], "round": c2["round_type"],
                 **{p["name"]: p["fill_rate"] for p in c2["allocations"]["producers"]}})
    # The two raw clearing summaries come back as well so a caller (R0) can
    # measure per-order quantities the CSV rows above do not carry.
    return m, c, c2


async def block_sd_sweep(st):
    """B: supply/demand ratio sweep, 100 participants, 5 seeds."""
    rows = []
    ratios = [round(0.60 + 0.05*i, 2) for i in range(21)]
    for index, (ratio, seed) in enumerate(itertools.product(ratios, range(5))):
        scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                      sd_ratio=ratio, pair_density=0.25)
        r = await run_case(st, scen, community="B-sd-sweep", index=index,
                           preferences=prefs(), execute=False)
        c = r["clearing"]
        if c.get("status") != "cleared":
            continue
        prod = c["allocations"]["producers"]; cons = c["allocations"]["consumers"]
        rows.append({
            "sd_ratio": ratio, "seed": seed,
            "price": c["clearing_price_ct_per_kwh"],
            "ratio_actual": c["ratio"], "round": c["round_type"],
            "supply": c["total_supply_kwh"], "demand": c["total_demand_kwh"],
            "traded": min(c["total_supply_kwh"], c["total_demand_kwh"]),
            "seller_fill_mean": statistics.mean(p["fill_rate"] for p in prod),
            "buyer_fill_mean": statistics.mean(cc["fill_rate"] for cc in cons),
            "seller_fill_min": min(p["fill_rate"] for p in prod),
            "seller_fill_max": max(p["fill_rate"] for p in prod),
        })
    return rows


async def block_pairs(st):
    """C1: allocation effect. pair density x order, multipliers off."""
    rows = []
    grid = itertools.product((0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
                             ("pro_rata_first", "preferences_first"),
                             range(5))
    for index, (density, order, seed) in enumerate(grid):
        scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                      sd_ratio=1.25, pair_density=density)
        r = await run_case(st, scen, community="C1-pairs", index=index,
                           preferences=prefs(order=order), execute=False)
        c = r["clearing"]
        prod = c["allocations"]["producers"]
        matched = [p["fill_rate"] for p in prod if p["preference_matched"]]
        unmatched = [p["fill_rate"] for p in prod if not p["preference_matched"]]
        rows.append({
            "pair_density": density, "order": order, "seed": seed,
            "price": c["clearing_price_ct_per_kwh"],
            "n_pairs": len(c["preferences"]["mutual_pairs"]),
            "fill_matched": statistics.mean(matched) if matched else None,
            "fill_unmatched": statistics.mean(unmatched) if unmatched else None,
            "fill_all": statistics.mean(p["fill_rate"] for p in prod),
            "fill_sd": statistics.pstdev([p["fill_rate"] for p in prod]),
            "traded_total": round(sum(p["allocated_kwh"] for p in prod), 6),
        })
    return rows


async def block_multipliers(st):
    """C2: mode x sides x green_multiplier, guide reference scenario."""
    rows = []
    multipliers = [round(0.005*i, 4) for i in range(0, 25)]   # 0.000 .. 0.120
    grid = itertools.product(("multiplicative", "additive"),
                             ("seller", "both"), multipliers)
    for index, (mode, sides, gm) in enumerate(grid):
        r = await run_case(
            st, scenario.GUIDE, community="C2-multipliers", index=index,
            preferences=prefs(multipliers_enabled=True, mode=mode,
                              sides=sides, green_multiplier=gm,
                              grey_levy=0.10), execute=False)
        c = r["clearing"]; m = c["preferences"]["multipliers"]
        rows.append({
            "mode": mode, "sides": sides, "green_multiplier": gm,
            "price": c["clearing_price_ct_per_kwh"],
            "green_final": m["green_final_ct_per_kwh"],
            "grey_final": m["grey_final_ct_per_kwh"],
            "buyer_rate": m.get("buyer_final_ct_per_kwh"),
            "levy_collected_ct": m["levy_collected_ct"],
            "bonus_requested_ct": m["bonus_requested_ct"],
            "bonus_paid_ct": m["bonus_paid_ct"],
            "scale": m["scale"],
            "pool_surplus_ct": m["pool_surplus_ct"],
            "buyers_pay_ct": m["buyers_pay_ct"],
            "sellers_receive_ct": m.get("sellers_receive_ct"),
        })
    return rows


# =====================================================================
# The campaign: one process per run, five seeds per cell
# =====================================================================

# D-59. Five per cell, and they are written into the manifest one by one
# rather than as "seeds 0-4", so a single deviating cell can be re-run alone.
SEEDS = (0, 1, 2, 3, 4)

# Each replicate runs over its own week: the seed picks the dataset extension
# draw, which is the only stochastic input a profile run has. Recorded in the
# manifest as `dataset_extension_seed`.
EXTENSION_SEED_BASE = 4242

# D-76. The participation every cell runs at unless it is the axis being
# varied. It is set explicitly in `spec_for` rather than left to the
# `RunSpec` default, so each manifest records the value the run actually used
# instead of whatever the dataclass happened to default to when it ran -- the
# green-share axis below changes this per cell, and a reader comparing two
# manifests has to be able to see that from the entries alone.
REFERENCE_PARTICIPATION = 0.25

# The green-share axis as its own cell group (D-76). Deliberately not a factor
# over the four cells above: crossing them would turn block 1 into twenty
# cells and answer a question nobody asked, while what 5.1 needs is one axis
# varied against a fixed preference set. `green_share_025` therefore repeats
# the baseline cell's configuration under its own name -- the redundancy is
# the point, because the axis has to be readable as five comparable runs
# without the reader reaching into another cell group for its middle point.
GREEN_SHARES = (0.0, 0.25, 0.50, 0.75, 1.0)

# The delta baseline, named once here and once in the manifest, and never
# re-derived per table: pro-rata with preferences disabled, at the calibrated
# theta/steepness.
#
# None until T-19 fits it. Note what that means: `run_slot` then falls back to
# `runner.SIGMOID` and sends *that* as the trigger's `sigmoid_params`, which
# `resolve_community` merges over the configured community parameters. So the
# runs below are at the harness default (theta 1.0, steepness 2.5), not at the
# community configuration -- and every manifest entry records which, because
# the clearing response carries `sigmoid_params` back.
CALIBRATED_SIGMOID = None
BASELINE_PREFERENCES = dict(PREF_OFF)


def spec_for(cell: str, seed: int, **overrides) -> runs.RunSpec:
    """One run of one cell. Everything it depends on is in the spec."""
    params = dict(
        run_id=f"{cell}--seed-{seed}", seed=seed, cell=cell, cell_seeds=SEEDS,
        dataset_extension_seed=EXTENSION_SEED_BASE + seed,
        participation=REFERENCE_PARTICIPATION,
        sigmoid=CALIBRATED_SIGMOID, baseline=runs.BASELINE)
    params.update(overrides)
    return runs.RunSpec(**params)


def cells() -> list:
    """The campaign grid: every cell at all five seeds.

    The first cell is the delta baseline itself, so it is produced by the same
    code path as everything it is subtracted from.
    """
    grid = {
        "baseline_pro_rata": {"preferences": BASELINE_PREFERENCES},
        "preferences_first": {"preferences": prefs(order="preferences_first")},
        "pro_rata_first": {"preferences": prefs(order="pro_rata_first")},
        "multipliers_on": {"preferences": prefs(multipliers_enabled=True)},
    }
    # The green-share axis: five cells at the baseline preference set, varying
    # nothing but `participation`. Note that `green_share_000` fits on fewer
    # slots than the rest -- with no battery areas the community has no supply
    # at all in the evening periods the rule discharges into, so roughly 262
    # of 672 slots trade against roughly 353 elsewhere. Both round types still
    # occur and the ratio span is unchanged, so it passes pre-flight; but a
    # table that puts its per-slot means next to the other four cells is
    # comparing different numbers of slots and has to say so.
    grid.update({
        f"green_share_{int(share * 100):03d}": {
            "preferences": BASELINE_PREFERENCES, "participation": share}
        for share in GREEN_SHARES})
    return [spec_for(cell, seed, **overrides)
            for cell, overrides in grid.items() for seed in SEEDS]


def _worker(payload):
    """Spawn entry point. One process, one `Stack`, one empty store.

    Kept at module level and given only picklable arguments, because spawn
    re-imports this module in the child and looks the function up by name.
    """
    spec, sha = payload
    logging.disable(logging.WARNING)
    return runs.run_one(spec, sha=sha)


def run_parallel(specs, *, processes=None, sha=None) -> list:
    """One process per run, spawn start method."""
    sha = sha or repo_sha()
    context = multiprocessing.get_context("spawn")
    processes = processes or min(len(specs), multiprocessing.cpu_count())
    with context.Pool(processes) as pool:
        return pool.map(_worker, [(spec, sha) for spec in specs])


def run_sequential(specs, *, sha=None) -> list:
    """The same runs in this interpreter. The comparison path for the
    parallel one -- identical results, more wall clock."""
    sha = sha or repo_sha()
    return [runs.run_one(spec, sha=sha) for spec in specs]


def report(results) -> None:
    """Every run's guards, printed. `check_ratio_spread` runs inside
    `runs.execute_run`, so a cell that never varied never gets this far --
    a guard that is never called is not a guard."""
    for result in results:
        checks = result["checks"]
        print(f"  {result['run_id']:<32} "
              f"slots={checks['n_slots']:<4} "
              f"ratio_span={checks['ratio_span']:<10} "
              f"{checks['round_type_census']}")


async def main():
    """The profile campaign: every cell at five seeds, one process per run."""
    started = time.perf_counter()
    specs = cells()
    print(f"campaign: {len(specs)} runs "
          f"({len(specs) // len(SEEDS)} cells x {len(SEEDS)} seeds), "
          f"one process each")
    results = run_parallel(specs)
    report(results)
    print(f"\n{len(results)} runs in {time.perf_counter() - started:.1f}s")
    print(f"manifests under {OUT}")


async def pilot_main():
    t0 = time.perf_counter()
    st = stack.Stack()
    manifest = {"repo_sha": repo_sha(), "timestamp": int(time.time()),
                "python": sys.version.split()[0]}

    print("A golden run")
    grows = []; mult, _c_a, _c_b = await block_golden(st, grows)
    write("A_golden.csv", grows)
    manifest["golden_multipliers"] = mult

    print("B supply/demand sweep")
    write("B_sd_sweep.csv", await block_sd_sweep(st))

    print("C1 pair density")
    write("C1_pairs.csv", await block_pairs(st))

    print("C2 multiplier grid")
    write("C2_multipliers.csv", await block_multipliers(st))

    manifest["runtime_sec"] = round(time.perf_counter() - t0, 1)
    write_manifest("pilot_campaign", params=manifest, seeds=list(range(5)),
                   wall_sec=manifest["runtime_sec"],
                   output_files=["A_golden.csv", "B_sd_sweep.csv",
                                 "C1_pairs.csv", "C2_multipliers.csv"])
    print("manifest:", manifest)
    await st.close()


if __name__ == "__main__":
    # Guarded, and it has to be: under spawn the child re-imports this module,
    # and unguarded top-level work would start the campaign again in every
    # worker.
    if "--pilot" in sys.argv:
        # The 02.09. synthetic blocks. They are what plots.py reads, so they
        # stay runnable; they are not the profile campaign.
        asyncio.run(pilot_main())
    else:
        asyncio.run(main())
