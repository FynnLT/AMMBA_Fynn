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
import asyncio, csv, dataclasses, functools, itertools, json, logging
import multiprocessing, platform, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
import deviations, runner, runs, scenario, stack
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

# The out-of-sample week T-19 validated against, and deliberately the same
# draw as `EXTENSION_SEED_BASE`: seed 0 of every campaign cell runs the week
# the fit was *not* fitted on.
OOS_EXTENSION_SEED = 4242

# D-76. The participation every cell runs at unless it is the axis being
# varied. It is set explicitly in `spec_for` rather than left to the
# `RunSpec` default, so each manifest records the value the run actually used
# instead of whatever the dataclass happened to default to when it ran -- the
# green-share axis below changes this per cell, and a reader comparing two
# manifests has to be able to see that from the entries alone.
REFERENCE_PARTICIPATION = 0.25

# D-71/D-81. The preference density axis of block 1's second pass: the share
# of households that name a preferred partner. The first pass had no axis at
# all -- `make_scenario_from_profiles` posted no preference, so every
# preference cell cleared a book with zero pairs and `preferences_first` and
# `pro_rata_first` were the same run.
DENSITIES = (0.20, 0.50, 0.80)

# The reference reciprocity rate, and the two sensitivity points around it.
# Only a *mutual* nomination gets priority (the clearing node matches pairs,
# not one-sided wishes), so this is the parameter that decides how much of
# the named volume the mechanism can act on at all.
REFERENCE_MUTUAL = 0.50
MUTUAL_SENSITIVITY = (0.25, 0.75)

# D-79. The green multiplier of the `multipliers_on` cells.
#
# The run plan says 0.03, from the crossover argument on the pilot mix. The
# campaign mix is different: at participation 0.25 the battery (grey) share of
# posted energy is 20.9 % (D-76), so grey/green ~ 0.264 and the crossover sits
# at 0.10 * 0.264 ~ 0.026 on the week's *average* -- 0.03 is above it, where
# the levy no longer funds the bonus and the two formulations start to
# coincide (trap 1). Hence 0.02.
#
# NOTE, measured 17.09. and reported rather than acted on: on this week the
# average is the only level at which the argument holds at all. The battery
# rule discharges into 18:15..22:00 and PV runs 08:00..17:45, so **no slot
# carries both green and grey supply** -- 0 of 91 grey slots, at participation
# 0.25 and at 1.0 alike. Per slot `green_alloc_kwh` or `grey_alloc_kwh` is
# always 0, `crossover_mult` is therefore None everywhere, and the green bonus
# is unfundable at every value of G. What `multipliers_on` measures on this
# week is the grey levy and the pool surplus it leaves behind, not the
# bonus/levy trade-off. That finding is what D-83 and `OVERLAP_EVENING_PERIODS`
# answer; the reference cells stay as they are.
#
# SECOND NOTE, measured on the D-83 overlap window where the crossover can be
# read per slot at all, and again reported rather than acted on: **0.02 is far
# below the crossover there, not near it.** Between 15:00 and 19:00 the
# battery dominates, so `grey/green` runs 1.09 .. 207 (median 5.25) and
# `crossover_mult = 0.10 * grey/green` runs 0.109 .. 20.7 (median 0.52). The
# levy therefore over-funds the bonus by a factor of 5 to 80 in every overlap
# slot, `bonus_scale` is 1 in all 70 (the 23 slots reading 0.999... are the
# 6-decimal rounding of `green_final`, not a shortfall), and both formulations
# pay the full `p * (1 + G)` on the green side -- which is why they are
# indistinguishable in `green_final_ct` and differ everywhere else.
#
# Separating them on the green side would need G above the median crossover,
# i.e. a bonus over 50 %, which is not a parameter anyone would defend. The
# green-side coincidence is therefore a property of this supply mix and
# belongs in 5.2 as one; the formulation comparison itself is carried by the
# grey rate, the levy and the pool surplus, which do separate in all 70 slots.
GREEN_MULTIPLIER = 0.02
GREY_LEVY = 0.10
LEVY_CAP = 0.20

# D-83. The declared scenario variant the formulation comparison runs on.
#
# The finding above is the reason it exists: on the reference window the two
# formulations cannot be told apart, because neither pays a bonus. This window
# is 15:00-19:00 (periods 61..76, labelled by their end), so periods 61..71
# overlap the PV that runs to 17:45 and a slot can carry both a green and a
# grey offer. Same week, same 15 units, same charging rule -- only the
# discharge window moves.
#
# **It is compared within its own group, never against `baseline_pro_rata`.**
# A different battery window is a different supply curve and therefore a
# different ratio distribution: a delta against the reference cells would be
# reading the window change, not the multiplier. `mult_overlap_off` is that
# group's own baseline and is the only thing the two formulation cells are
# subtracted from. Measured at seed 0: the group's census is 124/159/389
# (supply-/demand-limited/no-trade) against 235/118/319 on the reference
# window, which is how large that difference is and why the two groups may
# not be put in one table.
#
# Measured at seed 0, 70 overlap slots (10 per day, every day of the week):
#
# * the bonus is funded and paid in all 70, so the group does what it exists
#   for -- `bonus_paid_ct` is 60.21 ct over the week against 0.00 on the
#   reference window at any `green_multiplier`;
# * the two formulations separate in `grey_final_ct`, `levy_collected_ct`,
#   `sellers_receive_ct` and `pool_surplus_ct` in **70 of 70** slots. The
#   multiplicative levy collects 1356.85 ct to fund a 60.21 ct bonus and the
#   pool retains 1296.64 ct; the additive one collects exactly what the bonus
#   costs and leaves 0.0003 ct. That over-collection is the formulation
#   property `MultiplierResult` documents, and it is what D-38 compares;
# * they do **not** separate in `green_final_ct` (2 of 70, and those two by
#   1e-6, which is rounding). See `GREEN_MULTIPLIER` for why.
OVERLAP_EVENING_PERIODS = tuple(range(61, 77))

# The green-share axis as its own cell group (D-76). Deliberately not a factor
# over the mechanism cells above: crossing them would turn block 1 into fifty
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
# Fitted 17.09.2026 (T-19, D-78): componentwise median over five independent
# weeks at alpha = 0.5, rounded. Sent as the trigger's `sigmoid_params` and
# recorded in every manifest; the same values sit in configuration.yaml.
CALIBRATED_SIGMOID = {"k_upper": 40.0, "k_lower": 8.0, "theta": 1.0, "steepness": 0.6}
BASELINE_PREFERENCES = dict(PREF_OFF)


# ---------------------------------------------------------------- block 2

# D-80. The accidental noise floor, two sigma below the deadband, so roughly
# 95 % of accidental deviations fall inside it. Chosen and stated as chosen in
# 5.1, not sourced; deriving it from a PV forecast-error reference would be
# better and is a literature search away.
SIGMA = deviations.DEFAULT_SIGMA

# D-26/D-44. The deadband relative to the trade's own quantity, not absolute:
# an absolute eta makes the penalty a function of installation size.
ETA_RELATIVE = 0.10

# D-72's two axes: how hard a single deviator deviates, and how many deviate
# at the reference share. `k = 1 / 5 / 10` is the minimum the decision names;
# `k = 2` is the extra point and the first to cut if the budget is short.
DEVIATION_SHARES = (0.10, 0.25, 0.50)
REFERENCE_SHARE = 0.25
COALITION_SIZES = (2, 5, 10)
DEVIATION_SEED = 20260919


# ----------------------------------------------------------- the sweep

# D-26/B-10. The parameter axis block 2 does not carry. All 70 block-2 runs
# sat at one point of the (gamma, eta) plane -- gamma at the execution node's
# configured value, `eta_relative = 0.10` -- while the published framework
# constrains gamma only by `gamma > 1` and eta only by `eta_t >= 0`, with no
# derivation and no sensitivity analysis anywhere. This group is that
# sensitivity analysis.
#
# The gamma axis. Written out rather than read off the configuration, unlike
# every other use of gamma in this file: these four are the *axis*, and each
# cell is named after the value it holds, so a `b2_short_g110` that silently
# followed a re-configured node would be a cell whose name lied. 1.1 is the
# node's configured value today (`execution_gamma()`), which is what makes
# the first point of the axis the point block 2 already ran at.
#
# `Phi = gamma * K_upper * shortfall` is linear in gamma by construction, so
# this axis holds no surprise in its magnitude and 5.3 must not present one.
# What it measures is the ratio of the two penalty channels at a comparable
# deviation, whether the redistribution stays budget-balanced as the pool
# grows, and the one sharp RQ2 statement the plane supports: raising gamma
# deters under-delivery and does nothing at all against withholding.
SWEEP_GAMMAS = (1.1, 1.5, 2.0, 3.0)

# The eta axis, and the substantive one: eta moves `W_sell = max(0, forecast
# - traded - eta)`, so at `eta_relative = share` the deviation is
# extinguished exactly. `0.10` is deliberately absent -- `b2_sell_s25` *is*
# that point, both axes are read against it, and re-running it into `out/`
# would append a second entry to the manifest the 17./18.09. campaign is
# written against (`write_manifest` appends, it does not replace).
#
# `b2_eta000` will show a far larger penalty population than the other three,
# and it is not the deviator: at `eta_relative = 0` the deadband no longer
# absorbs the accidental layer, so every seller whose delivery noise is
# negative incurs a shortfall. That is the deadband doing its job and is a
# 5.3 result. The noise stays on, and the eta axis is read from the
# deviator's own `_areas.csv` rows, filtered to `manifest.deviation.deviators`.
SWEEP_ETAS = (0.00, 0.05, 0.15, 0.25)


def execution_gamma() -> float:
    """The execution node's own configured gamma, read rather than restated.

    A campaign that hard-codes it records a number the service may not be
    using, and the manifest then describes a run that did not happen.
    """
    return stack.Stack().cfg_exe.penalty_gamma


def spec_for(cell: str, seed: int, **overrides) -> runs.RunSpec:
    """One run of one cell. Everything it depends on is in the spec."""
    params = dict(
        run_id=f"{cell}--seed-{seed}", seed=seed, cell=cell, cell_seeds=SEEDS,
        dataset_extension_seed=EXTENSION_SEED_BASE + seed,
        participation=REFERENCE_PARTICIPATION,
        sigmoid=CALIBRATED_SIGMOID, baseline=runs.BASELINE)
    params.update(overrides)
    return runs.RunSpec(**params)


def block1_grid() -> dict:
    """Block 1's second pass: 19 cells.

    The first pass had four mechanism cells and they produced byte-identical
    slot CSVs, for two reasons that are both closed now: the profile path
    posted no preference at all (D-71), and the slot CSV projected nothing the
    mechanism changes (Code To-Do 3.11). The order axis is therefore crossed
    with a *density* axis here -- "preferences first" is not a treatment
    unless there are preferences to serve first.

    The last group before the green-share axis, `mult_overlap_*`, runs on a
    declared scenario variant of its own (D-83) and is read only against its
    own baseline; see `OVERLAP_EVENING_PERIODS`.
    """
    grid = {"baseline_pro_rata": {"preferences": BASELINE_PREFERENCES}}

    # The two orders at three densities. Same densities on both, so a delta is
    # read down a column and never across two differently populated books.
    for order in ("preferences_first", "pro_rata_first"):
        prefix = ("prefs_first" if order == "preferences_first"
                  else "pro_rata_first")
        for density in DENSITIES:
            grid[f"{prefix}_d{int(density * 100):03d}"] = {
                "preferences": prefs(order=order, multipliers_enabled=False),
                "named_share": density, "mutual_share": REFERENCE_MUTUAL}

    # D-71 sensitivity: the reciprocity rate at the reference density. Only a
    # mutual nomination gets priority, so this is the parameter that decides
    # how much of the named volume the mechanism can act on at all.
    for mutual in MUTUAL_SENSITIVITY:
        grid[f"prefs_first_d050_m{int(mutual * 100):03d}"] = {
            "preferences": prefs(order="preferences_first",
                                 multipliers_enabled=False),
            "named_share": 0.50, "mutual_share": mutual}

    # The two multiplier formulations, identical in everything but `mode`
    # (D-46: they differ at every measured parameter point, and the comparison
    # is only a comparison if nothing else moves between them).
    for mode in ("multiplicative", "additive"):
        name = ("multipliers_on" if mode == "multiplicative"
                else "multipliers_on_additive")
        grid[name] = {
            "preferences": prefs(multipliers_enabled=True, mode=mode,
                                 sides="seller",
                                 green_multiplier=GREEN_MULTIPLIER,
                                 grey_levy=GREY_LEVY, levy_cap=LEVY_CAP),
            "named_share": 0.50, "mutual_share": REFERENCE_MUTUAL}

    # D-83. The formulation comparison, on its own declared scenario variant
    # and with its own baseline. `multipliers_on` / `multipliers_on_additive`
    # above stay on the reference window, where they measure the levy and the
    # pool surplus 5.2 reports; the bonus is measured here, because here there
    # is a slot in which it can be funded at all.
    overlap = {"named_share": 0.50, "mutual_share": REFERENCE_MUTUAL,
               "evening_periods": OVERLAP_EVENING_PERIODS}
    grid["mult_overlap_off"] = {
        **overlap, "preferences": prefs(multipliers_enabled=False)}
    for mode in ("multiplicative", "additive"):
        grid[f"mult_overlap_{mode}"] = {
            **overlap,
            "preferences": prefs(multipliers_enabled=True, mode=mode,
                                 sides="seller",
                                 green_multiplier=GREEN_MULTIPLIER,
                                 grey_levy=GREY_LEVY, levy_cap=LEVY_CAP)}

    # The green-share axis: five cells at the baseline preference set, varying
    # nothing but `participation`. Note that `green_share_000` fits on fewer
    # slots than the rest -- with no battery areas the community has no supply
    # at all in the evening periods the rule discharges into, so roughly 262
    # of 672 slots trade against roughly 353 elsewhere. Both round types still
    # occur and the ratio span is unchanged, so it passes pre-flight; but a
    # table that puts its per-slot means next to the other cells is comparing
    # different numbers of slots and has to say so.
    grid.update({
        f"green_share_{int(share * 100):03d}": {
            "preferences": BASELINE_PREFERENCES, "participation": share}
        for share in GREEN_SHARES})
    return grid


def block2_grid() -> dict:
    """Block 2: 14 cells, every one of them executing (D-72).

    `RunSpec.execute` defaulted to False and no cell set it, so no campaign
    run has ever executed. These do. `gamma` is read off the execution node's
    own configuration rather than restated here, so the manifest records the
    number the service actually used.

    Two references, not one. `b2_noise_only` is the accidental layer on its
    own at the campaign sigma, and `b2_noise_off` is the same cell at
    `sigma = 0` (D-84).

    The second one exists because the buyer externality carries no eta
    deadband -- that is the mechanism's own specification, not an oversight
    in the harness ("No eta tolerance for buyers in the spec",
    `penalties.py`). A buyer in a DEMAND_LIMITED round is filled completely,
    so `actual = allocated * (1 + eps)` exceeds its bid whenever `eps > 0`
    and is penalised as under-reporting. Measured on the reference week at
    sigma = 0.05: a penalty pool in 147 of 353 executed slots against 29 for
    the named deviator alone, with up to 38 "deviators" in a single slot.
    **The noise stays on in every arm** -- a buyer side that tolerates no
    forecast error is a finding for 5.3 and Chapter 6, not something to
    configure away -- and `b2_noise_off` is what separates that floor from
    the named deviator's signal.

    Preferences are off throughout -- block 2 measures the penalty layer, and
    crossing it with the preference axis would answer a question nobody asked.
    """
    gamma = execution_gamma()

    def cell(arm, share, k):
        return {"preferences": BASELINE_PREFERENCES, "execute": True,
                "gamma": gamma, "eta_relative": ETA_RELATIVE,
                "deviation": {"arm": arm, "share": share, "k": k,
                              "sigma": SIGMA, "seed": DEVIATION_SEED}}

    grid = {"b2_noise_only": cell(deviations.NONE, 0.0, 0),
            "b2_noise_off": {**cell(deviations.NONE, 0.0, 0),
                             "deviation": {"arm": deviations.NONE,
                                           "share": 0.0, "k": 0,
                                           "sigma": 0.0,
                                           "seed": DEVIATION_SEED}}}
    for arm, prefix in ((deviations.SELLER_ARM, "b2_sell"),
                        (deviations.BUYER_ARM, "b2_buy")):
        # How hard one deviator deviates.
        for share in DEVIATION_SHARES:
            grid[f"{prefix}_s{int(share * 100):02d}"] = cell(arm, share, 1)
        # How many deviate, at the reference share.
        for k in COALITION_SIZES:
            grid[f"{prefix}_k{k:02d}"] = cell(arm, REFERENCE_SHARE, k)
    return grid


def sweep_grid() -> dict:
    """The (gamma, eta) sweep: 12 cells, block 2's reference configuration
    with exactly one parameter moved (D-26, B-10).

    Every cell runs at `BASELINE_PREFERENCES`, `execute = True`, the campaign
    `SIGMA`, `DEVIATION_SEED`, `REFERENCE_PARTICIPATION` and
    `CALIBRATED_SIGMOID` -- i.e. `b2_sell_s25`'s configuration -- so a
    difference measured along either axis is the parameter and nothing else.

    **The reference point is not in this grid.** `eta_relative = 0.10` at the
    configured gamma *is* `b2_sell_s25`, both axes are read against it, and
    re-running it would append a second entry to a manifest the 17./18.09.
    campaign is written against.

    Four groups, and none of them is a duplicate of another:

    * `b2_eta*` -- the eta axis on the withholding arm, where eta is the
      deadband of `W_sell` and extinguishes the deviation exactly at
      `eta_relative = share`. `b2_eta025` is therefore an expected zero and
      not an empty cell; `b2_sell_s10` at `eta_relative = 0.10` is the same
      arithmetic already measured in block 2.
    * `b2_short_g*` -- the gamma axis on the under-delivery arm, which is the
      only arm gamma is observable in at all (`deviations.py`, fact 4). The
      four cells share an arm, a share and a deviation seed, so the
      deviator's measurements are identical across them and the four
      `shortfall_penalty_ct` values stand in the exact ratio of the gammas.
    * `b2_sell_s25_g*` -- the same gamma axis **on the withholding arm**, at
      the three non-default gammas of `SWEEP_GAMMAS` (`b2_sell_s25` is its
      first point). The contrast against `b2_short_g*` is the point of the
      group: the deviator withholds and delivers exactly what it traded, so
      its own penalty is untouched by gamma at every one of these four
      points, while the honest sellers around it carry delivery noise that
      the shortfall term *does* scale with gamma. One cell would show the
      deviator unmoved; three show that the rest of the community is not,
      and that the two effects are separable.

      Table 5.7 reported four gammas on this arm and measured two of them,
      the other two following from the exact linearity of `Phi`. The
      linearity is an argument and belongs in the text; a derived number in
      a results table reads as data, so all four are now run.
    * one control. `b2_sell_s20` fills the gap in block 2's share axis, so
      the eta axis has a `share - eta` neighbour on the other side.
    """
    gamma = execution_gamma()

    def cell(arm, share, *, eta_relative, gamma=gamma):
        return {"preferences": BASELINE_PREFERENCES, "execute": True,
                "gamma": gamma, "eta_relative": eta_relative,
                "deviation": {"arm": arm, "share": share, "k": 1,
                              "sigma": SIGMA, "seed": DEVIATION_SEED}}

    grid = {f"b2_eta{int(eta * 100):03d}":
            cell(deviations.SELLER_ARM, REFERENCE_SHARE, eta_relative=eta)
            for eta in SWEEP_ETAS}
    grid.update({
        f"b2_short_g{int(g * 100):03d}":
        cell(deviations.SELLER_SHORTFALL_ARM, REFERENCE_SHARE,
             eta_relative=ETA_RELATIVE, gamma=g)
        for g in SWEEP_GAMMAS})
    # The controls, all at the reference eta. The gamma list is read off
    # `SWEEP_GAMMAS` and the default is taken out by comparing against
    # `execution_gamma()` rather than by restating it as 1.1: "the configured
    # value" is a fact about the service, and a cell that restated it would
    # keep its name after the node was re-configured.
    for g in SWEEP_GAMMAS:
        if g == gamma:
            # The default gamma at the reference eta *is* `b2_sell_s25`.
            continue
        grid[f"b2_sell_s25_g{int(g * 100):03d}"] = cell(
            deviations.SELLER_ARM, REFERENCE_SHARE,
            eta_relative=ETA_RELATIVE, gamma=g)
    grid["b2_sell_s20"] = cell(deviations.SELLER_ARM, 0.20,
                               eta_relative=ETA_RELATIVE)
    return grid


def _fit_module():
    """`calibration/fit.py`, resolved relative to this file.

    `conftest.py` puts `calibration/` on the path for the test suite, but
    `campaign.py` is also run as a script from inside `harness/`, where it is
    not importable. Resolved from `HARNESS_DIR` rather than from the working
    directory, for the same reason `stack.REPO` is: the harness has to run on
    Windows from wherever it was started.
    """
    directory = str(HARNESS_DIR.parent / "calibration")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import fit
    return fit


def calibration_cells() -> list:
    """The six weeks T-19 fitted on, from one command (Code To-Do 4.6).

    They were produced ad hoc through `runs.run_one`; their manifests are
    complete, but nothing reproduced them. The five stability seeds are
    imported from the fit itself rather than restated here -- two lists that
    have to agree is one list too many -- and 4242 is the out-of-sample week.

    One seed per run, not five: a calibration week *is* its extension seed, so
    `seed = 0` and `cell_seeds = (0,)`. `sigmoid = None` as well, because
    these runs are what the calibrated band is fitted *from* and must not be
    started at it; the clearing node falls back to its own configuration.
    Every field here matches `out/calib-20260917/manifest.json`, which is the
    specification for this function.
    """
    fit = _fit_module()

    def spec(run_id, extension_seed):
        return runs.RunSpec(
            run_id=run_id, seed=0, cell_seeds=(0,), cell="calibration",
            dataset_extension_seed=extension_seed,
            participation=REFERENCE_PARTICIPATION,
            preferences=dict(PREF_OFF), sigmoid=None, execute=False,
            baseline=runs.BASELINE)

    return ([spec(f"calib-{seed}", seed)
             for seed in fit.DEFAULT_STABILITY_SEEDS]
            + [spec("oos-4242", OOS_EXTENSION_SEED)])


def cells(block=None) -> list:
    """The campaign grid: every cell at all five seeds.

    `block` selects a group -- 1, 2, "sweep", "calibration", or None for the
    union. The first cell of block 1 is the delta baseline itself, so it is
    produced by the same code path as everything it is subtracted from.
    """
    if block == "calibration":
        return calibration_cells()
    grid = {}
    if block in (None, 1):
        grid.update(block1_grid())
    if block in (None, 2):
        grid.update(block2_grid())
    if block in (None, "sweep"):
        grid.update(sweep_grid())
    specs = [spec_for(cell, seed, **overrides)
             for cell, overrides in grid.items() for seed in SEEDS]
    if block is None:
        specs += calibration_cells()
    return specs


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
    with context.Pool(processes, maxtasksperchild=1) as pool:
       return pool.map(_worker, [(spec, sha) for spec in specs], chunksize=1)


def run_sequential(specs, *, sha=None) -> list:
    """The same runs in this interpreter. The comparison path for the
    parallel one -- identical results, more wall clock."""
    sha = sha or repo_sha()
    return [runs.run_one(spec, sha=sha) for spec in specs]


def _csv_column_sum(path, column) -> float:
    """Sum of one numeric column of a run's slot CSV, blanks skipped."""
    total = 0.0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get(column)
            if value not in (None, ""):
                total += float(value)
    return total


def report(results) -> None:
    """Every run's guards, printed. `check_ratio_spread` runs inside
    `runs.execute_run`, so a cell that never varied never gets this far --
    a guard that is never called is not a guard.

    Beyond the guards, four result columns: `pairs_posted` for block 1, the
    summed `penalty_pool_ct` for block 2, the summed `shortfall_penalty_ct`
    for the sweep and the summed `bonus_paid_ct` for the D-83 overlap group.
    All four exist so an empty result is visible in the console rather than
    only six weeks later in a CSV -- the first pass printed clean guards for
    four cells that had produced the same file, and nothing on this line said
    so. `bonus_paid_ct` is the one that says whether an overlap cell actually
    found a slot to pay a bonus in; `shortfall_penalty_ct` is the one that
    says whether a gamma cell found a shortfall to price at all, which on
    every arm but `sellers_underdeliver` it does not.
    """
    for result in results:
        checks = result["checks"]
        posted = _csv_column_sum(result["csv"], "pairs_posted")
        pool = _csv_column_sum(result["csv"], "penalty_pool_ct")
        shortfall = _csv_column_sum(result["csv"], "shortfall_penalty_ct")
        bonus = _csv_column_sum(result["csv"], "bonus_paid_ct")
        print(f"  {result['run_id']:<32} "
              f"slots={checks['n_slots']:<4} "
              f"ratio_span={checks['ratio_span']:<10} "
              f"pairs_posted={posted:<10.0f} "
              f"penalty_pool_ct={pool:<12.3f} "
              f"shortfall_penalty_ct={shortfall:<12.3f} "
              f"bonus_paid_ct={bonus:<10.3f} "
              f"{checks['round_type_census']}")


def parse_block(argv) -> object:
    """`--block 1`, `--block 2`, `--block sweep`, `--calibration`, default
    all."""
    if "--calibration" in argv:
        return "calibration"
    if "--block" in argv:
        value = argv[argv.index("--block") + 1]
        if value in ("calibration", "sweep"):
            return value
        if value not in ("1", "2"):
            raise SystemExit(f"--block takes 1, 2, sweep or calibration, "
                             f"not {value!r}")
        return int(value)
    return None


def parse_out(argv) -> str:
    """`--out <dir>`: write this campaign somewhere other than `out/`.

    `write_manifest` appends, so re-running a cell into the directory that
    already holds it adds a second entry to that run's manifest rather than
    replacing it. That is the right behaviour for a re-run of a *campaign*
    and the wrong one for reproducing a week that has already been fitted on
    -- `--calibration` into the live `out/` would append to the manifests of
    the five weeks T-19 used. Hence the redirect, rather than a note in a
    README telling the next person to remember.
    """
    if "--out" not in argv:
        return None
    return argv[argv.index("--out") + 1]


def redirect(specs, out_dir):
    """Every spec's `out_dir` set to `<out_dir>/<run-id>`.

    Set on the spec rather than passed beside it, so the manifest records the
    directory the run actually wrote into.
    """
    if out_dir is None:
        return specs
    root = Path(out_dir)
    return [dataclasses.replace(spec, out_dir=str(root / spec.run_id))
            for spec in specs]

def resume(specs, out_dir=None) -> list:
    """`--resume`: skip every run whose manifest already exists.

    The manifest is written last in `execute_run`, after the CSVs and the
    pre-flight checks, so its presence means the run completed. A directory
    without one is an aborted run and is re-done.
    """
    root = Path(out_dir) if out_dir else OUT
    todo = [s for s in specs if not (root / s.run_id / "manifest.json").exists()]
    print(f"resume: {len(specs) - len(todo)} runs already complete, "
          f"{len(todo)} to run")
    return todo

async def main(block=None, out_dir=None):
    """The profile campaign: every cell at five seeds, one process per run."""
    started = time.perf_counter()
    specs = redirect(cells(block), out_dir)
    if "--resume" in sys.argv:
        specs = resume(specs, out_dir)
    n_cells = len({spec.cell for spec in specs})
    name = {1: "block 1", 2: "block 2", "sweep": "the gamma/eta sweep",
            "calibration": "calibration"}.get(block, "all blocks")
    print(f"campaign ({name}): {len(specs)} runs over {n_cells} cells, "
          f"one process each")
    if not specs:
        print("nothing to run")
        return
    results = run_parallel(specs)
    report(results)
    print(f"\n{len(results)} runs in {time.perf_counter() - started:.1f}s")
    print(f"manifests under {Path(out_dir) if out_dir else OUT}")


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
        asyncio.run(main(parse_block(sys.argv), parse_out(sys.argv)))
