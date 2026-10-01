"""The combined IR cells: preferences, origin adjustment and execution (D-91).

Individual rationality was measured on two halves of the artifact: block 1
runs the preference-first allocation and the energy-origin adjustment
without executing, block 2 and the sweep execute without preferences. The
`ir_*` cells of `campaign.ir_grid()` run both at once. This stage evaluates
them on their own: `dr_analysis.py` hands a group run here before any
existing accumulator sees it, so every table that exists without the group
reproduces byte for byte.

**Decomposition.** Per area row, from `dr3.realised` -- the same u, rates,
penalties and compensation; there is no second formula for u:

    adj       = (r_seller - p) q for sellers (< 0 levy, > 0 bonus), 0 for
                buyers (`sides = "seller"`: buyers settle at the price)
    u_no_adj  = u - adj: the row settled at the uniform price, penalties
                and compensation kept (the block-2 world)
    u_no_exec = (r_seller - K_lower) q for sellers, (K_upper - r_buyer) q
                for buyers: adjustments kept, no penalty, no compensation,
                actual = q (the block-1 world, as `dr3` reads block 1)

A violation is u < -`dr3.IR_TOL` and falls into exactly one class:
`exec_only` (only u_no_adj violates: the penalty layer alone would),
`adj_only` (only u_no_exec does: the levy alone would), `both_alone`
(either alone would) and `interaction` (neither does: only the combination
violates). `rescued` is u >= -tol with u_no_adj < -tol: the bonus prevents a
violation the penalty alone would cause. `dr3.causes` answers a different
question -- which single removal restores u -- and counts an interaction
under both `levy` and `shortfall`; its table is written beside this one.

**Checks.** G1 (completeness) short-circuits: without the whole group there
is nothing to compare. G2-G6 are hard and all evaluated, so `checks.json`
names every failure; any one withholds the `ir_combined_*` tables while the
other tables are written and the exit code is 1. G7 is reported: the tables
are written and a failure makes the exit code 1.
"""
import ast
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import discovery
import dr2
import dr3
import dr4

PREFIX = "ir_"
#: Seeds every cell is expected at (D-59).
SEEDS = (0, 1, 2, 3, 4)
#: The campaign's accidental layer (`campaign.SIGMA`).
SIGMA = 0.05
TOL = dr3.IR_TOL
#: A seller rate differs from the price beyond the six-decimal rounding of
#: the recorded rates.
RATE_TOL = 1e-6
#: G5: the same market, to this tolerance.
MARKET_TOL = 1e-9
#: The reference penalty setting, as `b2_noise_off` ran it.
REFERENCE_GAMMA, REFERENCE_ETA_RELATIVE = 1.1, 0.10

REPO = Path(__file__).resolve().parents[2]
GRID_FILE = "evaluation/harness/campaign.py"


def belongs(cell) -> bool:
    """Whether a cell is one of the combined IR cells: the `ir_` prefix,
    which no other group uses (`test_campaign.py` asserts that)."""
    return str(cell).startswith(PREFIX)


def is_table(name) -> bool:
    return str(name).startswith("ir_combined")


@dataclass(frozen=True)
class Cell:
    """What a combined cell prescribes, as `campaign.ir_grid()` builds it."""
    #: The block-1 cell its preference and window configuration comes from.
    source: str
    #: The run G5 compares its markets with, at the same seed.
    counterpart: str
    mode: str
    grey_levy: float
    gamma: float
    eta_relative: float
    #: On the D-83 overlap window, where a slot carries green and grey
    #: supply and the bonus is paid (G4).
    bonus: bool
    #: G5 leaves the four rate and origin-flow columns out: the levy moved.
    rates_move: bool = False


_OVERLAP = "mult_overlap_multiplicative"
CELLS = {
    "ir_ref_mult": Cell("multipliers_on", "multipliers_on", "multiplicative",
                        0.10, REFERENCE_GAMMA, REFERENCE_ETA_RELATIVE,
                        bonus=False),
    "ir_overlap_mult": Cell(_OVERLAP, _OVERLAP, "multiplicative", 0.10,
                            REFERENCE_GAMMA, REFERENCE_ETA_RELATIVE,
                            bonus=True),
    "ir_overlap_add": Cell("mult_overlap_additive", "mult_overlap_additive",
                           "additive", 0.10, REFERENCE_GAMMA,
                           REFERENCE_ETA_RELATIVE, bonus=True),
    "ir_overlap_mult_levy020": Cell(_OVERLAP, "ir_overlap_mult",
                                    "multiplicative", 0.20, REFERENCE_GAMMA,
                                    REFERENCE_ETA_RELATIVE, bonus=True,
                                    rates_move=True),
    "ir_overlap_mult_eta005": Cell(_OVERLAP, "ir_overlap_mult",
                                   "multiplicative", 0.10, REFERENCE_GAMMA,
                                   0.05, bonus=True),
    "ir_overlap_mult_g150": Cell(_OVERLAP, "ir_overlap_mult",
                                 "multiplicative", 0.10, 1.5,
                                 REFERENCE_ETA_RELATIVE, bonus=True),
}

#: G5: the slot columns clearing decides, and the area rows it allocates.
SLOT_COLUMNS = ("clearing_price_ct", "traded_kwh", "n_pairs", "pair_kwh",
                "green_final_ct", "grey_final_ct", "buyer_final_ct",
                "levy_collected_ct", "bonus_paid_ct")
RATE_COLUMNS = ("green_final_ct", "grey_final_ct", "levy_collected_ct",
                "bonus_paid_ct")
AREA_KEYS = ("slot", "area_uuid", "side", "energy_type", "preference_matched")
AREA_VALUES = ("requested_kwh", "allocated_kwh")

GROUPS = ("seller_green", "seller_grey", "buyer", "all")
CLASSES = ("exec_only", "adj_only", "both_alone", "interaction")
TYPES = (dr2.HOUSEHOLD, dr2.BATTERY)

TABLES = ("ir_combined_census", "ir_combined_ir", "ir_combined_settlement",
          "ir_combined_eq48", "ir_combined_budget", "ir_combined_coverage",
          "ir_combined_decomposition", "ir_combined_participants")
G1 = "ir_combined_g1_completeness"
G2 = "ir_combined_g2_configuration"
G3 = "ir_combined_g3_provenance"
G4 = "ir_combined_g4_exercised"
G5 = "ir_combined_g5_same_markets"
G6 = "ir_combined_g6_classification_closes"
G7 = "ir_combined_g7_budget_balance"
HARD = (G1, G2, G3, G4, G5, G6)


# ----------------------------------------------------- the decomposition

def decompose(rec) -> dict:
    """adj, u_no_adj and u_no_exec beside u, from `dr3.realised`."""
    seller, q, band = rec["seller"], rec["q"], rec["band"]
    adj = np.where(seller, (rec["r_seller"] - rec["price"]) * q, 0.0)
    return {"u": rec["u"], "adj": adj, "u_no_adj": rec["u"] - adj,
            "u_no_exec": np.where(seller,
                                  (rec["r_seller"] - band.k_lower) * q,
                                  (band.k_upper - rec["r_buyer"]) * q)}


def classify(parts, tol=TOL) -> dict:
    """Boolean masks: the violation, its class, and `rescued`."""
    violation = parts["u"] < -tol
    without_adj = parts["u_no_adj"] < -tol
    without_exec = parts["u_no_exec"] < -tol
    return {"violation": violation,
            "exec_only": violation & without_adj & ~without_exec,
            "adj_only": violation & without_exec & ~without_adj,
            "both_alone": violation & without_adj & without_exec,
            "interaction": violation & ~without_adj & ~without_exec,
            "rescued": ~violation & without_adj}


def net_rates(rec) -> np.ndarray:
    """The settlement rate per row, as `dr3.DR3.add` computes it for
    `dr3_settlement`: seller (r s - Phi - phi + c) / s, buyer
    (r_b q + phi - c) / q. NaN where q = 0."""
    q = rec["q"]
    with np.errstate(divide="ignore", invalid="ignore"):
        seller = (rec["r_seller"] * q - rec["shortfall"]
                  - rec["externality"] + rec["compensation"]) / q
        buyer = (rec["r_buyer"] * q + rec["externality"]
                 - rec["compensation"]) / q
    return np.where(rec["seller"], seller, buyer)


# --------------------------------------------------------- the markets

@dataclass
class Market:
    """What clearing decided in one run, in a canonical row order (G5)."""
    slots: np.ndarray
    columns: dict
    keys: dict
    values: dict
    missing: list = field(default_factory=list)


def market(data) -> Market:
    slots, areas = data.slots, data.areas
    order = np.argsort(slots.ints("slot"), kind="stable")
    missing = [c for c in SLOT_COLUMNS if not slots.has(c)]
    columns = {c: slots.floats(c)[order] for c in SLOT_COLUMNS
               if slots.has(c)}
    keys = {"slot": areas.ints("slot")}
    for name in AREA_KEYS[1:]:
        keys[name] = np.array(areas.raw(name), dtype=str)
    rows = np.lexsort(tuple(keys[name] for name in reversed(AREA_KEYS)))
    return Market(slots=slots.ints("slot")[order], columns=columns,
                  keys={k: v[rows] for k, v in keys.items()},
                  values={v: areas.floats(v)[rows] for v in AREA_VALUES},
                  missing=missing)


def _max_deviation(a, b) -> float:
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        return float("inf")
    both = ~nan_a
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


def compare_markets(mine: Market, theirs: Market, skip=()) -> dict:
    """The largest deviation per slot column and per area value; any
    difference in the slot set or the area keys is a mismatch."""
    out = {"slot_columns": {}, "area_values": {}, "mismatch": []}
    for m in (mine, theirs):
        if m.missing:
            out["mismatch"].append(f"slot CSV lacks {', '.join(m.missing)}")
    if (len(mine.slots) != len(theirs.slots)
            or (mine.slots != theirs.slots).any()):
        out["mismatch"].append(f"slot sets differ ({len(mine.slots)} vs "
                               f"{len(theirs.slots)} slots)")
    else:
        for column in SLOT_COLUMNS:
            if column in skip or column not in mine.columns \
                    or column not in theirs.columns:
                continue
            out["slot_columns"][column] = _max_deviation(
                mine.columns[column], theirs.columns[column])
    n_mine, n_theirs = len(mine.keys["slot"]), len(theirs.keys["slot"])
    if n_mine != n_theirs:
        out["mismatch"].append(f"{n_mine} area rows vs {n_theirs}")
    else:
        for name in AREA_KEYS:
            differ = mine.keys[name] != theirs.keys[name]
            if differ.any():
                out["mismatch"].append(f"area {name} differs in "
                                       f"{int(differ.sum())} rows")
        for name in AREA_VALUES:
            out["area_values"][name] = _max_deviation(mine.values[name],
                                                      theirs.values[name])
    deviations = (list(out["slot_columns"].values())
                  + list(out["area_values"].values()))
    out["max_deviation"] = max(deviations) if deviations else 0.0
    out["passed"] = not out["mismatch"] and out["max_deviation"] <= MARKET_TOL
    return out


# ---------------------------------------------------- the run-level checks

def completeness(runs, cells, seeds) -> dict:
    """G1: every cell at every seed, and nothing else in the group."""
    have = defaultdict(set)
    for run in runs:
        have[run.cell].add(run.seed)
    missing = []
    for cell in cells:
        if not have.get(cell):
            missing.append(f"{cell} (no run)")
            continue
        missing += [f"{cell} seed {s}" for s in seeds if s not in have[cell]]
    unexpected = sorted(f"{cell} seed {s}" for cell, found in have.items()
                        for s in found if cell not in cells or s not in seeds)
    return {"passed": not missing and not unexpected,
            "n_runs": len(runs), "n_expected": len(cells) * len(seeds),
            "missing": missing, "unexpected": unexpected}


def _executed(slots) -> np.ndarray:
    if not slots.has("round_type_exec"):
        return np.zeros(slots.n, dtype=bool)
    return np.array([v not in ("", "None")
                     for v in slots.raw("round_type_exec")])


def configuration(data, spec) -> dict:
    """G2 for one run, from its manifest and its slot CSV. The battery
    window is compared with the source run in `IRCombined.run`."""
    run, slots = data.run, data.slots
    problems = []
    if not run.executed:
        problems.append("config.execute is not true")
    prefs = run.config.get("preferences") or {}
    expected = {"enabled": True, "order": "preferences_first",
                "multipliers_enabled": True, "sides": "seller",
                "mode": spec.mode, "grey_levy": spec.grey_levy}
    for key, value in expected.items():
        if prefs.get(key) != value:
            problems.append(f"config.preferences.{key} = "
                            f"{prefs.get(key)!r}, not {value!r}")
    plan = run.deviation
    if plan.get("arm") != "none":
        problems.append(f"deviation.arm = {plan.get('arm')!r}, not 'none'")
    if plan.get("sigma") != SIGMA:
        problems.append(f"deviation.sigma = {plan.get('sigma')!r}, not "
                        f"{SIGMA}")
    if run.deviators:
        problems.append(f"deviators {list(run.deviators)}")

    cleared = ~np.isnan(slots.floats("clearing_price_ct"))
    for column, value in (("pref_enabled", "True"),
                          ("pref_order", "preferences_first"),
                          ("mult_enabled", "True"),
                          ("mult_mode", spec.mode),
                          ("mult_sides", "seller")):
        if not slots.has(column):
            problems.append(f"slot CSV has no {column}")
            continue
        off = cleared & (slots.strings(column) != value)
        if off.any():
            problems.append(f"{column} is not {value!r} in {int(off.sum())} "
                            f"cleared slots")

    executed = _executed(slots)
    applied = {}
    if not executed.any():
        problems.append("no executed slot")
    if data.param_source != discovery.SLOT_CSV:
        problems.append(f"applied gamma / eta not in the slot CSV "
                        f"(param_source {data.param_source})")
    else:
        for column, value in (("gamma_eff", spec.gamma),
                              ("eta_relative_eff", spec.eta_relative)):
            values = slots.floats(column)[executed]
            applied[column] = sorted(set(values[~np.isnan(values)].tolist()))
            off = np.isnan(values) | (np.abs(values - value) > 1e-12)
            if off.any():
                problems.append(f"{column} is not {value} in "
                                f"{int(off.sum())} executed slots "
                                f"({applied[column]})")
    return {"problems": problems, "applied": applied,
            "battery_window": run.entry.get("battery_window")}


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args],
                                   text=True, stderr=subprocess.DEVNULL)


def defines_grid(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "ir_grid"
               for node in tree.body)


def provenance(runs, git=None) -> dict:
    """G3: one SHA for the whole group, a commit of this checkout, and one
    whose `campaign.py` defines the grid the runs claim to come from."""
    git = git or _git
    shas = sorted({str(r.repo_sha) for r in runs})
    multi = sorted(r.run_id for r in runs if r.n_entries != 1)
    problems = []
    if len(shas) != 1:
        problems.append(f"{len(shas)} different repo_sha: {shas}")
    if multi:
        problems.append(f"{len(multi)} manifest(s) with more than one entry: "
                        f"{multi[:5]}")
    if len(shas) == 1:
        sha = shas[0]
        try:
            git("cat-file", "-e", f"{sha}^{{commit}}")
        except (OSError, subprocess.CalledProcessError):
            problems.append(f"{sha} is not a commit of this checkout")
        else:
            try:
                source = git("show", f"{sha}:{GRID_FILE}")
            except (OSError, subprocess.CalledProcessError):
                problems.append(f"{GRID_FILE} does not exist at {sha}")
            else:
                if not defines_grid(source):
                    problems.append(f"{GRID_FILE} at {sha} does not define "
                                    f"ir_grid")
    return {"passed": not problems,
            "repo_sha": shas[0] if len(shas) == 1 else shas,
            "n_runs": len(runs), "problems": problems}


# -------------------------------------------------------------- the stage

@dataclass
class Stage:
    checks: dict
    findings: dict
    #: None when no group run was found or a hard check stopped the stage.
    tables: dict | None
    cells: list
    repo_sha: object
    wall_sec: float


def _new_group():
    return {"n": 0, "n_levy": 0, "n_bonus": 0, "n_penalised": 0,
            "n_adj_and_penalty": 0, "n_violations": 0,
            **{f"n_{name}": 0 for name in CLASSES}, "n_rescued": 0,
            "min_u": np.inf, "sum_negative_u": 0.0, "rates": [],
            "n_below": 0}


class IRCombined:
    """The group's runs, and the block-1 counterparts G5 compares with.

    `add` takes a group run, `observe` every other run (it keeps only the
    counterparts' markets and manifests); `run` evaluates G1-G7 and builds
    the tables. Inactive -- no group run in the run set -- `observe` keeps
    nothing.
    """

    def __init__(self, *, active=True, cells=None, seeds=None):
        self.active = active
        self.cells = CELLS if cells is None else cells
        self.seeds = tuple(SEEDS if seeds is None else seeds)
        self.counterparts = ({c.source for c in self.cells.values()}
                             | {c.counterpart for c in self.cells.values()})
        self.runs = []
        self.census = []
        self.acc3, self.acc4 = dr3.DR3(), dr4.DR4()
        self.markets = {}
        self.manifests = {}
        self.config = {}
        self.exercise = {}
        self.groups = defaultdict(_new_group)
        self.weeks = defaultdict(list)

    # ------------------------------------------------------------ input

    def observe(self, data) -> None:
        run = data.run
        if not self.active or run.cell not in self.counterparts:
            return
        self.markets[(run.cell, run.seed)] = market(data)
        self.manifests[(run.cell, run.seed)] = run

    def add(self, data, census=None) -> None:
        run = data.run
        self.runs.append(run)
        if census is not None:
            self.census.append(census)
        self.acc3.add(data)
        self.acc4.add(data)
        self.markets[(run.cell, run.seed)] = market(data)
        self.manifests[(run.cell, run.seed)] = run
        spec = self.cells.get(run.cell)
        self.config[run.run_id] = (
            configuration(data, spec) if spec is not None else
            {"problems": [f"{run.cell} is not a cell of the group"],
             "applied": {}, "battery_window": None})
        self.exercise[run.run_id] = self._exercise(data)
        rec = dr3.realised(data)
        parts = decompose(rec)
        masks = classify(parts)
        self._decompose(run, rec, parts, masks)
        self._participants(data, parts, masks)

    def _exercise(self, data) -> dict:
        slots = data.slots

        def total(column):
            return (float(np.nansum(slots.floats(column)))
                    if slots.has(column) else 0.0)

        pool = (slots.floats_or_zero("penalty_pool_ct")
                if slots.has("penalty_pool_ct") else np.zeros(slots.n))
        spec = self.cells.get(data.run.cell)
        return {"bonus_expected": bool(spec is not None and spec.bonus),
                "levy_collected_ct": total("levy_collected_ct"),
                "bonus_paid_ct": total("bonus_paid_ct"),
                "n_slots_with_pool": int((_executed(slots)
                                          & (pool > 0)).sum())}

    def _decompose(self, run, rec, parts, masks) -> None:
        seller, grey, q = rec["seller"], rec["grey"], rec["q"]
        diff = rec["r_seller"] - rec["price"]
        levy = seller & (diff < -RATE_TOL)
        bonus = seller & (diff > RATE_TOL)
        penalised = (rec["shortfall"] + rec["externality"]) > TOL
        rates = net_rates(rec)
        traded = q > 0
        below = seller & traded & (rates < rec["band"].k_lower)
        u = parts["u"]
        members = {"seller_green": seller & ~grey,
                   "seller_grey": seller & grey,
                   "buyer": ~seller, "all": np.ones(len(u), dtype=bool)}
        for group, mask in members.items():
            acc = self.groups[(run.cell, group)]
            acc["n"] += int(mask.sum())
            acc["n_levy"] += int((mask & levy).sum())
            acc["n_bonus"] += int((mask & bonus).sum())
            acc["n_penalised"] += int((mask & penalised).sum())
            acc["n_adj_and_penalty"] += int(
                (mask & (levy | bonus) & penalised).sum())
            acc["n_violations"] += int((mask & masks["violation"]).sum())
            for name in CLASSES + ("rescued",):
                acc[f"n_{name}"] += int((mask & masks[name]).sum())
            if mask.any():
                acc["min_u"] = min(acc["min_u"], float(u[mask].min()))
            acc["sum_negative_u"] += float(u[mask & masks["violation"]].sum())
            acc["n_below"] += int((mask & below).sum())
            if group != "all":
                acc["rates"].append(rates[mask & traded])

    def _participants(self, data, parts, masks) -> None:
        """U = sum of u over the week, per run and area (both sides)."""
        areas = np.array(data.areas.raw("area_uuid"), dtype=str)
        ids, inverse = np.unique(areas, return_inverse=True)
        week = np.bincount(inverse, weights=parts["u"], minlength=len(ids))
        violated = np.bincount(inverse, weights=masks["violation"]
                               .astype(float), minlength=len(ids)) > 0
        for area, total, hit in zip(ids, week, violated):
            self.weeks[(data.run.cell, dr2.participant_type(area))].append(
                (float(total), bool(hit)))

    # ----------------------------------------------------------- tables

    def decomposition_table(self) -> list:
        out = []
        cells = sorted({cell for cell, _group in self.groups})
        for cell in cells:
            for group in GROUPS:
                acc = self.groups[(cell, group)]
                rates = (np.concatenate(acc["rates"]) if acc["rates"]
                         else np.array([]))
                stats = (dr3._quantiles(rates) if group != "all"
                         else dict.fromkeys(("min", "p10", "median")))
                out.append({
                    "cell": cell, "group": group,
                    **{key: acc[key] for key in (
                        "n", "n_levy", "n_bonus", "n_penalised",
                        "n_adj_and_penalty", "n_violations")},
                    **{f"n_{name}": acc[f"n_{name}"]
                       for name in CLASSES + ("rescued",)},
                    "min_u_ct": acc["min_u"] if acc["n"] else None,
                    "sum_negative_u_ct": acc["sum_negative_u"],
                    "net_rate_min_ct": stats["min"],
                    "net_rate_p10_ct": stats["p10"],
                    "net_rate_median_ct": stats["median"],
                    "n_net_rate_below_k_lower": (None if group == "buyer"
                                                 else acc["n_below"])})
        return out

    def participants_table(self) -> list:
        out = []
        cells = sorted({cell for cell, _type in self.weeks}
                       | {cell for cell, _group in self.groups})
        for cell in cells:
            for kind in TYPES:
                items = self.weeks.get((cell, kind), [])
                week = np.array([w for w, _hit in items], dtype=float)
                hit = np.array([h for _w, h in items], dtype=bool)
                stats = dr3._quantiles(week)
                out.append({"cell": cell, "type": kind,
                            "n_participant_weeks": len(items),
                            "n_with_violating_slot": int(hit.sum()),
                            "n_negative_week": int((week < -TOL).sum()),
                            "min_week_u_ct": stats["min"],
                            "p10_week_u_ct": stats["p10"],
                            "median_week_u_ct": stats["median"]})
        return out

    def tables(self) -> dict:
        return {"ir_combined_census": self.census,
                "ir_combined_ir": self.acc3.ir_table(),
                "ir_combined_settlement": self.acc3.settlement_table(),
                "ir_combined_eq48": self.acc3.eq48_table(),
                "ir_combined_budget": self.acc4.layers_table(),
                "ir_combined_coverage": self.acc4.coverage_table(),
                "ir_combined_decomposition": self.decomposition_table(),
                "ir_combined_participants": self.participants_table()}

    # ----------------------------------------------------------- checks

    def configuration_check(self) -> dict:
        """G2, with the battery window read against the source run."""
        problems, applied, windows = {}, defaultdict(dict), {}
        for run in self.runs:
            found = self.config[run.run_id]
            mine = list(found["problems"])
            spec = self.cells.get(run.cell)
            if spec is not None:
                source = self.manifests.get((spec.source, run.seed))
                theirs = (source.entry.get("battery_window")
                          if source is not None else None)
                if source is None:
                    mine.append(f"source run {spec.source} seed {run.seed} "
                                f"not in the run set")
                elif found["battery_window"] is None \
                        or found["battery_window"] != theirs:
                    mine.append(f"battery_window {found['battery_window']} "
                                f"is not the source's {theirs}")
                windows[run.cell] = found["battery_window"]
            for column, values in found["applied"].items():
                applied[column].setdefault(run.cell, set()).update(values)
            if mine:
                problems[run.run_id] = mine
        return {"passed": not problems, "n_runs": len(self.runs),
                "n_runs_with_problems": len(problems),
                "problems": dict(sorted(problems.items())[:10]),
                "gamma_eff": {c: sorted(v) for c, v in
                              sorted(applied["gamma_eff"].items())},
                "eta_relative_eff": {c: sorted(v) for c, v in
                                     sorted(applied["eta_relative_eff"]
                                            .items())},
                "battery_window": dict(sorted(windows.items()))}

    def markets_check(self) -> dict:
        """G5, per cell pair and seed; the largest deviation per pair."""
        pairs = {}
        for cell, spec in sorted(self.cells.items()):
            skip = RATE_COLUMNS if spec.rates_move else ()
            result = {"counterpart": spec.counterpart,
                      "compared": [c for c in SLOT_COLUMNS if c not in skip]
                      + [f"area {k}" for k in AREA_KEYS + AREA_VALUES],
                      "n_seeds": 0, "max_deviation": 0.0,
                      "max_slot_column_deviation": {}, "max_area_deviation":
                      {}, "problems": []}
            for seed in self.seeds:
                mine = self.markets.get((cell, seed))
                theirs = self.markets.get((spec.counterpart, seed))
                if mine is None or theirs is None:
                    absent = cell if mine is None else spec.counterpart
                    result["problems"].append(
                        f"seed {seed}: {absent} not in the run set")
                    continue
                found = compare_markets(mine, theirs, skip)
                result["n_seeds"] += 1
                result["max_deviation"] = max(result["max_deviation"],
                                              found["max_deviation"])
                for into, key in (("max_slot_column_deviation",
                                   "slot_columns"),
                                  ("max_area_deviation", "area_values")):
                    for name, value in found[key].items():
                        result[into][name] = max(result[into].get(name, 0.0),
                                                 value)
                result["problems"] += [f"seed {seed}: {m}"
                                       for m in found["mismatch"]]
                if found["max_deviation"] > MARKET_TOL:
                    worst = max(list(found["slot_columns"].items())
                                + list(found["area_values"].items()),
                                key=lambda item: item[1])
                    result["problems"].append(
                        f"seed {seed}: {worst[0]} deviates by {worst[1]:.3e}")
            result["passed"] = not result["problems"]
            pairs[f"{cell} <-> {spec.counterpart}"] = result
        return {"passed": all(p["passed"] for p in pairs.values()),
                "tolerance": MARKET_TOL,
                "max_deviation": max((p["max_deviation"]
                                      for p in pairs.values()), default=0.0),
                "pairs": pairs}

    def exercised_check(self, decomposition) -> dict:
        """G4: the levy, the bonus and the penalty layer each actually did
        something, and they met in at least one row per cell."""
        problems = []
        for run_id, sums in sorted(self.exercise.items()):
            if not sums["levy_collected_ct"] > 0:
                problems.append(f"{run_id}: no levy collected")
            if sums["bonus_expected"] and not sums["bonus_paid_ct"] > 0:
                problems.append(f"{run_id}: no bonus paid")
            if not sums["n_slots_with_pool"] > 0:
                problems.append(f"{run_id}: no executed slot with a "
                                f"penalty pool")
        together = {row["cell"]: row["n_adj_and_penalty"]
                    for row in decomposition if row["group"] == "all"}
        problems += [f"{cell}: no row with an adjustment and a penalty"
                     for cell, n in sorted(together.items()) if not n > 0]

        def least(key, only_bonus=False):
            values = [v[key] for v in self.exercise.values()
                      if v["bonus_expected"] or not only_bonus]
            return min(values) if values else None

        return {"passed": not problems, "problems": problems[:20],
                "min_levy_collected_ct_per_run": least("levy_collected_ct"),
                "min_bonus_paid_ct_per_run_bonus_cells":
                    least("bonus_paid_ct", only_bonus=True),
                "min_slots_with_pool_per_run": least("n_slots_with_pool"),
                "n_adj_and_penalty_by_cell": dict(sorted(together.items()))}

    def closes_check(self, decomposition, ir_rows) -> dict:
        """G6: the four classes partition the violations, and the
        violations are the ones `dr3` counts."""
        problems = []
        largest = 0
        for row in decomposition:
            gap = row["n_violations"] - sum(row[f"n_{c}"] for c in CLASSES)
            largest = max(largest, abs(gap))
            if gap:
                problems.append(f"{row['cell']} {row['group']}: "
                                f"{row['n_violations']} violations, classes "
                                f"sum to {row['n_violations'] - gap}")
        dr3_any = defaultdict(int)
        for row in ir_rows:
            if row["cause"] == "any":
                dr3_any[(row["cell"], row["side"])] += row["n_violations"]
        mine = defaultdict(int)
        for row in decomposition:
            side = {"seller_green": "seller", "seller_grey": "seller",
                    "buyer": "buyer"}.get(row["group"])
            if side:
                mine[(row["cell"], side)] += row["n_violations"]
        for key in sorted(set(mine) | set(dr3_any)):
            gap = mine.get(key, 0) - dr3_any.get(key, 0)
            largest = max(largest, abs(gap))
            if gap:
                problems.append(f"{key[0]} {key[1]}: {mine.get(key, 0)} here, "
                                f"{dr3_any.get(key, 0)} in ir_combined_ir")
        return {"passed": not problems, "max_abs_difference": largest,
                "n_rows": len(decomposition), "problems": problems[:20]}

    def budget_check(self, layers) -> dict:
        """G7: budget balance under the combined configuration, and the
        single-deviator coverage check on the group."""
        externality = [r for r in layers if r["layer"] == "externality"]
        imbalanced = {r["cell"]: int(r["n_imbalanced"] or 0)
                      for r in externality}
        coverage = self.acc4.single_deviator_check()
        return {"passed": bool(externality)
                and not any(imbalanced.values()) and coverage["passed"],
                "n_imbalanced_by_cell": dict(sorted(imbalanced.items())),
                "max_abs_balance_ct": max((r["max_abs_balance_ct"] or 0.0
                                           for r in externality),
                                          default=None),
                "single_deviator_coverage": coverage}

    # -------------------------------------------------------------- run

    def run(self, log=print, err=print) -> Stage:
        started = time.perf_counter()

        def elapsed():
            return round(time.perf_counter() - started, 1)

        if not self.runs:
            return Stage({"ir_combined": {"passed": True, "skipped": True,
                                          "reason": "no runs"}},
                         {}, None, [], None, elapsed())
        cells = sorted({r.cell for r in self.runs})
        shas = sorted({str(r.repo_sha) for r in self.runs})
        repo_sha = shas[0] if len(shas) == 1 else shas
        checks = {G1: completeness(self.runs, list(self.cells), self.seeds)}

        def summary(stopped):
            return {"passed": not stopped, "skipped": False,
                    "n_runs": len(self.runs), "cells": cells,
                    "stopped_by": stopped}

        if not checks[G1]["passed"]:
            what = checks[G1]["missing"] + [
                f"unexpected {u}" for u in checks[G1]["unexpected"]]
            err(f"IR combined cells stopped: G1 incomplete group, "
                f"{', '.join(what)}")
            err("no ir_combined table written")
            checks["ir_combined"] = summary([G1])
            return Stage(checks, {}, None, cells, repo_sha, elapsed())

        checks[G2] = self.configuration_check()
        checks[G3] = provenance(self.runs)
        checks[G5] = self.markets_check()
        tables = self.tables()
        decomposition = tables["ir_combined_decomposition"]
        checks[G4] = self.exercised_check(decomposition)
        checks[G6] = self.closes_check(decomposition,
                                       tables["ir_combined_ir"])
        checks[G7] = self.budget_check(tables["ir_combined_budget"])
        log(f"IR combined cells: {len(self.runs)} runs over {len(cells)} "
            f"cells; G5 max deviation {checks[G5]['max_deviation']:.3e}")
        findings = {"cells": {}}
        for row in decomposition:
            if row["group"] == "all":
                findings["cells"][row["cell"]] = {
                    key: row[key] for key in (
                        "n", "n_violations", *[f"n_{c}" for c in CLASSES],
                        "n_rescued", "n_adj_and_penalty", "min_u_ct")}
                log(f"  {row['cell']}: {row['n_violations']} violations of "
                    f"{row['n']} (interaction {row['n_interaction']}, "
                    f"exec_only {row['n_exec_only']}, adj_only "
                    f"{row['n_adj_only']}, both_alone {row['n_both_alone']}), "
                    f"rescued {row['n_rescued']}")
        stopped = [name for name in HARD if not checks[name]["passed"]]
        checks["ir_combined"] = summary(stopped)
        if stopped:
            for name in stopped:
                err(f"IR combined cells stopped: {name}: "
                    f"{checks[name].get('problems') or checks[name]}")
            err("no ir_combined table written")
            tables = None
        findings["wall_sec"] = elapsed()
        return Stage(checks, findings, tables, cells, repo_sha, elapsed())
