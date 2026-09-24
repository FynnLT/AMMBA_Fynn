"""Which recorded runs the analysis reads, and how it reads them (A0).

A run is a directory that holds `manifest.json`, `<run_id>_slots.csv` and
`<run_id>_areas.csv`. Everything else under `out/` -- the N-curve, the pilot
CSVs, an earlier analysis -- has no area CSV and is skipped, with the reason
recorded rather than silently dropped.

**Read-only.** Nothing here opens a file for writing. The CSVs are read as
the harness wrote them (`aggregates.write_slot_csv` / `write_area_csv`): the
columns are projections of what the artifact answered, and this module only
converts them to arrays.
"""
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class DiscoveryError(RuntimeError):
    """The run set cannot be analysed as it stands."""


class MissingColumns(DiscoveryError):
    """A run's CSVs lack columns the analysis reads."""

    def __init__(self, run, columns):
        super().__init__(f"{run.path}: missing {', '.join(columns)}")
        self.run = run
        self.columns = columns


@dataclass(frozen=True)
class Run:
    """One recorded run, as its last manifest entry describes it."""
    path: Path
    run_id: str
    cell: str
    seed: int
    repo_sha: str
    entry: dict
    n_entries: int
    sigmoid: dict
    sigmoid_source: str

    @property
    def config(self) -> dict:
        return self.entry.get("config") or {}

    @property
    def executed(self) -> bool:
        return bool(self.config.get("execute"))

    @property
    def deviation(self) -> dict:
        """The realised plan (`manifest.deviation`), not the configured one:
        it carries the drawn deviator ids."""
        return self.entry.get("deviation") or {}

    @property
    def arm(self) -> str | None:
        return self.deviation.get("arm")

    @property
    def deviators(self) -> tuple:
        return tuple(self.deviation.get("deviators") or ())

    @property
    def slots_path(self) -> Path:
        return self.path / f"{self.run_id}_slots.csv"

    @property
    def areas_path(self) -> Path:
        return self.path / f"{self.run_id}_areas.csv"


@dataclass
class Discovery:
    runs: list = field(default_factory=list)
    #: (directory, reason) for every manifest directory that is not a run.
    skipped: list = field(default_factory=list)
    #: Run directories whose manifest holds more than one entry (a re-run
    #: was appended); the last entry is the one used.
    multi_entry: list = field(default_factory=list)


def _manifest_entries(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _sigmoid(entry: dict) -> tuple:
    """The band the node applied, and where it was read from.

    `sigmoid_effective` is the node's own echo (runs from 17.09. on);
    `config.sigmoid` is what the harness asked for, the only record older
    runs have. Neither: the run cannot be priced, so the analysis stops.
    """
    effective = entry.get("sigmoid_effective")
    if effective:
        return dict(effective), "sigmoid_effective"
    configured = (entry.get("config") or {}).get("sigmoid")
    if configured:
        return dict(configured), "config.sigmoid"
    return None, None


def discover(roots, *, exclude=()) -> Discovery:
    """Every run below `roots`, calibration runs left out.

    Two directories with the same `(cell, seed)` stop the analysis: which of
    them a table should read is not a choice this script may make.
    """
    found = Discovery()
    excluded = {Path(p).resolve() for p in exclude}
    by_key = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise DiscoveryError(f"--runs {root} is not a directory")
        for dirpath, dirnames, filenames in os.walk(root):
            directory = Path(dirpath)
            if directory.resolve() in excluded:
                dirnames[:] = []
                continue
            # Archive folders (`_old`, `_superseded`, ...) hold earlier runs
            # of the same cells; reading them would be a duplicate
            # (cell, seed). Not walked, but listed.
            for name in sorted(d for d in dirnames if d.startswith("_")):
                found.skipped.append((str(directory / name),
                                      "archive directory (leading _)"))
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("_"))
            if "manifest.json" not in filenames:
                continue
            try:
                entries = _manifest_entries(directory / "manifest.json")
            except (json.JSONDecodeError, OSError) as exc:
                found.skipped.append((str(directory),
                                      f"unreadable manifest: {exc}"))
                continue
            if not entries:
                found.skipped.append((str(directory), "empty manifest"))
                continue
            entry = entries[-1]
            run_id = entry.get("run_id")
            if not run_id or not (directory / f"{run_id}_areas.csv").exists():
                found.skipped.append((str(directory), "no area CSV"))
                continue
            if not (directory / f"{run_id}_slots.csv").exists():
                found.skipped.append((str(directory), "no slot CSV"))
                continue
            cell = entry.get("cell") or (entry.get("config") or {}).get("cell")
            if cell == "calibration":
                found.skipped.append((str(directory), "calibration run"))
                continue
            seed = entry.get("seed")
            if cell is None or seed is None:
                found.skipped.append((str(directory),
                                      "manifest names no cell or seed"))
                continue
            sigmoid, source = _sigmoid(entry)
            if sigmoid is None:
                raise DiscoveryError(
                    f"{directory}: manifest carries neither sigmoid_effective "
                    f"nor config.sigmoid, so the run cannot be priced")
            key = (cell, int(seed))
            if key in by_key:
                raise DiscoveryError(
                    f"two runs for cell {cell!r} seed {seed}:\n"
                    f"  {by_key[key]}\n  {directory}\n"
                    f"which one a table reads is not the analysis's choice")
            by_key[key] = directory
            if len(entries) > 1:
                found.multi_entry.append(str(directory))
            found.runs.append(Run(
                path=directory, run_id=run_id, cell=cell, seed=int(seed),
                repo_sha=entry.get("repo_sha"), entry=entry,
                n_entries=len(entries), sigmoid=sigmoid,
                sigmoid_source=source))
    found.runs.sort(key=lambda r: (r.cell, r.seed))
    return found


# --------------------------------------------------------------- the tables

class Table:
    """A CSV as columns of strings, converted on demand.

    Column access is by name, so a CSV written by a later harness with more
    columns reads the same; a missing column is an error at the point of use,
    not a silent NaN.
    """

    def __init__(self, path: Path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
        self.path = Path(path)
        self.header = header
        self.n = len(rows)
        columns = list(zip(*rows)) if rows else [()] * len(header)
        self._raw = {name: columns[i] for i, name in enumerate(header)}
        self._cache = {}

    def has(self, name: str) -> bool:
        return name in self._raw

    def inject(self, name: str, values) -> None:
        """Add a column the file does not carry (strings, one per row)."""
        if name in self._raw:
            raise ValueError(f"{self.path.name} already has {name!r}")
        values = tuple(values)
        if len(values) != self.n:
            raise ValueError(f"{name}: {len(values)} values for {self.n} rows")
        self._raw[name] = values

    def raw(self, name: str):
        if name not in self._raw:
            raise KeyError(f"{self.path.name} has no column {name!r}")
        return self._raw[name]

    def floats(self, name: str) -> np.ndarray:
        """Empty cells are NaN: "not recorded" stays distinguishable from 0."""
        key = ("f", name)
        if key not in self._cache:
            self._cache[key] = np.array(
                [float(v) if v not in ("", "None") else np.nan
                 for v in self.raw(name)], dtype=float)
        return self._cache[key]

    def floats_or_zero(self, name: str) -> np.ndarray:
        return np.nan_to_num(self.floats(name), nan=0.0)

    def bools(self, name: str) -> np.ndarray:
        key = ("b", name)
        if key not in self._cache:
            self._cache[key] = np.array([v == "True" for v in self.raw(name)],
                                        dtype=bool)
        return self._cache[key]

    def strings(self, name: str) -> np.ndarray:
        key = ("s", name)
        if key not in self._cache:
            self._cache[key] = np.array(self.raw(name), dtype=object)
        return self._cache[key]

    def ints(self, name: str) -> np.ndarray:
        key = ("i", name)
        if key not in self._cache:
            self._cache[key] = np.array([int(v) for v in self.raw(name)],
                                        dtype=np.int64)
        return self._cache[key]


#: The columns the analysis reads, by CSV and by run kind. Checked on the
#: header before anything is computed: a run written by an older harness
#: that lacks one is named, not half-read.
REQUIRED_SLOT_COLUMNS = ("slot", "total_supply_kwh", "total_demand_kwh",
                         "clearing_price_ct", "traded_kwh")
REQUIRED_EXECUTED_SLOT_COLUMNS = (
    "round_type_exec", "penalty_pool_ct", "compensated_ct",
    "budget_balance_ct", "n_deviators", "shortfall_penalty_ct")
#: The applied penalty parameters (`execution.penalty_params`). Slot CSVs
#: written before 91790e9 (18.09.) do not carry them; for those runs they are
#: taken from the manifest's `config.gamma` / `config.eta_relative`, but only
#: when both are set explicitly -- `None` there means "whatever the node was
#: configured with that day", which is not a record of what it applied.
PARAM_COLUMNS = ("gamma_eff", "eta_relative_eff", "eta_mode",
                 "k_sho_ct_per_kwh")
SLOT_CSV, MANIFEST_CONFIG = "slot_csv", "manifest_config"

REQUIRED_BLOCK1_SLOT_COLUMNS = ("pref_enabled", "pref_order", "pair_kwh")
REQUIRED_AREA_COLUMNS = ("slot", "area_uuid", "side", "energy_type",
                         "requested_kwh", "allocated_kwh", "fill_rate",
                         "preference_matched", "named", "partner")
REQUIRED_EXECUTED_AREA_COLUMNS = (
    "actual_kwh", "deliverable_kwh", "shortfall_penalty_ct",
    "externality_penalty_ct", "total_penalty_ct", "compensation_ct")


def missing_columns(run: Run, slots: "Table", areas: "Table") -> list:
    slot_cols = REQUIRED_SLOT_COLUMNS + (
        REQUIRED_EXECUTED_SLOT_COLUMNS + PARAM_COLUMNS if run.executed
        else REQUIRED_BLOCK1_SLOT_COLUMNS)
    area_cols = REQUIRED_AREA_COLUMNS + (
        REQUIRED_EXECUTED_AREA_COLUMNS if run.executed else ())
    return ([f"{run.slots_path.name}:{c}" for c in slot_cols
             if not slots.has(c)]
            + [f"{run.areas_path.name}:{c}" for c in area_cols
               if not areas.has(c)])


def _param_fallback(run: Run, slots: Table) -> str | None:
    """Fill the applied penalty parameters from the manifest, where the slot
    CSV predates them. Returns the source, None when neither has them."""
    absent = [c for c in PARAM_COLUMNS if not slots.has(c)]
    if not absent:
        return SLOT_CSV
    gamma = run.config.get("gamma")
    eta = run.config.get("eta_relative")
    if gamma is None or eta is None:
        return None
    k_upper = float(run.sigmoid["k_upper"])
    executed = [v not in ("", "None") for v in slots.raw("round_type_exec")]
    # The harness hands `eta_relative` to the node whenever it is set, so
    # the node ran in relative mode (`execution.py`, eta_mode).
    value = {"gamma_eff": repr(float(gamma)),
             "eta_relative_eff": repr(float(eta)),
             "eta_mode": "relative",
             "k_sho_ct_per_kwh": repr(round(float(gamma) * k_upper, 6))}
    for column in absent:
        slots.inject(column, [value[column] if on else "" for on in executed])
    return MANIFEST_CONFIG


@dataclass
class RunData:
    """One run's two CSVs, joined on the slot."""
    run: Run
    slots: Table
    areas: Table
    #: For every area row, the index of its slot's row in `slots`.
    slot_index: np.ndarray
    #: Where the applied gamma / eta came from: `slot_csv`, or
    #: `manifest_config` for slot CSVs that predate the columns; None for a
    #: run that did not execute.
    param_source: str | None = None

    @classmethod
    def load(cls, run: Run) -> "RunData":
        slots = Table(run.slots_path)
        areas = Table(run.areas_path)
        param_source = (_param_fallback(run, slots)
                        if run.executed and slots.has("round_type_exec")
                        else None)
        missing = missing_columns(run, slots, areas)
        if missing:
            raise MissingColumns(run, missing)
        position = {int(s): i for i, s in enumerate(slots.ints("slot"))}
        try:
            index = np.array([position[int(s)] for s in areas.ints("slot")],
                             dtype=np.int64)
        except KeyError as exc:
            raise DiscoveryError(
                f"{run.areas_path.name} has a slot the slot CSV does not: "
                f"{exc}") from exc
        return cls(run=run, slots=slots, areas=areas, slot_index=index,
                   param_source=param_source)

    def slot_value(self, name: str) -> np.ndarray:
        """A slot column broadcast onto the area rows."""
        return self.slots.floats(name)[self.slot_index]
