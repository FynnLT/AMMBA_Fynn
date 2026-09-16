"""The Faia et al. energy-community workbook as market profiles.

Reads `evaluation/data/raw/EC_EV_dataset (fixed error).xlsx` (see
`evaluation/data/PROVENANCE.md` for source, DOI, licence and the SHA-256 the
figures were produced from), selects the reference community, and builds the
672-slot week the campaign runs over.

Nothing here writes into `data/raw/`. The workbook is read-only for every
script in this harness; a run that modifies it is a bug.

Two conversions are explicit because getting either wrong is silent:

* **Values are kW, not kWh.** Energy per period is `value * 0.25 h`. Reading
  the sheets as kWh inflates every quantity by four.
* **Period `n` covers `[(n-1)*900 s, n*900 s)` and is labelled by its end.**
  Period 33 is 08:15, period 76 is 19:00.

What the workbook does *not* contain is a grey selling side: 200 players have
PV, 150 have storage, all 150 of those also have PV, and none has storage
without it. `BESS` is a parameter sheet, not a time series -- there is no
charging history to read. `make_battery_areas` models one; see its docstring
for what is data and what is assumption (D-75).
"""
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

PERIOD_SEC = 900
PERIODS_PER_DAY = 96
HOURS_PER_PERIOD = PERIOD_SEC / 3600.0      # 0.25 h -- the kW -> kWh factor
PLAYERS = 250

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_WORKBOOK = DATA_DIR / "raw" / "EC_EV_dataset (fixed error).xlsx"

# Sheet rows of `BESS`, by their label with the workbook's own quoting stripped.
_BESS_ROWS = {
    "Model": "model",
    "Capacity (kW)": "capacity_kwh",
    "Charge (kW)": "charge_kw",
    "Discharge (kW)": "discharge_kw",
    "Eficiency": "efficiency",          # the workbook's spelling
    "Initial (kW)": "initial_kwh",
    "Final (kW)": "final_kwh",
}


class ProfileError(ValueError):
    """The workbook is not shaped the way this loader requires."""


# --------------------------------------------------------------- unit helpers

def kw_to_kwh(kw: float) -> float:
    """Energy delivered over one 900 s period by a constant `kw`.

    The single place this factor appears. `Load`, `PV` and the `BESS` power
    rows are kW; the artifact trades kWh.
    """
    return kw * HOURS_PER_PERIOD


def period_window_sec(period: int) -> tuple[int, int]:
    """Seconds after midnight spanned by `period`, end-exclusive."""
    if not 1 <= period <= PERIODS_PER_DAY:
        raise ProfileError(f"period {period} outside 1..{PERIODS_PER_DAY}")
    return ((period - 1) * PERIOD_SEC, period * PERIOD_SEC)


def period_label(period: int) -> str:
    """Clock time the workbook names the period by: its *end*.

    The EV departure table pairs clock times with period numbers this way --
    period 33 is 08:15, period 76 is 19:00 -- so the label is `n * 900 s`
    while the delivery window starts at `(n - 1) * 900 s`.
    """
    end = period_window_sec(period)[1]
    return f"{end // 3600 % 24:02d}:{end % 3600 // 60:02d}"


def period_to_slot(period: int, day_start: int, day: int = 0) -> int:
    """Delivery slot of `period` on `day`, as the artifact's `time_slot`.

    The artifact's slot is the *start* of the delivery window, so this is
    `(period - 1) * 900` after the day's midnight, not the period's label.
    """
    start = period_window_sec(period)[0]
    return int(day_start) + day * PERIODS_PER_DAY * PERIOD_SEC + start


# ------------------------------------------------------------------ the sheets

def _number(value, where: str) -> float:
    """Strict conversion. openpyxl hands these cells back as strings, so a
    silent `float(None) -> nan` or a stray label would otherwise travel all
    the way into an order book."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip().strip("'").replace(",", ".")
        try:
            return float(text)
        except ValueError:
            pass
    raise ProfileError(f"{where}: expected a number, got {value!r}")


def _label(value) -> str:
    """Row and column labels carry the workbook's own literal quotes."""
    return "" if value is None else str(value).strip().strip("'").strip()


@dataclass(frozen=True)
class BatterySpec:
    """One row set of the `BESS` sheet.

    `Capacity`, `Initial` and `Final` are labelled kW in the workbook but are
    energies: `Initial` is exactly half of `Capacity` for every unit
    (3.2/6.4, 5.85/11.7, 3.3/6.6), which is a state of charge, not a power.
    They are read as kWh and named so. `Charge`/`Discharge` are powers and
    keep their kW name.
    """
    player: int
    model: int
    capacity_kwh: float
    charge_kw: float
    discharge_kw: float
    efficiency: float
    initial_kwh: float
    final_kwh: float

    @property
    def charge_kwh_per_period(self) -> float:
        return kw_to_kwh(self.charge_kw)

    @property
    def discharge_kwh_per_period(self) -> float:
        return kw_to_kwh(self.discharge_kw)


@dataclass(frozen=True)
class PlayerFlags:
    pv: bool
    ess: bool


@dataclass(frozen=True)
class Community:
    """The whole workbook: 250 players, 96 periods, values in kW."""
    players: tuple
    load_kw: dict
    pv_kw: dict
    flags: dict
    bess: dict
    source: Path

    def day_kwh(self, players=None) -> "DayProfile":
        """The measured day for `players`, converted to kWh once."""
        chosen = tuple(self.players if players is None else players)
        unknown = [p for p in chosen if p not in self.load_kw]
        if unknown:
            raise ProfileError(f"unknown players: {unknown}")
        return DayProfile(
            players=chosen,
            load_kwh={p: [kw_to_kwh(v) for v in self.load_kw[p]] for p in chosen},
            pv_kwh={p: [kw_to_kwh(v) for v in self.pv_kw[p]] for p in chosen},
            source=self.source)


@dataclass(frozen=True)
class DayProfile:
    """One measured day, 96 slots, kWh per slot."""
    players: tuple
    load_kwh: dict
    pv_kwh: dict
    source: Path

    n_slots: int = field(default=PERIODS_PER_DAY, init=False)


@dataclass(frozen=True)
class ProfileWeek:
    """`days * 96` slots, kWh per slot, built from one measured day.

    `seed` is the extension seed and belongs in the run manifest: without it
    the data ship but do not regenerate.
    """
    players: tuple
    load_kwh: dict
    pv_kwh: dict
    days: int
    seed: int
    sigma_load: float
    sigma_pv: float
    source: Path

    @property
    def n_slots(self) -> int:
        return self.days * PERIODS_PER_DAY


def _read_matrix(worksheet, sheet_name: str) -> tuple:
    """`Load` / `PV` layout: row 1 is the player id, column 1 the period."""
    rows = list(worksheet.iter_rows(min_row=1, max_row=PERIODS_PER_DAY + 1,
                                    max_col=PLAYERS + 1, values_only=True))
    if len(rows) != PERIODS_PER_DAY + 1:
        raise ProfileError(f"{sheet_name}: expected {PERIODS_PER_DAY + 1} rows, "
                           f"got {len(rows)}")
    ids = [int(_number(v, f"{sheet_name}!header")) for v in rows[0][1:]]
    if len(ids) != PLAYERS:
        raise ProfileError(f"{sheet_name}: expected {PLAYERS} players, "
                           f"got {len(ids)}")
    series = {pid: [0.0] * PERIODS_PER_DAY for pid in ids}
    for row in rows[1:]:
        period = int(_number(row[0], f"{sheet_name}!period"))
        for pid, value in zip(ids, row[1:]):
            series[pid][period - 1] = _number(
                value, f"{sheet_name}!player {pid} period {period}")
    return tuple(ids), series


def _read_flags(worksheet) -> dict:
    """`General Information` puts the player table under a header at row 4."""
    rows = list(worksheet.iter_rows(min_row=4, max_row=4 + PLAYERS,
                                    max_col=3, values_only=True))
    header = [_label(v) for v in rows[0]]
    if header[:3] != ["Player ID", "PV", "ESS"]:
        raise ProfileError(f"General Information: unexpected header {header[:3]}")
    flags = {}
    for row in rows[1:]:
        pid = int(_number(row[0], "General Information!Player ID"))
        flags[pid] = PlayerFlags(
            pv=bool(_number(row[1], f"General Information!PV player {pid}")),
            ess=bool(_number(row[2], f"General Information!ESS player {pid}")))
    return flags


def _read_bess(worksheet) -> dict:
    """`BESS` is a parameter sheet: one column per player, one row per field."""
    rows = list(worksheet.iter_rows(min_row=1, max_row=8, max_col=PLAYERS + 1,
                                    values_only=True))
    ids = [int(_number(v, "BESS!header")) for v in rows[0][1:]]
    fields = {}
    for row in rows[1:]:
        key = _BESS_ROWS.get(_label(row[0]))
        if key is None:
            continue
        fields[key] = [_number(v, f"BESS!{_label(row[0])}") for v in row[1:]]
    missing = set(_BESS_ROWS.values()) - set(fields)
    if missing:
        raise ProfileError(f"BESS: missing rows {sorted(missing)}")
    specs = {}
    for index, pid in enumerate(ids):
        if fields["capacity_kwh"][index] <= 0:
            continue                      # no unit installed: the sheet zeroes it
        specs[pid] = BatterySpec(
            player=pid, model=int(fields["model"][index]),
            capacity_kwh=fields["capacity_kwh"][index],
            charge_kw=fields["charge_kw"][index],
            discharge_kw=fields["discharge_kw"][index],
            efficiency=fields["efficiency"][index],
            initial_kwh=fields["initial_kwh"][index],
            final_kwh=fields["final_kwh"][index])
    return specs


def load_community(path=DEFAULT_WORKBOOK) -> Community:
    """Read `Load`, `PV`, `General Information` and `BESS`.

    `read_only=True`: the twelve EV sheets are out of scope and the workbook
    is 1.4 MB. The twelve are never opened.
    """
    import openpyxl

    path = Path(path)
    if not path.is_file():
        raise ProfileError(
            f"workbook not found at {path}. It is committed under "
            f"evaluation/data/raw/; see evaluation/data/PROVENANCE.md.")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ids, load = _read_matrix(workbook["Load"], "Load")
        pv_ids, pv = _read_matrix(workbook["PV"], "PV")
        if pv_ids != ids:
            raise ProfileError("Load and PV disagree on the player ids")
        flags = _read_flags(workbook["General Information"])
        bess = _read_bess(workbook["BESS"])
    finally:
        workbook.close()

    if set(flags) != set(ids):
        raise ProfileError("General Information and Load disagree on the players")
    return Community(players=ids, load_kw=load, pv_kw=pv, flags=flags,
                     bess=bess, source=path)


# ------------------------------------------------------- the reference community

def select_players(flags: dict, n: int = 100, seed: int = 0) -> list:
    """D-70: `n` of the 250 players, drawn by a recorded seed.

    Reproducible from the seed alone: the draw runs over the *sorted* ids, so
    it does not depend on the order the workbook happened to be read in. The
    chosen ids go into the run manifest.
    """
    population = sorted(flags)
    if n > len(population):
        raise ProfileError(f"cannot draw {n} of {len(population)} players")
    return sorted(random.Random(seed).sample(population, n))


def community_composition(flags: dict, players) -> dict:
    """What the draw actually produced -- for the manifest, not for a claim."""
    chosen = [flags[p] for p in players]
    return {"n_players": len(chosen),
            "n_with_pv": sum(1 for f in chosen if f.pv),
            "n_with_ess": sum(1 for f in chosen if f.ess),
            "n_with_both": sum(1 for f in chosen if f.pv and f.ess),
            "n_with_ess_only": sum(1 for f in chosen if f.ess and not f.pv)}


# ------------------------------------------------------------- battery areas

# Default windows of the declared rule. Periods are labelled by their end, so
# 1..24 is 00:15..06:00 and 73..88 is 18:15..22:00.
NIGHT_PERIODS = tuple(range(1, 25))
EVENING_PERIODS = tuple(range(73, 89))


@dataclass(frozen=True)
class BatteryArea:
    """A storage unit as its own market area, separate from its household."""
    area_uuid: str
    name: str
    player: int
    spec: BatterySpec
    discharge_kwh: tuple           # offered to the community, per period
    grid_charge_kwh: tuple         # drawn from the grid, per period
    rule: str
    energy_type: str = "grey"

    def slot_energy(self, index: int) -> float:
        return self.discharge_kwh[index % len(self.discharge_kwh)]


def _night_charge_evening_discharge(spec: BatterySpec, *, night, evening,
                                    initial_kwh):
    """Charge from the grid overnight, discharge into the community at the
    evening peak. Returns (discharge per period, grid draw per period).

    Efficiency applies on the charge side: the workbook gives one figure per
    unit (0.95), and stored energy is `grid_draw * efficiency`.
    """
    discharge = [0.0] * PERIODS_PER_DAY
    grid = [0.0] * PERIODS_PER_DAY
    stored = min(max(initial_kwh, 0.0), spec.capacity_kwh)
    night, evening = set(night), set(evening)

    for period in range(1, PERIODS_PER_DAY + 1):
        index = period - 1
        if period in night:
            headroom = spec.capacity_kwh - stored
            stored_gain = min(spec.charge_kwh_per_period * spec.efficiency,
                              headroom)
            if stored_gain > 0:
                grid[index] = stored_gain / spec.efficiency
                stored += stored_gain
        elif period in evening:
            out = min(spec.discharge_kwh_per_period, stored)
            if out > 0:
                discharge[index] = out
                stored -= out
    return discharge, grid


_RULES = {"night_charge_evening_discharge": _night_charge_evening_discharge}


def make_battery_areas(bess: dict, players, *,
                       rule: str = "night_charge_evening_discharge",
                       participation: float = 1.0, seed: int = 0,
                       night_periods=NIGHT_PERIODS,
                       evening_periods=EVENING_PERIODS,
                       initial_soc_fraction: float = 0.0) -> list:
    """Storage units as their own market areas, with a declared charging rule.

    **The charging rule is a modelling assumption, not data (D-75), and 5.1
    has to repeat it as one.** The workbook holds no charging history: `BESS`
    is a parameter sheet. What is data here is the per-unit capacity, charge
    and discharge power and efficiency; what is assumed is *when* the unit
    charges and discharges. The rule as declared:

        charge from the grid during `night_periods`, discharge into the
        community during `evening_periods`, respecting the per-unit power
        limit, the capacity, and the efficiency on the charge side.

    Its parameters are left free so the assumption can be varied and the
    sensitivity reported rather than asserted.

    A battery area is separate from its household, which is why it can sell
    at all: under netting (D-61) a battery inside its household's net
    position would simply disappear. Its offers carry `energy_type: "grey"`,
    because the energy was bought from the grid. **No variant of this labels
    PV output as grey.**

    `participation` is the green-share axis: the fraction of battery areas
    that take part in a run, drawn by `seed`.
    """
    if rule not in _RULES:
        raise ProfileError(f"unknown battery rule {rule!r}; "
                           f"known: {sorted(_RULES)}")
    if not 0.0 <= participation <= 1.0:
        raise ProfileError(f"participation must be in [0, 1], got {participation}")

    eligible = sorted(p for p in players if p in bess)
    n_taking_part = int(round(participation * len(eligible)))
    taking_part = sorted(random.Random(seed).sample(eligible, n_taking_part))

    areas = []
    for player in taking_part:
        spec = bess[player]
        discharge, grid = _RULES[rule](
            spec, night=night_periods, evening=evening_periods,
            initial_kwh=initial_soc_fraction * spec.capacity_kwh)
        areas.append(BatteryArea(
            area_uuid=f"battery-{player:03d}", name=f"Battery {player:03d}",
            player=player, spec=spec, discharge_kwh=tuple(discharge),
            grid_charge_kwh=tuple(grid), rule=rule))
    return areas


# ---------------------------------------------------------------- the week

def fit_sigma(daily_totals) -> float:
    """Lognormal sigma from the between-player spread of daily totals.

    The dataset holds one day, so it carries no day-to-day variation of its
    own. This takes the spread it *does* carry -- how much households differ
    from each other -- as the scale of the synthetic day-to-day draw. It is a
    stand-in, and an upper bound: households differ from each other by more
    than one household differs from itself between days. `sigma_scale` on
    `extend_to_week` is there to say so quantitatively.
    """
    logs = [math.log(t) for t in daily_totals if t > 0]
    if len(logs) < 2:
        return 0.0
    return statistics.pstdev(logs)


def extend_to_week(day: DayProfile, *, seed: int, days: int = 7,
                   sigma_scale: float = 1.0) -> ProfileWeek:
    """D-73: `days * 96` slots from one measured day.

    One multiplicative draw per player, per day and per channel (load and
    generation separately), lognormal with the sigma fitted by `fit_sigma`
    and a mean of exactly 1, so the week's expected total is `days` times the
    measured day rather than drifting upward with sigma. Results are clipped
    at zero.

    **The extension carries its own seed and it goes in the manifest.**
    Without it the data ship but do not regenerate. The risk this guards
    against is a week that is one day repeated seven times unchanged: the
    calibration fits theta and steepness against the distribution of the
    supply/demand ratio over slots, and seven identical days carry the same
    information as one.
    """
    if days < 1:
        raise ProfileError(f"days must be >= 1, got {days}")

    sigma_load = fit_sigma([sum(v) for v in day.load_kwh.values()]) * sigma_scale
    sigma_pv = fit_sigma([sum(v) for v in day.pv_kwh.values()]) * sigma_scale
    rng = random.Random(seed)

    load, pv = {}, {}
    for player in day.players:
        load_series, pv_series = [], []
        for _ in range(days):
            # mu = -sigma^2 / 2 makes E[multiplier] = 1.
            m_load = rng.lognormvariate(-0.5 * sigma_load ** 2, sigma_load)
            m_pv = rng.lognormvariate(-0.5 * sigma_pv ** 2, sigma_pv)
            load_series += [max(0.0, v * m_load) for v in day.load_kwh[player]]
            pv_series += [max(0.0, v * m_pv) for v in day.pv_kwh[player]]
        load[player] = load_series
        pv[player] = pv_series

    return ProfileWeek(players=tuple(day.players), load_kwh=load, pv_kwh=pv,
                       days=days, seed=seed, sigma_load=sigma_load,
                       sigma_pv=sigma_pv, source=day.source)


def slot_times(week: ProfileWeek, day_start: int) -> list:
    """Delivery slots of a week, contiguous and 900 s apart."""
    return [day_start + i * PERIOD_SEC for i in range(week.n_slots)]
