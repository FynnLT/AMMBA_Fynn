"""DR2 for the full artifact: preference allocation and origin settlement in
the replay (A2, D-90).

Table 4.3 carries a second DR2 prediction: in the full artifact, preference
and origin adjustments create no additional profitable quantity deviations
(numerical, no DSIC claim). `dr2.py` replays the proportional reference cell
only; this module replays the block-1 cells that run the preference-aware
allocation (mutual pairs first, pro-rata residual) and the green bonus /
grey levy, and compares every participant-slot with the proportional replay
of **the same order book**.

**Population.** Every participant-slot of the cells in `CELLS`, all seeds.
Block 1 has no deviator, so the recorded report is the truthful type (A = D
= `requested_kwh`). Nominations and energy types are held fixed: only one
participant's quantity report changes (Section 4.2.1). The eight cases, the
grid and the thresholds are those of `dr2`.

**Two engines, one convention.** The *full* engine runs the cell's own
`PreferenceConfig`; the *baseline* is the same engine with
`PreferenceConfig(enabled=False)` on the same book: pro rata, every rate the
rounded price. Both price at the unrounded sigmoid, settle at the six-decimal
rates the artifact writes into the trades, and charge the penalties as
`replay.replay` does, so a difference between them is caused by the
adjustments and not by a rounding convention.

**The artifact's functions are the reference.** `replay_full` rebuilds the
slot's order book (F1) and calls `apply_preference_allocation` and
`apply_energy_type_multipliers` of the clearing node through `stack.CLR`, and
the execution node's penalty functions through `replay`. `evaluate` is the
same arithmetic in closed form on numpy arrays, for the sweep: the pair set
does not depend on quantities, so one deviating report moves only its own
pair quantity, the residual and its own side's remainder sum. Check R5 holds
`evaluate` to `replay_full`, and R4 holds `replay_full` to the recorded runs.

Settlement follows `run_clearing` down to its rounding: the rates are
computed from the green / grey totals of the trades' `selected_energy`, which
`trade_builder` rounds to six decimals per trade, and from the clearing price
rounded to six decimals; the rates themselves are rounded to six decimals
with Python's `round` (see `round6`).

Block 1 was cleared, not executed, so it never applied gamma and eta_rel. The
penalties are charged at the reference setting gamma = 1.1, eta_rel = 0.10,
read from `b2_noise_off` where that cell was discovered.
"""
import contextlib
import functools
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import dr2
import replay as R
import stack  # noqa: E402  (on the path once `replay` is imported)

_PREFS = stack.CLR["preferences"]
_CONFIG = stack.CLR["config"]

apply_preference_allocation = _PREFS.apply_preference_allocation
apply_energy_type_multipliers = _PREFS.apply_energy_type_multipliers
mutual_pairs = _PREFS._mutual_pairs
PreferenceConfig = _CONFIG.PreferenceConfig

#: The clearing node's own tolerances and rounding, read rather than
#: restated.
PREF_EPSILON = _PREFS.EPSILON
DECIMALS = _PREFS._ROUND
GREEN, GREY, MIXED = _PREFS.GREEN, _PREFS.GREY, _PREFS.MIXED
PREFERENCES_FIRST, PRO_RATA_FIRST = _CONFIG.PREFERENCE_ORDERS

#: F0: the six block-1 cells, in the order of the task.
CELLS = ("prefs_first_d050", "pro_rata_first_d050", "multipliers_on",
         "multipliers_on_additive", "mult_overlap_multiplicative",
         "mult_overlap_additive")
#: Seeds a cell is expected to have when its manifests do not say.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
REFERENCE_CELL = "b2_noise_off"
#: The reference penalty setting (Chapter 5): what `b2_noise_off` ran at.
REFERENCE_GAMMA = 1.1
REFERENCE_ETA_RELATIVE = 0.10
#: Any non-empty name: `apply_energy_type_multipliers` tells a seller trade
#: from a buyer trade by which side the pool is on.
POOL_ID = "AMM_POOL_dr2_full"

R4_TOL = 1e-5
R5_POINTS = 5_000
R5_ALLOCATION_TOL = 1e-9
R5_RATE_TOL = 1e-6
R5_UTILITY_TOL = 1e-6
C1_TOL = 1e-9
C2_ALLOCATION_TOL = 1e-9
C2_UTILITY_TOL = 1e-5
N_EXAMPLES = 20

#: Rows per sweep chunk, and elements per (rows x grid x sellers) block of
#: the green / grey totals.
CHUNK = 2048
TENSOR_ELEMENTS = 1 << 21

class PopulationError(RuntimeError):
    """A run does not satisfy the preconditions of the analysis (F0)."""


# ---------------------------------------------------------------- rounding

def round6(values):
    """`round(x, 6)` on an array, with Python's rounding exactly.

    `np.round` scales, rounds half to even and scales back, which parts from
    Python's correctly rounded `round` on values next to a half-way point --
    and the rates sit there often: p * 0.9 of a six-decimal price has a
    seventh decimal of 5 in one slot in ten. Values within 1e-4 of a
    half-way point (in units of the sixth decimal) are rounded by Python
    itself; everywhere else the two agree.
    """
    x = np.asarray(values, dtype=float)
    out = np.array(np.round(x, DECIMALS))
    scaled = x * 10.0 ** DECIMALS
    with np.errstate(invalid="ignore"):
        near = np.abs(scaled - np.floor(scaled) - 0.5) < 1e-4
    if near.any():
        where = np.flatnonzero(near)
        out.flat[where] = [round(float(v), DECIMALS) for v in x.flat[where]]
    return out if out.ndim else float(out)


def _near_half(values) -> np.ndarray:
    """Values whose seventh decimal sits on a rounding boundary (within
    1e-9 of a half-way point at six decimals)."""
    scaled = np.asarray(values, dtype=float) * 10.0 ** DECIMALS
    with np.errstate(invalid="ignore"):
        return np.abs(scaled - np.floor(scaled) - 0.5) < 1e-3


# ------------------------------------------------------- F1: the order book

@dataclass(frozen=True)
class Book:
    """One slot's order book as `run_clearing` passed it to
    `apply_preference_allocation`. The order dicts are never handed to the
    artifact themselves, only copies of them."""
    slot: int
    bids: tuple
    offers: tuple

    def orders(self) -> tuple:
        """Offers, then bids: the member order of the population."""
        return self.offers + self.bids


def build_book(slot, areas, sides, energy_types, energies, partners) -> Book:
    """F1: the book of one slot from its area rows.

    Sellers carry `attributes.energy_type`, buyers no attributes (their rows
    say `mixed`, the trade-level label). A nomination is passed on only if
    the named area posts on the opposite side in this slot: the `partner`
    column holds the named partner whether or not it could be posted, and
    passing an unpostable one would only make `parse_preferred_partner`
    warn.
    """
    sellers = {a for a, s in zip(areas, sides) if s == R.SELLER}
    buyers = {a for a, s in zip(areas, sides) if s == R.BUYER}
    offers, bids = [], []
    for area, side, energy_type, energy, partner in zip(
            areas, sides, energy_types, energies, partners):
        area = str(area)
        order = {"area_uuid": area, "order_id": f"{int(slot)}-{area}",
                 "energy": float(energy)}
        counterparties = buyers if side == R.SELLER else sellers
        if side == R.SELLER:
            order["attributes"] = {"energy_type": str(energy_type)}
        if partner and partner in counterparties:
            order["requirements"] = {"preferred_partner": str(partner)}
        (offers if side == R.SELLER else bids).append(order)
    return Book(int(slot), tuple(bids), tuple(offers))


def _copy(orders) -> list:
    """A deep copy of an order list. The artifact writes into the order
    dicts (`allocated_energy`, `preference_matched`, ...); the nested dicts
    are copied too so nothing it holds is shared with the book."""
    return [{key: dict(value) if isinstance(value, dict) else value
             for key, value in order.items()} for order in orders]


@contextlib.contextmanager
def _quiet():
    """The clearing node logs every pair and every slot without grey volume
    at INFO / WARNING; over a sweep that is noise. Errors stay visible."""
    logger = _PREFS.logger
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


# ------------------------------------------------- F2: the scalar reference

def _trade(order: dict, seller: bool) -> dict:
    """The fields of a trade `apply_energy_type_multipliers` reads.
    `selected_energy` is rounded as `trade_builder` rounds it."""
    area = order["area_uuid"]
    return {"buyer": POOL_ID if seller else area,
            "seller": area if seller else POOL_ID,
            "parameters": {
                "selected_energy": round(order["allocated_energy"], DECIMALS),
                "energy_type": order["energy_type"] if seller else MIXED}}


def clear_book(bids, offers, cfg, clearing_price) -> tuple:
    """Steps 5 to 6.3 of `run_clearing` on copies of the book: allocation,
    trades and the multiplier block at the (rounded) clearing price.

    Returns (PreferenceResult, trades, MultiplierResult); the trades are in
    `build_all_trades` order, buyers first.
    """
    bids, offers = _copy(bids), _copy(offers)
    supply = sum(o["energy"] for o in offers)
    demand = sum(b["energy"] for b in bids)
    with _quiet():
        allocation = apply_preference_allocation(
            bids, offers, traded_quantity=min(supply, demand),
            total_supply_kwh=supply, total_demand_kwh=demand, cfg=cfg)
        trades = ([_trade(b, False) for b in allocation.bids]
                  + [_trade(o, True) for o in allocation.offers])
        multipliers = apply_energy_type_multipliers(
            trades, clearing_price=clearing_price, pool_id=POOL_ID, cfg=cfg)
    return allocation, trades, multipliers


@dataclass(frozen=True)
class FullOutcome:
    supply: float
    demand: float
    ratio: float
    #: Unrounded: what the penalties are computed at, as in `replay.replay`.
    price: float
    #: `round(price, 6)`: what `run_clearing` settles and multiplies at.
    price_rounded: float
    traded: float
    round_type: str
    allocation: float
    rate: float
    matched: bool
    shortfall_ct: float
    externality_ct: float
    gross_utility: float
    #: Every order's allocation, offers then bids (`Book.orders` order).
    allocations: tuple = ()
    n_pairs: int = 0
    pair_kwh: float = 0.0
    multipliers: object = None

    @property
    def penalty_ct(self) -> float:
        return self.shortfall_ct + self.externality_ct

    @property
    def utility(self) -> float:
        return self.gross_utility - self.penalty_ct


def replay_full(book: Book, cfg, band: R.Band, gamma: float,
                eta_relative: float, participant: str, new_report: float,
                truth: float) -> FullOutcome:
    """The slot with `participant`'s report replaced by `new_report`,
    everyone else and every nomination fixed. `truth` is A (seller) or D
    (buyer). Scalar, and built from the artifact's functions throughout.
    """
    def swap(orders):
        return tuple(dict(o, energy=float(new_report))
                     if o["area_uuid"] == participant else o for o in orders)

    seller = any(o["area_uuid"] == participant for o in book.offers)
    if not seller and not any(b["area_uuid"] == participant
                              for b in book.bids):
        raise KeyError(f"{participant} has no order in slot {book.slot}")
    bids, offers = swap(book.bids), swap(book.offers)
    supply = sum(o["energy"] for o in offers)
    demand = sum(b["energy"] for b in bids)
    ratio = supply / demand
    price = band.price(ratio)
    price_rounded = round(price, DECIMALS)
    traded = min(supply, demand)
    round_type = R.determine_round_type(supply, demand)

    result, trades, multipliers = clear_book(bids, offers, cfg,
                                             price_rounded)
    orders = tuple(result.offers) + tuple(result.bids)
    own = next(o for o in orders if o["area_uuid"] == participant)
    allocation = own["allocated_energy"]
    trade = next(t for t in trades
                 if t["seller" if seller else "buyer"] == participant)
    rate = trade["parameters"]["final_energy_rate"]

    # The penalties exactly as `replay.replay` charges them.
    shortfall_ct = externality_ct = 0.0
    if seller:
        eta = eta_relative * allocation
        shortfall_ct = R.seller_shortfall_penalty(
            allocation, min(allocation, truth), band.k_upper, gamma,
            eta)["penalty_ct"]
        if round_type == R.SUPPLY_LIMITED:
            ext = R.seller_externality_penalty(
                allocation, truth, eta, supply, demand, traded, price,
                band.as_sigmoid())
            if ext is not None:
                externality_ct = ext["penalty_ct"]
        gross = float(R.seller_gross_utility(rate, allocation, band.k_lower))
    else:
        if round_type == R.DEMAND_LIMITED:
            ext = R.buyer_externality_penalty(
                new_report, truth, supply, demand, traded, price,
                band.as_sigmoid())
            if ext is not None:
                externality_ct = ext["penalty_ct"]
        gross = float(R.buyer_gross_utility(rate, allocation, truth,
                                            band.k_upper))
    return FullOutcome(
        supply=supply, demand=demand, ratio=ratio, price=price,
        price_rounded=price_rounded, traded=traded, round_type=round_type,
        allocation=allocation, rate=rate,
        matched=bool(own["preference_matched"]),
        shortfall_ct=shortfall_ct, externality_ct=externality_ct,
        gross_utility=gross,
        allocations=tuple(o["allocated_energy"] for o in orders),
        n_pairs=len(result.pairs),
        pair_kwh=sum(p.energy_kwh for p in result.pairs),
        multipliers=multipliers)


# ----------------------------------------------------------- the population

@dataclass
class SlotInput:
    """One slot of one run, as the population is assembled from it."""
    book: Book
    cfg: object
    band: R.Band
    run_id: str = ""
    seed: int = -1
    #: area -> the named partner (the `partner` column, posted or not).
    named: dict = field(default_factory=dict)
    #: area -> (allocated_kwh, preference_matched) as recorded.
    recorded_rows: dict = field(default_factory=dict)
    #: slot CSV column -> recorded value.
    recorded: dict = field(default_factory=dict)


#: Slot CSV columns R4 compares, and the extra ones it reports.
R4_SLOT_COLUMNS = ("n_pairs", "pair_kwh", "green_alloc_kwh", "grey_alloc_kwh",
                   "green_final_ct", "grey_final_ct", "buyer_final_ct",
                   "clearing_price_ct", "total_supply_kwh",
                   "total_demand_kwh")


def preference_config(run):
    """The run's `PreferenceConfig`, after the F0 preconditions."""
    prefs = run.config.get("preferences")
    if not isinstance(prefs, dict):
        raise PopulationError(f"{run.run_id}: manifest config carries no "
                              f"preferences block")
    if prefs.get("sides") != "seller":
        raise PopulationError(
            f"{run.run_id}: preferences.sides is {prefs.get('sides')!r}, not "
            f"'seller'; buyers would not settle at the uniform price")
    if prefs.get("order") not in (PREFERENCES_FIRST, PRO_RATA_FIRST):
        raise PopulationError(
            f"{run.run_id}: preferences.order is {prefs.get('order')!r}")
    return PreferenceConfig(**prefs)


def book_digest(data) -> str:
    """sha256 over the sorted (slot, area, side, requested_kwh) rows, as
    written: two runs replay the same order books iff the digests agree."""
    areas = data.areas
    rows = sorted(zip(areas.raw("slot"), areas.raw("area_uuid"),
                      areas.raw("side"), areas.raw("requested_kwh")))
    return hashlib.sha256("\n".join(",".join(row) for row in rows)
                          .encode("utf-8")).hexdigest()


@dataclass
class CellRun:
    """What this stage reads of one run, taken while the run is loaded.

    The stage runs after DR2, and holding thirty runs' CSVs until then
    would more than double what the analysis keeps in memory; the columns
    below are a fraction of that.
    """
    run: object
    slot_index: np.ndarray
    areas: dict
    #: R4_SLOT_COLUMNS -> per-slot array (NaN where the CSV lacks one).
    slots: dict
    book_digest: str


def extract(data) -> CellRun:
    a, s = data.areas, data.slots
    return CellRun(
        run=data.run, slot_index=data.slot_index,
        areas={"slot": a.ints("slot"), "area_uuid": a.strings("area_uuid"),
               "side": a.strings("side"),
               "energy_type": a.strings("energy_type"),
               "requested_kwh": a.floats("requested_kwh"),
               "allocated_kwh": a.floats("allocated_kwh"),
               "preference_matched": a.bools("preference_matched"),
               "partner": a.strings("partner")},
        slots={name: (s.floats(name) if s.has(name) else np.full(s.n, np.nan))
               for name in R4_SLOT_COLUMNS},
        book_digest=book_digest(data))


def slot_inputs(cell_run: CellRun) -> list:
    """Every cleared slot of one run as a `SlotInput`, preconditions
    asserted."""
    run, areas = cell_run.run, cell_run.areas
    cfg = preference_config(run)
    band = R.Band.from_dict(run.sigmoid)
    slot = areas["slot"]
    area = areas["area_uuid"]
    side = areas["side"]
    energy_type = areas["energy_type"]
    requested = areas["requested_kwh"]
    allocated = areas["allocated_kwh"]
    matched = areas["preference_matched"]
    partner = areas["partner"]
    recorded = cell_run.slots
    groups = defaultdict(list)
    for i, s in enumerate(cell_run.slot_index):
        groups[int(s)].append(i)
    out = []
    for s, rows in sorted(groups.items()):
        rows = np.array(rows)
        names = area[rows].tolist()
        if len(set(names)) != len(names):
            dup = sorted({a for a in names if names.count(a) > 1})
            raise PopulationError(
                f"{run.run_id}: slot {int(slot[rows[0]])} has more than one "
                f"order for {dup[:5]}")
        book = build_book(slot[rows[0]], area[rows], side[rows],
                          energy_type[rows], requested[rows], partner[rows])
        out.append(SlotInput(
            book=book, cfg=cfg, band=band, run_id=run.run_id, seed=run.seed,
            named={area[i]: partner[i] for i in rows if partner[i]},
            recorded_rows={area[i]: (float(allocated[i]), bool(matched[i]))
                           for i in rows},
            recorded={name: float(values[s])
                      for name, values in recorded.items()}))
    return out


def assemble(slots, gamma: float, eta_relative: float) -> dict:
    """Arrays over every participant-slot of `slots`, members in
    `Book.orders` order, plus per-slot matrices of the sellers and of all
    members for the closed form.

    Everything the closed form holds fixed is computed here once, at the
    truthful reports and with the artifact's own summation order: the
    mutual pairs (`_mutual_pairs`), their quantities, the pair total and the
    remainder sum of each side.
    """
    rows = defaultdict(list)
    per_slot = defaultdict(list)
    books, cfgs, bands = [], [], []
    for s, entry in enumerate(slots):
        book, cfg = entry.book, entry.cfg
        bids, offers = list(book.bids), list(book.offers)
        with _quiet():
            pairs = mutual_pairs(bids, offers) if cfg.enabled else []
        prio = bool(cfg.enabled and cfg.order == PREFERENCES_FIRST)
        n_off = len(offers)
        members = offers + bids
        pq = [0.0] * len(members)
        partner = [-1] * len(members)
        quantities = []
        for b, o in pairs:
            q = min(bids[b]["energy"], offers[o]["energy"]) if prio else 0.0
            quantities.append(q)
            pq[o] = pq[n_off + b] = q
            partner[o], partner[n_off + b] = n_off + b, o
        supply = sum(o["energy"] for o in offers)
        demand = sum(b["energy"] for b in bids)
        sum_pq = sum(quantities)
        if sum_pq > min(supply, demand) + PREF_EPSILON:
            raise PopulationError(f"{entry.run_id}: slot {book.slot}: pairs "
                                  f"oversubscribe the market, which the "
                                  f"closed form does not cover")
        # `_distribute_residual`: remainder = energy - (0.0 + pair quantity).
        rest_s = sum(o["energy"] - (0.0 + pq[k]) for k, o in enumerate(offers))
        rest_b = sum(b["energy"] - (0.0 + pq[n_off + k])
                     for k, b in enumerate(bids))
        active = bool(cfg.enabled and cfg.multipliers_enabled)
        base = len(rows["claim"])
        books.append(book)
        cfgs.append(cfg)
        bands.append(entry.band)
        per_slot["start"].append(base)
        per_slot["n_offers"].append(n_off)
        per_slot["n_members"].append(len(members))
        for name in R4_SLOT_COLUMNS:
            per_slot[f"rec_{name}"].append(entry.recorded.get(name, np.nan))
        for k, order in enumerate(members):
            seller = k < n_off
            area = order["area_uuid"]
            etype = ((order.get("attributes") or {}).get("energy_type", GREEN)
                     if seller else MIXED)
            m = partner[k]
            rec_alloc, rec_matched = entry.recorded_rows.get(area,
                                                             (np.nan, None))
            rows["run"].append(entry.run_id)
            rows["seed"].append(entry.seed)
            rows["slot"].append(book.slot)
            rows["area"].append(area)
            rows["is_seller"].append(seller)
            rows["energy_type"].append(etype)
            rows["claim"].append(order["energy"])
            rows["matched"].append(m >= 0)
            rows["named_partner"].append(entry.named.get(area, ""))
            rows["partner_row"].append(base + m if m >= 0 else -1)
            rows["partner_pos"].append(m)
            rows["partner_claim"].append(members[m]["energy"] if m >= 0
                                         else 0.0)
            rows["pq"].append(pq[k])
            rows["prio"].append(prio)
            rows["slot_id"].append(s)
            rows["pos"].append(k if seller else -1)
            rows["pos_all"].append(k)
            rows["supply"].append(supply)
            rows["demand"].append(demand)
            rows["sum_pq"].append(sum_pq)
            rows["rest_s"].append(rest_s)
            rows["rest_b"].append(rest_b)
            rows["k_upper"].append(entry.band.k_upper)
            rows["k_lower"].append(entry.band.k_lower)
            rows["theta"].append(entry.band.theta)
            rows["steepness"].append(entry.band.steepness)
            rows["mult_active"].append(active)
            rows["additive"].append(cfg.mode == "additive")
            rows["green_multiplier"].append(float(cfg.green_multiplier))
            rows["grey_levy"].append(float(cfg.grey_levy))
            rows["levy_cap"].append(float(cfg.levy_cap))
            rows["allocated"].append(rec_alloc)
            rows["matched_recorded"].append(rec_matched)

    pop = {key: np.array(values) for key, values in rows.items()}
    for key in ("run", "area", "energy_type", "named_partner",
                "matched_recorded"):
        pop[key] = np.array(rows[key], dtype=object)
    for key in ("seed", "slot", "slot_id", "pos", "pos_all", "partner_row",
                "partner_pos"):
        pop[key] = np.array(rows[key], dtype=np.int64)
    for key in ("is_seller", "matched", "prio", "mult_active", "additive"):
        pop[key] = np.array(rows[key], dtype=bool)
    n = len(pop["claim"])
    pop["n"] = n
    pop["gamma"] = np.full(n, float(gamma))
    pop["eta_relative"] = np.full(n, float(eta_relative))
    pop["is_green"] = pop["is_seller"] & (pop["energy_type"] != GREY)
    pop["is_grey"] = pop["is_seller"] & (pop["energy_type"] == GREY)
    pop["is_battery"] = np.array([dr2.participant_type(a) == dr2.BATTERY
                                  for a in pop["area"]], dtype=bool)
    pop["regime"] = R.regime_vec(pop["supply"], pop["demand"])
    short = ((pop["is_seller"] & (pop["regime"] == R.SL))
             | (~pop["is_seller"] & (pop["regime"] == R.DL)))
    pop["role"] = np.where(short, "short", "long")
    pop["books"], pop["cfgs"], pop["bands"] = books, cfgs, bands
    for key, values in per_slot.items():
        pop[f"slot_{key}"] = np.array(values)
    for kind in (GREEN, GREY):
        has = np.zeros(len(books), dtype=bool)
        np.logical_or.at(has, pop["slot_id"], pop[f"is_{kind}"])
        pop[f"slot_has_{kind}"] = has

    # Per-slot matrices, padded: the sellers (S_*, for the green / grey
    # totals) and all members (M_*, for `member_allocations`).
    n_slots = len(books)
    n_off = pop["slot_n_offers"] if n_slots else np.zeros(0, int)
    n_mem = pop["slot_n_members"] if n_slots else np.zeros(0, int)
    rem = pop["claim"] - (0.0 + pop["pq"])
    columns = {"S": {"pq": pop["pq"], "rem": rem, "e": pop["claim"],
                     "green": pop["is_green"], "grey": pop["is_grey"]},
               "M": {"pq": pop["pq"], "e": pop["claim"],
                     "seller": pop["is_seller"],
                     "valid": np.ones(n, dtype=bool)}}
    for prefix, counts in (("S", n_off), ("M", n_mem)):
        width = int(counts.max()) if n_slots else 0
        for key, values in columns[prefix].items():
            matrix = np.zeros((n_slots, width), dtype=values.dtype)
            for s in range(n_slots):
                start = pop["slot_start"][s]
                matrix[s, :counts[s]] = values[start:start + counts[s]]
            pop[f"{prefix}_{key}"] = matrix
    return pop


def build_population(cell_runs, gamma: float, eta_relative: float) -> dict:
    return assemble([entry for cell_run in cell_runs
                     for entry in slot_inputs(cell_run)],
                    gamma, eta_relative)


# --------------------------------------------------- F3: the closed form

def _col(pop, idx, key):
    return pop[key][idx][:, None]


def rates_vec(green_alloc, grey_alloc, price, *, active, additive,
              green_multiplier, grey_levy, levy_cap) -> tuple:
    """`_multiplicative_rates` / `_additive_rates` and the rules of
    `apply_energy_type_multipliers` around them, on arrays, in the
    artifact's operation order. Returns (green, grey, green_raw, grey_raw):
    rounded to six decimals, and before that rounding."""
    g, y, p = green_alloc, grey_alloc, price
    eps = PREF_EPSILON
    with np.errstate(divide="ignore", invalid="ignore"):
        levy_eff = np.minimum(grey_levy, levy_cap)
        levy_collected = y * p * levy_eff
        bonus_requested = g * p * green_multiplier
        scale = np.where(bonus_requested > eps,
                         np.minimum(1.0, levy_collected / bonus_requested),
                         1.0)
        mult_green = p * (1 + green_multiplier * scale)
        mult_grey = p * (1 - levy_eff)

        bonus = p * green_multiplier
        levy = bonus * g / y
        cap = p * levy_cap
        capped = levy > cap
        add_green = p + np.where(capped, bonus * (cap / levy), bonus)
        add_grey = p - np.where(capped, cap, levy)
        none = (y <= eps) | (g <= eps)
        add_green = np.where(none, p, add_green)
        add_grey = np.where(none, p, add_grey)

    green = np.where(additive, add_green, mult_green)
    grey = np.where(additive, add_grey, mult_grey)
    green = np.where(active, green, p)
    grey = np.where(active, grey, p)
    # A side with no volume settles at the untouched price.
    green = np.where(g <= eps, p, green)
    grey = np.where(y <= eps, p, grey)
    return round6(green), round6(grey), green, grey


def _others(pq, rem, energy, rest, residual):
    """`_distribute_residual` + `_clamp_allocations` for orders whose own
    report and pair are unchanged: pq + rem / rest * residual, in [0, e]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(rest > PREF_EPSILON, rem / rest * residual, 0.0)
    return np.minimum(np.maximum(pq + share, 0.0), energy)


def _blocks(n_rows, width, g):
    step = max(1, TENSOR_ELEMENTS // max(1, width * g))
    for start in range(0, n_rows, step):
        yield slice(start, start + step)


def seller_totals(pop, idx, allocation, rest_s, residual) -> tuple:
    """(G', Y'): the green and grey totals of the seller trades' rounded
    `selected_energy`, in trade order, with seller rows `idx` deviating.

    Every other seller keeps its pair quantity; its allocation moves only
    with the residual and the seller-side remainder sum, so it is
    `_others(...)` at the deviated (residual, rest_s). Summed with
    `cumsum`, which adds left to right like Python's `sum` over the trades.
    """
    n, g = allocation.shape
    green = np.empty((n, g))
    grey = np.empty((n, g))
    slot_id = pop["slot_id"][idx]
    pos = pop["pos"][idx]
    width_all = pop["slot_n_offers"][slot_id]
    for block in _blocks(n, int(width_all.max()) if n else 1, g):
        sl = slot_id[block]
        if not len(sl):
            continue
        width = int(width_all[block].max())
        pq = pop["S_pq"][sl, :width][:, None, :]
        rem = pop["S_rem"][sl, :width][:, None, :]
        energy = pop["S_e"][sl, :width][:, None, :]
        alloc = _others(pq, rem, energy, rest_s[block][..., None],
                        residual[block][..., None])
        m = len(sl)
        alloc[np.arange(m)[:, None], np.arange(g)[None, :],
              pos[block][:, None]] = allocation[block]
        alloc = round6(alloc)
        is_green = pop["S_green"][sl, :width][:, None, :]
        is_grey = pop["S_grey"][sl, :width][:, None, :]
        green[block] = np.cumsum(np.where(is_green, alloc, 0.0),
                                 axis=-1)[..., -1]
        grey[block] = np.cumsum(np.where(is_grey, alloc, 0.0),
                                axis=-1)[..., -1]
    return green, grey


def _totals(pop, rows, allocation, rest_s, residual, price, params) -> tuple:
    """(G', Y') as far as the deviating sellers' own rates depend on them.

    Where a slot has sellers of one energy type only, the other total is
    zero and the deviator's own rounded trade is a lower bound of its own
    type's total. That bound decides the rate exactly wherever it clears
    the thresholds `apply_energy_type_multipliers` tests (Y' > eps for the
    grey levy; G' > eps and G' p m > eps, which makes the unfunded bonus
    zero, for the green rate; the additive formulation settles at p with
    either total at zero). Only the remaining rows -- every row of a slot
    with both types, and the rare one whose own trade rounds to zero -- get
    the exact totals from `seller_totals`.
    """
    eps = PREF_EPSILON
    slot = pop["slot_id"][rows]
    mixed = pop["slot_has_green"][slot] & pop["slot_has_grey"][slot]
    green_dev = pop["is_green"][rows][:, None]
    own = round6(allocation)
    green_alloc = np.where(green_dev, own, 0.0)
    grey_alloc = np.where(green_dev, 0.0, own)
    additive = params["additive"]
    decisive = np.where(
        green_dev,
        additive | ((own > eps)
                    & (own * price * params["green_multiplier"] > eps)),
        additive | (own > eps))
    exact = np.where(mixed | ~decisive.all(axis=1))[0]
    if len(exact):
        green_alloc[exact], grey_alloc[exact] = seller_totals(
            pop, rows[exact], allocation[exact], rest_s[exact],
            residual[exact])
    return green_alloc, grey_alloc


def _state(pop, idx, rel, direction, full, truth):
    """The deviated aggregates, pair quantity and own allocation, 2-D."""
    idx = np.asarray(idx)
    rel = np.asarray(rel, dtype=float).reshape(len(idx), -1)
    c = functools.partial(_col, pop, idx)
    claim = c("claim")
    new = claim * (1.0 + direction * rel)
    truth = (claim if truth is None
             else np.asarray(truth, dtype=float).reshape(len(idx), -1))
    is_seller = c("is_seller")
    delta = new - claim
    supply = np.where(is_seller, c("supply") + delta, c("supply"))
    demand = np.where(is_seller, c("demand"), c("demand") + delta)
    traded = np.minimum(supply, demand)
    if full:
        paired = c("prio") & c("matched")
        pq0 = c("pq")
        pq_new = np.where(paired, np.minimum(new, c("partner_claim")), 0.0)
        sum_pq = c("sum_pq") - pq0 + pq_new
        rest_s0, rest_b0 = c("rest_s"), c("rest_b")
    else:
        paired = np.zeros_like(is_seller)
        pq0 = np.zeros_like(claim)
        pq_new = np.zeros_like(new)
        sum_pq = 0.0
        rest_s0, rest_b0 = c("supply"), c("demand")
    residual = traded - sum_pq
    rem_new = new - pq_new
    rest_own = np.where(is_seller, rest_s0, rest_b0) - (claim - pq0) + rem_new
    # The partner's side: its pair quantity follows the deviator's.
    rem_partner0 = c("partner_claim") - pq0
    rem_partner = c("partner_claim") - pq_new
    rest_other = (np.where(is_seller, rest_b0, rest_s0)
                  - np.where(paired, rem_partner0, 0.0)
                  + np.where(paired, rem_partner, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(rest_own > PREF_EPSILON,
                         rem_new / rest_own * residual, 0.0)
    allocation = np.minimum(np.maximum(pq_new + share, 0.0), new)
    return {"idx": idx, "rel": rel, "claim": claim, "new": new,
            "truth": truth, "is_seller": is_seller, "supply": supply,
            "demand": demand, "traded": traded, "paired": paired,
            "pq_new": pq_new, "residual": residual, "rest_own": rest_own,
            "rest_other": rest_other, "rem_partner": rem_partner,
            "allocation": allocation}


def evaluate(pop, idx, rel, direction, band=None, *, full=True, truth=None):
    """The replay of rows `idx` at relative deviation `rel` (1-D per row, or
    2-D rows x grid), in the full engine or (`full=False`) its proportional
    baseline. Same signature as `dr2.evaluate`, so `dr2.thresholds` can
    bisect on it; there is no band sensitivity here.

    Penalties are rounded where the artifact rounds them; `penalty_raw` is
    the unrounded sum, which the threshold search reads.
    """
    if band is not None:
        raise ValueError("the full engine has no band sensitivity")
    one_d = np.ndim(rel) == 1
    st = _state(pop, idx, rel, direction, full, truth)
    idx = st["idx"]
    c = functools.partial(_col, pop, idx)
    k_upper, k_lower = c("k_upper"), c("k_lower")
    theta, steepness = c("theta"), c("steepness")
    supply, demand, traded = st["supply"], st["demand"], st["traded"]
    is_seller, allocation = st["is_seller"], st["allocation"]
    truth, new = st["truth"], st["new"]
    ratio = supply / demand
    price = R.sigmoid_vec(ratio, k_upper, k_lower, theta, steepness)
    price_rounded = round6(price)
    regime = R.regime_vec(supply, demand)

    rate = np.array(np.broadcast_to(price_rounded, allocation.shape))
    rate_raw = rate.copy()
    if full:
        need = np.where(pop["is_seller"][idx] & pop["mult_active"][idx])[0]
        if len(need):
            rows = idx[need]
            p = price_rounded[need]
            params = {"additive": _col(pop, rows, "additive"),
                      "green_multiplier": _col(pop, rows, "green_multiplier"),
                      "grey_levy": _col(pop, rows, "grey_levy"),
                      "levy_cap": _col(pop, rows, "levy_cap")}
            green_alloc, grey_alloc = _totals(
                pop, rows, allocation[need], st["rest_own"][need],
                st["residual"][need], p, params)
            green, grey, green_raw, grey_raw = rates_vec(
                green_alloc, grey_alloc, p, active=True, **params)
            is_green = _col(pop, rows, "is_green")
            rate[need] = np.where(is_green, green, grey)
            rate_raw[need] = np.where(is_green, green_raw, grey_raw)

    # The penalties as `replay_vec` charges them, at this allocation.
    gamma, eta_rel = c("gamma"), c("eta_relative")
    eta = eta_rel * allocation
    shortfall = np.maximum(0.0, allocation - np.minimum(allocation, truth)
                           - eta)
    shortfall = np.where(is_seller, shortfall, 0.0)
    shortfall_raw = gamma * k_upper * shortfall
    shortfall_ct = round6(shortfall_raw)

    withheld = np.maximum(0.0, truth - allocation - eta)
    seller_ext = is_seller & (regime == R.SL) & (withheld > R.EPSILON)
    p_cf_seller = R.sigmoid_vec((supply + withheld) / demand, k_upper,
                                k_lower, theta, steepness)
    seller_ext_raw = np.where(
        seller_ext, np.maximum(0.0, (price - p_cf_seller) * traded), 0.0)
    underreported = np.maximum(0.0, truth - new)
    buyer_ext = (~is_seller) & (regime == R.DL) & (underreported > R.EPSILON)
    p_cf_buyer = R.sigmoid_vec(supply / (demand + underreported), k_upper,
                               k_lower, theta, steepness)
    buyer_ext_raw = np.where(
        buyer_ext, np.maximum(0.0, (p_cf_buyer - price) * traded), 0.0)
    externality_raw = seller_ext_raw + buyer_ext_raw
    externality_ct = round6(externality_raw)

    gross = np.where(is_seller,
                     R.seller_gross_utility(rate, allocation, k_lower),
                     R.buyer_gross_utility(rate, allocation, truth, k_upper))
    penalty = shortfall_ct + externality_ct
    out = {"supply": supply, "demand": demand, "ratio": ratio,
           "price": price, "price_rounded": price_rounded, "traded": traded,
           "regime": regime, "allocation": allocation, "rate": rate,
           "rate_raw": rate_raw, "shortfall_ct": shortfall_ct,
           "externality_ct": externality_ct, "penalty_ct": penalty,
           "penalty_raw": shortfall_raw + externality_raw,
           "gross_utility": gross, "utility": gross - penalty}
    if one_d:
        out = {key: value[:, 0] for key, value in out.items()}
    return out


def member_allocations(pop, idx, rel, direction, *, full=True):
    """Every order's allocation in the slot of each row of `idx`, that row
    deviating: rows x grid x members (`Book.orders` order, NaN padding).

    F3 for any k: pq_k' + rem_k' R' / (the remainder sum of k's side), in
    [0, e_k']. Only the deviator and its partner change their own pair
    quantity and remainder.
    """
    st = _state(pop, idx, rel, direction, full, None)
    idx = st["idx"]
    n, g = st["allocation"].shape
    slot_id = pop["slot_id"][idx]
    width = int(pop["slot_n_members"][slot_id].max()) if n else 0
    pq = pop["M_pq"][slot_id, :width][:, None, :]
    if not full:
        pq = np.zeros_like(pq)
    rem = pop["M_e"][slot_id, :width][:, None, :] - (0.0 + pq)
    energy = pop["M_e"][slot_id, :width][:, None, :]
    seller = pop["M_seller"][slot_id, :width][:, None, :]
    own_seller = st["is_seller"][..., None]
    rest = np.where(seller == own_seller, st["rest_own"][..., None],
                    st["rest_other"][..., None])
    alloc = _others(pq, rem, energy, rest, st["residual"][..., None])
    rows, grid = np.arange(n)[:, None], np.arange(g)[None, :]
    partner = pop["partner_pos"][idx]
    has = st["paired"][:, 0] & (partner >= 0)
    if has.any():
        r = np.where(has)[0]
        rest_p = st["rest_other"][r]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(rest_p > PREF_EPSILON,
                             st["rem_partner"][r] / rest_p
                             * st["residual"][r], 0.0)
        partner_claim = pop["partner_claim"][idx[r]][:, None]
        alloc[r[:, None], grid, partner[r][:, None]] = np.minimum(
            np.maximum(st["pq_new"][r] + share, 0.0), partner_claim)
    alloc[rows, grid, pop["pos_all"][idx][:, None]] = st["allocation"]
    valid = pop["M_valid"][slot_id, :width][:, None, :]
    return np.where(valid, alloc, np.nan)


# ---------------------------------------------------------------- the sweep

def sweep(pop, idx, direction, chunk=CHUNK) -> dict:
    """Net gain over the grid in both engines, rows x grid, and the C1 / C2
    maxima on the way."""
    n, g = len(idx), len(dr2.GRID)
    zero = np.zeros(n)
    engines = {}
    for name, full in (("full", True), ("base", False)):
        at0 = evaluate(pop, idx, zero, direction, full=full)
        engines[name] = {"net": np.empty((n, g)), "u0": at0["utility"],
                         "regime0": at0["regime"],
                         "alloc0": at0["allocation"]}
    # The threshold search runs on the full engine only.
    engines["full"]["penalty_raw"] = np.empty((n, g))
    engines["full"]["regime"] = np.empty((n, g), dtype=np.int8)
    c1 = c2_alloc = c2_utility = 0.0
    for start in range(0, n, chunk):
        rows = slice(start, start + chunk)
        sub = idx[rows]
        rel = np.broadcast_to(dr2.GRID, (len(sub), g))
        results = {}
        for name, full in (("full", True), ("base", False)):
            res = evaluate(pop, sub, rel, direction, full=full)
            acc = engines[name]
            acc["net"][rows] = res["utility"] - acc["u0"][rows, None]
            if full:
                acc["penalty_raw"][rows] = res["penalty_raw"]
                acc["regime"][rows] = res["regime"]
            results[name] = res
        c1 = max(c1, float(np.abs(engines["full"]["net"][rows]
                                  - engines["base"]["net"][rows]).max()))
        claim = pop["claim"][sub][:, None]
        vec = R.replay_vec(
            is_seller=_col(pop, sub, "is_seller"),
            supply=_col(pop, sub, "supply"), demand=_col(pop, sub, "demand"),
            own_report=claim, new_report=claim * (1.0 + direction * rel),
            truth=claim, k_upper=_col(pop, sub, "k_upper"),
            k_lower=_col(pop, sub, "k_lower"), theta=_col(pop, sub, "theta"),
            steepness=_col(pop, sub, "steepness"),
            gamma=_col(pop, sub, "gamma"),
            eta_relative=_col(pop, sub, "eta_relative"))
        c2_alloc = max(c2_alloc, float(np.abs(
            vec["allocation"] - results["base"]["allocation"]).max()))
        c2_utility = max(c2_utility, float(np.abs(
            vec["utility"] - results["base"]["utility"]).max()))
    return {**engines, "c1_max": c1, "c2_allocation_max": c2_alloc,
            "c2_utility_max": c2_utility}


# ------------------------------------------------------------- the tables

def group_masks(pop, idx, side):
    yield "all", np.ones(len(idx), bool)
    yield "matched", pop["matched"][idx]
    yield "unmatched", ~pop["matched"][idx]
    if side == R.SELLER:
        yield GREEN, pop["is_green"][idx]
        yield GREY, pop["is_grey"][idx]
    yield dr2.HOUSEHOLD, ~pop["is_battery"][idx]
    yield dr2.BATTERY, pop["is_battery"][idx]


def _relative(values, u0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / np.where(u0 > 0, u0, np.nan)[:, None]


def analyse_cell(cell, pop, log=print) -> dict:
    """dr2_full_cases, dr2_full_curves, the example candidates and the C1 /
    C2 maxima of one cell."""
    cases, curves, candidates = [], [], []
    c1 = c2_alloc = c2_utility = 0.0
    shares_base = {}
    full_eval = functools.partial(evaluate, full=True)
    for side, role, direction, idx in dr2.case_rows(pop):
        name = dr2.case_name(side, role, direction)
        if not len(idx):
            continue
        res = sweep(pop, idx, direction)
        c1 = max(c1, res["c1_max"])
        c2_alloc = max(c2_alloc, res["c2_allocation_max"])
        c2_utility = max(c2_utility, res["c2_utility_max"])
        full, base = res["full"], res["base"]
        thr = dr2.thresholds(pop, idx, direction, full,
                             evaluate_fn=full_eval)
        net_full = thr["net_max"]
        kbase = base["net"].argmax(axis=1)
        net_base = base["net"][np.arange(len(idx)), kbase]
        star_base = dr2.GRID[kbase].astype(float)
        diff = net_full - net_base
        pos_full = net_full > dr2.NET_TOL
        pos_base = net_base > dr2.NET_TOL
        short_seller = side == R.SELLER and role == "short"
        violated = (dr2.proposition2_violations(full, thr) if short_seller
                    else None)
        rel_full = net_full / np.where(full["u0"] > 0, full["u0"], np.nan)
        shares_base[name] = float(pos_base.mean())
        for group, mask in group_masks(pop, idx, side):
            if not mask.any():
                continue
            both = pos_full[mask] & pos_base[mask]
            d = diff[mask]
            cases.append({
                "cell": cell, "case": name,
                "table_4_2_row": dr2.CASES.index((side, role, direction)) + 1,
                "group": group, "n": int(mask.sum()),
                "share_profitable_full": float(pos_full[mask].mean()),
                "share_profitable_base": float(pos_base[mask].mean()),
                "n_additional": int((pos_full[mask] & ~pos_base[mask]).sum()),
                "n_removed": int((~pos_full[mask] & pos_base[mask]).sum()),
                "n_gain_increased": int((both & (d > dr2.NET_TOL)).sum()),
                "n_gain_decreased": int((both & (d < -dr2.NET_TOL)).sum()),
                "net_max_full_ct_median": dr2._q(net_full[mask], 0.5),
                "net_max_full_ct_p90": dr2._q(net_full[mask], 0.9),
                "net_max_full_ct_max": dr2._max(net_full[mask]),
                "net_max_base_ct_median": dr2._q(net_base[mask], 0.5),
                "net_max_base_ct_p90": dr2._q(net_base[mask], 0.9),
                "net_max_base_ct_max": dr2._max(net_base[mask]),
                "diff_ct_median": dr2._q(d, 0.5),
                "diff_ct_p90": dr2._q(d, 0.9),
                "diff_ct_max": dr2._max(d),
                "diff_ct_min": float(d.min()),
                "net_max_full_rel_median": dr2._q(rel_full[mask], 0.5),
                "delta_star_full_median": float(np.median(
                    thr["delta_star"][mask])),
                "delta_zero_full_median": float(np.median(
                    thr["delta_zero"][mask])),
                "share_switch_any": float(thr["switch_any"][mask].mean()),
                "prop2_n_violations_full": (int(violated[mask].sum())
                                            if short_seller else None),
            })

        rel_f = _relative(full["net"], full["u0"])
        rel_b = _relative(base["net"], base["u0"])
        diff_grid = full["net"] - base["net"]
        diff_rel = rel_f - rel_b
        for k, delta in enumerate(dr2.GRID):
            curves.append({
                "cell": cell, "case": name, "delta_rel": float(delta),
                "n": len(idx),
                "net_full_ct_median": dr2._q(full["net"][:, k], 0.5),
                "net_full_ct_p90": dr2._q(full["net"][:, k], 0.9),
                "net_base_ct_median": dr2._q(base["net"][:, k], 0.5),
                "net_base_ct_p90": dr2._q(base["net"][:, k], 0.9),
                "diff_ct_median": dr2._q(diff_grid[:, k], 0.5),
                "diff_ct_p90": dr2._q(diff_grid[:, k], 0.9),
                "net_full_rel_median": dr2._q(rel_f[:, k], 0.5),
                "net_full_rel_p90": dr2._q(rel_f[:, k], 0.9),
                "net_base_rel_median": dr2._q(rel_b[:, k], 0.5),
                "net_base_rel_p90": dr2._q(rel_b[:, k], 0.9),
                "diff_rel_median": dr2._q(diff_rel[:, k], 0.5),
                "diff_rel_p90": dr2._q(diff_rel[:, k], 0.9),
            })

        top = np.where(diff > dr2.NET_TOL)[0]
        for i in top:
            candidates.append((float(diff[i]), int(idx[i]), name,
                               float(net_full[i]), float(net_base[i]),
                               float(thr["delta_star"][i]),
                               float(star_base[i]),
                               float(full["alloc0"][i]),
                               float(base["alloc0"][i])))
        log(f"    {name}: n = {len(idx)}, additional "
            f"{int((pos_full & ~pos_base).sum())}, removed "
            f"{int((~pos_full & pos_base).sum())}")
    return {"cases": cases, "curves": curves,
            "examples": examples(cell, pop, candidates),
            "c1_max": c1, "c2_allocation_max": c2_alloc,
            "c2_utility_max": c2_utility, "shares_base": shares_base}


def examples(cell, pop, candidates) -> list:
    """The participant-slots with the largest positive diff_ct."""
    candidates = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))
    out = []
    for rank, (diff, i, case, full, base, star_full, star_base, alloc_full,
               alloc_base) in enumerate(candidates[:N_EXAMPLES], 1):
        out.append({
            "cell": cell, "rank": rank, "run_id": pop["run"][i],
            "seed": int(pop["seed"][i]), "slot": int(pop["slot"][i]),
            "area": pop["area"][i], "case": case,
            "energy_type": pop["energy_type"][i],
            "claim_kwh": float(pop["claim"][i]),
            "partner": pop["named_partner"][i],
            "matched": bool(pop["matched"][i]),
            "pair_kwh": float(pop["pq"][i]),
            "alloc0_full_kwh": alloc_full, "alloc0_base_kwh": alloc_base,
            "net_max_full_ct": full, "net_max_base_ct": base,
            "diff_ct": diff, "delta_star_full": star_full,
            "delta_star_base": star_base})
    return out


# -------------------------------------------------------------- the checks

def r4(pop) -> dict:
    """The full engine reproduces the recorded block-1 outcome at delta = 0:
    `replay_full`'s clearing of every reconstructed book against every
    recorded row and slot, and `evaluate` at delta = 0 against the rows."""
    worst = defaultdict(float)
    mismatch_matched = 0
    n_slots = len(pop["books"])
    for s in range(n_slots):
        book, cfg, band = pop["books"][s], pop["cfgs"][s], pop["bands"][s]
        supply = sum(o["energy"] for o in book.offers)
        demand = sum(b["energy"] for b in book.bids)
        price = band.price(supply / demand)
        result, _trades, mult = clear_book(book.bids, book.offers, cfg,
                                           round(price, DECIMALS))
        start = pop["slot_start"][s]
        for k, order in enumerate(tuple(result.offers) + tuple(result.bids)):
            row = start + k
            worst["allocated_kwh"] = max(worst["allocated_kwh"], abs(
                order["allocated_energy"] - pop["allocated"][row]))
            mismatch_matched += int(bool(order["preference_matched"])
                                    != pop["matched_recorded"][row])
        got = {"n_pairs": len(result.pairs),
               "pair_kwh": sum(p.energy_kwh for p in result.pairs),
               "green_alloc_kwh": mult.green_alloc_kwh,
               "grey_alloc_kwh": mult.grey_alloc_kwh,
               "green_final_ct": mult.green_final_ct_per_kwh,
               "grey_final_ct": mult.grey_final_ct_per_kwh,
               "buyer_final_ct": mult.buyer_final_ct_per_kwh,
               "clearing_price_ct": round(price, DECIMALS),
               "total_supply_kwh": supply, "total_demand_kwh": demand}
        for name, value in got.items():
            worst[name] = max(worst[name], abs(
                value - pop[f"slot_rec_{name}"][s]))

    # The closed form at delta = 0, every row.
    idx = np.arange(pop["n"])
    vec = evaluate(pop, idx, np.zeros(pop["n"]), +1, full=True)
    worst["vectorised_allocated_kwh"] = float(np.abs(
        vec["allocation"] - pop["allocated"]).max()) if pop["n"] else 0.0
    slot = pop["slot_id"]
    recorded_rate = np.where(
        pop["is_seller"],
        np.where(pop["is_grey"], pop["slot_rec_grey_final_ct"][slot],
                 pop["slot_rec_green_final_ct"][slot]),
        pop["slot_rec_buyer_final_ct"][slot]) if pop["n"] else np.zeros(0)
    worst["vectorised_rate_ct"] = float(np.abs(
        vec["rate"] - recorded_rate).max()) if pop["n"] else 0.0
    # A NaN in a maximum means a recorded value is missing: not a pass.
    worst = {name: float(value) for name, value in worst.items()}
    passed = (n_slots > 0 and mismatch_matched == 0
              and all(np.isfinite(v) and v <= R4_TOL
                      for v in worst.values()))
    return {"passed": bool(passed), "tolerance": R4_TOL,
            "n_slots": n_slots, "n_rows": int(pop["n"]),
            "n_preference_matched_mismatch": mismatch_matched,
            "max_deviation_by_quantity": worst}


def r5(pop, seed=20260925, points=None) -> dict:
    """The closed form against `replay_full`, on random (participant-slot,
    delta) points in both directions, in both engines: one draw at truth =
    claim, one with the truth off the claim by up to 30 % so the shortfall
    and both externality paths are exercised. `points` per draw, default
    `R5_POINTS`."""
    points = R5_POINTS if points is None else points
    rng = np.random.default_rng(seed)
    base_cfg = PreferenceConfig(enabled=False)
    worst = defaultdict(float)
    counts = defaultdict(int)
    for hard in (False, True):
        rows = rng.integers(0, pop["n"], size=points)
        rel = rng.uniform(-0.5, 0.5, size=points)
        claim = pop["claim"][rows]
        truth = (claim * (1.0 + rng.uniform(-0.3, 0.3, size=points))
                 if hard else claim)
        new = claim * (1.0 + rel)
        for full in (True, False):
            engine = "full" if full else "base"
            vec = evaluate(pop, rows, rel, +1, full=full, truth=truth)
            members = member_allocations(pop, rows, rel, +1, full=full)[:, 0]
            counts[f"{engine}_n_rate_on_boundary"] += int(
                _near_half(vec["rate_raw"]).sum())
            for i, row in enumerate(rows):
                s = pop["slot_id"][row]
                # Python floats throughout: `round` on a numpy float
                # rounds the numpy way.
                ref = replay_full(pop["books"][s],
                                  pop["cfgs"][s] if full else base_cfg,
                                  pop["bands"][s], float(pop["gamma"][row]),
                                  float(pop["eta_relative"][row]),
                                  pop["area"][row], float(new[i]),
                                  float(truth[i]))
                n_mem = pop["slot_n_members"][s]
                deviations = {
                    "allocation": abs(ref.allocation - vec["allocation"][i]),
                    "slot_allocations": float(np.abs(
                        np.array(ref.allocations) - members[i, :n_mem]).max()),
                    "rate": abs(ref.rate - vec["rate"][i]),
                    "utility": abs(ref.utility - vec["utility"][i]),
                    "price": abs(ref.price - vec["price"][i])}
                for name, value in deviations.items():
                    key = f"{engine}_{name}"
                    worst[key] = max(worst[key], float(value))
                counts[f"{engine}_n_regime_mismatch"] += int(
                    R.REGIME_CODE[ref.round_type] != vec["regime"][i])
                counts[f"{engine}_n_rate_mismatch"] += int(
                    deviations["rate"] > 1e-12)
                counts[f"{engine}_n_points"] += 1
                if full:
                    counts["hits_pair_priority"] += int(
                        pop["prio"][row] and pop["matched"][row])
                    counts["hits_rate_adjusted"] += int(
                        abs(ref.rate - ref.price_rounded) > 1e-12)
                counts["hits_shortfall"] += int(ref.shortfall_ct > 0)
                counts["hits_seller_externality"] += int(
                    ref.externality_ct > 0 and pop["is_seller"][row])
                counts["hits_buyer_externality"] += int(
                    ref.externality_ct > 0 and not pop["is_seller"][row])
    tolerances = {"allocation": R5_ALLOCATION_TOL,
                  "slot_allocations": R5_ALLOCATION_TOL,
                  "rate": R5_RATE_TOL, "utility": R5_UTILITY_TOL,
                  "price": R5_ALLOCATION_TOL}
    passed = pop["n"] > 0 and all(
        worst[f"{engine}_{name}"] <= tol
        for engine in ("full", "base") for name, tol in tolerances.items()
    ) and counts["full_n_regime_mismatch"] == 0 \
        and counts["base_n_regime_mismatch"] == 0
    return {"passed": bool(passed), "tolerances": tolerances,
            "n_points_per_draw": points, "max_deviation": dict(worst),
            "counts": dict(counts)}


def c3(cell_runs, reference, main_cases, shares_base) -> dict:
    """Which cells replay the order books of b2_noise_off, and where they
    do, whether the baseline reproduces the main DR2 share per case."""
    ref_books = {data.run.seed: book_digest(data) for data in reference}
    main = {r["case"]: r["share_net_max_pos"] for r in main_cases
            if r["type"] == "all"}
    out, passed = {}, True
    for cell, runs in sorted(cell_runs.items()):
        identical = {int(c.run.seed): c.book_digest == ref_books.get(
            c.run.seed) for c in runs}
        entry = {"identical_by_seed": identical,
                 "identical": bool(identical) and all(identical.values())}
        if entry["identical"]:
            differences = {
                case: {"base": share, "dr2_cases": main.get(case)}
                for case, share in shares_base.get(cell, {}).items()
                if main.get(case) is None or abs(share - main[case]) > 1e-12}
            entry["share_differences"] = differences
            passed &= not differences
        out[cell] = entry
    return {"passed": bool(passed), "reference_seeds": sorted(ref_books),
            "cells": out}


# ------------------------------------------- the run set and the setting

def expected_seeds(runs) -> tuple:
    seeds = set()
    for run in runs:
        seeds.update(int(s) for s in (run.config.get("cell_seeds") or ()))
    return tuple(sorted(seeds)) or DEFAULT_SEEDS


def missing_runs(runs, cells) -> list:
    """`cell` or `cell seed N` for every requested cell and seed that no
    discovered run carries."""
    by_cell = defaultdict(list)
    for run in runs:
        if run.cell in cells:
            by_cell[run.cell].append(run)
    missing = []
    for cell in cells:
        found = by_cell.get(cell)
        if not found:
            missing.append(f"{cell} (no run)")
            continue
        have = {r.seed for r in found}
        missing += [f"{cell} seed {s}" for s in expected_seeds(found)
                    if s not in have]
    return missing


def penalty_parameters(reference) -> dict:
    """gamma and eta_rel for the replay: the ones `b2_noise_off` applied,
    asserted to be the reference setting; the constants without it."""
    if not reference:
        return {"gamma": REFERENCE_GAMMA,
                "eta_relative": REFERENCE_ETA_RELATIVE,
                "source": f"constants ({REFERENCE_CELL} not discovered)"}
    gammas, etas, sources = set(), set(), set()
    for data in reference:
        for column, into in (("gamma_eff", gammas),
                             ("eta_relative_eff", etas)):
            values = data.slots.floats(column)
            into.update(values[~np.isnan(values)].tolist())
        sources.add(str(data.param_source))
    if gammas != {REFERENCE_GAMMA} or etas != {REFERENCE_ETA_RELATIVE}:
        raise PopulationError(
            f"{REFERENCE_CELL} applied gamma {sorted(gammas)} and eta_rel "
            f"{sorted(etas)}, not the reference {REFERENCE_GAMMA} / "
            f"{REFERENCE_ETA_RELATIVE}")
    return {"gamma": REFERENCE_GAMMA, "eta_relative": REFERENCE_ETA_RELATIVE,
            "source": f"{REFERENCE_CELL} gamma_eff / eta_relative_eff "
                      f"(param_source {', '.join(sorted(sources))}), "
                      f"{len(reference)} runs"}


# ------------------------------------------------------------- the stage

TABLES = ("dr2_full_cases", "dr2_full_curves", "dr2_full_examples")


@dataclass
class Stage:
    checks: dict
    findings: dict
    #: None when a hard check stopped the stage.
    tables: dict | None
    params: dict
    wall_sec: float


def _is_control(pop) -> bool:
    """pro_rata_first with the multipliers off: pairs detected, no priority,
    no origin adjustment -- the full engine must equal its baseline."""
    return all(cfg.order == PRO_RATA_FIRST
               and not (cfg.enabled and cfg.multipliers_enabled)
               for cfg in pop["cfgs"])


def run(cell_runs, reference, main_cases, cells, log=print,
        err=print) -> Stage:
    """The whole stage: population, R4 / R5, the sweep, C1 - C3.

    One cell at a time, so only one population is held. Hard: the
    population (missing runs, preconditions), R4, R5 and C1; the first
    failure stops the stage and no `dr2_full_*` table is returned, the
    checks up to it are.
    """
    started = time.perf_counter()
    checks, findings, params = {}, {"cells": list(cells)}, {}
    r4s, r5s, c1s, c2s, shares = {}, {}, {}, {}, {}

    def elapsed():
        return round(time.perf_counter() - started, 1)

    def collect():
        checks["R4"] = {"passed": bool(r4s) and all(
            r["passed"] for r in r4s.values()), "tolerance": R4_TOL,
            "by_cell": r4s}
        checks["R5"] = {"passed": bool(r5s) and all(
            r["passed"] for r in r5s.values()), "by_cell": r5s}
        checks["C1"] = {"passed": all(v["max_abs_net_diff_ct"] <= C1_TOL
                                      for v in c1s.values()),
                        "tolerance": C1_TOL, "control_cells": c1s}
        if not c1s:
            checks["C1"].update(skipped=True, reason=(
                "no pro_rata_first cell with the multipliers off was "
                "analysed"))
        checks["C2"] = {
            "passed": all(v["max_allocation_deviation_kwh"]
                          <= C2_ALLOCATION_TOL
                          and v["max_utility_deviation_ct"] <= C2_UTILITY_TOL
                          for v in c2s.values()),
            "tolerance_allocation_kwh": C2_ALLOCATION_TOL,
            "tolerance_utility_ct": C2_UTILITY_TOL, "by_cell": c2s}

    def stop(name, reason):
        if name == "dr2_full_population":
            checks[name] = {"passed": False, "reason": reason}
        else:
            collect()
        err(f"DR2 full artifact stopped: {reason}")
        err("no dr2_full table written")
        return Stage(checks, findings, None, params, elapsed())

    missing = missing_runs([c.run for c in cell_runs], cells)
    if missing:
        return stop("dr2_full_population",
                    f"missing {', '.join(missing)}; pass --full-cells to "
                    f"restrict the set")
    by_cell = defaultdict(list)
    for cell_run in cell_runs:
        by_cell[cell_run.run.cell].append(cell_run)
    try:
        params.update(penalty_parameters(reference))
        # Every run's preconditions before any cell is computed.
        for cell_run in cell_runs:
            preference_config(cell_run.run)
    except PopulationError as exc:
        return stop("dr2_full_population", str(exc))
    findings["penalty_parameters"] = dict(params)
    checks["dr2_full_population"] = {
        "passed": True, "n_runs": {c: len(by_cell[c]) for c in cells}}
    log(f"DR2 full artifact: gamma = {params['gamma']}, eta_rel = "
        f"{params['eta_relative']} ({params['source']})")

    tables = {name: [] for name in TABLES}
    for cell in cells:
        try:
            pop = build_population(by_cell[cell], params["gamma"],
                                   params["eta_relative"])
        except PopulationError as exc:
            checks["dr2_full_population"] = {"passed": False,
                                             "reason": str(exc)}
            return stop("dr2_full_population", str(exc))
        checks["dr2_full_population"].setdefault("n_rows", {})[cell] = int(
            pop["n"])
        r4s[cell] = r4(pop)
        r5s[cell] = r5(pop)
        log(f"  {cell}: {pop['n']} participant-slots; R4 "
            f"{'passed' if r4s[cell]['passed'] else 'FAILED'}, R5 "
            f"{'passed' if r5s[cell]['passed'] else 'FAILED'} ({elapsed()}s)")
        failed = [name for name, result in (("R4", r4s[cell]),
                                            ("R5", r5s[cell]))
                  if not result["passed"]]
        if failed:
            return stop(failed[0], f"{' and '.join(failed)} failed in "
                                   f"{cell}")
        result = analyse_cell(cell, pop, log=log)
        tables["dr2_full_cases"] += result["cases"]
        tables["dr2_full_curves"] += result["curves"]
        tables["dr2_full_examples"] += result["examples"]
        shares[cell] = result["shares_base"]
        c2s[cell] = {
            "max_allocation_deviation_kwh": result["c2_allocation_max"],
            "max_utility_deviation_ct": result["c2_utility_max"]}
        if _is_control(pop):
            c1s[cell] = {"max_abs_net_diff_ct": result["c1_max"]}
            if result["c1_max"] > C1_TOL:
                return stop("C1", f"C1 failed in {cell}: max |net_full - "
                                  f"net_base| = {result['c1_max']:.3e} ct")
        log(f"  {cell}: done ({elapsed()}s)")
        del pop

    collect()
    checks["C3"] = c3(by_cell, reference, main_cases, shares)
    for key, column in (("n_additional_by_cell", "n_additional"),
                        ("n_removed_by_cell", "n_removed")):
        findings[key] = {
            cell: sum(r[column] for r in tables["dr2_full_cases"]
                      if r["cell"] == cell and r["group"] == "all")
            for cell in cells}
    findings["wall_sec"] = elapsed()
    for cell in cells:
        log(f"  {cell}: n_additional = "
            f"{findings['n_additional_by_cell'][cell]}, n_removed = "
            f"{findings['n_removed_by_cell'][cell]}")
    return Stage(checks, findings, tables, params, elapsed())
