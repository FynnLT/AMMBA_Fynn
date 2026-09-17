"""Per-slot aggregates for one run: a tidy CSV and the round-type census.

**These are projections of the clearing response, not computations.** Every
field below is read off what the artifact answered. Recomputing the ratio from
the scenario would make the calibration measure the generator instead of the
artifact -- the two agree only as long as netting, rounding and the order
filter all agree, and the moment they do not, the fitted theta describes the
harness.

The first block-1 campaign projected eight quantities, all of them invariant
under preference order and under the multipliers, so four mechanism cells
wrote byte-identical CSVs. Everything the mechanism changes is in the clearing
response -- `preferences.mutual_pairs`, `preferences.multipliers` and the
per-area `allocations` rows -- and is copied out here (Code To-Do 3.11), never
recomputed beside it.
"""
import csv
from pathlib import Path

SUPPLY_LIMITED = "SUPPLY_LIMITED"
DEMAND_LIMITED = "DEMAND_LIMITED"
BALANCED = "BALANCED"

#: The eight quantities of the first pass. Unchanged, and in their original
#: order, so a CSV written by this module stays readable by the calibration
#: (`calibration/fit.py` reads `ratio` and `clearing_price_ct` by name).
BASE_FIELDS = ("slot", "total_supply_kwh", "total_demand_kwh", "ratio",
               "clearing_price_ct", "traded_kwh", "n_producers",
               "n_consumers")

#: What the preference mechanism does to a slot, read off
#: `clearing["preferences"]`. `pref_order` is the response's `order`, renamed
#: because "order" in a CSV of a market run reads as an order book entry.
PREFERENCE_FIELDS = ("pref_enabled", "pref_order", "n_pairs", "pair_kwh",
                     "pairs_rationed")

#: The multiplier block, `clearing["preferences"]["multipliers"]`. Prefixed
#: only where the response's own name is ambiguous next to the preference
#: block above (`enabled`, `mode`, `sides`, `scale`).
MULTIPLIER_FIELDS = ("mult_enabled", "mult_mode", "mult_sides",
                     "green_alloc_kwh", "grey_alloc_kwh", "levy_collected_ct",
                     "bonus_requested_ct", "bonus_paid_ct", "bonus_scale",
                     "green_final_ct", "grey_final_ct", "buyer_final_ct",
                     "buyers_pay_ct", "sellers_receive_ct", "pool_surplus_ct")

#: From the scenario, carried through the record by `runner.run_slot`: a pair
#: that could not be posted leaves no trace in any response, because from the
#: artifact's side it never existed.
SCENARIO_FIELDS = ("pairs_posted", "pairs_unpostable")

#: The single derived field, and the reason it is allowed: it is the guard for
#: the green-multiplier decision (D-79), and it has to be readable per slot
#: rather than against the week's average mix.
DERIVED_FIELDS = ("crossover_mult",)

#: The execution response's own `redistribution` block plus the round-level
#: penalty counts (Code To-Do 3.4). All None on a slot that did not execute.
#:
#: `budget_balance_ct` in particular is **copied, never recomputed**: it is
#: the node's own statement of whether the rule's Sum (p - p_cf) * q_i closed,
#: and the +-0 test in 5.3 reads this column. A harness that recomputed it
#: would be testing its own arithmetic against itself.
EXECUTION_FIELDS = ("round_type_exec", "total_penalties_ct", "penalty_pool_ct",
                    "compensated_ct", "budget_balance_ct", "n_excluded",
                    "n_harmed", "harmed_side", "n_deviators", "n_shortfall",
                    "withheld_kwh", "underreported_kwh",
                    "shortfall_penalty_ct", "externality_penalty_ct")

FIELDS = (BASE_FIELDS + PREFERENCE_FIELDS + MULTIPLIER_FIELDS
          + SCENARIO_FIELDS + DERIVED_FIELDS + EXECUTION_FIELDS)

#: One row per allocation row per slot. `named` and `partner` come from the
#: run's held preference draw, everything else off `clearing["allocations"]`;
#: the last seven are the execution node's per-participant result row and are
#: empty on a run that did not execute.
AREA_FIELDS = ("slot", "area_uuid", "side", "energy_type", "requested_kwh",
               "allocated_kwh", "fill_rate", "preference_matched", "named",
               "partner", "actual_kwh", "deliverable_kwh",
               "deliverable_source", "shortfall_penalty_ct",
               "externality_penalty_ct", "total_penalty_ct",
               "compensation_ct")

#: Per-area execution columns, copied straight off an `execution["results"]`
#: row under the same names it uses.
_AREA_EXECUTION_KEYS = ("actual_kwh", "deliverable_kwh", "deliverable_source",
                        "shortfall_penalty_ct", "externality_penalty_ct",
                        "total_penalty_ct")


def _execution_row(execution: dict | None) -> dict:
    """The round-level execution projection, or None in every field.

    A slot that did not execute is not a slot with zero penalties: the first
    is a gap and the second is a measurement, and 5.3 must not sum them.
    """
    if not execution or execution.get("status") != "executed":
        return dict.fromkeys(EXECUTION_FIELDS)

    redistribution = execution.get("redistribution") or {}
    results = execution.get("results") or []
    sellers = [r for r in results if r.get("role") == "seller"]
    buyers = [r for r in results if r.get("role") == "buyer"]

    def total(rows, key):
        return round(sum(float(r.get(key) or 0.0) for r in rows), 6)

    return {
        # Named apart from the clearing response's `round_type`: the two are
        # computed from the same totals by two services, and a column that
        # silently merged them would hide a disagreement rather than show it.
        "round_type_exec": execution.get("round_type"),
        "total_penalties_ct": execution.get("total_penalties_ct"),
        "penalty_pool_ct": redistribution.get("penalty_pool_ct"),
        "compensated_ct": redistribution.get("compensated_ct"),
        "budget_balance_ct": redistribution.get("budget_balance_ct"),
        "n_excluded": len(redistribution.get("excluded_deviators") or []),
        "n_harmed": len(redistribution.get("rows") or []),
        "harmed_side": redistribution.get("harmed_side"),
        "n_deviators": sum(1 for r in results
                           if float(r.get("externality_penalty_ct") or 0.0) > 0),
        "n_shortfall": sum(1 for r in results
                           if float(r.get("shortfall_penalty_ct") or 0.0) > 0),
        "withheld_kwh": total(sellers, "externality_kwh"),
        "underreported_kwh": total(buyers, "externality_kwh"),
        "shortfall_penalty_ct": total(results, "shortfall_penalty_ct"),
        "externality_penalty_ct": total(results, "externality_penalty_ct"),
    }


def _crossover_mult(multipliers: dict, clearing_price) -> float | None:
    """`green_multiplier` at which this slot's levy stops funding the bonus.

    The bonus the pool is asked for is `green_alloc * p * green_multiplier`
    and the levy it has to be funded from is `levy_collected_ct`, so the two
    meet at `levy_collected_ct / (p * green_alloc_kwh)`. Above it the levy no
    longer covers the bonus, `bonus_scale` drops below 1 and the two
    formulations start to coincide -- trap 1 of the multiplier note, and the
    reason `G = 0.02` rather than 0.03 (D-79).

    Equal to the run plan's `grey_levy * grey_alloc / green_alloc`, but read
    off the response rather than off the configuration: `levy_collected_ct`
    already carries the *effective* levy, so a slot where `levy_cap` binds
    reports the crossover that actually applied there.

    None where the question has no answer in this slot: no green volume to
    pay a bonus on, no grey volume to fund it from, or no levy configured.
    """
    green = multipliers.get("green_alloc_kwh") or 0.0
    grey = multipliers.get("grey_alloc_kwh") or 0.0
    levy = multipliers.get("levy_collected_ct") or 0.0
    price = clearing_price or 0.0
    if not (green and grey and price and levy):
        return None
    return round(levy / (price * green), 6)


def slot_row(record: dict) -> dict:
    """One record of `runner.run_sequence` as one CSV row.

    A slot that did not clear (one side empty) keeps its supply and demand --
    the clearing node reports both on the no-trade path -- and carries None
    for price, ratio and traded quantity. Dropping those rows would hide the
    empty half of the week from the census.

    The preference and multiplier fields are None wherever the response
    carries no `preferences` block at all, which is exactly the no-trade path
    (`clearing._expire_one_sided_market` returns without one). A cleared slot
    with preferences disabled *does* carry the block, reporting
    `enabled: False` and the round's green/grey split -- that is a measured
    zero and is written as one, not as a gap.
    """
    clearing = record["clearing"]
    allocations = clearing.get("allocations") or {}
    preferences = clearing.get("preferences") or {}
    multipliers = preferences.get("multipliers") or {}
    pairs = preferences.get("mutual_pairs")
    scen = record.get("scenario") or {}
    price = clearing.get("clearing_price_ct_per_kwh")

    def pref(key):
        return preferences.get(key) if preferences else None

    def mult(key):
        return multipliers.get(key) if multipliers else None

    return {
        "slot": record["time_slot"],
        "total_supply_kwh": clearing.get("total_supply_kwh"),
        "total_demand_kwh": clearing.get("total_demand_kwh"),
        "ratio": clearing.get("ratio"),
        "clearing_price_ct": price,
        "traded_kwh": clearing.get("traded_quantity_kwh"),
        "n_producers": len(allocations.get("producers") or []),
        "n_consumers": len(allocations.get("consumers") or []),

        "pref_enabled": pref("enabled"),
        "pref_order": pref("order"),
        "n_pairs": len(pairs) if pairs is not None else None,
        "pair_kwh": (round(sum(p.get("energy_kwh") or 0.0 for p in pairs), 6)
                     if pairs is not None else None),
        "pairs_rationed": pref("pairs_rationed"),

        "mult_enabled": mult("enabled"),
        "mult_mode": mult("mode"),
        "mult_sides": mult("sides"),
        "green_alloc_kwh": mult("green_alloc_kwh"),
        "grey_alloc_kwh": mult("grey_alloc_kwh"),
        "levy_collected_ct": mult("levy_collected_ct"),
        "bonus_requested_ct": mult("bonus_requested_ct"),
        "bonus_paid_ct": mult("bonus_paid_ct"),
        # The response calls it `scale`; renamed here because a column called
        # "scale" in a slot CSV names nothing.
        "bonus_scale": mult("scale"),
        "green_final_ct": mult("green_final_ct_per_kwh"),
        "grey_final_ct": mult("grey_final_ct_per_kwh"),
        "buyer_final_ct": mult("buyer_final_ct_per_kwh"),
        "buyers_pay_ct": mult("buyers_pay_ct"),
        "sellers_receive_ct": mult("sellers_receive_ct"),
        "pool_surplus_ct": mult("pool_surplus_ct"),

        "pairs_posted": scen.get("pairs_posted"),
        "pairs_unpostable": scen.get("pairs_unpostable"),

        "crossover_mult": (_crossover_mult(multipliers, price)
                           if multipliers else None),

        **_execution_row(record.get("execution")),
    }


def slot_rows(records) -> list:
    return [slot_row(record) for record in records]


def write_slot_csv(path, records) -> Path:
    """One tidy CSV per run, one row per slot, in slot order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(slot_rows(records), key=lambda r: r["slot"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ------------------------------------------------------------ the area CSV

def area_rows(records, preferences: dict = None) -> list:
    """One row per allocation row per slot.

    The slot CSV can only carry round-level quantities, and the preference
    mechanism is a *distributional* one: what `preferences_first` changes
    against `pro_rata_first` is which areas get filled, at an unchanged
    round-level traded quantity. That difference is only visible per area.

    A slot that did not clear has no allocations and writes nothing here --
    roughly 317 of 672 slots on the reference week. `named` and `partner` are
    the run's held draw (`scenario.draw_preferences`), not a per-slot fact:
    `preference_matched` off the response says whether the pair was *served*,
    `named` says whether there was one to serve.

    Where the slot also executed, the participant's own execution result row
    is copied onto the same line: what it delivered, what it could have
    delivered and through which channel that was known, its two penalties and
    the compensation the redistribution rule computed for it. `compensation_ct`
    comes from `redistribution.rows`, which only carries the harmed set, so it
    is empty for a deviator and for anyone who traded nothing.
    """
    preferences = preferences or {}
    rows = []
    for record in records:
        allocations = (record["clearing"].get("allocations") or {})
        execution = record.get("execution") or {}
        results = {r.get("area_uuid"): r
                   for r in execution.get("results") or []}
        compensation = {
            r.get("area_uuid"): r.get("compensation_ct")
            for r in (execution.get("redistribution") or {}).get("rows") or []}
        for side, key in (("seller", "producers"), ("buyer", "consumers")):
            for entry in allocations.get(key) or []:
                area = entry.get("area_uuid")
                result = results.get(area) or {}
                rows.append({
                    "slot": record["time_slot"],
                    "area_uuid": area,
                    "side": side,
                    "energy_type": entry.get("energy_type"),
                    "requested_kwh": entry.get("requested_kwh"),
                    "allocated_kwh": entry.get("allocated_kwh"),
                    "fill_rate": entry.get("fill_rate"),
                    "preference_matched": entry.get("preference_matched"),
                    "named": area in preferences,
                    "partner": preferences.get(area),
                    **{k: result.get(k) for k in _AREA_EXECUTION_KEYS},
                    "compensation_ct": compensation.get(area),
                })
    return rows


def write_area_csv(path, records, preferences: dict = None) -> Path:
    """`<run-id>_areas.csv`: one row per allocation row per slot.

    Roughly 115 rows per cleared slot over roughly 355 tradeable slots, so
    about 40k rows per run -- written per run rather than per campaign, and
    only for slots that cleared.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(area_rows(records, preferences),
                  key=lambda r: (r["slot"], r["side"], r["area_uuid"] or ""))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AREA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def round_type_census(records) -> dict:
    """Counts of supply-limited and demand-limited slots, read off the
    clearing response's own `round_type`.

    `no_trade` is counted separately rather than folded into either: a slot
    with no supply at all is not a supply-limited round, it is a slot the
    mechanism never ran on, and the two must not be summed in 5.1.
    """
    census = {SUPPLY_LIMITED: 0, DEMAND_LIMITED: 0, BALANCED: 0, "no_trade": 0}
    for record in records:
        clearing = record["clearing"]
        if clearing.get("status") != "cleared":
            census["no_trade"] += 1
            continue
        round_type = clearing.get("round_type")
        census[round_type] = census.get(round_type, 0) + 1
    return census


def ratios(records) -> list:
    """The ratios the artifact reported, cleared slots only."""
    return [r["clearing"]["ratio"] for r in records
            if r["clearing"].get("status") == "cleared"
            and r["clearing"].get("ratio") is not None]


def ratio_span(records) -> float:
    """Span of the reported supply/demand ratio over a sequence."""
    values = ratios(records)
    return (max(values) - min(values)) if values else 0.0
