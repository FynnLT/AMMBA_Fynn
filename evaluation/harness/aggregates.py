"""Per-slot aggregates for one run: a tidy CSV and the round-type census.

**These are projections of the clearing response, not computations.** Every
field below is read off what the artifact answered. Recomputing the ratio from
the scenario would make the calibration measure the generator instead of the
artifact -- the two agree only as long as netting, rounding and the order
filter all agree, and the moment they do not, the fitted theta describes the
harness.
"""
import csv
from pathlib import Path

SUPPLY_LIMITED = "SUPPLY_LIMITED"
DEMAND_LIMITED = "DEMAND_LIMITED"
BALANCED = "BALANCED"

FIELDS = ("slot", "total_supply_kwh", "total_demand_kwh", "ratio",
          "clearing_price_ct", "traded_kwh", "n_producers", "n_consumers")


def slot_row(record: dict) -> dict:
    """One record of `runner.run_sequence` as one CSV row.

    A slot that did not clear (one side empty) keeps its supply and demand --
    the clearing node reports both on the no-trade path -- and carries None
    for price, ratio and traded quantity. Dropping those rows would hide the
    empty half of the week from the census.
    """
    clearing = record["clearing"]
    allocations = clearing.get("allocations") or {}
    return {
        "slot": record["time_slot"],
        "total_supply_kwh": clearing.get("total_supply_kwh"),
        "total_demand_kwh": clearing.get("total_demand_kwh"),
        "ratio": clearing.get("ratio"),
        "clearing_price_ct": clearing.get("clearing_price_ct_per_kwh"),
        "traded_kwh": clearing.get("traded_quantity_kwh"),
        "n_producers": len(allocations.get("producers") or []),
        "n_consumers": len(allocations.get("consumers") or []),
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
