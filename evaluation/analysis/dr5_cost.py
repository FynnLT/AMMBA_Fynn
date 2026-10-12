"""DR5, second criterion: the cost of priority (D-97, fixed 12.10.2026).

In every slot, on the rationed (long) side, the fill rate r of a participant
not served first may fall at most 10 % below the pro-rata fill rate of that
slot, r_pro = allocated / claimed on that side (under pure pro rata every
claimant on the rationed side receives that share). Reported for 5 %, 10 %
and 20 %. Block-1 runs only (not executed), preference-first cells only.

Post-hoc analysis of recorded runs: reads `out/`, writes one JSON and one
CSV into a fresh directory. Touches none of the services.

    python dr5_cost.py [--runs ../out] [--dest ../out/analysis/<id>]
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import discovery  # noqa: E402

# Regime as in the Execution Node (penalties.determine_round_type, _EPSILON =
# 1e-9) and replay.regime_vec; repeated here so that this script does not
# import the service stack.
SL, BAL, DL = -1, 0, 1
EPSILON = 1e-9


def regime_vec(supply, demand):
    return np.where(supply < demand - EPSILON, SL,
                    np.where(supply > demand + EPSILON, DL, BAL))

BOUNDS = (0.05, 0.10, 0.20)
PRIMARY = 0.10


def analyse(data, acc):
    run = data.run
    if run.executed:
        return
    slots, areas, idx = data.slots, data.areas, data.slot_index
    enabled = slots.strings("pref_enabled") == "True"
    order = slots.strings("pref_order")
    regime = regime_vec(slots.floats("total_supply_kwh"),
                          slots.floats("total_demand_kwh"))
    side = areas.strings("side")
    requested = areas.floats("requested_kwh")
    allocated = areas.floats("allocated_kwh")
    fill = areas.floats("fill_rate")
    matched = areas.bools("preference_matched")
    groups = defaultdict(list)
    for i in range(areas.n):
        groups[(int(idx[i]), side[i])].append(i)
    if run.run_id not in acc["_totals"]["runs"]:
        t = acc["_totals"]
        t["runs"].add(run.run_id)
        traded = np.nan_to_num(slots.floats("traded_kwh"))  # empty = no trade
        price = slots.floats("clearing_price_ct")
        cell = t["cells"].setdefault(run.cell, {"runs": 0, "traded_kwh": 0.0,
                                         "value_ct": 0.0, "trading_slots": 0,
                                         "slots": 0})
        cell["runs"] += 1
        cell["slots"] += int(len(traded))
        cell["trading_slots"] += int((traded > 0).sum())
        cell["traded_kwh"] += float(traded.sum())
        cell["value_ct"] += float((np.nan_to_num(price) * traded).sum())
    for (s, side_name), members in groups.items():
        if not (enabled[s] and order[s] == "preferences_first"):
            continue
        long_side = ((side_name == "seller" and regime[s] == DL)
                     or (side_name == "buyer" and regime[s] == SL))
        if not long_side:
            continue
        members = np.array(members)
        claimed = float(requested[members].sum())
        if claimed <= 0:
            continue
        r_pro = float(allocated[members].sum()) / claimed
        served = matched[members]
        rest = members[~served]
        a = acc[(run.cell, side_name)]
        a["slots"] += 1
        a["runs"].add(run.run_id)
        if not served.any():
            continue                      # priority did not act: r == r_pro
        a["slots_priority"] += 1
        if not len(rest):
            a["slots_all_served"] += 1
            continue
        r = float(np.mean(fill[rest]))
        shifted = max(0.0, r_pro - r) * float(requested[rest].sum())
        a["shifted_kwh"] += shifted
        supply_s = float(slots.floats("total_supply_kwh")[s])
        demand_s = float(slots.floats("total_demand_kwh")[s])
        short_over_long = (min(supply_s, demand_s) / max(supply_s, demand_s)
                           if max(supply_s, demand_s) > 0 else 0.0)
        spread = float(np.ptp(fill[rest]))
        a["max_spread_rest"] = max(a["max_spread_rest"], spread)
        rel = 0.0 if r_pro <= 0 else max(0.0, 1.0 - r / r_pro)
        a["rel"].append(rel)
        a["abs_pp"].append(max(0.0, r_pro - r))
        a["participant_slots"] += len(rest)
        for b in BOUNDS:
            if rel > b + 1e-9:
                a[f"slots_over_{b}"] += 1
                a[f"pslots_over_{b}"] += len(rest)
                a[f"shifted_kwh_over_{b}"] += shifted
                if b == PRIMARY:
                    a["short_over_long_violating"].append(short_over_long)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE.parent / "out"))
    ap.add_argument("--dest", default=None)
    args = ap.parse_args(argv)
    try:
        sha = subprocess.run(["git", "--no-optional-locks", "rev-parse",
                              "--short", "HEAD"], cwd=HERE, text=True,
                             capture_output=True).stdout.strip()
    except OSError:
        sha = "unknown"
    dest = Path(args.dest) if args.dest else (
        HERE.parent / "out" / "analysis" / f"dr5-cost-{sha or 'nosha'}")
    if dest.exists() and any(dest.iterdir()):
        print(f"refusing to overwrite {dest}", file=sys.stderr)
        return 2
    found = discovery.discover([Path(args.runs)], exclude=[dest])
    acc = defaultdict(lambda: defaultdict(float) | {
        "rel": [], "abs_pp": [], "runs": set(), "max_spread_rest": 0.0,
        "short_over_long_violating": []})
    acc["_totals"] = {"runs": set(), "cells": {}}
    for run in found.runs:
        if run.executed:
            continue
        try:
            data = discovery.RunData.load(run)
        except discovery.MissingColumns:
            continue
        analyse(data, acc)
    rows = []
    totals = acc.pop("_totals")
    for (cell, side), a in sorted(acc.items()):
        if not a["slots_priority"]:
            continue
        rel = np.array(a["rel"])
        pp = np.array(a["abs_pp"])
        row = {"cell": cell, "side": side, "n_runs": len(a["runs"]),
               "rationed_slots": int(a["slots"]),
               "slots_priority_acted": int(a["slots_priority"]),
               "slots_all_served_first": int(a["slots_all_served"]),
               "participant_slots_not_served_first": int(a["participant_slots"]),
               "rel_reduction_median": float(np.median(rel)) if len(rel) else 0.0,
               "rel_reduction_p95": float(np.quantile(rel, 0.95)) if len(rel) else 0.0,
               "rel_reduction_max": float(rel.max()) if len(rel) else 0.0,
               "abs_pp_max": float(pp.max()) if len(pp) else 0.0,
               "max_spread_within_not_served": a["max_spread_rest"],
               "shifted_kwh_per_week": a["shifted_kwh"] / max(1, len(a["runs"])),
               "shifted_kwh_per_week_over_10pct": a[f"shifted_kwh_over_{PRIMARY}"] / max(1, len(a["runs"])),
               "traded_kwh_per_week": totals["cells"][cell]["traded_kwh"] / totals["cells"][cell]["runs"],
               "short_over_long_median_violating": float(np.median(a["short_over_long_violating"])) if a["short_over_long_violating"] else None,
               "short_over_long_p90_violating": float(np.quantile(a["short_over_long_violating"], 0.9)) if a["short_over_long_violating"] else None}
        for b in BOUNDS:
            row[f"slots_over_{int(b*100)}pct"] = int(a[f"slots_over_{b}"])
            row[f"pslots_over_{int(b*100)}pct"] = int(a[f"pslots_over_{b}"])
        rows.append(row)
    met = all(r[f"slots_over_{int(PRIMARY*100)}pct"] == 0 for r in rows)
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "dr5_cost.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (dest / "dr5_cost.json").write_text(json.dumps(
        {"repo_sha": sha, "criterion": "D-97", "primary_bound": PRIMARY,
         "bounds": BOUNDS, "dr5_cost_met": met, "rows": rows,
         "n_runs_discovered": len(found.runs),
         "cells_totals": totals["cells"]}, indent=2), encoding="utf-8")
    for c, t in sorted(totals["cells"].items()):
        print(c, t, "value per trading slot ct:", round(t["value_ct"] / max(1, t["trading_slots"]), 2))
    for r in rows:
        print(r)
    print("DR5 cost-of-priority criterion met:", met, "->", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
