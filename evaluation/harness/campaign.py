"""Pilot campaign. Writes tidy CSVs into out/.

Paths are derived from this file's own location so the harness runs on
Windows as well as on the Linux sandbox the pilot used.
"""
import asyncio, csv, json, logging, os, platform, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import stack, scenario, runner
logging.disable(logging.WARNING)

HARNESS_DIR = Path(__file__).resolve().parent
REPO = stack.REPO                      # the pinned clone next to the harness
OUT = HARNESS_DIR.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
SHA = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              text=True).strip()
MANIFEST = OUT / "manifest.json"

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
        "repo_sha": SHA,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "params": params,
        "seeds": seeds,
        "wall_sec": round(wall_sec, 3),
        "output_files": [str(Path(f).name) for f in output_files],
    }
    entry.update(extra)
    runs = []
    if MANIFEST.exists():
        try:
            runs = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
        if not isinstance(runs, list):
            runs = [runs]
    runs.append(entry)
    MANIFEST.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"  -> manifest.json  (+{run_id})")
    return entry


async def block_golden(st, rows):
    """A: reproduce the implementation-guide reference numbers."""
    r = await runner.run_slot(st, scenario.GUIDE, preferences=PREF_OFF, execute=False)
    c = r["clearing"]
    rows.append({"case": "baseline_pro_rata", "price": c["clearing_price_ct_per_kwh"],
                 "ratio": c["ratio"], "round": c["round_type"],
                 **{p["name"]: p["fill_rate"] for p in c["allocations"]["producers"]}})
    r2 = await runner.run_slot(st, scenario.GUIDE,
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
    for ratio in ratios:
        for seed in range(5):
            scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                          sd_ratio=ratio, pair_density=0.25)
            r = await runner.run_slot(st, scen, preferences=prefs(), execute=False)
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
    for density in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
        for order in ("pro_rata_first", "preferences_first"):
            for seed in range(5):
                scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                              sd_ratio=1.25, pair_density=density)
                r = await runner.run_slot(st, scen, preferences=prefs(order=order),
                                          execute=False)
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
    grid = [round(0.005*i, 4) for i in range(0, 25)]   # 0.000 .. 0.120
    for mode in ("multiplicative", "additive"):
        for sides in ("seller", "both"):
            for gm in grid:
                r = await runner.run_slot(
                    st, scenario.GUIDE,
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


async def main():
    t0 = time.perf_counter()
    st = stack.Stack()
    manifest = {"repo_sha": SHA, "timestamp": int(time.time()),
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
    # Guarded so R0/R1 can import `block_golden` without running the whole
    # pilot campaign as a side effect of the import.
    asyncio.run(main())
