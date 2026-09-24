"""The pilot: the 02.09. synthetic blocks, which `plots.py` reads.

    python pilot.py

Moved out of `campaign.py`, unchanged: the four blocks (A golden run, B
supply/demand sweep, C1 pair density, C2 multiplier grid) and the pilot's own
manifest writer, which appends to `out/manifest.json`. That writer is not the
profile campaign's -- a campaign run writes one manifest per run directory
through `runs.write_manifest` -- and keeping both in one module was two
manifest conventions in one file.

It imports what it shares with the campaign from `campaign`, in that
direction only: `campaign` does not import this module.
"""
import asyncio, csv, itertools, json, statistics, sys, time, platform
from datetime import datetime, timezone
from pathlib import Path
import runner, scenario, stack
from campaign import OUT, PREF_OFF, prefs, repo_sha

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

MANIFEST = OUT / "manifest.json"


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
    asyncio.run(pilot_main())
