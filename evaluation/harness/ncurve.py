"""The N-curve: off-chain clearing-cycle time against the number of orders in
a slot.

The feasibility statement of 5.4 is a contrast, not a capacity: the off-chain
clearing cycle is O(N) in the size of the order book while `clearMarket` is
O(1) by construction, so the settlement layer does not grow with the
community and the off-chain layer does. This runner measures the first half
of that sentence. The pilot measured four points and its successor
`harness/legacy/blocke.py` is frozen at the 02.09. runner signature; nothing
in the current harness measures it.

**Three decisions govern what this may claim** (19.09.):

* **D-85 -- N is the number of orders in a slot.** The harness assumption
  "one order per area and slot" holds at every N and is not bypassed,
  relaxed or caught here: N orders means N market areas, and every generated
  book goes through `scenario._assert_one_order_per_area` before it is
  posted.
* **D-86 -- the mock-store threshold is measured, not set.** Every point
  reports the time spent inside the clearing cycle separately from the time
  spent in off-chain store calls (`stack.TimedTransport`), so "above this N
  the curve is measuring the store" is a result this file produces rather
  than a caveat the chapter has to add.
* **D-87 -- no saturation figure.** This runner reports measured points and
  the share of the 900 s slot they occupy. It does not fit a line, does not
  extrapolate a capacity and does not print a participant count. The
  complexity statement is the output; a capacity number would be an
  extrapolation from an in-process measurement with no network, no uvicorn
  and no real chain behind it.

**Why this is not a campaign cell.** `runs.RunSpec` is built around the
profile week: it carries `n_players`, a workbook, an extension seed and 672
slots, and `runs.write_manifest` requires `player_ids` and
`dataset_extension_seed`. The N-curve needs synthetic books at N up to 1e5, a
handful of slots per point and no execution layer, so forcing it through
`RunSpec` would mean inventing values for fields that have no meaning here.
It gets its own runner and its own manifest, with the same provenance
content.

    cd evaluation/harness
    python ncurve.py                       # the default points
    python ncurve.py --points 5,10,20,50,100 --repeats 1
    python ncurve.py --max-n 1000
"""
import argparse
import asyncio
import csv
import json
import logging
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import runner
import runs
import scenario
import stack

HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out" / "ncurve"

# The slot length, and therefore the stopping rule: a cycle that does not fit
# in the slot it is clearing has answered the feasibility question, and every
# larger N would only answer it again more slowly. Read off `runner` rather
# than restated, and kept as a module-level name so a test can shrink it.
SLOT_SEC = runner.SLOT_SEC

# 1e6 is deliberately absent. At one order per area (D-85) it is 1e6 market
# areas in an in-memory store, and the run would likely exhaust memory rather
# than time -- which measures the mock, not the cycle. It is reachable as
# `--points 1000000`, as a deliberate act, and the stopping rule covers it.
N_POINTS = (5, 10, 20, 50, 100, 400, 800, 1_000, 10_000, 100_000)

# The pilot's 40/60 split, so the points here are comparable with the pilot's
# N in {6, 100, 400, 800}.
PRODUCER_SHARE = 0.4
SD_RATIO = 1.25

REPEATS = 3
# At and above this N one slot per point, automatically: three repetitions of
# 1e5 orders is setup time spent to sharpen a median that the curve does not
# turn on. The CSV records how many were actually run.
BIG_N = 10_000
BIG_N_REPEATS = 1

COMMUNITY = "ncurve"
COMMUNITY_NAME = "N-curve"
BASE_SLOT = runs.DAY_START

FIELDS = ("n", "n_prod", "n_cons", "n_areas", "rep", "repeats", "seed",
          "slot", "market_id", "n_orders", "n_trades", "status",
          "t_setup_s", "t_clear_s", "t_store_in_clear_s",
          "n_store_calls_in_clear", "slot_share_pct")


def n_split(n: int) -> tuple[int, int]:
    """Producers and consumers for a point of the curve, summing to `n`.

    The consumer side is the remainder rather than a second rounding, so
    `n_prod + n_cons == n` holds exactly at every point -- N is the number of
    orders in the slot (D-85) and a curve whose x axis was off by one at odd
    N would be reporting a different N than it names.

    Both sides are held at one or more: a point with an empty side is not a
    market, and `round(0.4 * 5) == 2` only by luck of the rounding.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"a market needs at least two orders, got N={n}")
    n_prod = min(max(round(PRODUCER_SHARE * n), 1), n - 1)
    return n_prod, n - n_prod


def repeats_for(n: int, repeats: int) -> int:
    """One slot per point at the large N, whatever `--repeats` asked for."""
    return BIG_N_REPEATS if n >= BIG_N else repeats


def make_book(n: int, seed: int) -> dict:
    """One synthetic order book of exactly `n` orders.

    No preferences and no deviations: the curve measures the clearing cycle,
    and a preferred-partner requirement changes what the cycle does rather
    than how much of it there is.

    The one-order-per-area assumption is asserted on the book that was
    actually built, not assumed of the generator (D-85). `make_scenario`
    names its areas `gen-*` and `load-*` and cannot collide today; the
    assertion is what keeps that a checked fact rather than a reading of the
    generator.
    """
    n_prod, n_cons = n_split(n)
    scen = scenario.make_scenario(seed=seed, n_prod=n_prod, n_cons=n_cons,
                                  sd_ratio=SD_RATIO)
    scenario._assert_one_order_per_area(scen)
    return scen


async def post_book(st, scen, areas, *, market_id, slot) -> str:
    """`POST /market` plus `POST /orders-normalized` -- harness setup.

    This is the first half of `runner.run_slot`, written out rather than
    reused because the two halves have to be timed apart: `run_slot` posts
    the book and clears it in one call, and D-86 needs the setup *out* of the
    cycle figure. The order shape is `run_slot`'s, minus the
    `requirements` branch this curve never takes.
    """
    market = await st.post("/market", {
        "community_uuid": COMMUNITY, "community_name": COMMUNITY_NAME,
        "time_slot": slot, "community_areas": areas,
        "market_id": market_id})
    mid = market["market_id"]
    orders = [{"order_type": "Offer", "created_by": p["name"],
               "area_uuid": p["area_uuid"], "market_id": mid,
               "time_slot": slot, "energy": p["energy"], "energy_rate": 8.0,
               "attributes": {"energy_type": p.get("energy_type", "green")}}
              for p in scen["producers"]]
    orders += [{"order_type": "Bid", "created_by": c["name"],
                "area_uuid": c["area_uuid"], "market_id": mid,
                "time_slot": slot, "energy": c["energy"],
                "energy_rate": 40.0}
               for c in scen["consumers"]]
    await st.post("/orders-normalized", orders)
    return mid


async def measure_slot(st, n: int, *, seed: int, slot: int) -> dict:
    """One slot at N orders, timed in three parts.

    The store is wiped first. `store.query_orders` is a linear scan over
    every order the store holds, so without the reset each point would be
    clearing its own book out of the accumulated books of all the smaller
    ones -- the curve would measure the cumulative campaign, not N.
    `Stack.reset()` deliberately leaves the transport counters alone; they
    are reset here, around the section being timed.
    """
    scen = make_book(n, seed)
    areas = runner.areas_for(scen)
    market_id = runner.market_id_for(COMMUNITY, slot)

    await st.reset()

    started = time.perf_counter()
    mid = await post_book(st, scen, areas, market_id=market_id, slot=slot)
    t_setup = time.perf_counter() - started

    trigger = {"market_id": mid, "community_uuid": COMMUNITY,
               "time_slot": slot, "community_name": COMMUNITY_NAME,
               "sigmoid_params": dict(runner.SIGMOID)}

    # Reset immediately before and read immediately after: anything between
    # these two lines is attributed to the cycle, and the whole point of the
    # split is that the attribution is exact.
    st.transport.reset()
    started = time.perf_counter()
    clearing = await st.clear(trigger)
    t_clear = time.perf_counter() - started
    store_elapsed = st.transport.elapsed
    store_calls = st.transport.n_requests

    n_prod, n_cons = n_split(n)
    return {"n": n, "n_prod": n_prod, "n_cons": n_cons,
            "n_areas": len(areas), "seed": seed, "slot": slot,
            "market_id": mid, "n_orders": n_prod + n_cons,
            "n_trades": clearing.get("num_trades"), "status": "ok",
            "t_setup_s": round(t_setup, 6),
            "t_clear_s": round(t_clear, 6),
            "t_store_in_clear_s": round(store_elapsed, 6),
            "n_store_calls_in_clear": store_calls,
            "slot_share_pct": round(100 * t_clear / SLOT_SEC, 6)}


def median_of(rows, key):
    values = [r[key] for r in rows if r["status"] == "ok"]
    return statistics.median(values) if values else None


def print_table(rows) -> None:
    """N, the two medians, the store's share of the cycle, the slot share.

    Nothing else, and in particular no fit, no extrapolated capacity and no
    participant count (D-87): the statement this table supports is that the
    cycle grows with N while the settlement layer does not, and a fitted
    line would turn a measurement into a projection.
    """
    print(f"\n{'N':>8} {'slots':>6} {'t_clear_s':>12} {'t_store_s':>12} "
          f"{'store %':>9} {'% of 900 s slot':>17}  status")
    for n in dict.fromkeys(r["n"] for r in rows):
        group = [r for r in rows if r["n"] == n]
        clear = median_of(group, "t_clear_s")
        store = median_of(group, "t_store_in_clear_s")
        if clear is None:
            print(f"{n:>8} {len(group):>6} {'-':>12} {'-':>12} {'-':>9} "
                  f"{'-':>17}  {group[0]['status']}")
            continue
        share = 100 * store / clear if clear else float("nan")
        print(f"{n:>8} {len(group):>6} {clear:12.4f} {store:12.4f} "
              f"{share:8.1f}% {100 * clear / SLOT_SEC:16.4f}%  "
              f"{group[0]['status']}")


def write_csv(path: Path, rows) -> None:
    """One row per slot, never per point.

    The median is a step of the analysis and belongs in the analysis: a CSV
    that carried only the medians could not be re-read at a different
    summary statistic, and could not show the spread the medians came from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def write_manifest(path: Path, entry: dict) -> None:
    """One object, not a list.

    `runs.write_manifest` appends, which is right for a campaign cell that
    may legitimately be re-run into the same directory and wrong here: this
    runner refuses to start into a directory that already holds a manifest,
    so there is never a second entry to append. Written as a bare object so
    that stays structurally true rather than merely observed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def parse_points(text: str) -> tuple:
    return tuple(int(part) for part in text.replace(" ", "").split(",")
                 if part)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Off-chain clearing-cycle time against the number of "
                    "orders in a slot. Reports measured points; it does not "
                    "fit, extrapolate or report a capacity (D-87).")
    parser.add_argument("--points", type=parse_points, default=N_POINTS,
                        help="comma-separated N values (default: "
                             f"{','.join(str(n) for n in N_POINTS)})")
    parser.add_argument("--max-n", type=int, default=None,
                        help="drop every point above this N")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"slots per point (default {REPEATS}; "
                             f"forced to {BIG_N_REPEATS} at N >= {BIG_N})")
    parser.add_argument("--seed", type=int, default=0,
                        help="base scenario seed; repetition r uses seed + r")
    parser.add_argument("--out", default=str(OUT),
                        help=f"output directory (default {OUT})")
    return parser.parse_args(argv)


async def main(argv=None) -> int:
    args = parse_args(argv)
    # Part of the measurement, not cosmetics: the three services log a
    # handful of INFO lines per slot, and at the small points that is a
    # measurable share of a cycle of a few milliseconds. WARNING and above
    # still reach the console, so a run that goes wrong still says so.
    logging.disable(logging.INFO)
    out_dir = Path(args.out)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        # Refuse rather than append: the alternative is a directory whose CSV
        # is from one invocation and whose provenance is from another.
        print(f"refusing to start: {manifest_path} already exists; "
              f"pass --out with a fresh directory", file=sys.stderr)
        return 2

    points = [n for n in args.points
              if args.max_n is None or n <= args.max_n]
    if not points:
        print("no points to run", file=sys.stderr)
        return 2

    started = time.perf_counter()
    st = stack.Stack()
    rows, status, stopped_at = [], "complete", None
    try:
        for n in points:
            repeats = repeats_for(n, args.repeats)
            try:
                for rep in range(repeats):
                    slot = BASE_SLOT + SLOT_SEC * len(rows)
                    row = await measure_slot(st, n, seed=args.seed + rep,
                                             slot=slot)
                    row.update(rep=rep, repeats=repeats)
                    rows.append(row)
                    print(f"N={n:<8} rep={rep}  "
                          f"setup={row['t_setup_s']:9.3f}s  "
                          f"clear={row['t_clear_s']:9.3f}s  "
                          f"store={row['t_store_in_clear_s']:9.3f}s "
                          f"({row['n_store_calls_in_clear']} calls)  "
                          f"{row['slot_share_pct']:.4f}% of slot")
                    if row["t_clear_s"] > SLOT_SEC:
                        status, stopped_at = "slot_exceeded", n
                        break
            except MemoryError:
                # Recorded, not raised: a stopping rule that fails the script
                # loses every point already measured.
                rows.append({"n": n, "repeats": repeats, "status": "memory"})
                status, stopped_at = "memory", n
                print(f"stopped: MemoryError at N={n}")
                break
            if status != "complete":
                if status == "slot_exceeded":
                    print(f"stopped: t_clear exceeded the slot length "
                          f"at N={n}")
                break
    finally:
        await st.close()

    wall = time.perf_counter() - started
    write_csv(out_dir / "ncurve.csv", rows)
    write_manifest(manifest_path, {
        "run_id": f"ncurve-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "repo_sha": runs.repo_sha(),
        "seed": args.seed,
        "n_points": [int(n) for n in points],
        "n_points_measured": sorted({int(r["n"]) for r in rows}),
        "repeats": args.repeats,
        "repeats_large_n": BIG_N_REPEATS,
        "large_n_threshold": BIG_N,
        "producer_share": PRODUCER_SHARE,
        "sd_ratio": SD_RATIO,
        "slot_sec": SLOT_SEC,
        "status": status,
        "stopped_at_n": stopped_at,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_sec": round(wall, 1),
        "output_files": ["ncurve.csv"],
    })

    print_table(rows)
    print(f"\n{len(rows)} slots in {wall:.1f}s; {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
