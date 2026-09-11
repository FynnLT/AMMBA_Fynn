"""Block E — off-chain cycle time vs participant count (in-process, so this
is a LOWER bound on real latency: no network, no uvicorn, no real chain)."""
import asyncio, csv, logging, os, statistics, time
from pathlib import Path
import stack, scenario, runner
logging.disable(logging.WARNING)
HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

async def main():
    st = stack.Stack(); rows = []
    for n in (6, 25, 50, 100, 200, 400, 800):
        for rep in range(5):
            scen = scenario.make_scenario(seed=rep, n_prod=max(1, int(n*0.4)),
                                          n_cons=max(1, int(n*0.6)),
                                          sd_ratio=0.85, pair_density=0.25)
            t0 = time.perf_counter()
            r = await runner.run_slot(st, scen, preferences={"enabled": True,
                                      "multipliers_enabled": True}, execute=False)
            t1 = time.perf_counter()
            await st.execute({"market_id": r["market_id"],
                              "community_uuid": r["community"],
                              "time_slot": r["time_slot"]})
            t2 = time.perf_counter()
            rows.append({"participants": n, "rep": rep,
                         "clearing_ms": round((t1-t0)*1000, 2),
                         "execution_ms": round((t2-t1)*1000, 2),
                         "total_ms": round((t2-t0)*1000, 2),
                         "n_trades": r["clearing"]["num_trades"]})
    keys = list(rows[0].keys())
    with open(OUT / "E_latency.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{'n':>5} {'clearing ms':>12} {'execution ms':>13} {'total ms':>10} {'% of 900 s slot':>16}")
    for n in (6, 25, 50, 100, 200, 400, 800):
        g = [r for r in rows if r["participants"] == n]
        tot = statistics.median(r["total_ms"] for r in g)
        print(f"{n:5d} {statistics.median(r['clearing_ms'] for r in g):12.1f} "
              f"{statistics.median(r['execution_ms'] for r in g):13.1f} {tot:10.1f} "
              f"{tot/900000*100:15.4f}%")
    await st.close()

if __name__ == "__main__":
    asyncio.run(main())
