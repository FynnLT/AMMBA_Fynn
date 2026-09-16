"""Block D, corrected: withholding economics need the outside value of the
withheld energy. Without it, withholding is pure volume loss."""
# FROZEN at the 02.09.2026 test run. Runs against `git checkout fe52d63`,
# not against the current runner signature: `run_slot` now requires an
# explicit community and slot, and these scripts call it positionally.
# No Chapter 5 result depends on them -- they produced the 02.09. test-run
# figures only. Do not adapt them; do not import them from live code.
import asyncio, csv, logging, os, statistics, random
from pathlib import Path
import stack, scenario, runner
logging.disable(logging.WARNING)
HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
SIG = {"k_upper": 28.5, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}
P = stack.penalties; sp = stack.sigmoid_price
BASE_OFFERS = [3.0, 2.5, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5]
BASE_BIDS   = [2.5, 2.2, 2.0, 1.8, 1.6, 1.5, 1.4, 1.2, 1.0, 0.8]
D_TOTAL = sum(BASE_BIDS); S_FULL = sum(BASE_OFFERS)
SLOTS_PER_YEAR = 96 * 365

def write(name, rows):
    keys = list(rows[0].keys())
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"  -> {name}  ({len(rows)} rows)")

def d1(outside_values=(0.0, 8.0, 15.147225, 28.5)):
    rows = []
    C = BASE_OFFERS[0]
    p_h = sp(S_FULL / D_TOTAL, **SIG)
    for v in outside_values:
        for w in [round(0.05*i, 3) for i in range(0, 51)]:      # 0 .. 2.5 kWh
            offers = list(BASE_OFFERS); offers[0] = C - w
            S = sum(offers); traded = min(S, D_TOTAL)
            p_w = round(sp(S / D_TOTAL, **SIG), 6)
            o = P.seller_externality_penalty(C - w, C, 0.0, S, D_TOTAL, traded,
                                             p_w, SIG)
            pen = o["penalty_ct"] if o else 0.0
            gain = (C - w) * p_w + w * v - C * p_h
            rows.append({"v_ct_per_kwh": v, "withheld_kwh": w,
                         "withheld_share": round(w / C, 4),
                         "price_honest": round(p_h, 6), "price_withhold": p_w,
                         "gain_ct": round(gain, 4),
                         "penalty_oracle_ct": round(pen, 6),
                         "penalty_meter_ct": 0.0,
                         "net_with_penalty_ct": round(gain - pen, 4),
                         "net_meter_ct": round(gain, 4)})
    return rows

def d3(sigma=0.01, n=40000, seed=7, v=28.5):
    rng = random.Random(seed); rows = []
    traded = 3.0
    noise = [rng.gauss(0, sigma) for _ in range(n)]
    for eta_rel in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025,
                    0.03, 0.04, 0.05, 0.075, 0.10, 0.15]:
        eta = eta_rel * traded
        fp = sum(1 for z in noise if max(0.0, -z * traded) > eta) / n
        pen_mean = statistics.mean(
            P.seller_shortfall_penalty(traded, traded*(1+z), 28.5, 1.1, eta)["penalty_ct"]
            for z in noise)
        free_gain = eta * v          # keep eta kWh, no penalty
        rows.append({"eta_rel": eta_rel, "eta_kwh": round(eta, 5),
                     "false_positive_share": round(fp, 5),
                     "mean_wrong_penalty_ct": round(pen_mean, 5),
                     "free_kwh_per_slot": round(eta, 5),
                     "free_gain_ct_per_slot": round(free_gain, 4),
                     "free_gain_eur_per_year": round(
                         free_gain * SLOTS_PER_YEAR / 100.0, 2)})
    return rows

async def d_pipeline_proof(st):
    """End-to-end proof that the meter-only path cannot see withholding."""
    rows = []
    C, w = 3.0, 1.0
    offers = list(BASE_OFFERS); offers[0] = C - w
    scen = {"producers": [{"area_uuid": f"gen-{i}", "name": f"Gen {i}",
                           "energy": e, "energy_type": "green"}
                          for i, e in enumerate(offers)],
            "consumers": [{"area_uuid": f"load-{i}", "name": f"Load {i}",
                           "energy": e} for i, e in enumerate(BASE_BIDS)]}
    for label, meas in (("withholder delivers exactly what he sold", {}),
                        ("same seller under-delivers 0.5 kWh",
                         {"gen-0": C - w - 0.5})):
        r = await runner.run_slot(st, scen, preferences={"enabled": False,
                                  "multipliers_enabled": False},
                                  measurements=meas, gamma=1.1, eta=0.0)
        ex = r["execution"]
        g0 = [x for x in ex["results"] if x["area_uuid"] == "gen-0"][0]
        rows.append({"case": label, "round_type": ex["round_type"],
                     "traded_kwh": g0["traded_kwh"], "actual_kwh": g0["actual_kwh"],
                     "shortfall_kwh": g0["shortfall_kwh"],
                     "shortfall_penalty_ct": g0["shortfall_penalty_ct"],
                     "externality_kwh": g0["externality_kwh"],
                     "externality_penalty_ct": g0["externality_penalty_ct"],
                     "total_penalty_ct": g0["total_penalty_ct"],
                     "market_total_penalties_ct": ex["total_penalties_ct"]})
    return rows

async def main():
    st = stack.Stack()
    print("D1 withholding economics"); write("D1_withholding.csv", d1())
    print("D3 eta");                   write("D3_eta.csv", d3())
    print("D5 pipeline proof");        write("D5_pipeline_proof.csv", await d_pipeline_proof(st))
    await st.close()

if __name__ == "__main__":
    asyncio.run(main())
