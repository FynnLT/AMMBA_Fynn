"""Block D — penalties. Pipeline where it matters, direct function calls
for the parameter sweeps (they are pure functions)."""
# FROZEN at the 02.09.2026 test run. Runs against `git checkout fe52d63`,
# not against the current runner signature: `run_slot` now requires an
# explicit community and slot, and these scripts call it positionally.
# No Chapter 5 result depends on them -- they produced the 02.09. test-run
# figures only. Do not adapt them; do not import them from live code.
import asyncio, csv, json, logging, os, statistics, random
from pathlib import Path
import stack, scenario, runner
logging.disable(logging.WARNING)
HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
SIG = {"k_upper": 28.5, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}
P = stack.penalties
sp = stack.sigmoid_price

def write(name, rows):
    keys = list(rows[0].keys())
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"  -> {name}  ({len(rows)} rows)")

# supply-limited base market: 8 sellers, 10 buyers, S < D
BASE_OFFERS = [3.0, 2.5, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5]     # 13.0 kWh
BASE_BIDS   = [2.5, 2.2, 2.0, 1.8, 1.6, 1.5, 1.4, 1.2, 1.0, 0.8]  # 16.0 kWh
D_TOTAL = sum(BASE_BIDS)

def scen_from(offers, bids):
    return {"producers": [{"area_uuid": f"gen-{i}", "name": f"Gen {i}",
                           "energy": round(e, 6), "energy_type": "green"}
                          for i, e in enumerate(offers) if e > 0],
            "consumers": [{"area_uuid": f"load-{i}", "name": f"Load {i}",
                           "energy": round(e, 6)} for i, e in enumerate(bids)]}

async def d1_meter_vs_oracle(st):
    """Withholding: pipeline (meter only) vs oracle deliverable channel."""
    rows = []
    C = BASE_OFFERS[0]                       # the potential withholder, 3.0 kWh
    for w in [round(0.1*i, 2) for i in range(0, 26)]:   # 0 .. 2.5 kWh withheld
        offers = list(BASE_OFFERS); offers[0] = C - w
        S = sum(offers)
        p_w = sp(S / D_TOTAL, **SIG)
        # honest reference
        p_h = sp(sum(BASE_OFFERS) / D_TOTAL, **SIG)
        traded = min(S, D_TOTAL)
        # ---- variant "meter": run the real pipeline; deliver == traded ----
        scen = scen_from(offers, BASE_BIDS)
        r = await runner.run_slot(st, scen, preferences={"enabled": False,
                                  "multipliers_enabled": False},
                                  measurements=None, gamma=1.1, eta=0.0)
        ex = r["execution"]
        meter_pen = sum(row["total_penalty_ct"] for row in ex["results"]) if ex else 0.0
        # ---- variant "oracle": same numbers, deliverable = full capacity ----
        o = P.seller_externality_penalty(C - w, C, 0.0, S, D_TOTAL, traded,
                                         round(p_w, 6), SIG)
        oracle_pen = o["penalty_ct"] if o else 0.0
        rev_honest = C * p_h
        rev_withhold = (C - w) * p_w
        rows.append({"withheld_kwh": w, "withheld_share": round(w / C, 4),
                     "supply": round(S, 4), "price_honest": round(p_h, 6),
                     "price_withhold": round(p_w, 6),
                     "rev_honest_ct": round(rev_honest, 4),
                     "rev_withhold_ct": round(rev_withhold, 4),
                     "gain_ct": round(rev_withhold - rev_honest, 4),
                     "penalty_meter_ct": round(meter_pen, 6),
                     "penalty_oracle_ct": round(oracle_pen, 6),
                     "net_meter_ct": round(rev_withhold - rev_honest - meter_pen, 4),
                     "net_oracle_ct": round(rev_withhold - rev_honest - oracle_pen, 4),
                     "round_type": ex["round_type"] if ex else None})
    return rows

def d2_gamma():
    """Under-delivery: sell C, deliver C-s. Gain = s*v (energy kept)."""
    rows = []
    C, p = 3.0, 15.147225
    for v_label, v in (("FiT 8.0", 8.0), ("clearing 15.15", p),
                       ("retail 28.5", 28.5)):
        for gamma in [round(0.5 + 0.05*i, 2) for i in range(0, 41)]:  # 0.5..2.5
            s = 1.0
            pen = P.seller_shortfall_penalty(C, C - s, 28.5, gamma, 0.0)
            rows.append({"outside_value": v_label, "v_ct_per_kwh": v,
                         "gamma": gamma, "shortfall_kwh": s,
                         "gain_ct": round(s * v, 4),
                         "penalty_ct": pen["penalty_ct"],
                         "net_ct": round(s * v - pen["penalty_ct"], 4)})
    return rows

def d3_eta(sigma=0.01, n=20000, seed=7):
    """Relative eta: false positives on honest agents vs free withholding."""
    rng = random.Random(seed)
    rows = []
    traded = 3.0
    noise = [rng.gauss(0, sigma) for _ in range(n)]           # meter class B ~1%
    for eta_rel in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        eta = eta_rel * traded
        # honest: delivery deviates by measurement noise only
        fp = sum(1 for z in noise if max(0.0, -z) * traded > eta) / n
        pen_mean = statistics.mean(
            P.seller_shortfall_penalty(traded, traded * (1 + z), 28.5, 1.1, eta)["penalty_ct"]
            for z in noise)
        # strategic: withhold exactly eta, free of charge
        S = sum(BASE_OFFERS); S_w = S - eta
        p_h = sp(S / D_TOTAL, **SIG); p_w = sp(S_w / D_TOTAL, **SIG)
        free_gain = (traded - eta) * p_w - traded * p_h + eta * 8.0
        rows.append({"eta_rel": eta_rel, "eta_kwh": round(eta, 4),
                     "false_positive_share": round(fp, 5),
                     "mean_wrong_penalty_ct": round(pen_mean, 4),
                     "free_withholding_kwh": round(eta, 4),
                     "free_gain_ct": round(free_gain, 4)})
    return rows

def d4_multiagent():
    """Are individually computed penalties additive? (ssrn leaves this open)"""
    rows = []
    S_full = sum(BASE_OFFERS)
    for k in (1, 2, 3, 5, 8):
        for w in (0.2, 0.5, 1.0):
            ws = [w] * k
            S_obs = S_full - sum(ws)
            if S_obs <= 0: continue
            traded = min(S_obs, D_TOTAL)
            p_obs = round(sp(S_obs / D_TOTAL, **SIG), 6)
            indiv = []
            for wi in ws:
                o = P.seller_externality_penalty(1.0, 1.0 + wi, 0.0, S_obs,
                                                 D_TOTAL, traded, p_obs, SIG)
                indiv.append(o["penalty_ct"] if o else 0.0)
            p_joint = sp((S_obs + sum(ws)) / D_TOTAL, **SIG)
            joint = max(0.0, (p_obs - p_joint) * traded)
            rows.append({"n_deviators": k, "withheld_each_kwh": w,
                         "sum_individual_ct": round(sum(indiv), 4),
                         "joint_ct": round(joint, 4),
                         "over_collection_ct": round(sum(indiv) - joint, 4),
                         "over_collection_pct": round(
                             100 * (sum(indiv) - joint) / joint, 2) if joint else None})
    return rows

async def main():
    st = stack.Stack()
    print("D1 meter vs oracle"); write("D1_meter_vs_oracle.csv", await d1_meter_vs_oracle(st))
    print("D2 gamma sweep");     write("D2_gamma.csv", d2_gamma())
    print("D3 eta sweep");       write("D3_eta.csv", d3_eta())
    print("D4 multi-agent");     write("D4_multiagent.csv", d4_multiagent())
    await st.close()

if __name__ == "__main__":
    asyncio.run(main())
