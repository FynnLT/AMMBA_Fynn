"""Runs R0 (golden-run regression) and R1 (theta/B, configured vs published).

Driver for TASK-eval-R0-R1-2026-09-02. Everything it writes goes to `out/`
next to the harness; the pinned clone in `AMMBA_Fynn/` is only ever read from.

R0 must pass before R1 runs: R1 compares two parameter sets, and that is only
interpretable if the code still computes what the vault records.
"""
# FROZEN at the 02.09.2026 test run. Runs against `git checkout fe52d63`,
# not against the current runner signature: `run_slot` now requires an
# explicit community and slot, and these scripts call it positionally.
# No Chapter 5 result depends on them -- they produced the 02.09. test-run
# figures only. Do not adapt them; do not import them from live code.
import asyncio, statistics, sys, time
from pathlib import Path

import stack, scenario, runner, campaign
from campaign import OUT, PREF_OFF, prefs, write, write_manifest, block_golden

# --- R1 arms -----------------------------------------------------------
BASE_BAND = {"k_upper": 28.5, "k_lower": 8.0}
ARMS = {
    "configured": {**BASE_BAND, "theta": 1.0, "steepness": 2.5},
    "published":  {**BASE_BAND, "theta": 1.8, "steepness": 2.95},
}

# --- R0 reference values ------------------------------------------------
# The six quantities the vault declares as the regression guard (measured at
# 41fed16 on 25.08.). They populate the `expected` column only; every
# `measured` value comes from the run.
EXPECTED = [
    ("clearing_price_ct_per_kwh_ratio_1.25", "15.147225"),
    ("pro_rata_seller_fill_pct_a",           "80.0000"),
    ("matched_seller_fill_pct_b",            "96.8750"),
    ("unmatched_seller_fill_pct_b",          "68.7500"),
    ("subsidy_scale_0.10_0.10",              "0.37931"),
    ("green_final_rate_ct_per_kwh_b",        "15.721775"),
    ("grey_final_rate_ct_per_kwh_b",         "13.632503"),
]


def _decimals(literal):
    return len(literal.split(".")[1]) if "." in literal else 0


def _mean_fill_pct(producers, matched):
    vals = [p["fill_rate"] for p in producers
            if bool(p["preference_matched"]) is matched]
    return statistics.mean(vals) * 100.0 if vals else float("nan")


async def run_r0(st):
    """R0 - reproduce the six reference quantities at the pinned SHA."""
    t0 = time.perf_counter()
    golden_rows = []
    mult, c_a, c_b = await block_golden(st, golden_rows)

    prod_a = c_a["allocations"]["producers"]
    prod_b = c_b["allocations"]["producers"]

    measured = {
        "clearing_price_ct_per_kwh_ratio_1.25": c_a["clearing_price_ct_per_kwh"],
        "pro_rata_seller_fill_pct_a": statistics.mean(
            p["fill_rate"] for p in prod_a) * 100.0,
        "matched_seller_fill_pct_b": _mean_fill_pct(prod_b, True),
        "unmatched_seller_fill_pct_b": _mean_fill_pct(prod_b, False),
        "subsidy_scale_0.10_0.10": mult["scale"],
        "green_final_rate_ct_per_kwh_b": mult["green_final_ct_per_kwh"],
        "grey_final_rate_ct_per_kwh_b": mult["grey_final_ct_per_kwh"],
    }
    # Guard on quantity 2: "the pro-rata fill" is only a single number if every
    # seller really is filled identically. Report the spread, do not hide it
    # inside a mean.
    spread_a = (max(p["fill_rate"] for p in prod_a)
                - min(p["fill_rate"] for p in prod_a))

    rows, all_match = [], True
    for name, exp_literal in EXPECTED:
        dp = _decimals(exp_literal)
        exp = float(exp_literal)
        meas = measured[name]
        match = round(meas, dp) == round(exp, dp)
        all_match = all_match and match
        rows.append({"quantity": name, "expected": exp_literal,
                     "measured": "{:.{}f}".format(meas, dp),
                     "measured_full": repr(meas),
                     "delta": "{:+.{}f}".format(meas - exp, dp + 3),
                     "match": match})
    write("R0_golden.csv", rows)

    wall = time.perf_counter() - t0
    write_manifest("R0_golden", params={
        "scenario": "scenario.GUIDE (12.5 kWh supply / 10 kWh demand)",
        "sigmoid": dict(runner.SIGMOID),
        "arm_a_preferences": PREF_OFF,
        "arm_b_preferences": prefs(multipliers_enabled=True),
        "ratio_measured": c_a["ratio"],
        "round_type": c_a["round_type"],
        "pro_rata_fill_spread_a": round(spread_a, 9),
    }, seeds=[None], wall_sec=wall,
        output_files=["R0_golden.csv"], all_match=bool(all_match))
    return rows, all_match, spread_a


async def _price(st, scen, arm):
    r = await runner.run_slot(st, scen, preferences=PREF_OFF, execute=False,
                              sigmoid=ARMS[arm])
    c = r["clearing"]
    if c.get("status") != "cleared":
        return None, c
    return c["clearing_price_ct_per_kwh"], c


async def run_r1a(st):
    """R1 Part A - the two-point check on the reference scenario."""
    t0 = time.perf_counter()
    fields = ["arm", "theta", "steepness", "ratio", "clearing_price_ct",
              "round_type", "delta_ct", "delta_pct"]
    rows, price = [], {}
    for arm in ("configured", "published"):
        p, c = await _price(st, scenario.GUIDE, arm)
        price[arm] = p
        rows.append({"arm": arm, "theta": ARMS[arm]["theta"],
                     "steepness": ARMS[arm]["steepness"], "ratio": c["ratio"],
                     "clearing_price_ct": p, "round_type": c["round_type"],
                     "delta_ct": "", "delta_pct": ""})
    d = price["published"] - price["configured"]
    rows.append({"arm": "delta_published_minus_configured", "theta": "",
                 "steepness": "", "ratio": rows[0]["ratio"],
                 "clearing_price_ct": "", "round_type": "",
                 "delta_ct": round(d, 6),
                 "delta_pct": round(100.0 * d / price["configured"], 4)})
    write("R1a_two_point.csv", [{k: r[k] for k in fields} for r in rows])

    wall = time.perf_counter() - t0
    write_manifest("R1a_two_point",
                   params={"scenario": "scenario.GUIDE", "arms": ARMS,
                           "preferences": PREF_OFF, "execute": False},
                   seeds=[None], wall_sec=wall,
                   output_files=["R1a_two_point.csv"])
    return rows, price, d, wall


SD_RATIOS = [0.8, 1.0, 1.25, 1.5, 2.0]
SEEDS = [0, 1, 2]


async def run_r1b(st):
    """R1 Part B - five ratios x three seeds x two arms, price curve only."""
    t0 = time.perf_counter()
    rows = []
    for target in SD_RATIOS:
        for seed in SEEDS:
            for arm in ("configured", "published"):
                # Regenerated per arm from the same seed, so both arms see an
                # identical order book.
                scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                              sd_ratio=target, pair_density=0.0)
                p, c = await _price(st, scen, arm)
                if p is None:
                    continue
                rows.append({"arm": arm, "theta": ARMS[arm]["theta"],
                             "steepness": ARMS[arm]["steepness"],
                             "sd_ratio_target": target,
                             "ratio_actual": c["ratio"], "seed": seed,
                             "clearing_price_ct": p,
                             "round_type": c["round_type"]})
    write("R1b_ratio_sweep.csv", rows)
    fig = plot_r1(rows)
    wall = time.perf_counter() - t0
    write_manifest("R1b_ratio_sweep",
                   params={"generator": "scenario.make_scenario",
                           "n_prod": 40, "n_cons": 60, "pair_density": 0.0,
                           "sd_ratio_targets": SD_RATIOS,
                           "arms": ARMS, "preferences": PREF_OFF,
                           "execute": False},
                   seeds=SEEDS, wall_sec=wall,
                   output_files=["R1b_ratio_sweep.csv", fig.name])
    return rows, wall


def plot_r1(rows):
    """out/fig_R1_theta.png - pilot plotting style (plots.py rcParams)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GREY = "#1565c0", "#ef6c00", "#757575"
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for arm, col, mk in (("configured", BLUE, "o"), ("published", ORANGE, "s")):
        sub = [r for r in rows if r["arm"] == arm]
        targets = sorted({r["sd_ratio_target"] for r in sub})
        xs = [statistics.mean(r["ratio_actual"] for r in sub
                              if r["sd_ratio_target"] == t) for t in targets]
        ys = [statistics.mean(r["clearing_price_ct"] for r in sub
                              if r["sd_ratio_target"] == t) for t in targets]
        ax.plot(xs, ys, color=col, marker=mk, ms=4, lw=1.6,
                label="{} (theta {}, B {})".format(
                    arm, ARMS[arm]["theta"], ARMS[arm]["steepness"]))
        ax.scatter([r["ratio_actual"] for r in sub],
                   [r["clearing_price_ct"] for r in sub],
                   color=col, s=10, alpha=.45, zorder=3)
    ax.axhline(28.5, color=GREY, ls=":", lw=1)
    ax.axhline(8.0, color=GREY, ls=":", lw=1)
    ax.axvline(1.0, color="k", lw=.8, ls="--")
    ax.set_xlabel("supply / demand (actual)")
    ax.set_ylabel("clearing price [ct/kWh]")
    ax.set_title("Clearing price: configured vs published theta / B\n"
                 "(100 participants, mean over 3 seeds, points = seeds)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    out = OUT / "fig_R1_theta.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("  -> {}".format(out.name))
    return out


async def main():
    st = stack.Stack()
    try:
        print("repo SHA {}".format(campaign.SHA))
        print("R0 golden-run regression")
        rows, all_match, spread = await run_r0(st)
        width = max(len(r["quantity"]) for r in rows)
        for r in rows:
            print("  {:<{}}  expected {:>12}  measured {:>12}  "
                  "delta {:>13}  {}".format(
                      r["quantity"], width, r["expected"], r["measured"],
                      r["delta"], "MATCH" if r["match"] else "MISMATCH"))
        print("  (seller fill spread in arm (a): {:.2e})".format(spread))
        if not all_match:
            print("\nR0 FAILED - stopping before R1, as specified.")
            return 1

        print("\nR1a two-point check")
        r1a, price, d, wall_a = await run_r1a(st)
        print("  configured  {} ct/kWh".format(price["configured"]))
        print("  published   {} ct/kWh".format(price["published"]))
        print("  delta       {:+.6f} ct/kWh ({:+.4f} %)".format(
            d, 100.0 * d / price["configured"]))
        print("  part A wall time {:.2f} s".format(wall_a))

        if wall_a > 15 * 60:
            print("Part A took over 15 minutes - skipping Part B, as specified.")
            return 0
        print("\nR1b ratio sweep")
        r1b, wall_b = await run_r1b(st)
        for target in SD_RATIOS:
            per = {arm: statistics.mean(r["clearing_price_ct"] for r in r1b
                                        if r["arm"] == arm
                                        and r["sd_ratio_target"] == target)
                   for arm in ("configured", "published")}
            act = statistics.mean(r["ratio_actual"] for r in r1b
                                  if r["sd_ratio_target"] == target)
            gap = per["published"] - per["configured"]
            print("  target {:<5} actual {:.4f}  configured {:8.4f}  "
                  "published {:8.4f}  gap {:+8.4f} ct ({:+7.2f} %)".format(
                      target, act, per["configured"], per["published"], gap,
                      100 * gap / per["configured"]))
        print("  part B wall time {:.2f} s".format(wall_b))
        return 0
    finally:
        await st.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
