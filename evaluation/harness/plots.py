import csv, os, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"
FIG = HARNESS_DIR.parent / "out" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
GREEN, GREY, BLUE, RED, ORANGE = "#2e7d32", "#757575", "#1565c0", "#c62828", "#ef6c00"

def load(n): return list(csv.DictReader(open(OUT / n, encoding="utf-8")))
def f(r, k): return float(r[k]) if r[k] not in ("", "None", None) else float("nan")

# ---------------------------------------------------------------- Fig 1: S/D
rows = load("B_sd_sweep.csv")
ratios = sorted({float(r["sd_ratio"]) for r in rows})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
price = [statistics.mean(f(r, "price") for r in rows if float(r["sd_ratio"]) == x) for x in ratios]
psd   = [statistics.pstdev([f(r, "price") for r in rows if float(r["sd_ratio"]) == x]) for x in ratios]
ax[0].axvspan(0.55, 1.0, color=RED, alpha=.06); ax[0].axvspan(1.0, 1.65, color=BLUE, alpha=.06)
ax[0].errorbar(ratios, price, yerr=psd, color=BLUE, marker="o", ms=3, lw=1.4, capsize=2)
ax[0].axhline(28.5, color=GREY, ls=":", lw=1); ax[0].axhline(8.0, color=GREY, ls=":", lw=1)
ax[0].axvline(1.0, color="k", lw=.8, ls="--")
ax[0].text(0.72, 26.5, "supply-limited", color=RED, fontsize=8)
ax[0].text(1.15, 26.5, "demand-limited", color=BLUE, fontsize=8)
ax[0].set_xlabel("supply / demand"); ax[0].set_ylabel("clearing price [ct/kWh]")
ax[0].set_title("Clearing price over the S/D ratio\n(100 participants, mean ± sd over 5 seeds)", fontsize=9)
sf = [statistics.mean(f(r, "seller_fill_mean") for r in rows if float(r["sd_ratio"]) == x) for x in ratios]
bf = [statistics.mean(f(r, "buyer_fill_mean") for r in rows if float(r["sd_ratio"]) == x) for x in ratios]
ax[1].plot(ratios, sf, color=GREEN, marker="o", ms=3, lw=1.4, label="sellers")
ax[1].plot(ratios, bf, color=ORANGE, marker="s", ms=3, lw=1.4, label="buyers")
ax[1].axvline(1.0, color="k", lw=.8, ls="--"); ax[1].legend(frameon=False)
ax[1].set_xlabel("supply / demand"); ax[1].set_ylabel("mean fill rate")
ax[1].set_title("Who gets rationed", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_sd_sweep.png"); plt.close(fig)

# ------------------------------------------------------- Fig 2: pair density
rows = load("C1_pairs.csv")
dens = sorted({float(r["pair_density"]) for r in rows})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for order, col, mk in (("preferences_first", GREEN, "o"), ("pro_rata_first", GREY, "s")):
    m = [statistics.mean([f(r, "fill_matched") for r in rows
                          if float(r["pair_density"]) == d and r["order"] == order
                          and r["fill_matched"] not in ("", "None")] or [float("nan")]) for d in dens]
    u = [statistics.mean([f(r, "fill_unmatched") for r in rows
                          if float(r["pair_density"]) == d and r["order"] == order
                          and r["fill_unmatched"] not in ("", "None")] or [float("nan")]) for d in dens]
    ax[0].plot(dens, m, color=col, marker=mk, ms=4, lw=1.6,
               label=f"{order} · matched")
    ax[0].plot(dens, u, color=col, marker=mk, ms=4, lw=1.6, ls="--", alpha=.7,
               label=f"{order} · unmatched")
ax[0].axhline(0.8, color=RED, lw=.8, ls=":")
ax[0].text(0.02, 0.806, "pro-rata baseline 80 %", color=RED, fontsize=7.5)
ax[0].set_xlabel("share of participants in a mutual pair")
ax[0].set_ylabel("mean seller fill rate")
ax[0].set_title("Priority advantage shrinks as pairing spreads", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
spread = [statistics.mean([f(r, "fill_sd") for r in rows
                           if float(r["pair_density"]) == d and r["order"] == "preferences_first"])
          for d in dens]
trad = [statistics.mean([f(r, "traded_total") for r in rows
                         if float(r["pair_density"]) == d and r["order"] == "preferences_first"])
        for d in dens]
ax[1].bar([str(d) for d in dens], spread, color=GREEN, alpha=.8)
ax[1].set_xlabel("pair density"); ax[1].set_ylabel("sd of seller fill rates")
ax2 = ax[1].twinx(); ax2.plot([str(d) for d in dens], trad, color=BLUE, marker="o", ms=4)
ax2.set_ylabel("traded volume [kWh]", color=BLUE); ax2.set_ylim(0, max(trad)*1.3); ax2.grid(False)
ax[1].set_title("Dispersion rises, volume does not move", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_pair_density.png"); plt.close(fig)

# --------------------------------------------------------- Fig 3: multipliers
rows = load("C2_multipliers.csv")
gm = sorted({float(r["green_multiplier"]) for r in rows})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for (mode, sides), col, ls in ((("multiplicative", "seller"), BLUE, "-"),
                               (("multiplicative", "both"), BLUE, "--"),
                               (("additive", "seller"), ORANGE, "-"),
                               (("additive", "both"), ORANGE, "--")):
    sub = [r for r in rows if r["mode"] == mode and r["sides"] == sides]
    sub.sort(key=lambda r: float(r["green_multiplier"]))
    x = [float(r["green_multiplier"]) for r in sub]
    ax[0].plot(x, [f(r, "pool_surplus_ct") for r in sub], color=col, ls=ls, lw=1.6,
               label=f"{mode[:4]} · {sides}")
    ax[1].plot(x, [f(r, "grey_final") for r in sub], color=col, ls=ls, lw=1.6)
    ax[2].plot(x, [f(r, "buyer_rate") for r in sub], color=col, ls=ls, lw=1.6,
               label=f"{mode[:4]} · {sides}")
for a in ax: a.axvline(0.037931, color=RED, lw=.9, ls=":")
ax[0].axhline(0, color="k", lw=.8)
ax[0].text(0.0395, 3.4, "crossover\n0.0379", color=RED, fontsize=7)
ax[0].set_xlabel("green_multiplier"); ax[0].set_ylabel("pool surplus [ct]")
ax[0].set_title("Zero-sum violation", fontsize=9); ax[0].legend(frameon=False, fontsize=7)
ax[1].axhline(15.147225, color=GREY, ls=":", lw=1)
ax[1].set_xlabel("green_multiplier"); ax[1].set_ylabel("grey seller rate [ct/kWh]")
ax[1].set_title("What grey producers receive", fontsize=9)
ax[2].axhline(15.147225, color=GREY, ls=":", lw=1)
ax[2].set_xlabel("green_multiplier"); ax[2].set_ylabel("buyer rate [ct/kWh]")
ax[2].set_title("What buyers pay", fontsize=9); ax[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_multipliers.png"); plt.close(fig)

# ------------------------------------------------------- Fig 4: withholding
rows = load("D1_withholding.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for v, col, lab in ((0.0, GREY, "v = 0 (energy wasted)"),
                    (8.0, BLUE, "v = 8.0 ct (feed-in tariff)"),
                    (28.5, GREEN, "v = 28.5 ct (retail, self-consumption)")):
    sub = sorted([r for r in rows if abs(f(r, "v_ct_per_kwh") - v) < 1e-9],
                 key=lambda r: f(r, "withheld_share"))
    ax[0].plot([f(r, "withheld_share") for r in sub], [f(r, "gain_ct") for r in sub],
               color=col, lw=1.6, label=lab)
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xlabel("withheld share of own capacity"); ax[0].set_ylabel("gain without penalty [ct]")
ax[0].set_title("Withholding only pays if the energy\nhas outside value", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
sub = sorted([r for r in rows if abs(f(r, "v_ct_per_kwh") - 28.5) < 1e-9],
             key=lambda r: f(r, "withheld_share"))
x = [f(r, "withheld_share") for r in sub]
ax[1].plot(x, [f(r, "gain_ct") for r in sub], color=GREEN, lw=1.6, label="gain (v = retail)")
ax[1].plot(x, [f(r, "net_meter_ct") for r in sub], color=RED, lw=1.8, ls="--",
           label="net · meter-only penalty (= gain)")
ax[1].plot(x, [f(r, "net_with_penalty_ct") for r in sub], color=BLUE, lw=1.8,
           label="net · with deliverable channel")
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xlabel("withheld share of own capacity"); ax[1].set_ylabel("net gain [ct]")
ax[1].set_title("The penalty only bites with a second\nmeasurement channel", fontsize=9)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_withholding.png"); plt.close(fig)

# --------------------------------------------------------------- Fig 5: eta
rows = load("D3_eta.csv")
x = [f(r, "eta_rel")*100 for r in rows]
fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
ax1.plot(x, [f(r, "false_positive_share")*100 for r in rows], color=RED, marker="o",
         ms=3.5, lw=1.7, label="honest sellers penalised [%]")
ax1.set_xlabel("η as share of the traded quantity [%]")
ax1.set_ylabel("honest sellers wrongly penalised [%]", color=RED)
ax1.tick_params(axis="y", labelcolor=RED)
ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(x, [f(r, "free_gain_eur_per_year") for r in rows], color=BLUE, marker="s",
         ms=3.5, lw=1.7, label="undeterred withholding [€/a]")
ax2.set_ylabel("free withholding per seller [€/year]", color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax1.axvline(1.0, color=GREY, ls=":", lw=1)
ax1.text(1.08, 40, "meter class B\nσ ≈ 1 %", fontsize=7, color=GREY)
ax1.set_title("η trades false positives against free withholding", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_eta.png"); plt.close(fig)

# --------------------------------------------------- Fig 6: non-additivity
rows = load("D4_multiagent.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for w, col in ((0.2, BLUE), (0.5, ORANGE), (1.0, RED)):
    sub = sorted([r for r in rows if abs(f(r, "withheld_each_kwh") - w) < 1e-9],
                 key=lambda r: int(r["n_deviators"]))
    ax.plot([int(r["n_deviators"]) for r in sub],
            [f(r, "over_collection_pct") for r in sub], color=col, marker="o",
            ms=4, lw=1.6, label=f"each withholds {w} kWh")
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("number of simultaneous deviators")
ax.set_ylabel("Σ individual penalties vs joint [%]")
ax.set_title("Individually computed penalties under-collect\nwhen deviators coincide", fontsize=9)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_multiagent.png"); plt.close(fig)

# ------------------------------------------------------ Fig 7: cost/latency
lat = load("E_latency.csv"); gas = load("F_gas.csv")
ns = sorted({int(r["participants"]) for r in lat})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
med = [statistics.median([f(r, "total_ms") for r in lat if int(r["participants"]) == n]) for n in ns]
clr = [statistics.median([f(r, "clearing_ms") for r in lat if int(r["participants"]) == n]) for n in ns]
exe = [statistics.median([f(r, "execution_ms") for r in lat if int(r["participants"]) == n]) for n in ns]
ax[0].plot(ns, clr, color=BLUE, marker="o", ms=4, lw=1.6, label="clearing")
ax[0].plot(ns, exe, color=ORANGE, marker="s", ms=4, lw=1.6, label="execution")
ax[0].plot(ns, med, color="k", marker="^", ms=4, lw=1.4, label="total")
ax[0].set_xlabel("participants"); ax[0].set_ylabel("median wall time [ms]")
ax[0].set_title("Off-chain cycle scales linearly\n(900 000 ms slot budget)", fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5)
cm = sorted([r for r in gas if r["op"] == "clearMarket"], key=lambda r: int(r["participants"]))
ax[1].plot([int(r["participants"]) for r in cm], [int(r["gas"]) for r in cm],
           color=GREEN, marker="o", ms=4, lw=1.6)
ax[1].set_xscale("log")
ax[1].set_ylim(0, 260000)
ax[1].set_xlabel("participants (log scale)"); ax[1].set_ylabel("gas used by clearMarket")
ax[1].set_title("On-chain cost is O(1):\n216 242 – 216 314 gas over 6 … 3 200", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_feasibility.png"); plt.close(fig)

print("figures:", sorted(os.listdir(FIG)))
