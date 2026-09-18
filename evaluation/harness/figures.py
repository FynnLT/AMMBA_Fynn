"""Chapter 5 figures, built from the campaign runs under `evaluation/out`.

`plots.py` is the pilot's figure script: it reads the four synthetic campaign
CSVs of 02.09. and is left untouched, because none of its figures is reported
as a result (Chapter 5.1.5). This module reads the per-run directories of the
17.-18.09. campaign instead -- one manifest, one slot CSV and one area CSV per
run, five runs per cell, 43 cells.

Three rules, the same three the harness itself follows:

* **Projection, never computation.** Every quantity here is a sum, a mean or a
  count over columns the services produced. No price, penalty or balance is
  recomputed, so a figure states what the artifact answered.
* **Block 2 is read from the deviator's own area rows**, never from
  `penalty_pool_ct`. That column is dominated by the buyer-side noise floor,
  against which one deviator's signal disappears; the deviator ids are in each
  manifest and `<run-id>_areas.csv` carries the penalty per area.
* **A replicate is a week, not a noise draw.** The five seeds of a cell are
  five different synthetic weeks (`dataset_extension_seed = 4242 + seed`), so
  every figure carries the spread across them, and where a comparison is
  possible within a seed the figure says so in its caption.

Colours are a validated categorical set (blue / red / purple) and every series
also carries its own marker and line style, so the figures survive greyscale
printing and colour-vision deficiency without relying on hue.

Usage
-----
    python figures.py                 # all figures
    python figures.py 3 5             # only figures 3 and 5
    python figures.py --refresh       # rebuild the cached per-run summary
    python figures.py --list          # what each figure shows

Figures land in `evaluation/out/figures/` as PNG at 300 dpi and as PDF. The
per-run summary is cached beside them; delete it or pass `--refresh` after a
new campaign run.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARNESS_DIR = Path(__file__).resolve().parent
OUT = HARNESS_DIR.parent / "out"
FIG = OUT / "figures"
CACHE = FIG / "summary.json"

SEEDS = (0, 1, 2, 3, 4)

#: Below this a penalty in ct is six-decimal rounding residue on the forecast
#: against the trade record, amplified because the externality is priced on
#: the round's whole traded volume. It is not a deviation.
DUST_CT = 1e-3

# --------------------------------------------------------------- appearance

C1, C2, C3 = "#2166ac", "#b2182b", "#7b3294"   # validated categorical set
INK, MUTED, GRID = "#222222", "#6b6b6b", "#cccccc"
STYLE = ((C1, "o", "-"), (C2, "s", "--"), (C3, "^", "-."))

plt.rcParams.update({
    # Arial on the machine the thesis is written on, DejaVu as the fallback
    # anywhere else. A "findfont: Font family 'Arial' not found" on a run is
    # therefore a real signal -- the figures of that run are not set in the
    # font of the surrounding text -- and not warning noise to ignore.
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,          # the grid belongs behind the marks
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "figure.dpi": 300,
})


# ------------------------------------------------------------- reading runs

def run_dir(cell, seed):
    return OUT / f"{cell}--seed-{seed}"


def _float(row, key):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[-1] if isinstance(data, list) else data


def summarise_run(cell, seed):
    """Everything the figures need from one run, in one pass over its files."""
    directory = run_dir(cell, seed)
    manifest = _manifest(directory / "manifest.json")
    config = manifest.get("config") or {}
    census = (manifest.get("checks") or {}).get("round_type_census") or {}
    deviation = manifest.get("deviation") or {}
    deviators = set(deviation.get("deviators") or [])

    out = {
        "cell": cell, "seed": seed,
        "extension_seed": manifest.get("dataset_extension_seed"),
        "participation": config.get("participation"),
        "named_share": config.get("named_share"),
        "mutual_share": config.get("mutual_share"),
        "gamma": config.get("gamma"),
        "eta_relative": config.get("eta_relative"),
        "share": (config.get("deviation") or {}).get("share"),
        "k": (config.get("deviation") or {}).get("k"),
        "arm": (config.get("deviation") or {}).get("arm"),
        "n_battery": manifest.get("n_battery_areas"),
        "supply_limited": census.get("SUPPLY_LIMITED", 0),
        "demand_limited": census.get("DEMAND_LIMITED", 0),
        "no_trade": census.get("no_trade", 0),
        "repo_sha": manifest.get("repo_sha"),
    }

    sums = dict.fromkeys(
        ("pairs_posted", "pairs_unpostable", "n_pairs", "pair_kwh",
         "levy_collected_ct", "bonus_paid_ct", "pool_surplus_ct",
         "green_alloc_kwh", "grey_alloc_kwh", "traded_kwh",
         "withheld_kwh", "underreported_kwh", "penalty_pool_ct",
         "shortfall_penalty_ct", "externality_penalty_ct"), 0.0)
    prices, grey_rate, buyer_rate = [], [], []
    slots_traded = slots_executed = 0
    max_balance = 0.0

    with open(directory / f"{cell}--seed-{seed}_slots.csv",
              newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in sums:
                value = _float(row, key)
                if value is not None:
                    sums[key] += value
            price = _float(row, "clearing_price_ct")
            if price is not None:
                slots_traded += 1
                prices.append(price)
            for target, key in ((grey_rate, "grey_final_ct"),
                                (buyer_rate, "buyer_final_ct")):
                value = _float(row, key)
                if value is not None:
                    target.append(value)
            if _float(row, "total_penalties_ct") is not None:
                slots_executed += 1
            # The +-0 test reads the balance only where there was something to
            # distribute: with no harmed counterparty `redistribution` returns
            # the pool unchanged as the balance, which is its documented empty
            # branch and not a violated invariant.
            balance, harmed = _float(row, "budget_balance_ct"), _float(row, "n_harmed")
            if balance is not None and harmed:
                max_balance = max(max_balance, abs(balance))

    out.update({f"sum_{k}": v for k, v in sums.items()})
    out.update({
        "slots_traded": slots_traded,
        "slots_executed": slots_executed,
        "price_mean": statistics.mean(prices) if prices else None,
        "grey_final_mean": statistics.mean(grey_rate) if grey_rate else None,
        "buyer_final_mean": statistics.mean(buyer_rate) if buyer_rate else None,
        "max_balance_ct": max_balance,
    })

    # Per-area: the fill-rate asymmetry of block 1 and the deviator's own
    # penalties in block 2. Both are invisible at slot level -- the first
    # because the round-level traded quantity does not change with the
    # allocation order, the second because of the noise floor.
    fills = {("seller", True): [], ("seller", False): [],
             ("buyer", True): [], ("buyer", False): []}
    dev = {"ext_ct": 0.0, "short_ct": 0.0, "ext_slots": 0, "short_slots": 0}
    with open(directory / f"{cell}--seed-{seed}_areas.csv",
              newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fill = _float(row, "fill_rate")
            if fill is not None:
                key = (row["side"], row["preference_matched"] == "True")
                if key in fills:
                    fills[key].append(fill)
            if row["area_uuid"] in deviators:
                externality = _float(row, "externality_penalty_ct") or 0.0
                shortfall = _float(row, "shortfall_penalty_ct") or 0.0
                dev["ext_ct"] += externality
                dev["short_ct"] += shortfall
                dev["ext_slots"] += externality > DUST_CT
                dev["short_slots"] += shortfall > DUST_CT

    for (side, matched), values in fills.items():
        name = f"fill_{side}_{'matched' if matched else 'unmatched'}"
        out[name] = statistics.mean(values) if values else None
        out[f"n_{name}"] = len(values)
    out.update({f"dev_{k}": v for k, v in dev.items()})
    return out


def summary(refresh=False):
    """Per-run summaries for every cell present in `out/`, cached as JSON."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    cells = sorted(p.name.rsplit("--seed-", 1)[0] for p in OUT.glob("*--seed-0")
                   if (p / "manifest.json").exists())
    data = {}
    for cell in cells:
        runs = {}
        for seed in SEEDS:
            if (run_dir(cell, seed) / "manifest.json").exists():
                runs[str(seed)] = summarise_run(cell, seed)
        if runs:
            data[cell] = runs
        print(f"  summarised {cell} ({len(runs)} runs)")
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


# ------------------------------------------------------------- aggregation

def series(data, cell, key):
    """One value per seed, in seed order, skipping runs that lack the key."""
    return [run[key] for _, run in sorted(data[cell].items())
            if run.get(key) is not None]


def mean_sd(data, cell, key):
    values = series(data, cell, key)
    if not values:
        return float("nan"), 0.0
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def points(data, cells, key):
    """(means, sds) over a list of cells, for one quantity."""
    pairs = [mean_sd(data, cell, key) for cell in cells]
    return [m for m, _ in pairs], [s for _, s in pairs]


def clipped(means, sds, floor=1e-3):
    """Asymmetric error bars that stay positive, for logarithmic axes."""
    lower = [min(sd, max(m - floor, 0.0)) for m, sd in zip(means, sds)]
    return [lower, list(sds)]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _bars(ax, groups, labels, values, errors, ylabel, width=0.36):
    """Grouped bars: one group per x position, one bar per label."""
    positions = range(len(groups))
    offset = -width / 2 * (len(labels) - 1)
    for index, label in enumerate(labels):
        colour, _, _ = STYLE[index]
        xs = [p + offset + index * width for p in positions]
        ax.bar(xs, values[index], width * 0.92, yerr=errors[index], label=label,
               color=colour, edgecolor="white", linewidth=0.6,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2},
               hatch=("" if index == 0 else ("//" if index == 1 else "..")))
    ax.set_xticks(list(positions))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)


# ----------------------------------------------------------------- figures

def fig1(data):
    """Preference density: what the allocation order changes, and where."""
    densities = ("0.20", "0.50", "0.80")
    prefs = [f"prefs_first_d0{d}" for d in ("20", "50", "80")]
    pro = [f"pro_rata_first_d0{d}" for d in ("20", "50", "80")]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    ax = axes[0]
    for cells, label, (colour, marker, line) in ((prefs, "preferences first", STYLE[0]),
                                                 (pro, "pro rata first", STYLE[1])):
        means, sds = points(data, cells, "sum_pair_kwh")
        ax.errorbar(densities, means, yerr=sds, color=colour, marker=marker,
                    linestyle=line, markersize=5, linewidth=1.6, capsize=2,
                    label=label)
    ax.set_xlabel("named share"); ax.set_ylabel("cleared pair energy [kWh/week]")
    ax.set_title("(a) Energy allocated to preferred pairs")
    ax.legend(loc="upper left")

    for axis, cells, title in ((axes[1], prefs, "(b) preferences first"),
                               (axes[2], pro, "(c) pro rata first")):
        matched, matched_sd = points(data, cells, "fill_seller_matched")
        other, other_sd = points(data, cells, "fill_seller_unmatched")
        _bars(axis, densities, ("matched", "unmatched"),
              [matched, other], [matched_sd, other_sd], "mean seller fill rate")
        axis.set_xlabel("named share")
        axis.set_title(title)
        # Fill rates are bounded by 1, so the headroom is free and the legend
        # never has to sit on a bar.
        axis.set_ylim(0, 1.18)
    axes[1].legend(loc="upper center", ncol=2, columnspacing=1.2)
    save(fig, "fig1_preference_density")


def fig2(data):
    """Reciprocity: only a mutual nomination can be given priority."""
    cells = ("prefs_first_d050_m025", "prefs_first_d050", "prefs_first_d050_m075")
    groups = ("0.25", "0.50", "0.75")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for axis, key, ylabel, title in (
            (axes[0], "sum_pair_kwh", "cleared pair energy [kWh/week]",
             "(a) Energy reaching preferred pairs"),
            (axes[1], "sum_n_pairs", "cleared pairs [count/week]",
             "(b) Pairs cleared")):
        means, sds = points(data, cells, key)
        axis.bar(groups, means, 0.55, yerr=sds, color=C1, edgecolor="white",
                 linewidth=0.6,
                 error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
        axis.set_xlabel("mutual share"); axis.set_ylabel(ylabel)
        axis.set_title(title)
    save(fig, "fig2_reciprocity")


def fig3(data):
    """The two multiplier formulations, on both supply mixes."""
    keys = ("sum_levy_collected_ct", "sum_bonus_paid_ct", "sum_pool_surplus_ct")
    groups = ("levy collected", "bonus paid", "pool surplus")
    # One y-axis across both panels: the two supply mixes are measured in the
    # same unit, and separate scales would make the smaller one look like the
    # larger one's equal.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for axis, cells, title in (
            (axes[0], ("multipliers_on", "multipliers_on_additive"),
             "(a) Reference window: no slot carries both types"),
            (axes[1], ("mult_overlap_multiplicative", "mult_overlap_additive"),
             "(b) Overlap variant: green and grey in one slot")):
        values, errors = [], []
        for cell in cells:
            means, sds = zip(*(mean_sd(data, cell, key) for key in keys))
            values.append(list(means)); errors.append(list(sds))
        _bars(axis, groups, ("multiplicative", "additive"), values, errors,
              "ct per week")
        axis.set_title(title, fontsize=8.5)
        # A bar of height zero is invisible, and "zero" is the result in four
        # of the six columns of panel (a): label it rather than leave a gap.
        for index, row in enumerate(values):
            for position, value in enumerate(row):
                if value < 1.0:
                    axis.text(position - 0.18 + index * 0.36, 40, "0",
                              ha="center", fontsize=7.5, color=MUTED)
    axes[0].set_ylim(0, 3100)
    axes[1].set_ylabel("")          # one shared scale, one label
    axes[0].legend(loc="upper center", ncol=2, columnspacing=1.2)
    save(fig, "fig3_multiplier_formulation")


def fig4(data):
    """The declared battery rule: how much of the market rests on it."""
    cells = [f"green_share_{p:03d}" for p in (0, 25, 50, 75, 100)]
    labels = ("0.00", "0.25", "0.50", "0.75", "1.00")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    grey, grey_sd = points(data, cells, "sum_grey_alloc_kwh")
    green, _ = points(data, cells, "sum_green_alloc_kwh")
    shares = [100 * g / (g + n) if (g + n) else 0.0 for g, n in zip(grey, green)]
    axes[0].plot(labels, shares, color=C1, marker="o", markersize=5, linewidth=1.6)
    axes[0].set_xlabel("battery participation")
    axes[0].set_ylabel("grey share of allocated energy [%]")
    axes[0].set_title("(a) Share supplied by the declared rule")
    axes[0].set_ylim(-4, 62)
    axes[0].annotate("reference case", xy=(1, shares[1]), xytext=(1.1, 8),
                     color=MUTED, fontsize=8, ha="left",
                     arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.8})

    for index, (key, label) in enumerate((("supply_limited", "supply-limited"),
                                          ("demand_limited", "demand-limited"))):
        colour, marker, line = STYLE[index]
        means, sds = points(data, cells, key)
        axes[1].errorbar(labels, means, yerr=sds, color=colour, marker=marker,
                         linestyle=line, markersize=5, linewidth=1.6, capsize=2,
                         label=label)
    axes[1].set_xlabel("battery participation")
    axes[1].set_ylabel("slots per week")
    axes[1].set_title("(b) Round types available to block 2")
    axes[1].legend(loc="center right")
    save(fig, "fig4_battery_participation")


def fig5(data):
    """Block 2: the deviator's own penalty over both axes of D-72."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    seller_shares = (("b2_sell_s10", 0.10), ("b2_sell_s20", 0.20),
                     ("b2_sell_s25", 0.25), ("b2_sell_s50", 0.50))
    buyer_shares = (("b2_buy_s10", 0.10), ("b2_buy_s25", 0.25),
                    ("b2_buy_s50", 0.50))
    for index, (rows, label) in enumerate(((seller_shares, "seller withholds"),
                                           (buyer_shares, "buyer under-reports"))):
        colour, marker, line = STYLE[index]
        cells = [c for c, _ in rows if c in data]
        xs = [x for c, x in rows if c in data]
        means, sds = points(data, cells, "dev_ext_ct")
        axes[0].errorbar(xs, means, yerr=clipped(means, sds), color=colour,
                         marker=marker, linestyle=line, markersize=5,
                         linewidth=1.6, capsize=2, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks([0.10, 0.20, 0.25, 0.50])
    axes[0].set_xticklabels(["0.10", "0.20", "0.25", "0.50"])
    axes[0].set_xlabel("deviation share"); axes[0].set_ylabel("deviator penalty [ct/week]")
    axes[0].set_title("(a) One deviator, rising share")
    axes[0].legend(loc="upper left")
    # The lowest seller point is not a small penalty but an arithmetic zero:
    # W_sell = q(s - eta) vanishes at s = eta = 0.10, and what remains is
    # rounding residue. Saying so on the figure keeps it from being read as a
    # measurement.
    axes[0].annotate("$s = \\eta$: nothing is withheld",
                     xy=(0.105, mean_sd(data, "b2_sell_s10", "dev_ext_ct")[0]),
                     xytext=(0.135, 4e-3), color=MUTED, fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "color": MUTED,
                                 "linewidth": 0.8})

    sizes = [1, 2, 5, 10]
    for index, (prefix, reference, label) in enumerate(
            (("b2_sell", "b2_sell_s25", "seller withholds"),
             ("b2_buy", "b2_buy_s25", "buyer under-reports"))):
        colour, marker, line = STYLE[index]
        cells = [reference] + [f"{prefix}_k{k}" for k in ("02", "05", "10")]
        means, sds = points(data, cells, "dev_ext_ct")
        axes[1].errorbar(sizes, means, yerr=clipped(means, sds), color=colour,
                         marker=marker, linestyle=line, markersize=5,
                         linewidth=1.6, capsize=2, label=label)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xticks(sizes); axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].set_xlabel("simultaneous deviators k")
    axes[1].set_ylabel("penalty of the coalition [ct/week]")
    axes[1].set_title("(b) Coalition size at share 0.25")
    save(fig, "fig5_deviation_axes")


def fig6(data):
    """The deadband: what it does to the deviation, and to the honest side."""
    cells = ("b2_eta000", "b2_eta005", "b2_sell_s25", "b2_eta015", "b2_eta025")
    etas = [0.00, 0.05, 0.10, 0.15, 0.25]
    ticks = ["0.00", "0.05", "0.10", "0.15", "0.25"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))

    means, sds = points(data, cells, "sum_withheld_kwh")
    axes[0].errorbar(etas, means, yerr=sds, color=C1, marker="o", linestyle="-",
                     markersize=5, linewidth=1.6, capsize=2)
    axes[0].set_xlim(-0.02, 0.30)
    axes[0].set_xticks(etas); axes[0].set_xticklabels(ticks)
    axes[0].set_xlabel("deadband $\\eta$ (relative)")
    axes[0].set_ylabel("withheld energy [kWh/week]")
    axes[0].set_title("(a) The deviation, at a withholding share of 0.25")
    axes[0].annotate("$\\eta = s$", xy=(0.25, 0.0), xytext=(0.238, max(means) * 0.30),
                     color=MUTED, fontsize=8, ha="center",
                     arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.8})

    # The last point is exactly zero and a logarithmic axis cannot carry it.
    # Plotted at the axis floor as an open marker and labelled, rather than
    # floored to a small positive number that would read as a measurement.
    floor, floor_sd = points(data, cells, "sum_shortfall_penalty_ct")
    finite = [(x, v, s) for x, v, s in zip(etas, floor, floor_sd) if v > 0]
    axes[1].errorbar([x for x, _, _ in finite], [v for _, v, _ in finite],
                     yerr=[s for _, _, s in finite], color=C2, marker="s",
                     linestyle="--", markersize=5, linewidth=1.6, capsize=2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-1, 1e4)
    axes[1].plot([0.25], [1e-1], color=C2, marker="s", markerfacecolor="white",
                 markeredgewidth=1.2, markersize=5, clip_on=False)
    axes[1].annotate("exactly 0", xy=(0.25, 1e-1), xytext=(0.183, 3.2e-1),
                     color=MUTED, fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.8})
    axes[1].set_xlim(-0.02, 0.30)
    axes[1].set_xticks(etas); axes[1].set_xticklabels(ticks)
    axes[1].set_xlabel("deadband $\\eta$ (relative)")
    axes[1].set_ylabel("shortfall penalties [ct/week]")
    axes[1].set_title("(b) The honest side: accidental shortfall")
    save(fig, "fig6_deadband")


def fig7(data):
    """Gamma prices the shortfall channel, and only the shortfall channel."""
    cells = ("b2_short_g110", "b2_short_g150", "b2_short_g200", "b2_short_g300")
    gammas = [1.1, 1.5, 2.0, 3.0]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))

    means, sds = points(data, cells, "dev_short_ct")
    ax.errorbar(gammas, means, yerr=sds, color=C1, marker="o", linestyle="-",
                markersize=5, linewidth=1.6, capsize=2,
                label="deviator, under-delivery arm")

    control = ("b2_sell_s25", "b2_sell_s25_g300")
    control_means, control_sds = points(data, control, "sum_shortfall_penalty_ct")
    ax.errorbar([1.1, 3.0], control_means, yerr=control_sds, color=C2,
                marker="s", linestyle="--", markersize=5, linewidth=1.6,
                capsize=2, label="honest sellers, withholding arm")

    dev_withheld, _ = points(data, control, "dev_short_ct")
    ax.plot([1.1, 3.0], [max(v, 0.0) for v in dev_withheld], color=C3,
            marker="^", linestyle=":", markersize=5, linewidth=1.6,
            label="deviator, withholding arm")

    ax.set_xticks(gammas)
    ax.set_xticklabels([f"{g:g}" for g in gammas])
    ax.set_ylim(-12, 215)
    ax.set_xlabel("penalty factor $\\gamma$")
    ax.set_ylabel("shortfall penalties [ct/week]")
    ax.set_title("Who pays when $\\gamma$ rises")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig7_gamma")


FIGURES = {
    1: (fig1, "preference density: pair energy and the fill-rate asymmetry"),
    2: (fig2, "reciprocity: cleared pair energy and pair count over the mutual share"),
    3: (fig3, "multiplier formulations on the reference window and the overlap variant"),
    4: (fig4, "battery participation: grey share and the round types it leaves"),
    5: (fig5, "block 2: deviator penalty over the share and coalition axes"),
    6: (fig6, "the deadband: deviation extinguished, honest shortfall absorbed"),
    7: (fig7, "gamma: the shortfall channel, the deviator and the honest side"),
}


def main(argv):
    if "--list" in argv:
        for number, (_, description) in FIGURES.items():
            print(f"  {number}  {description}")
        return
    wanted = [int(a) for a in argv if a.isdigit()] or sorted(FIGURES)
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        raise SystemExit(f"no such figure: {unknown}; try --list")
    print("summary:")
    data = summary(refresh="--refresh" in argv)
    print("figures:")
    for number in wanted:
        function, _ = FIGURES[number]
        try:
            function(data)
        except KeyError as error:
            # A cell that was never run is a missing figure, not a crash: the
            # sweep group in particular may not be present in an older `out/`.
            print(f"  fig{number} skipped -- no runs for cell {error}")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main(sys.argv[1:])
