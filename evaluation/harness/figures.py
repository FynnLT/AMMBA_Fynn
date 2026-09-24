"""Chapter 5 figures, built from the runs and measurements under `evaluation/out`.

`plots.py` is the pilot's figure script: it reads the four synthetic campaign
CSVs of 02.09. and is left untouched, because none of its figures is reported
as a result (Chapter 5.1.5). This module reads three kinds of source instead:

* the per-run directories of the 17.-18.09. campaign -- one manifest, one slot
  CSV and one area CSV per run, five runs per cell, 45 cells, 225 runs;
* the DR2 curves that `evaluation/analysis/dr_analysis.py` wrote
  (`out/analysis/<date>-<sha>/dr2_curves.csv` and `dr2_cases.csv`);
* the two scalability measurements, the scripted gas series
  (`out/gas-*/gas_series.csv`) and the N-curve (`out/ncurve-*/ncurve.csv`).

Three rules, the same three the harness itself follows:

* **Projection, never computation.** Every quantity here is a sum, a mean, a
  median or a count over columns the services or the analysis produced. No
  price, penalty, balance or gain is recomputed, so a figure states what the
  artifact (or the replay with the artifact's own functions) answered.
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
printing and colour-vision deficiency without relying on hue. No figure has a
second y-axis: two measures of different scale are two panels.

Which figure is which in Chapter 5 (restructured by DR, 24.09.):

    module  file                          chapter
    1       fig1_preference_density       Figure 5.1
    2       fig2_reciprocity              Figure 5.2
    8       fig8_dr2_curves               Figure 5.3
    6       fig6_deadband                 Figure 5.4
    7       fig7_gamma                    Figure 5.5
    3       fig3_multiplier_formulation   Figure 5.6
    9       fig9_scalability              Figure 5.7
    4       fig4_battery_participation    not used
    5       fig5_deviation_axes           not used (see its docstring)

Usage
-----
    python figures.py                 # the seven figures of Chapter 5
    python figures.py --all           # all nine, including the two unused
    python figures.py 7 8 9           # only these
    python figures.py --refresh       # rebuild the cached per-run summary
    python figures.py --list          # what each figure shows
    python figures.py 8 --analysis ../out/analysis/20260924-0e65c00

`--analysis`, `--ncurve` and `--gas` pin a source directory; without them the
newest one under `out/` is used (by the manifest's `timestamp_utc` where there
is one, by directory name otherwise), and the directory actually read is
printed with every figure that uses it.

Figures land in `evaluation/out/figures/` as PNG at 300 dpi and as PDF. The
per-run summary is cached beside them. Cells that appear in `out/` after the
cache was written are summarised and added on the next call; pass `--refresh`
when a run that is already in the cache has been re-run.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

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
    """Per-run summaries for every cell present in `out/`, cached as JSON.

    The cache is extended, not trusted blindly: a cell or seed that is in
    `out/` but not in the cache is summarised and added. A cache written
    before the gamma cells were run would otherwise make fig7 silently drop
    them. What the cache cannot notice is a run that was re-run in place;
    that needs `--refresh`.
    """
    data = {}
    if CACHE.exists() and not refresh:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    cells = sorted(p.name.rsplit("--seed-", 1)[0] for p in OUT.glob("*--seed-0")
                   if (p / "manifest.json").exists())
    changed = False
    for cell in cells:
        runs = dict(data.get(cell, {}))
        missing = [seed for seed in SEEDS
                   if str(seed) not in runs
                   and (run_dir(cell, seed) / "manifest.json").exists()]
        if not missing:
            continue
        for seed in missing:
            runs[str(seed)] = summarise_run(cell, seed)
        data[cell] = dict(sorted(runs.items()))
        changed = True
        print(f"  summarised {cell} ({len(missing)} new of {len(runs)} runs)")
    if changed or not CACHE.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  {len(data)} cells, {sum(len(r) for r in data.values())} runs")
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


# ------------------------------------------------ analysis and measurements

#: Source directories pinned on the command line (--analysis, --ncurve, --gas).
PINNED = {}


def _timestamp(directory):
    """The manifest's `timestamp_utc`, or "" where there is no manifest."""
    path = directory / "manifest.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ""
    data = data[-1] if isinstance(data, list) else data
    return data.get("timestamp_utc") or ""


def _checks_passed(directory):
    """An analysis output counts only if its R1-R3 checks passed."""
    try:
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return bool(data.get("checks_passed"))


def source(kind, pattern, filename, accept=None):
    """The file a figure reads: pinned, or the newest match under `out/`."""
    if kind in PINNED:
        directory = Path(PINNED[kind]).resolve()
    else:
        found = [p for p in OUT.glob(pattern)
                 if (p / filename).exists() and (accept is None or accept(p))]
        if not found:
            raise FileNotFoundError(f"no {filename} under out/{pattern}")
        directory = max(found, key=lambda p: (_timestamp(p), p.name))
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(str(path))
    shown = directory.relative_to(OUT) if directory.is_relative_to(OUT) else directory
    print(f"    {kind}: {shown}")
    return path


def read_rows(path):
    """The rows of a CSV that may start with a BOM and carry `#` lines."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        lines = [line for line in handle
                 if line.strip() and not line.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def _thousands(value, _position=None):
    return f"{value:,.0f}"


def _plain(value, _position=None):
    """Tick labels on a logarithmic axis as plain numbers: 0.01 ... 100,000."""
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


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
    """The declared battery rule: how much of the market rests on it.

    Not used in Chapter 5 (24.09.); the battery-participation cells are
    reported in the text of Section 5.1. Built only with `--all` or `4`.
    """
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
    """Block 2: the deviator's own penalty over both axes of D-72.

    Not used in Chapter 5 (24.09.). `dev_ext_ct` sums the deviating household's
    externality penalty over all its area rows, so it includes the 0.094 ct per
    week it pays as a buyer on its own delivery noise in other slots; and panel
    (b), a coalition's penalty on log axes, reads as the superadditivity that
    the analysis refuted (per penalised kWh the coalitions pay 2.61-2.67 ct
    against 3.14 ct alone). Tables 5.4 and 5.9 report both axes instead. Kept
    for the record of the 18.09. draft; built only with `--all` or `5`.
    """
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
    """Gamma prices the shortfall channel, and only the shortfall channel.

    All four values of gamma on all three series (Table 5.6). The honest
    sellers are read from the withholding arm, whose deviator carries no
    shortfall, so the slot sum of shortfall penalties is theirs alone.
    """
    gammas = [1.1, 1.5, 2.0, 3.0]
    under = ("b2_short_g110", "b2_short_g150", "b2_short_g200", "b2_short_g300")
    withhold = ("b2_sell_s25", "b2_sell_s25_g150", "b2_sell_s25_g200",
                "b2_sell_s25_g300")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    # Colour follows the entity, as in the 18.09. version of this figure; the
    # legend follows the vertical order of the lines at the right edge.
    series_spec = (
        (withhold, "sum_shortfall_penalty_ct", "honest sellers, withholding arm", STYLE[1]),
        (under, "dev_short_ct", "deviator, under-delivery arm", STYLE[0]),
        (withhold, "dev_short_ct", "deviator, withholding arm", STYLE[2]),
    )
    top = 0.0
    for cells, key, label, (colour, marker, line) in series_spec:
        means, sds = points(data, cells, key)
        means = [max(m, 0.0) for m in means]     # -0.0 prints as a minus sign
        ax.errorbar(gammas, means, yerr=sds, color=colour, marker=marker,
                    linestyle=line, markersize=5, linewidth=1.6, capsize=2,
                    label=label)
        # Selective direct labels: the value at the right edge only.
        ax.text(3.08, means[-1], f"{means[-1]:,.1f}", va="center",
                fontsize=7.5, color=INK)
        top = max(top, max(m + s for m, s in zip(means, sds)))

    ax.set_xticks(gammas)
    ax.set_xticklabels([f"{g:g}" for g in gammas])
    ax.set_xlim(0.95, 3.35)
    ax.set_ylim(-0.06 * top, 1.12 * top)
    ax.set_xlabel("penalty factor $\\gamma$")
    ax.set_ylabel("shortfall penalties [ct/week]")
    ax.set_title("Who pays when $\\gamma$ rises")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig7_gamma")


#: The eight cases in the order of Table 5.3 (= the rows of Table 4.2):
#: sellers in the upper row, buyers in the lower, short side left of long.
DR2_CASES = (
    ("seller_short_withhold", 1, "short side, withholding"),
    ("seller_short_overreport", 2, "short side, over-reporting"),
    ("seller_long_withhold", 3, "long side, withholding"),
    ("seller_long_overreport", 4, "long side, over-reporting"),
    ("buyer_short_underreport", 5, "short side, under-reporting"),
    ("buyer_short_overreport", 6, "short side, over-reporting"),
    ("buyer_long_underreport", 7, "long side, under-reporting"),
    ("buyer_long_overreport", 8, "long side, over-reporting"),
)


def fig8(_data):
    """DR2: net gain over the size of the deviation, the eight cases of Table 5.3.

    Read from `dr2_curves.csv` of the analysis output: per case and grid point
    the median and the 90th percentile over the participant-slots of the
    honest reference cell. The gain is shown relative to the participant's
    truthful payoff, so that the eight panels share one scale: a deviation
    that only forgoes margin reads as a line of slope -1, an over-report that
    nothing prices as one of slope +1. The thresholds marked in the two
    short-side seller panels are the medians `dr2_cases.csv` reports.
    """
    curves_path = source("analysis", "analysis/*", "dr2_curves.csv",
                         accept=_checks_passed)
    curves = {}
    for row in read_rows(curves_path):
        curves.setdefault(row["case"], []).append(row)
    cases = {row["case"]: row for row in read_rows(curves_path.parent / "dr2_cases.csv")
             if row["type"] == "all"}

    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.0), sharex=True, sharey=True)
    for axis, (case, number, title) in zip(axes.flat, DR2_CASES):
        rows = sorted(curves[case], key=lambda r: float(r["delta_rel"]))
        delta = [float(r["delta_rel"]) for r in rows]
        axis.axhline(0.0, color=INK, linewidth=0.7, zorder=1)
        axis.plot(delta, [100 * float(r["net_rel_p90"]) for r in rows],
                  color=C2, linestyle="--", linewidth=1.4, zorder=2,
                  label="90th percentile")
        axis.plot(delta, [100 * float(r["net_rel_median"]) for r in rows],
                  color=C1, linestyle="-", linewidth=1.8, zorder=3,
                  label="median")
        axis.set_title(f"({number}) {title}\nn = {int(rows[0]['n']):,}",
                       fontsize=8)

    def mark(axis, x, text, y):
        axis.axvline(x, color=MUTED, linestyle=":", linewidth=1.0, zorder=1)
        axis.text(x + 0.012, y, text, fontsize=7, color=MUTED, va="center")

    withhold, over = cases["seller_short_withhold"], cases["seller_short_overreport"]
    mark(axes[0][0], float(withhold["delta_active_rel_median"]),
         f"penalty from {float(withhold['delta_active_rel_median']):.3f}", 44)
    mark(axes[0][1], float(over["delta_active_rel_median"]),
         f"penalty from {float(over['delta_active_rel_median']):.3f}", 44)
    mark(axes[0][1], float(over["delta_zero_rel_median"]),
         f"no gain from {float(over['delta_zero_rel_median']):.3f}", 30)
    # The long side is still rising where the tested domain ends: a lower
    # bound, which the figure says rather than leaves to the caption.
    for axis in (axes[0][3], axes[1][3]):
        axis.text(0.49, -38, "still rising\nat 0.5", fontsize=7, color=MUTED,
                  ha="right", va="center")

    axes[0][0].set_ylim(-75, 58)
    axes[0][0].set_yticks(range(-60, 60, 20))
    axes[0][0].set_xlim(0.0, 0.5)
    axes[0][0].set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0][0].set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    axes[0][0].set_ylabel("sellers\nnet gain [% of truthful payoff]")
    axes[1][0].set_ylabel("buyers\nnet gain [% of truthful payoff]")
    for axis in axes[1]:
        axis.set_xlabel("deviation [share of claim]")
    # Panel 3 has nothing above the zero line; the legend sits there.
    axes[0][2].legend(loc="upper right", fontsize=7.5)
    save(fig, "fig8_dr2_curves")


def fig9(_data):
    """DR7: on-chain gas and off-chain cycle over N, in two panels.

    Two measures of different unit and scale are two panels, never two
    y-axes on one. (a) The `n_axis` rows of the scripted gas series: one
    clearMarket per N, at the aggregates of N participants. (b) The N-curve:
    the in-process clearing cycle per slot over the number of orders, every
    repeat as an open marker and the median as the line (Table 5.10).
    """
    gas_path = source("gas", "gas-*", "gas_series.csv")
    ncurve_path = source("ncurve", "ncurve-*", "ncurve.csv")
    gas_rows = read_rows(gas_path)
    n_axis = sorted(((int(r["i"]), int(r["gas_used"])) for r in gas_rows
                     if r["series"] == "n_axis"))
    trade = [int(r["gas_used"]) for r in gas_rows if r["series"] == "trade"]

    cycles = {}
    for row in read_rows(ncurve_path):
        if row["status"] == "ok":
            cycles.setdefault(int(row["n"]), []).append(float(row["t_clear_s"]))
    ns = sorted(cycles)
    medians = [statistics.median(cycles[n]) for n in ns]
    per_order = [m / n for n, m in zip(ns, medians) if n >= 50]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))

    ax = axes[0]
    xs, ys = zip(*n_axis)
    ax.plot(xs, ys, color=C1, marker="o", linestyle="-", markersize=5,
            linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 260_000)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("participants N (aggregates anchored)")
    ax.set_ylabel("gas per clearMarket")
    ax.set_title("(a) On-chain: gas per window")
    flat = [y for x, y in n_axis if x >= 50]
    small = [(x, y) for x, y in n_axis if x < 50]
    note = f"{min(flat):,} gas" if min(flat) == max(flat) else \
        f"{min(flat):,} to {max(flat):,} gas"
    note += " for N ≥ 50"
    if small:
        note += "".join(f"\nN = {x}: {y:,}" for x, y in small)
    if trade:
        note += (f"\nfirst {len(trade)} clearings of a campaign week:"
                 f"\n{min(trade):,} to {max(trade):,}")
    ax.text(0.03, 0.40, note, transform=ax.transAxes, fontsize=7.5,
            color=INK, va="top")

    ax = axes[1]
    for n in ns:
        ax.plot([n] * len(cycles[n]), cycles[n], linestyle="none", marker="o",
                markersize=3.5, markerfacecolor="white", markeredgecolor=MUTED,
                markeredgewidth=0.8, zorder=2)
    ax.plot(ns, medians, color=C1, marker="o", linestyle="-", markersize=5,
            linewidth=1.6, zorder=3)
    rate = statistics.median(per_order)
    ax.plot([ns[0], ns[-1]], [rate * ns[0], rate * ns[-1]], color=MUTED,
            linestyle=":", linewidth=1.0, zorder=1)
    # The guide's label sits in the empty lower right, not on the data.
    ax.text(0.97, 0.05, f"dotted: {1000 * rate:.2f} ms per order \u00d7 N\n"
            f"(median for N \u2265 50)", transform=ax.transAxes, fontsize=7,
            color=MUTED, ha="right", va="bottom")
    ax.axhline(900, color=MUTED, linestyle="-.", linewidth=0.9, zorder=1)
    ax.text(ns[0], 900 * 1.35, "slot length, 900 s", fontsize=7, color=MUTED)
    for n, m in zip(ns, medians):
        if n == 100:
            ax.text(n * 1.4, m / 2.4, f"{m:,.2f} s", fontsize=7.5, color=INK,
                    ha="left", va="center")
        elif n == ns[-1]:
            ax.text(n / 1.5, m * 1.9, f"{m:,.0f} s", fontsize=7.5, color=INK,
                    ha="right", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ns[0] / 1.6, ns[-1] * 1.6)
    ax.set_ylim(min(min(v) for v in cycles.values()) / 2.5, 4000)
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.set_xlabel("orders per slot N")
    ax.set_ylabel("clearing cycle [s]")
    ax.set_title("(b) Off-chain: clearing cycle per slot")
    save(fig, "fig9_scalability")


#: number -> (function, needs the per-run summary, place in Chapter 5, what it shows)
FIGURES = {
    1: (fig1, True, "Figure 5.1", "preference density: pair energy and the fill-rate asymmetry"),
    2: (fig2, True, "Figure 5.2", "reciprocity: cleared pair energy and pair count over the mutual share"),
    3: (fig3, True, "Figure 5.6", "multiplier formulations on the reference window and the overlap variant"),
    4: (fig4, True, "not used", "battery participation: grey share and the round types it leaves"),
    5: (fig5, True, "not used", "block 2: deviator penalty over the share and coalition axes"),
    6: (fig6, True, "Figure 5.4", "the deadband: deviation extinguished, honest shortfall absorbed"),
    7: (fig7, True, "Figure 5.5", "gamma: the shortfall channel, the deviator and the honest side"),
    8: (fig8, False, "Figure 5.3", "DR2: net gain over the deviation, the eight cases of Table 5.3"),
    9: (fig9, False, "Figure 5.7", "DR7: gas per window and off-chain cycle over N, two panels"),
}

#: What a bare `python figures.py` builds: the figures of Chapter 5, in its order.
CHAPTER = (1, 2, 8, 6, 7, 3, 9)


def main(argv):
    parser = argparse.ArgumentParser(description="Chapter 5 figures.")
    parser.add_argument("numbers", nargs="*", type=int,
                        help="figure numbers (default: the seven of Chapter 5)")
    parser.add_argument("--all", action="store_true",
                        help="all figures, including the two Chapter 5 does not use")
    parser.add_argument("--list", action="store_true", help="what each figure shows")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the cached per-run summary from scratch")
    for kind in ("analysis", "ncurve", "gas"):
        parser.add_argument(f"--{kind}", metavar="DIR",
                            help=f"read this {kind} directory instead of the newest")
    args = parser.parse_args(argv)

    if args.list:
        for number, (_, _, chapter, description) in FIGURES.items():
            print(f"  {number}  {chapter:<11} {description}")
        return
    for kind in ("analysis", "ncurve", "gas"):
        if getattr(args, kind):
            PINNED[kind] = getattr(args, kind)
    wanted = sorted(FIGURES) if args.all else (args.numbers or list(CHAPTER))
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        raise SystemExit(f"no such figure: {unknown}; try --list")

    data = None
    if any(FIGURES[n][1] for n in wanted):
        print("summary:")
        data = summary(refresh=args.refresh)
    print("figures:")
    for number in wanted:
        function, _, chapter, _ = FIGURES[number]
        print(f"  fig{number} ({chapter})")
        try:
            function(data)
        except KeyError as error:
            # A cell that was never run is a missing figure, not a crash: the
            # sweep group in particular may not be present in an older `out/`.
            print(f"  fig{number} skipped -- no runs or rows for {error}")
        except FileNotFoundError as error:
            print(f"  fig{number} skipped -- {error}")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main(sys.argv[1:])
