"""Fitting theta and B: a coarse-to-fine grid over `objective.objective`.

**A grid, not a solver, and the argument is this thesis' own.** Every figure in
Chapter 5 is cited by `config + seed + SHA`, and a result that cannot be
regenerated from those three is not evidence. A numerical optimiser breaks that
chain quietly: L-BFGS-B or Nelder-Mead reaches a slightly different point when
SciPy changes its line search or its convergence test, so the fitted theta
becomes a function of a library version that no manifest records. A grid has no
such freedom. Its resolution is stated, its bounds are stated, it visits its
points in a fixed order, and re-running it on the same CSV six months from now
on a different machine returns the same pair or the input changed.

The cost of that choice is honest and small: the grid resolves theta and B to
the fine step, not to machine precision. For a parameter that goes into a
configuration file as a two-decimal number, and whose objective is flat enough
near the optimum that the sweep in `sweep()` matters more than the third
decimal, that is the right trade.

**The fine window is a function of the coarse step, never a fixed width.**
Widening the box without that coupling would make the fit coarser while
looking like it searched harder: the coarse pass would step further, the
refinement would keep looking in the same small neighbourhood, and an optimum
one coarse step away would be missed. `fit` therefore refines within
+/-(box width / (coarse - 1)) of the coarse winner, so the two passes scale
together and the reported resolution follows from `bounds`, `coarse` and
`fine` alone.

**A result on the edge of the box is not a fitted value**, and `FitResult`
says so with `at_bound`. At a large alpha the penalty term keeps rewarding
steepness past anything the data supports, and the optimum walks to whatever
the upper bound happens to be; reporting that number as a calibrated B would
be reporting the search box.

Run it on a slot CSV:

    python fit.py ../out/<run-id>/<run-id>_slots.csv --run-id calib-t19

which writes `theta_B.json` and `alpha_sweep.csv` under
`evaluation/out/<run-id>/`. It writes nothing into `configuration.yaml`:
setting the community parameters is a decision, and this module produces the
evidence for it, not the decision.
"""
import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from objective import band, load_slots, objective, sigmoid_price_unclamped

CALIBRATION_DIR = Path(__file__).resolve().parent
OUT = CALIBRATION_DIR.parent / "out"

# D-74: the penalty weight is reported along, not chosen silently. These are
# the points the sweep visits by default -- 0.0 for the degenerate case the
# penalty exists to rule out, then a spread dense enough around 0.5 to show
# where the steepness turns over.
DEFAULT_ALPHAS = (0.0, 0.25, 0.4, 0.5, 1.0, 2.0)

# The search box the paper uses: theta is a supply/demand ratio and B a
# steepness in the same units, both bounded below by 0. Stated here so the
# manifest can carry them, and read together with DEFAULT_COARSE -- the two
# fix the resolution between them, and changing one without the other changes
# how finely the fit resolves.
DEFAULT_BOUNDS = ((0.0, 50.0), (0.0, 50.0))

# 200 steps over a box of 50 is a coarse step of ~0.251 and therefore a fine
# window of ~+/-0.251; at DEFAULT_FINE that resolves to ~0.0129 per axis.
DEFAULT_COARSE = 200
DEFAULT_FINE = 40

# Default seeds for `stability`: five weeks, disjoint from the campaign's
# dataset extension seeds (4242 + 0..4), so a week that calibrates the
# parameters is never also a week that evaluates them.
DEFAULT_STABILITY_SEEDS = (20260917, 20260918, 20260919, 20260920, 20260921)


@dataclass(frozen=True)
class FitResult:
    """A fitted pair and everything needed to reproduce it.

    `at_bound` is true when either parameter came to rest on an edge of the
    search box. Such a row is the boundary of the search, not an optimum, and
    it must not be quoted as a calibrated parameter.
    """
    theta: float
    steepness: float
    objective: float
    alpha: float
    bounds: tuple
    coarse: int
    fine: int
    n_slots: int
    at_bound: bool = False

    def as_dict(self) -> dict:
        data = asdict(self)
        data["bounds"] = [list(pair) for pair in self.bounds]
        return data


def _axis(low: float, high: float, count: int) -> list:
    """`count` evenly spaced points from `low` to `high`, both included.

    Computed as `low + i * (high - low) / (count - 1)` rather than by
    accumulating a step, so the points do not drift and the last one is
    exactly `high`.
    """
    if count < 2:
        raise ValueError(f"a grid axis needs at least 2 points, got {count}")
    span = high - low
    return [low + index * span / (count - 1) for index in range(count)]


def _best_on_grid(slots, thetas, steepnesses, alpha) -> tuple:
    """The (theta, steepness, value) minimising the objective on the product.

    Ties keep the first point in iteration order -- ascending theta, then
    ascending steepness -- which is what makes two runs of the same grid
    return the same pair rather than merely the same value. The objective is
    flat in B wherever the curve has already saturated, so ties are not
    hypothetical here.
    """
    best_theta = best_steepness = None
    best_value = None
    for theta in thetas:
        for steepness in steepnesses:
            value = objective(slots, theta, steepness, alpha)
            if best_value is None or value < best_value:
                best_theta, best_steepness, best_value = theta, steepness, value
    return best_theta, best_steepness, best_value


def _on_edge(value: float, low: float, high: float, tolerance: float) -> bool:
    """Did this parameter come to rest on an edge of the box?

    Judged within one fine step rather than by equality: a value that the
    refinement placed 0.01 short of the bound is still the search box talking,
    and reporting only exact hits would hide most of them.
    """
    return value <= low + tolerance or value >= high - tolerance


def fit(slots, alpha: float, bounds=DEFAULT_BOUNDS, coarse: int = DEFAULT_COARSE,
        fine: int = DEFAULT_FINE) -> FitResult:
    """Fit theta and B by a coarse grid followed by a refinement.

    The coarse pass covers `bounds` at `coarse` points per axis. The fine pass
    re-grids one coarse step either side of the winner at `fine` points per
    axis, clipped to `bounds` so the refinement cannot leave the stated search
    region and quietly report a parameter nobody searched for. Tying the
    window to the coarse step is what keeps a wider box from producing a
    coarser answer -- see the module docstring.
    """
    if not slots:
        raise ValueError("cannot fit on zero slots")
    (theta_low, theta_high), (steep_low, steep_high) = bounds

    thetas = _axis(theta_low, theta_high, coarse)
    steepnesses = _axis(steep_low, steep_high, coarse)
    theta, steepness, _ = _best_on_grid(slots, thetas, steepnesses, alpha)

    theta_step = (theta_high - theta_low) / (coarse - 1)
    steep_step = (steep_high - steep_low) / (coarse - 1)
    fine_thetas = _axis(max(theta_low, theta - theta_step),
                        min(theta_high, theta + theta_step), fine)
    fine_steepnesses = _axis(max(steep_low, steepness - steep_step),
                             min(steep_high, steepness + steep_step), fine)
    theta, steepness, value = _best_on_grid(slots, fine_thetas,
                                            fine_steepnesses, alpha)

    at_bound = (_on_edge(theta, theta_low, theta_high,
                         fine_thetas[1] - fine_thetas[0]) or
                _on_edge(steepness, steep_low, steep_high,
                         fine_steepnesses[1] - fine_steepnesses[0]))

    return FitResult(theta=theta, steepness=steepness, objective=value,
                     alpha=alpha, bounds=bounds, coarse=coarse, fine=fine,
                     n_slots=len(slots), at_bound=at_bound)


def sweep(slots, alphas=DEFAULT_ALPHAS, **fit_kw) -> list:
    """One fit per penalty weight (D-74).

    Each row carries the price the fitted curve puts at ratio 1.0 -- supply
    exactly meeting demand, the one ratio every reader already has an
    intuition for. It makes the effect of alpha readable in the table itself:
    a fit that has collapsed onto the band midpoint and a fit that spans the
    band are two different numbers there, without anyone having to plot the
    curve to find out which happened.
    """
    k_upper, k_lower = band(slots)
    rows = []
    for alpha in alphas:
        result = fit(slots, alpha, **fit_kw)
        rows.append({
            "alpha": alpha,
            "theta": result.theta,
            "steepness": result.steepness,
            "objective": result.objective,
            "price_at_ratio_1_ct": sigmoid_price_unclamped(
                1.0, k_upper, k_lower, result.theta, result.steepness),
            "at_bound": result.at_bound,
            "n_slots": result.n_slots,
        })
    return rows


def slots_from_run(seed, k_upper: float = 40.0, k_lower: float = 8.0):
    """The conventional CSV for one calibration week: `out/calib-<seed>/`."""
    path = OUT / f"calib-{seed}" / f"calib-{seed}_slots.csv"
    return load_slots(path, k_upper=k_upper, k_lower=k_lower).slots


def stability(seeds=DEFAULT_STABILITY_SEEDS, alphas=DEFAULT_ALPHAS, *,
              slots_for_seed=slots_from_run, **fit_kw) -> dict:
    """Refit on several independently drawn weeks and report the spread.

    **Why this exists.** The penalty term of eq. (8) is evaluated at `r_min`
    and `r_max`, and on this artifact's data those are each a single slot out
    of roughly 350 -- a dawn slot with a sliver of PV against full household
    load, and one midday slot with an unusual surplus. The whole structure of
    the fit therefore rests on two observations, and a fitted pair that is
    stable within a week tells you nothing about that.

    The answer is not to replace min/max with a quantile: the paper says
    min/max, and quietly substituting something more comfortable would make
    the objective this thesis' own rather than the one it cites. The answer is
    to measure how far the result moves when the two anchoring observations
    are redrawn, which is exactly what a different dataset extension seed
    does. If theta and B move more between seeds at one alpha than they move
    between alphas at one seed, then the sweep is reporting noise and 5.1 has
    to say so.

    `slots_for_seed(seed)` resolves a seed to its slots, so the caller can
    point this at whatever run directory holds the weeks -- and so the test
    can drive it without a harness run.
    """
    per_seed = {seed: slots_for_seed(seed) for seed in seeds}
    for seed, slots in per_seed.items():
        if not slots:
            raise ValueError(f"seed {seed} resolved to zero tradeable slots")

    extremes = {seed: (min(s.ratio for s in slots),
                       max(s.ratio for s in slots))
                for seed, slots in per_seed.items()}

    rows, per_alpha = [], []
    for alpha in alphas:
        fits = {seed: fit(per_seed[seed], alpha, **fit_kw) for seed in seeds}
        thetas = [fits[seed].theta for seed in seeds]
        steepnesses = [fits[seed].steepness for seed in seeds]
        summary = {
            "alpha": alpha,
            "theta_min": min(thetas), "theta_median": statistics.median(thetas),
            "theta_max": max(thetas),
            "b_min": min(steepnesses), "b_median": statistics.median(steepnesses),
            "b_max": max(steepnesses),
            "any_at_bound": any(fits[seed].at_bound for seed in seeds),
        }
        per_alpha.append(summary)
        for seed in seeds:
            result = fits[seed]
            ratio_min, ratio_max = extremes[seed]
            # The per-alpha aggregates are repeated on every row so the CSV
            # can be read on its own, without a second file to join against.
            rows.append({"alpha": alpha, "seed": seed,
                         "theta": result.theta, "steepness": result.steepness,
                         "objective": result.objective,
                         "at_bound": result.at_bound,
                         "r_min": ratio_min, "r_max": ratio_max,
                         "n_slots": result.n_slots, **summary})
    return {"seeds": list(seeds), "rows": rows, "per_alpha": per_alpha,
            "extremes": {str(s): list(e) for s, e in extremes.items()}}


def write_stability_csv(out_dir, rows: list) -> str:
    """`stability.csv`: one row per (alpha, seed), tidy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stability.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_outputs(out_dir, result: FitResult, rows: list, **extra) -> dict:
    """`theta_B.json` and `alpha_sweep.csv`, in one run directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"fit": result.as_dict(), "alpha_sweep": rows, **extra}
    json_path = out_dir / "theta_B.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "alpha_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"theta_B": str(json_path), "alpha_sweep": str(csv_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit theta and B on a harness slot CSV (T-19).")
    parser.add_argument("csv", help="out/<run-id>/<run-id>_slots.csv")
    parser.add_argument("--run-id", default="calibration",
                        help="output directory under evaluation/out/")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="penalty weight of the reported fit")
    parser.add_argument("--k-upper", type=float, default=40.0)
    parser.add_argument("--k-lower", type=float, default=8.0)
    parser.add_argument("--coarse", type=int, default=DEFAULT_COARSE)
    parser.add_argument("--fine", type=int, default=DEFAULT_FINE)
    parser.add_argument("--stability-seeds", default="",
                        help="comma-separated extension seeds; also writes "
                             "stability.csv across those weeks")
    args = parser.parse_args(argv)

    loaded = load_slots(args.csv, k_upper=args.k_upper, k_lower=args.k_lower)
    print(f"{len(loaded.slots)} slots fitted, {loaded.dropped} dropped "
          f"(of {loaded.total}) -- dropped slots did not trade")
    if not loaded.slots:
        parser.error(f"{args.csv} holds no slot that traded")

    ratios = [s.ratio for s in loaded.slots]
    print(f"ratio range    : {min(ratios):.6f} .. {max(ratios):.6f}")

    result = fit(loaded.slots, args.alpha, coarse=args.coarse, fine=args.fine)
    print(f"fitted at alpha={args.alpha}: theta={result.theta:.6f}  "
          f"B={result.steepness:.6f}  objective={result.objective:.6f}")

    rows = sweep(loaded.slots, coarse=args.coarse, fine=args.fine)
    print("\n alpha      theta          B     objective   p(ratio=1)  at_bound")
    for row in rows:
        print(f"{row['alpha']:6.2f} {row['theta']:10.4f} "
              f"{row['steepness']:10.4f} {row['objective']:13.6f} "
              f"{row['price_at_ratio_1_ct']:12.4f}  "
              f"{'YES' if row['at_bound'] else '-'}")

    extra = {}
    if args.stability_seeds:
        seeds = [int(s) for s in args.stability_seeds.split(",") if s.strip()]
        spread = stability(seeds, coarse=args.coarse, fine=args.fine)
        print("\nstability across "
              f"{len(seeds)} weeks (seeds {', '.join(str(s) for s in seeds)}):")
        print(" alpha   theta min    median       max |      B min    median"
              "       max  at_bound")
        for summary in spread["per_alpha"]:
            print(f"{summary['alpha']:6.2f} {summary['theta_min']:10.4f} "
                  f"{summary['theta_median']:9.4f} {summary['theta_max']:9.4f} | "
                  f"{summary['b_min']:9.4f} {summary['b_median']:9.4f} "
                  f"{summary['b_max']:9.4f}  "
                  f"{'YES' if summary['any_at_bound'] else '-'}")
        path = write_stability_csv(OUT / args.run_id, spread["rows"])
        print(f"  -> stability: {path}")
        extra["stability"] = {"seeds": spread["seeds"],
                              "per_alpha": spread["per_alpha"],
                              "extremes": spread["extremes"]}

    written = write_outputs(
        OUT / args.run_id, result, rows,
        source_csv=str(Path(args.csv).resolve()),
        k_upper=args.k_upper, k_lower=args.k_lower,
        n_slots_dropped=loaded.dropped, n_slots_total=loaded.total,
        ratio_min=min(ratios), ratio_max=max(ratios), **extra)
    for name, path in written.items():
        print(f"  -> {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
