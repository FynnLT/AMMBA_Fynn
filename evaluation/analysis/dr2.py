"""DR2: the eight cases of Table 4.2, Propositions 1 and 2, the band
sensitivity and the named deviators of block 2 (A2).

**Population.** Every participant-slot of `b2_noise_off`: pro rata, sigma = 0
and no deviator, so the recorded report *is* the truthful type (A =
`requested_kwh` for a seller, D = `requested_kwh` for a buyer).

Each participant-slot admits two cases given its side and regime: withhold /
under-report, and over-report. The deviation runs over a grid of the relative
size delta / claim in [0, 0.5] at steps of 0.005, and the thresholds between
grid points are found by bisection.

**Named deviators.** The harness changes meter and forecast *after*
clearing, not the order book, so in a block-2 cell the recorded report is the
deviated one and the truthful counterfactual is the report plus the
deviation. Each deviator is replayed alone, everyone else at their recorded
report, with the scalar engine (the artifact's functions).
"""
import numpy as np

import replay as R

#: delta / claim, 0 .. 0.5 at 0.005.
GRID = np.round(np.arange(101) * 0.005, 3)
BISECTIONS = 40
#: A net gain below this is rounding: every penalty is reported at 1e-6 ct.
NET_TOL = 1e-6

#: Chapter 3.4.2: the band recomputed at several settings, bounds unchanged.
BAND_STEEPNESS = (0.6, 1.0, 2.5)
BAND_THETA = (0.8, 1.0, 1.2)

WITHHOLD, OVERREPORT = -1, +1
HOUSEHOLD, BATTERY = "household", "battery"

#: The eight rows of Table 4.2, in its order (row = position + 1): side,
#: side role, direction. The seller is the short side in a
#: supply-limited round, the buyer in a demand-limited one.
CASES = (
    ("seller", "short", WITHHOLD),
    ("seller", "short", OVERREPORT),
    ("seller", "long", WITHHOLD),
    ("seller", "long", OVERREPORT),
    ("buyer", "short", WITHHOLD),
    ("buyer", "short", OVERREPORT),
    ("buyer", "long", WITHHOLD),
    ("buyer", "long", OVERREPORT),
)


def case_name(side, role, direction) -> str:
    verb = ("overreport" if direction == OVERREPORT
            else "withhold" if side == "seller" else "underreport")
    return f"{side}_{role}_{verb}"


def participant_type(area: str) -> str:
    return BATTERY if str(area).startswith("battery-") else HOUSEHOLD


# ------------------------------------------------------------ population

def build_population(datas) -> dict:
    """Arrays over every participant-slot of the given (b2_noise_off) runs.

    The slot aggregates, the band and the penalty parameters are the
    recorded ones: `gamma_eff` / `eta_relative_eff` from the slot CSV, which
    is what the node applied, never the spec.
    """
    parts = []
    for data in datas:
        areas, slots, run = data.areas, data.slots, data.run
        idx = data.slot_index
        band = R.Band.from_dict(run.sigmoid)
        n = areas.n
        supply = slots.floats("total_supply_kwh")[idx]
        demand = slots.floats("total_demand_kwh")[idx]
        parts.append({
            "run": np.full(n, run.run_id, dtype=object),
            "seed": np.full(n, run.seed),
            "slot": areas.ints("slot"),
            "area": areas.strings("area_uuid"),
            "is_seller": areas.strings("side") == "seller",
            "claim": areas.floats("requested_kwh"),
            "allocated": areas.floats("allocated_kwh"),
            "supply": supply, "demand": demand,
            "price_recorded": slots.floats("clearing_price_ct")[idx],
            "k_upper": np.full(n, band.k_upper),
            "k_lower": np.full(n, band.k_lower),
            "theta": np.full(n, band.theta),
            "steepness": np.full(n, band.steepness),
            "gamma": slots.floats("gamma_eff")[idx],
            "eta_relative": slots.floats("eta_relative_eff")[idx],
        })
    pop = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    pop["regime"] = R.regime_vec(pop["supply"], pop["demand"])
    pop["is_battery"] = np.array([participant_type(a) == BATTERY
                                  for a in pop["area"]])
    short = ((pop["is_seller"] & (pop["regime"] == R.SL))
             | (~pop["is_seller"] & (pop["regime"] == R.DL)))
    pop["role"] = np.where(short, "short", "long")
    pop["n"] = len(pop["claim"])
    return pop


def _take(pop, idx, key, two_d):
    values = pop[key][idx]
    return values[:, None] if two_d else values


def evaluate(pop, idx, rel, direction, band=None):
    """The replay of rows `idx` at relative deviation `rel` (1-D per row, or
    2-D rows x grid). `band = (theta, steepness)` reprices at another band
    with the bounds unchanged."""
    two_d = np.ndim(rel) == 2
    claim = _take(pop, idx, "claim", two_d)
    if band is None:
        theta = _take(pop, idx, "theta", two_d)
        steepness = _take(pop, idx, "steepness", two_d)
    else:
        theta, steepness = band
    return R.replay_vec(
        is_seller=_take(pop, idx, "is_seller", two_d),
        supply=_take(pop, idx, "supply", two_d),
        demand=_take(pop, idx, "demand", two_d),
        own_report=claim, new_report=claim * (1.0 + direction * rel),
        truth=claim,
        k_upper=_take(pop, idx, "k_upper", two_d),
        k_lower=_take(pop, idx, "k_lower", two_d),
        theta=theta, steepness=steepness,
        gamma=_take(pop, idx, "gamma", two_d),
        eta_relative=_take(pop, idx, "eta_relative", two_d))


def sweep(pop, idx, direction, band=None, chunk=4096) -> dict:
    """Gross gain, penalty and net gain over the grid, rows x grid.

    Gains are against the truthful point of the same band, so the grid's
    first column is zero by construction.
    """
    n, g = len(idx), len(GRID)
    base = evaluate(pop, idx, np.zeros(n), direction, band)
    out = {"gross": np.empty((n, g)), "penalty": np.empty((n, g)),
           "penalty_raw": np.empty((n, g)), "net": np.empty((n, g)),
           "regime": np.empty((n, g), dtype=np.int8),
           "u0": base["utility"], "gross0": base["gross_utility"],
           "penalty0": base["penalty_ct"], "regime0": base["regime"]}
    for start in range(0, n, chunk):
        rows = slice(start, start + chunk)
        rel = np.broadcast_to(GRID, (len(idx[rows]), g))
        res = evaluate(pop, idx[rows], rel, direction, band)
        gross = res["gross_utility"] - base["gross_utility"][rows, None]
        penalty = res["penalty_ct"] - base["penalty_ct"][rows, None]
        out["gross"][rows] = gross
        out["penalty"][rows] = penalty
        out["penalty_raw"][rows] = res["penalty_raw"]
        out["net"][rows] = gross - penalty
        out["regime"][rows] = res["regime"]
    return out


def _net_at(pop, idx, rel, direction, band, base_utility, evaluate_fn=None):
    evaluate_fn = evaluate_fn or evaluate
    return evaluate_fn(pop, idx, rel, direction, band)["utility"] - base_utility


def thresholds(pop, idx, direction, res, band=None, evaluate_fn=None) -> dict:
    """delta_active, delta*, net_max, delta_zero and net(delta_active).

    `evaluate_fn` is the engine the bisections call, `evaluate` unless
    given; `dr2_full` passes its own with the same signature.
    """
    evaluate_fn = evaluate_fn or evaluate
    n = len(idx)
    rows = np.arange(n)
    u0 = res["u0"]

    # delta_active: the smallest delta at which any (unrounded) penalty is
    # non-zero; infinite if none on the grid.
    active = res["penalty_raw"] > 0.0
    has_active = active.any(axis=1)
    k = active.argmax(axis=1)
    lo = GRID[np.maximum(k - 1, 0)].astype(float)
    hi = GRID[k].astype(float)
    sel = np.where(has_active & (k > 0))[0]
    for _ in range(BISECTIONS):
        if not len(sel):
            break
        mid = 0.5 * (lo[sel] + hi[sel])
        on = evaluate_fn(pop, idx[sel], mid, direction,
                         band)["penalty_raw"] > 0.0
        hi[sel] = np.where(on, mid, hi[sel])
        lo[sel] = np.where(on, lo[sel], mid)
    delta_active = np.where(has_active, hi, np.inf)

    net = res["net"]
    kstar = net.argmax(axis=1)
    net_max = net[rows, kstar]
    delta_star = GRID[kstar].astype(float)

    # delta_zero: above the maximiser, the first delta with net <= 0.
    after = (np.arange(len(GRID))[None, :] > kstar[:, None]) & (net <= 0.0)
    has_zero = after.any(axis=1)
    j = after.argmax(axis=1)
    lo = GRID[np.maximum(j - 1, 0)].astype(float)
    hi = GRID[j].astype(float)
    profitable = net_max > NET_TOL
    sel = np.where(has_zero & profitable)[0]
    for _ in range(BISECTIONS):
        if not len(sel):
            break
        mid = 0.5 * (lo[sel] + hi[sel])
        nonpos = _net_at(pop, idx[sel], mid, direction, band, u0[sel],
                         evaluate_fn) <= 0.0
        hi[sel] = np.where(nonpos, mid, hi[sel])
        lo[sel] = np.where(nonpos, lo[sel], mid)
    delta_zero = np.where(profitable, np.where(has_zero, hi, np.inf), 0.0)

    finite = np.isfinite(delta_active)
    net_at_active = np.full(n, np.nan)
    if finite.any():
        sel = np.where(finite)[0]
        net_at_active[sel] = _net_at(pop, idx[sel], delta_active[sel],
                                     direction, band, u0[sel], evaluate_fn)

    switched = res["regime"] != res["regime0"][:, None]
    return {"delta_active": delta_active, "delta_star": delta_star,
            "net_max": net_max, "delta_zero": delta_zero,
            "net_at_active": net_at_active,
            "switch_any": switched.any(axis=1),
            "switch_at_max": switched[rows, kstar]}


def proposition2_violations(res, thr) -> np.ndarray:
    """Beyond delta_active and within the regime, a net gain above
    net(delta_active): the proposition predicts none."""
    beyond = ((GRID[None, :] > thr["delta_active"][:, None])
              & (res["regime"] == res["regime0"][:, None]))
    better = res["net"] > (thr["net_at_active"][:, None] + NET_TOL)
    return (beyond & better).any(axis=1)


# ------------------------------------------------------------ the tables

def _q(values, q):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.quantile(values, q)) if len(values) else None


def _max(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.max()) if len(values) else None


def case_rows(pop):
    """Row masks per case, BALANCED slots left out (no side is short)."""
    for side, role, direction in CASES:
        mask = ((pop["is_seller"] == (side == "seller"))
                & (pop["role"] == role) & (pop["regime"] != R.BAL))
        yield side, role, direction, np.where(mask)[0]


def analyse_cases(pop):
    """dr2_cases, dr2_curves and the Proposition 2 counts."""
    case_table, curve_table, prop2 = [], [], []
    for side, role, direction, idx in case_rows(pop):
        name = case_name(side, role, direction)
        if not len(idx):
            continue
        res = sweep(pop, idx, direction)
        thr = thresholds(pop, idx, direction, res)
        u0 = res["u0"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_base = np.where(u0 > 0, u0, np.nan)
        regime = R.REGIME_NAME[int(res["regime0"][0])]
        short_seller = side == "seller" and role == "short"
        violated = proposition2_violations(res, thr) if short_seller else None
        types = (("all", np.ones(len(idx), bool)),
                 (HOUSEHOLD, ~pop["is_battery"][idx]),
                 (BATTERY, pop["is_battery"][idx]))
        for type_name, mask in types:
            if not mask.any():
                continue
            net_max = thr["net_max"][mask]
            net_rel = net_max / rel_base[mask]
            case_table.append({
                "case": name,
                "table_4_2_row": CASES.index((side, role, direction)) + 1,
                "side": side, "side_role": role,
                "regime": regime,
                "direction": "overreport" if direction > 0 else "underreport",
                "type": type_name, "n": int(mask.sum()),
                "share_net_max_pos": float((net_max > NET_TOL).mean()),
                "net_max_ct_median": _q(net_max, 0.5),
                "net_max_ct_p90": _q(net_max, 0.9),
                "net_max_ct_max": _max(net_max),
                "net_max_rel_median": _q(net_rel, 0.5),
                "net_max_rel_p90": _q(net_rel, 0.9),
                "net_max_rel_max": _max(net_rel),
                "delta_star_rel_median": _q(thr["delta_star"][mask], 0.5),
                "delta_active_rel_median": float(np.median(
                    thr["delta_active"][mask])),
                "n_delta_active_inf": int(np.isinf(
                    thr["delta_active"][mask]).sum()),
                "delta_zero_rel_median": float(np.median(
                    thr["delta_zero"][mask])),
                "n_delta_zero_inf": int(np.isinf(
                    thr["delta_zero"][mask]).sum()),
                "net_at_active_ct_median": _q(thr["net_at_active"][mask], 0.5),
                "share_switch_at_max": float(thr["switch_at_max"][mask].mean()),
                "share_switch_any": float(thr["switch_any"][mask].mean()),
                "prop2_n_violations": (int(violated[mask].sum())
                                       if short_seller else None),
            })

        rel = res["gross"] / rel_base[:, None]
        rel_pen = res["penalty"] / rel_base[:, None]
        rel_net = res["net"] / rel_base[:, None]
        for k, delta in enumerate(GRID):
            curve_table.append({
                "case": name, "delta_rel": float(delta), "n": len(idx),
                "gross_ct_median": _q(res["gross"][:, k], 0.5),
                "gross_ct_p90": _q(res["gross"][:, k], 0.9),
                "penalty_ct_median": _q(res["penalty"][:, k], 0.5),
                "penalty_ct_p90": _q(res["penalty"][:, k], 0.9),
                "net_ct_median": _q(res["net"][:, k], 0.5),
                "net_ct_p90": _q(res["net"][:, k], 0.9),
                "gross_rel_median": _q(rel[:, k], 0.5),
                "gross_rel_p90": _q(rel[:, k], 0.9),
                "penalty_rel_median": _q(rel_pen[:, k], 0.5),
                "penalty_rel_p90": _q(rel_pen[:, k], 0.9),
                "net_rel_median": _q(rel_net[:, k], 0.5),
                "net_rel_p90": _q(rel_net[:, k], 0.9),
                "share_regime_switch": float(
                    (res["regime"][:, k] != res["regime0"]).mean()),
            })

        if short_seller:
            prop2.append({"case": name, "n": len(idx),
                          "n_active": int(np.isfinite(
                              thr["delta_active"]).sum()),
                          "n_violations": int(violated.sum())})
    return case_table, curve_table, prop2


def proposition2_condition(pop) -> list:
    """gamma (1 - eta_rel) K_upper > K_upper - K_lower, with the run values."""
    out = []
    seen = set()
    for gamma, eta, ku, kl in zip(pop["gamma"], pop["eta_relative"],
                                  pop["k_upper"], pop["k_lower"]):
        key = (float(gamma), float(eta), float(ku), float(kl))
        if key in seen:
            continue
        seen.add(key)
        lhs = key[0] * (1.0 - key[1]) * key[2]
        out.append({"gamma": key[0], "eta_relative": key[1],
                    "k_upper": key[2], "k_lower": key[3], "lhs": lhs,
                    "rhs": key[2] - key[3], "holds": lhs > key[2] - key[3]})
    return out


def conditions(pop, band=None, types=True) -> list:
    """Proposition 1 (long side) and the Section 4.3.4 condition (short-side
    sellers), at the truthful point, no sweep.

    Long side: epsilon_t against (1 - sigma) / sigma (Eqs. 4.6/4.7), with
    sigma the participant's share of its own side. `holds` counts epsilon_t >
    (1 - sigma)/sigma: withholding / under-reporting pays at the margin; the
    complement is where over-reporting does. Short-side sellers:
    epsilon_t * sigma against 1, `holds` counting epsilon_t * sigma > 1.
    """
    theta = pop["theta"] if band is None else np.full(pop["n"], band[0])
    steep = pop["steepness"] if band is None else np.full(pop["n"], band[1])
    ratio = pop["supply"] / pop["demand"]
    arg = -steep * (ratio - theta)
    e = np.exp(np.clip(arg, -R.EXP_ARG_LIMIT, R.EXP_ARG_LIMIT))
    slope = -(pop["k_upper"] - pop["k_lower"]) * steep * e / (1.0 + e) ** 2
    slope = np.where(np.abs(arg) > R.EXP_ARG_LIMIT, 0.0, slope)
    price = R.sigmoid_vec(ratio, pop["k_upper"], pop["k_lower"], theta, steep)
    margin = np.where(pop["is_seller"], price - pop["k_lower"],
                      pop["k_upper"] - price)
    eps = -slope * ratio / margin
    own_side = np.where(pop["is_seller"], pop["supply"], pop["demand"])
    sigma = pop["claim"] / own_side

    groups = (("seller", "long", R.DL), ("buyer", "long", R.SL),
              ("seller", "short", R.SL))
    type_masks = [("all", np.ones(pop["n"], bool))]
    if types:
        type_masks += [(HOUSEHOLD, ~pop["is_battery"]),
                       (BATTERY, pop["is_battery"])]
    rows = []
    for side, role, regime in groups:
        base = (pop["is_seller"] == (side == "seller")) & (pop["regime"] == regime)
        for type_name, tmask in type_masks:
            mask = base & tmask
            if not mask.any():
                continue
            if role == "long":
                condition = "eps > (1-sigma)/sigma"
                lhs, rhs = eps[mask], (1.0 - sigma[mask]) / sigma[mask]
            else:
                condition = "eps*sigma > 1"
                lhs, rhs = eps[mask] * sigma[mask], np.ones(mask.sum())
            gap = lhs - rhs
            closest = int(np.argmin(np.abs(gap)))
            rows.append({
                "side": side, "side_role": role,
                "regime": R.REGIME_NAME[regime], "type": type_name,
                "condition": condition, "n": int(mask.sum()),
                "n_holds": int((gap > 0).sum()),
                "n_fails": int((gap < 0).sum()),
                "sigma_max": float(sigma[mask].max()),
                "eps_median": float(np.median(eps[mask])),
                "eps_max": float(eps[mask].max()),
                "lhs_max": float(lhs.max()),
                "closest_margin": float(abs(gap[closest])),
                "closest_margin_signed": float(gap[closest]),
                "closest_sigma": float(sigma[mask][closest]),
            })
    return rows


def band_sensitivity(pop, main_cases, main_conditions) -> tuple:
    """Prop. 1 counts and per-case share with net_max > 0 at B x theta.

    Returns the table and whether the calibrated band's row reproduces the
    main tables exactly (it is the same computation at the same band).
    """
    table = []
    reproduces = True
    calibrated = (float(pop["theta"][0]), float(pop["steepness"][0]))
    main_share = {r["case"]: r["share_net_max_pos"] for r in main_cases
                  if r["type"] == "all"}
    main_holds = {(r["side"], r["side_role"]): r["n_holds"]
                  for r in main_conditions if r["type"] == "all"}
    for steep in BAND_STEEPNESS:
        for theta in BAND_THETA:
            band = (theta, steep)
            for row in conditions(pop, band=band, types=False):
                table.append({"steepness": steep, "theta": theta,
                              "kind": "condition",
                              "name": f"{row['side']}_{row['side_role']}: "
                                      f"{row['condition']}",
                              "n": row["n"], "n_positive": row["n_holds"],
                              "share": row["n_holds"] / row["n"]})
                if (theta, steep) == calibrated:
                    reproduces &= (row["n_holds"]
                                   == main_holds[(row["side"],
                                                  row["side_role"])])
            for side, role, direction, idx in case_rows(pop):
                if not len(idx):
                    continue
                name = case_name(side, role, direction)
                best = np.full(len(idx), -np.inf)
                for start in range(0, len(idx), 4096):
                    rows = idx[start:start + 4096]
                    base = evaluate(pop, rows, np.zeros(len(rows)), direction,
                                    band)["utility"]
                    rel = np.broadcast_to(GRID, (len(rows), len(GRID)))
                    net = (evaluate(pop, rows, rel, direction, band)["utility"]
                           - base[:, None])
                    best[start:start + len(rows)] = net.max(axis=1)
                positive = int((best > NET_TOL).sum())
                table.append({"steepness": steep, "theta": theta,
                              "kind": "case", "name": name, "n": len(idx),
                              "n_positive": positive,
                              "share": positive / len(idx)})
                if (theta, steep) == calibrated:
                    reproduces &= abs(positive / len(idx)
                                      - main_share[name]) < 1e-12
    return table, reproduces


# -------------------------------------------------------- named deviators

WITHHOLD_ARM = "sellers_withhold"
UNDERDELIVER_ARM = "sellers_underdeliver"
UNDERREPORT_ARM = "buyers_underreport"
NAMED_ARMS = (WITHHOLD_ARM, UNDERDELIVER_ARM, UNDERREPORT_ARM)


def named_deviators(data) -> list:
    """One row per named deviator of one run, summed over the week.

    * sellers_withhold: A = deliverable_kwh, supply-limited slots replayed.
      In a demand-limited slot the forecast `allocated * (1 + share)` can lie
      below the rationed seller's report, so the arm defines no order-book
      deviation there (and the node applies no externality): counted, not
      replayed.
    * buyers_underreport: D = actual_kwh, both regimes replayed.
    * sellers_underdeliver: read as an over-report on the short side, the
      truthful report being actual_kwh; supply-limited slots only.
    """
    run = data.run
    if run.arm not in NAMED_ARMS or not run.deviators:
        return []
    areas, slots = data.areas, data.slots
    band = R.Band.from_dict(run.sigmoid)
    area_ids = areas.strings("area_uuid")
    supply = slots.floats("total_supply_kwh")
    demand = slots.floats("total_demand_kwh")
    price = slots.floats("clearing_price_ct")
    gamma = slots.floats("gamma_eff")
    eta = slots.floats("eta_relative_eff")
    requested = areas.floats("requested_kwh")
    deliverable = areas.floats("deliverable_kwh")
    actual = areas.floats("actual_kwh")
    penalty = areas.floats_or_zero("total_penalty_ct")
    side_is_seller = areas.strings("side") == "seller"
    config = run.config
    out = []
    for deviator in run.deviators:
        # A household nets to one side per slot, so the deviator is on the
        # arm's side in some slots and on the other in the rest. Only the
        # arm's side carries the deviation; the other side carries the
        # accidental layer like everyone else and is reported apart.
        arm_seller = run.arm != UNDERREPORT_ARM
        mine = area_ids == deviator
        rows = np.where(mine & (side_is_seller == arm_seller))[0]
        other = np.where(mine & (side_is_seller != arm_seller))[0]
        acc = {"n_active_slots": len(rows), "n_other_side_slots": len(other),
               "n_replayed": 0, "n_not_replayed": 0,
               "n_regime_switch_excluded": 0,
               "gross_gain_ct": 0.0, "price_effect_ct": 0.0,
               "forgone_margin_ct": 0.0,
               "penalty_all_ct": float(penalty[rows].sum()),
               "penalty_in_regime_ct": 0.0,
               "penalty_other_side_ct": float(penalty[other].sum()),
               "max_price_replay_error": 0.0}
        for row in rows:
            s = data.slot_index[row]
            state = R.SlotState(supply[s], demand[s], band, gamma[s], eta[s])
            regime = R.determine_round_type(supply[s], demand[s])
            seller = bool(side_is_seller[row])
            if run.arm == UNDERREPORT_ARM:
                truth = actual[row]
            else:
                if regime != R.SUPPLY_LIMITED:
                    acc["n_not_replayed"] += 1
                    continue
                truth = (deliverable[row] if run.arm == WITHHOLD_ARM
                         else actual[row])
            side = R.SELLER if seller else R.BUYER
            own = requested[row]
            dev = R.replay(state, side, own, own, truth)
            cf = R.replay(state, side, own, truth, truth)
            gross = dev.gross_utility - cf.gross_utility
            if seller:
                price_effect = (dev.price - cf.price) * dev.allocation
            else:
                price_effect = (cf.price - dev.price) * dev.allocation
            acc["n_replayed"] += 1
            acc["max_price_replay_error"] = max(
                acc["max_price_replay_error"], float(abs(dev.price - price[s])))
            if cf.round_type != dev.round_type:
                # The propositions hold within the regime only: a slot whose
                # counterfactual switches it is counted and left out of the
                # sums, penalty included, so gross and penalty cover the
                # same slots.
                acc["n_regime_switch_excluded"] += 1
                continue
            acc["gross_gain_ct"] += float(gross)
            acc["price_effect_ct"] += float(price_effect)
            acc["forgone_margin_ct"] += float(price_effect - gross)
            acc["penalty_in_regime_ct"] += float(penalty[row])
        acc["net_gain_ct"] = acc["gross_gain_ct"] - acc["penalty_in_regime_ct"]
        out.append({
            "cell": run.cell, "seed": run.seed, "run_id": run.run_id,
            "arm": run.arm, "share": run.deviation.get("share"),
            "k": run.deviation.get("k"), "gamma": config.get("gamma"),
            "eta_relative": config.get("eta_relative"),
            "deviator": deviator, **acc,
            "slots_with_absent_deviator":
                run.deviation.get("slots_with_absent_deviator")})
    return out
