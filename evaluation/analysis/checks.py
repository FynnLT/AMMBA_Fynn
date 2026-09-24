"""R1-R3: the replay against the recorded runs (A1).

All three must pass before any DR table is written. Each returns a dict
with `passed`, the maximum deviation and the tolerance it was held to, so
`checks.json` carries the number and not only the verdict.
"""
import numpy as np

import dr2
import replay as R

R1_TOL = 1e-5
R2_TOL = 1e-4
R3_TOL = 1e-9
R3_POINTS = 10_000


def r1(pop) -> dict:
    """At delta = 0 the replay reproduces every recorded price and every
    allocation of the honest reference cell."""
    if not pop or not pop.get("n"):
        return {"passed": False, "reason": "no b2_noise_off run found",
                "tolerance": R1_TOL}
    idx = np.arange(pop["n"])
    res = dr2.evaluate(pop, idx, np.zeros(pop["n"]), dr2.WITHHOLD)
    price_dev = np.abs(res["price"] - pop["price_recorded"])
    alloc_dev = np.abs(res["allocation"] - pop["allocated"])
    worst = float(max(price_dev.max(), alloc_dev.max()))
    return {"passed": bool(worst <= R1_TOL), "tolerance": R1_TOL,
            "n_rows": int(pop["n"]),
            "n_slots": int(len(set(zip(pop["run"], pop["slot"])))),
            "n_runs": int(len(set(pop["run"]))),
            "max_price_deviation_ct": float(price_dev.max()),
            "max_allocation_deviation_kwh": float(alloc_dev.max())}


class R2:
    """Every recorded penalty, recomputed from the row's recorded quantities
    and the slot's recorded aggregates with the node's own call."""

    def __init__(self):
        self.n_rows = 0
        self.n_runs = 0
        self.max_dev = 0.0
        self.worst = None
        self.unsupported = []
        #: Per parameter source: the manifest-config runs have to pass on
        #: their own, not only inside a pooled maximum.
        self.by_source = {}

    def add(self, data) -> None:
        run = data.run
        if not run.executed:
            return
        areas, slots = data.areas, data.slots
        shortfall = areas.floats_or_zero("shortfall_penalty_ct")
        externality = areas.floats_or_zero("externality_penalty_ct")
        rows = np.where((shortfall > 0) | (externality > 0))[0]
        self.n_runs += 1
        source = self.by_source.setdefault(
            data.param_source, {"n_runs": 0, "n_rows": 0,
                                "max_deviation_ct": 0.0})
        source["n_runs"] += 1
        source["n_rows"] += len(rows)
        if not len(rows):
            return
        band = R.Band.from_dict(run.sigmoid)
        sigmoid = band.as_sigmoid()
        eta_mode = slots.strings("eta_mode")
        side = areas.strings("side")
        allocated = areas.floats("allocated_kwh")
        requested = areas.floats("requested_kwh")
        actual = areas.floats("actual_kwh")
        deliverable = areas.floats("deliverable_kwh")
        supply = slots.floats("total_supply_kwh")
        demand = slots.floats("total_demand_kwh")
        price = slots.floats("clearing_price_ct")
        gamma = slots.floats("gamma_eff")
        eta_rel = slots.floats("eta_relative_eff")
        round_type = slots.strings("round_type_exec")
        for row in rows:
            s = data.slot_index[row]
            if eta_mode[s] != "relative":
                # The absolute deadband is not in the CSV; the row cannot be
                # recomputed, and saying so beats guessing it.
                self.unsupported.append(f"{run.run_id}@{slots.ints('slot')[s]}")
                continue
            traded = allocated[row]
            e, d = supply[s], demand[s]
            q = min(e, d)
            got_short = got_ext = 0.0
            if side[row] == "seller":
                eta = eta_rel[s] * traded
                got_short = R.seller_shortfall_penalty(
                    traded, actual[row], band.k_upper, gamma[s],
                    eta)["penalty_ct"]
                if round_type[s] == R.SUPPLY_LIMITED:
                    ext = R.seller_externality_penalty(
                        traded, deliverable[row], eta, e, d, q, price[s],
                        sigmoid)
                    got_ext = ext["penalty_ct"] if ext else 0.0
            elif round_type[s] == R.DEMAND_LIMITED:
                ext = R.buyer_externality_penalty(
                    requested[row], actual[row], e, d, q, price[s], sigmoid)
                got_ext = ext["penalty_ct"] if ext else 0.0
            dev = max(abs(got_short - shortfall[row]),
                      abs(got_ext - externality[row]))
            self.n_rows += 1
            source["max_deviation_ct"] = max(source["max_deviation_ct"], dev)
            if dev > self.max_dev:
                self.max_dev = dev
                self.worst = {"run": run.run_id,
                              "slot": int(slots.ints("slot")[s]),
                              "area": str(areas.strings("area_uuid")[row])}

    def result(self) -> dict:
        passed = (self.max_dev <= R2_TOL and not self.unsupported
                  and self.n_runs > 0)
        out = {"passed": bool(passed), "tolerance": R2_TOL,
               "n_runs": self.n_runs, "n_rows": self.n_rows,
               "max_deviation_ct": self.max_dev, "worst": self.worst,
               "by_param_source": {
                   str(name): {**acc, "passed": bool(
                       acc["max_deviation_ct"] <= R2_TOL)}
                   for name, acc in self.by_source.items()}}
        if self.unsupported:
            out["reason"] = (f"{len(self.unsupported)} slot(s) ran with an "
                             f"absolute deadband, which the CSV does not "
                             f"record: {self.unsupported[:5]}")
        if not self.n_runs:
            out["reason"] = "no executed run found"
        return out


def r3(pop, seed=20260924, points=R3_POINTS) -> dict:
    """The vectorised engine against the artifact functions, on random
    (participant, delta) points.

    Two draws of `points` each: the DR2 use (truth = claim) at the recorded
    band, and a harder one -- truth off the claim by up to 30 %, deviation
    in either direction, at a band drawn from the sensitivity grid -- so
    the shortfall and both externality paths are exercised, not only the
    ones the honest cell happens to reach.
    """
    if not pop or not pop.get("n"):
        return {"passed": False, "reason": "no population", "tolerance": R3_TOL}
    rng = np.random.default_rng(seed)
    worst = {"price": 0.0, "allocation": 0.0, "shortfall_ct": 0.0,
             "externality_ct": 0.0, "utility": 0.0}
    regime_mismatch = 0
    total = 0
    for hard in (False, True):
        idx = rng.integers(0, pop["n"], size=points)
        rel = rng.uniform(-0.5, 0.5, size=points)
        claim = pop["claim"][idx]
        truth = (claim * (1.0 + rng.uniform(-0.3, 0.3, size=points))
                 if hard else claim)
        theta = pop["theta"][idx].copy()
        steep = pop["steepness"][idx].copy()
        if hard:
            theta = rng.choice(dr2.BAND_THETA, size=points)
            steep = rng.choice(dr2.BAND_STEEPNESS, size=points)
        new = claim * (1.0 + rel)
        vec = R.replay_vec(
            is_seller=pop["is_seller"][idx], supply=pop["supply"][idx],
            demand=pop["demand"][idx], own_report=claim, new_report=new,
            truth=truth, k_upper=pop["k_upper"][idx],
            k_lower=pop["k_lower"][idx], theta=theta, steepness=steep,
            gamma=pop["gamma"][idx], eta_relative=pop["eta_relative"][idx])
        for i in range(points):
            j = idx[i]
            state = R.SlotState(
                pop["supply"][j], pop["demand"][j],
                R.Band(pop["k_upper"][j], pop["k_lower"][j], theta[i],
                       steep[i]),
                pop["gamma"][j], pop["eta_relative"][j])
            side = R.SELLER if pop["is_seller"][j] else R.BUYER
            ref = R.replay(state, side, claim[i], new[i], truth[i])
            worst["price"] = max(worst["price"], abs(ref.price - vec["price"][i]))
            worst["allocation"] = max(worst["allocation"],
                                      abs(ref.allocation - vec["allocation"][i]))
            worst["shortfall_ct"] = max(worst["shortfall_ct"], abs(
                ref.shortfall_ct - vec["shortfall_ct"][i]))
            worst["externality_ct"] = max(worst["externality_ct"], abs(
                ref.externality_ct - vec["externality_ct"][i]))
            worst["utility"] = max(worst["utility"],
                                   abs(ref.utility - vec["utility"][i]))
            regime_mismatch += int(R.REGIME_CODE[ref.round_type]
                                   != vec["regime"][i])
        total += points
    max_dev = max(worst.values())
    return {"passed": bool(max_dev <= R3_TOL and regime_mismatch == 0),
            "tolerance": R3_TOL, "n_points": total,
            "max_deviation": max_dev,
            "max_deviation_by_quantity": worst,
            "n_regime_mismatch": regime_mismatch}
