"""DR4: balance per layer, and how much of the damage the pool covers (A4).

**Balance per layer**, per executed slot:

* externality: `penalty_pool_ct - compensated_ct`. The pool is distributed
  in full whenever the round's total damage exceeds epsilon, for any number
  of deviators; a round below it reports its pool undistributed and shows
  up here.
* shortfall: the retained sum of `shortfall_penalty_ct`, >= 0 by
  construction (weak balance).
* energy origin: `pool_surplus_ct` by mode and sides, over every cleared
  slot with the multipliers on (no executed cell runs them).

**Coverage** against a joint counterfactual that puts back every deviator's
quantity at once, priced with `sigmoid_price`:

* penalizable: W_j = [A_j - s_j - eta_j]^+ for the node's deviators, as the
  node computes it; buyers U_i = [actual - requested]^+;
* total: W_j = [A_j - s_j]^+ over the whole deviating side, deadband
  removed (identical for buyers, who have none).

damage = |p - p_cf,joint| * Q_harmed, Q_harmed the traded quantity of the
harmed side -- the whole side, since the deviators sit on the other one.
The deviating side is the regime's penalised side: sellers in a
supply-limited round, buyers in a demand-limited one.
"""
from collections import defaultdict

import numpy as np

import replay as R

BALANCE_TOL = 1e-6
COVERAGE_TOL = 1e-6
BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"),
           (6, 10, "6-10"), (11, 10**9, ">10"))


def bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


class DR4:
    def __init__(self):
        self.layers = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(list)
        self.coverage = defaultdict(lambda: {"c_pen": [], "c_tot": [],
                                             "pool": 0.0, "compensated": 0.0,
                                             "damage_pen": 0.0,
                                             "damage_tot": 0.0})
        self.single = {"n_slots": 0, "n_undistributed": 0,
                       "undistributed_max_pool_ct": 0.0,
                       "max_abs_deviation_ct": 0.0,
                       "max_ratio_deviation": 0.0, "failures": []}
        self.n_multi = 0
        self.n_multi_column = 0
        self.n_executed_slots = 0
        self.deviator_count_mismatch = []

    def add(self, data) -> None:
        run, slots = data.run, data.slots
        self._energy_origin(data)
        if not run.executed:
            return
        executed = np.array([v not in ("", "None")
                             for v in slots.raw("round_type_exec")])
        pool = slots.floats_or_zero("penalty_pool_ct")
        comp = slots.floats_or_zero("compensated_ct")
        balance = slots.floats_or_zero("budget_balance_ct")
        shortfall = slots.floats_or_zero("shortfall_penalty_ct")
        imbalance = pool - comp
        layer = self.layers[("externality", run.cell)]
        layer["n_slots"] += int(executed.sum())
        layer["n_slots_nonzero"] += int((executed & (pool > 0)).sum())
        layer["n_imbalanced"] += int(
            (executed & (np.abs(imbalance) > BALANCE_TOL)).sum())
        layer["n_undistributed"] += int(
            (executed & (pool > 0) & (comp == 0)).sum())
        layer["n_budget_balance_nonzero"] += int(
            (executed & (np.abs(balance) > BALANCE_TOL)).sum())
        layer["sum_pool_ct"] += float(pool[executed].sum())
        layer["sum_compensated_ct"] += float(comp[executed].sum())
        layer["sum_balance_ct"] += float(imbalance[executed].sum())
        layer["max_abs_balance_ct"] = max(
            layer["max_abs_balance_ct"],
            float(np.abs(imbalance[executed]).max()) if executed.any() else 0.0)
        layer = self.layers[("shortfall", run.cell)]
        layer["n_slots"] += int(executed.sum())
        layer["n_slots_nonzero"] += int((executed & (shortfall > 0)).sum())
        layer["sum_pool_ct"] += float(shortfall[executed].sum())
        layer["min_ct"] = min(layer.get("min_ct", np.inf),
                              float(shortfall[executed].min())
                              if executed.any() else np.inf)
        self.n_executed_slots += int(executed.sum())
        self._coverage(data, executed, pool, comp)

    def _energy_origin(self, data) -> None:
        slots = data.slots
        if not slots.has("mult_enabled"):
            return
        enabled = slots.bools("mult_enabled")
        if not enabled.any():
            return
        modes = slots.strings("mult_mode")
        sides = slots.strings("mult_sides")
        surplus = slots.floats("pool_surplus_ct")
        for i in np.where(enabled)[0]:
            self.energy[(data.run.cell, modes[i], sides[i])].append(surplus[i])

    def _coverage(self, data, executed, pool, comp) -> None:
        run, areas, slots = data.run, data.areas, data.slots
        band = R.Band.from_dict(run.sigmoid)
        n_slots = slots.n
        idx = data.slot_index
        seller = areas.strings("side") == "seller"
        allocated = areas.floats("allocated_kwh")
        requested = areas.floats("requested_kwh")
        deliverable = areas.floats_or_zero("deliverable_kwh")
        actual = areas.floats_or_zero("actual_kwh")
        ext = areas.floats_or_zero("externality_penalty_ct")
        eta = slots.floats("eta_relative_eff")[idx]
        supply = slots.floats("total_supply_kwh")
        demand = slots.floats("total_demand_kwh")
        price = slots.floats("clearing_price_ct")
        traded = slots.floats("traded_kwh")
        regime = R.regime_vec(supply, demand)
        row_regime = regime[idx]

        # The deviating side of each row's slot, and its two quantities.
        sl_side = seller & (row_regime == R.SL)
        dl_side = ~seller & (row_regime == R.DL)
        w_pen = np.where(sl_side, np.maximum(
            0.0, deliverable - allocated - eta * allocated), 0.0)
        w_tot = np.where(sl_side, np.maximum(0.0, deliverable - allocated), 0.0)
        u = np.where(dl_side, np.maximum(0.0, actual - requested), 0.0)
        node_dev = ext > 0
        pen_qty = np.where(node_dev, w_pen + u, 0.0)
        tot_qty = np.where(w_tot + u > R.EPSILON, w_tot + u, 0.0)

        n_dev = np.bincount(idx[node_dev], minlength=n_slots)
        sum_pen = np.bincount(idx, weights=pen_qty, minlength=n_slots)
        sum_tot = np.bincount(idx, weights=tot_qty, minlength=n_slots)
        deviators = set(run.deviators)
        area_ids = areas.strings("area_uuid")
        named_dev = np.bincount(
            idx[node_dev & np.array([a in deviators for a in area_ids])],
            minlength=n_slots)

        column = slots.floats("n_deviators")
        mismatch = executed & (np.nan_to_num(column, nan=-1) != n_dev)
        if mismatch.any():
            self.deviator_count_mismatch.append(
                (run.run_id, int(mismatch.sum())))
        self.n_multi += int((executed & (n_dev > 1)).sum())
        self.n_multi_column += int((executed & (column > 1)).sum())

        arm = run.arm or "none"
        for s in np.where(executed & ((n_dev > 0) | (sum_tot > R.EPSILON))
                          & (regime != R.BAL))[0]:
            damage = {}
            for key, qty in (("pen", sum_pen[s]), ("tot", sum_tot[s])):
                if qty <= R.EPSILON:
                    # Nothing put back: no counterfactual, no damage -- and
                    # not the six-decimal gap between the recorded price and
                    # a re-evaluated f(E / D_hat).
                    damage[key] = 0.0
                    continue
                if regime[s] == R.SL:
                    ratio = (supply[s] + qty) / demand[s]
                else:
                    ratio = supply[s] / (demand[s] + qty)
                p_cf = band.price(ratio)
                damage[key] = abs(price[s] - p_cf) * traded[s]
            c_pen = comp[s] / damage["pen"] if damage["pen"] > 0 else np.nan
            c_tot = comp[s] / damage["tot"] if damage["tot"] > 0 else np.nan
            if n_dev[s] == 0:
                klass = "none"
            elif named_dev[s] == n_dev[s]:
                klass = "all_named"
            elif named_dev[s] == 0:
                klass = "all_accidental"
            else:
                klass = "mixed"
            acc = self.coverage[(run.cell, arm, bucket(int(n_dev[s])), klass)]
            acc["c_pen"].append(c_pen)
            acc["c_tot"].append(c_tot)
            acc["pool"] += float(pool[s])
            acc["compensated"] += float(comp[s])
            acc["damage_pen"] += float(damage["pen"])
            acc["damage_tot"] += float(damage["tot"])

            if n_dev[s] == 1 and pool[s] > 0 and comp[s] == 0:
                # The one case the rule does not distribute: total damage
                # <= eps, the pool reported undistributed (budget_balance_ct
                # = pool). Counted, not held to C_pen = 1.
                self.single["n_undistributed"] += 1
                self.single["undistributed_max_pool_ct"] = max(
                    self.single["undistributed_max_pool_ct"], float(pool[s]))
            elif n_dev[s] == 1:
                # The node's counterfactual *is* the joint one here, so the
                # pool must cover the penalizable damage exactly. The
                # recorded amounts carry six decimals, hence the absolute
                # floor beside the ratio.
                deviation = abs(comp[s] - damage["pen"])
                ratio_dev = abs(c_pen - 1.0) if np.isfinite(c_pen) else np.inf
                self.single["n_slots"] += 1
                self.single["max_abs_deviation_ct"] = max(
                    self.single["max_abs_deviation_ct"], deviation)
                self.single["max_ratio_deviation"] = max(
                    self.single["max_ratio_deviation"], ratio_dev)
                if deviation > COVERAGE_TOL and ratio_dev > COVERAGE_TOL:
                    self.single["failures"].append(
                        {"run": run.run_id, "slot": int(slots.ints("slot")[s]),
                         "compensated_ct": float(comp[s]),
                         "damage_pen_ct": float(damage["pen"])})

    # ---------------------------------------------------------- results

    def single_deviator_check(self) -> dict:
        failures = self.single["failures"]
        return {"passed": not failures and not self.deviator_count_mismatch,
                "tolerance": COVERAGE_TOL,
                "n_slots": self.single["n_slots"],
                "n_undistributed_excluded": self.single["n_undistributed"],
                "undistributed_max_pool_ct":
                    self.single["undistributed_max_pool_ct"],
                "max_abs_deviation_ct": self.single["max_abs_deviation_ct"],
                "max_ratio_deviation": self.single["max_ratio_deviation"],
                "n_failures": len(failures), "failures": failures[:10],
                "deviator_count_mismatch": self.deviator_count_mismatch}

    def layers_table(self) -> list:
        out = []
        for (layer, cell), acc in sorted(self.layers.items()):
            out.append({"layer": layer, "cell": cell, "mode": None,
                        "sides": None, **{k: acc.get(k) for k in (
                            "n_slots", "n_slots_nonzero", "n_imbalanced",
                            "n_undistributed", "n_budget_balance_nonzero",
                            "sum_pool_ct", "sum_compensated_ct",
                            "sum_balance_ct", "max_abs_balance_ct")},
                        "min_ct": acc.get("min_ct"), "max_ct": None})
        for (cell, mode, sides), values in sorted(self.energy.items()):
            values = np.asarray(values, dtype=float)
            out.append({"layer": "energy_origin", "cell": cell, "mode": mode,
                        "sides": sides, "n_slots": len(values),
                        "n_slots_nonzero": int((np.abs(values)
                                                > BALANCE_TOL).sum()),
                        "sum_pool_ct": float(values.sum()),
                        "min_ct": float(values.min()),
                        "max_ct": float(values.max())})
        return out

    def coverage_table(self) -> list:
        order = {name: i for i, (_lo, _hi, name) in enumerate(BUCKETS)}
        out = []
        for (cell, arm, bucket_name, klass), acc in sorted(
                self.coverage.items(),
                key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]],
                                kv[0][3])):
            c_pen = np.array(acc["c_pen"], dtype=float)
            c_tot = np.array(acc["c_tot"], dtype=float)

            def stats(values, name):
                values = values[np.isfinite(values)]
                if not len(values):
                    return {f"{name}_median": None, f"{name}_min": None,
                            f"{name}_max": None}
                return {f"{name}_median": float(np.median(values)),
                        f"{name}_min": float(values.min()),
                        f"{name}_max": float(values.max())}

            out.append({"cell": cell, "arm": arm, "n_deviators": bucket_name,
                        "deviator_class": klass, "n_slots": len(c_pen),
                        **stats(c_pen, "c_pen"), **stats(c_tot, "c_tot"),
                        "sum_pool_ct": acc["pool"],
                        "sum_compensated_ct": acc["compensated"],
                        "sum_damage_pen_ct": acc["damage_pen"],
                        "sum_damage_tot_ct": acc["damage_tot"]})
        return out
