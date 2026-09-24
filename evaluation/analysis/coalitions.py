"""The coalition cells: nesting and additivity (A6).

`deviations.plan_deviations` draws `random.Random(seed).sample(households,
k)` with the same seed and the same household list in every coalition cell,
and `sample` returns a prefix-consistent draw: the k = 2 set contains the
k = 1 set, and so on. The draws are therefore nested, not independent per
cell. This module asserts that from the data rather than from the code:

1. the deviator sets of `*_s25` (k = 1), `*_k02`, `*_k05` and `*_k10` are
   nested per arm and identical across the five seeds;
2. per seed, the slot aggregates are identical across the four cells (the
   harness deviates after clearing, so the book cannot differ);
3. per deviator and seed, its summed externality penalty is identical across
   the cells it appears in: the node's counterfactual adds back only the
   deviator's own quantity.
"""
from collections import defaultdict

import numpy as np

import replay as R

ARMS = {"sellers_withhold": "b2_sell", "buyers_underreport": "b2_buy"}
SIZES = (("s25", 1), ("k02", 2), ("k05", 5), ("k10", 10))
#: What the code predicts (deviation seed 20260919, the 100-player draw).
EXPECTED = {1: ("player-194",),
            2: ("player-030",),
            5: ("player-041", "player-112", "player-196"),
            10: ("player-107", "player-127", "player-223", "player-241",
                 "player-248")}
PENALTY_TOL = 1e-6


def coalition_cells() -> dict:
    return {f"{prefix}_{suffix}": (arm, k)
            for arm, prefix in ARMS.items() for suffix, k in SIZES}


def expected_set(k: int) -> set:
    return {a for size, members in EXPECTED.items() if size <= k
            for a in members}


class Coalitions:
    def __init__(self):
        self.cells = coalition_cells()
        self.sets = {}
        self.aggregates = {}
        self.members = {}

    def add(self, data) -> None:
        run = data.run
        if run.cell not in self.cells:
            return
        arm, _k = self.cells[run.cell]
        key = (run.cell, run.seed)
        self.sets[key] = tuple(run.deviators)
        slots, areas = data.slots, data.areas
        supply = slots.floats("total_supply_kwh")
        demand = slots.floats("total_demand_kwh")
        self.aggregates[key] = (slots.ints("slot"), supply, demand,
                                slots.floats("clearing_price_ct"),
                                R.regime_vec(supply, demand))

        ids = areas.strings("area_uuid")
        seller = areas.strings("side") == "seller"
        ext = areas.floats_or_zero("externality_penalty_ct")
        allocated = areas.floats("allocated_kwh")
        eta = data.slot_value("eta_relative_eff")
        withheld = np.maximum(0.0, areas.floats_or_zero("deliverable_kwh")
                              - allocated - eta * allocated)
        under = np.maximum(0.0, areas.floats_or_zero("actual_kwh")
                           - areas.floats("requested_kwh"))
        penalized = np.where(seller, withheld, under)
        arm_side = seller if arm == "sellers_withhold" else ~seller
        members = {}
        for deviator in run.deviators:
            rows = (ids == deviator) & arm_side & (ext > 0)
            members[deviator] = (float(ext[rows].sum()),
                                 float(penalized[rows].sum()))
        self.members[key] = members

    # ---------------------------------------------------------- results

    def assertions(self) -> dict:
        out = {}
        for arm, prefix in ARMS.items():
            cells = [(f"{prefix}_{suffix}", k) for suffix, k in SIZES]
            seeds = sorted({seed for (cell, seed) in self.sets
                            if cell in dict(cells)})
            missing = [c for c, _k in cells
                       if not any((c, s) in self.sets for s in seeds)]
            if missing:
                out[f"a6_{arm}"] = {"passed": False,
                                    "reason": f"coalition cells missing: "
                                              f"{missing}"}
                continue

            # 1. nested, and the same set in every seed
            per_cell = {}
            identical = True
            for cell, _k in cells:
                found = {self.sets[(cell, s)] for s in seeds
                         if (cell, s) in self.sets}
                identical &= len(found) == 1
                per_cell[cell] = sorted(set().union(*map(set, found)))
            nested = all(set(per_cell[cells[i][0]])
                         <= set(per_cell[cells[i + 1][0]])
                         for i in range(len(cells) - 1))
            as_expected = all(set(per_cell[cell]) == expected_set(k)
                              for cell, k in cells)
            out[f"a6_{arm}_nested"] = {
                "passed": bool(nested and identical),
                "nested": bool(nested), "identical_across_seeds": identical,
                "sets": per_cell}
            out[f"a6_{arm}_matches_code"] = {
                "passed": bool(as_expected),
                "expected": {cell: sorted(expected_set(k))
                             for cell, k in cells}}

            # 2. slot aggregates identical per seed
            worst = 0.0
            regimes_equal = True
            compared = 0
            for seed in seeds:
                ref = self.aggregates.get((cells[0][0], seed))
                for cell, _k in cells[1:]:
                    other = self.aggregates.get((cell, seed))
                    if ref is None or other is None:
                        continue
                    compared += 1
                    if not np.array_equal(ref[0], other[0]):
                        worst = np.inf
                        continue
                    for a, b in zip(ref[1:4], other[1:4]):
                        if not np.array_equal(np.isnan(a), np.isnan(b)):
                            worst = np.inf
                            continue
                        diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
                        worst = max(worst, float(diff.max()))
                    regimes_equal &= bool(np.array_equal(ref[4], other[4]))
            out[f"a6_{arm}_aggregates_identical"] = {
                "passed": bool(worst == 0.0 and regimes_equal and compared),
                "max_deviation": worst, "round_types_identical": regimes_equal,
                "n_comparisons": compared}

            # 3. each member's penalty identical across the cells it is in
            worst = 0.0
            compared = 0
            for seed in seeds:
                seen = defaultdict(list)
                for cell, _k in cells:
                    for member, (pen, _kwh) in self.members.get(
                            (cell, seed), {}).items():
                        seen[member].append(pen)
                for values in seen.values():
                    if len(values) > 1:
                        compared += 1
                        worst = max(worst, max(values) - min(values))
            out[f"a6_{arm}_member_penalty_identical"] = {
                "passed": bool(worst <= PENALTY_TOL and compared),
                "tolerance": PENALTY_TOL, "max_deviation_ct": worst,
                "n_comparisons": compared}
        return out

    def table(self) -> list:
        out = []
        for arm, prefix in ARMS.items():
            first = EXPECTED[1][0]
            for suffix, k in SIZES:
                cell = f"{prefix}_{suffix}"
                seeds = sorted(s for (c, s) in self.members if c == cell)
                if not seeds:
                    continue
                for seed in seeds + ["all"]:
                    chosen = seeds if seed == "all" else [seed]
                    pen = kwh = pen1 = kwh1 = 0.0
                    for s in chosen:
                        for member, (p, q) in self.members[(cell, s)].items():
                            pen += p
                            kwh += q
                            if member == first:
                                pen1 += p
                                kwh1 += q
                    out.append({
                        "arm": arm, "cell": cell, "k": k, "seed": seed,
                        "deviators": " ".join(sorted(
                            self.sets[(cell, chosen[0])])),
                        "coalition_penalty_ct": pen,
                        "penalized_kwh": kwh,
                        "penalty_per_kwh_ct": pen / kwh if kwh else None,
                        "k1_member": first,
                        "k1_penalty_ct": pen1, "k1_penalized_kwh": kwh1,
                        "k1_penalty_per_kwh_ct": pen1 / kwh1 if kwh1 else None})
        return out
