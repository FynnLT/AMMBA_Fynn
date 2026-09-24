"""DR5: fairness of allocation, block 1 (A5).

**Classes of equals** (Chapter 2.2.5), per slot: same side; same
preference status (`served_first` where the run's order is
`preferences_first` and `preference_matched` holds, `not_served_first`
otherwise); and, for `served_first`, the same partner capacity share
c = min(1, partner_requested / own_requested), rounded to 1e-6.

The prediction is a Jain index of exactly 1 within every class. The short
side is filled completely and is trivially 1 -- reported, and marked. On the
long side the prediction is checked more strictly: a served-first member
fills c + (1 - c) * r, with r the slot's fill rate of the non-pair members.
"""
from collections import defaultdict

import numpy as np

import replay as R

SERVED, NOT_SERVED = "served_first", "not_served_first"
REASONS = ("order_pro_rata_first", "not_reciprocated", "partner_not_trading",
           "partner_same_side", "partner_never_in_run", "other")


def jain(values) -> float:
    """(sum x)^2 / (n * sum x^2); 1 for equal shares, 1/n when one takes all."""
    values = np.asarray(values, dtype=float)
    square = float((values ** 2).sum())
    if not len(values) or square == 0.0:
        return float("nan")
    return float(values.sum() ** 2 / (len(values) * square))


class DR5:
    def __init__(self):
        self.fair = defaultdict(lambda: {"slots": set(), "n_classes": 0,
                                         "n_singleton": 0,
                                         "n_members_nonsingleton": 0,
                                         "j_class": [], "residual": [],
                                         "residual_skipped": 0, "j_all": []})
        self.prefs = defaultdict(lambda: defaultdict(float))
        self.pair_kwh_max_dev = 0.0
        self.partner_rows_missing = 0

    def add(self, data) -> None:
        run = data.run
        if run.executed:
            return
        areas, slots = data.areas, data.slots
        idx = data.slot_index
        side = areas.strings("side")
        area = areas.strings("area_uuid")
        requested = areas.floats("requested_kwh")
        fill = areas.floats("fill_rate")
        matched = areas.bools("preference_matched")
        named = areas.bools("named")
        partner = areas.strings("partner")
        order = slots.strings("pref_order")[idx]
        enabled = slots.strings("pref_enabled")[idx] == "True"
        supply = slots.floats("total_supply_kwh")
        demand = slots.floats("total_demand_kwh")
        regime = R.regime_vec(supply, demand)

        served = enabled & (order == "preferences_first") & matched
        where = {(int(idx[i]), area[i]): i for i in range(areas.n)}
        partner_of = {}
        for i in np.where(named)[0]:
            partner_of.setdefault(area[i], partner[i])
        in_run = set(area.tolist())

        # --- capacity share of every served-first member
        capacity = np.full(areas.n, np.nan)
        for i in np.where(served)[0]:
            j = where.get((int(idx[i]), partner[i]))
            if j is None or side[j] == side[i]:
                self.partner_rows_missing += 1
                continue
            capacity[i] = min(1.0, requested[j] / requested[i])

        # --- classes and the residual check, per slot and side
        groups = defaultdict(list)
        for i in range(areas.n):
            groups[(int(idx[i]), side[i])].append(i)
        for (s, side_name), members in groups.items():
            members = np.array(members)
            short = ((side_name == "seller" and regime[s] == R.SL)
                     or (side_name == "buyer" and regime[s] == R.DL))
            role = ("balanced" if regime[s] == R.BAL
                    else "short" if short else "long")
            acc = self.fair[(run.cell, side_name, role)]
            acc["slots"].add((run.run_id, s))
            acc["j_all"].append(jain(fill[members]))
            classes = defaultdict(list)
            for i in members:
                if served[i]:
                    key = (SERVED, round(float(capacity[i]), 6)
                           if np.isfinite(capacity[i]) else None)
                else:
                    key = (NOT_SERVED, None)
                classes[key].append(i)
            for rows in classes.values():
                acc["n_classes"] += 1
                if len(rows) == 1:
                    acc["n_singleton"] += 1
                    continue
                acc["n_members_nonsingleton"] += len(rows)
                acc["j_class"].append(jain(fill[rows]))
            if role == "long":
                pair = members[served[members]]
                rest = members[~served[members]]
                if len(pair) and not len(rest):
                    acc["residual_skipped"] += len(pair)
                elif len(pair):
                    r = float(np.mean(fill[rest]))
                    c = capacity[pair]
                    ok = np.isfinite(c)
                    predicted = c[ok] + (1.0 - c[ok]) * r
                    acc["residual"].extend(
                        np.abs(fill[pair][ok] - predicted).tolist())

        # --- preference satisfaction, named participants only
        prefs = self.prefs[run.cell]
        seller_pair = defaultdict(float)
        for i in np.where(named)[0]:
            s = int(idx[i])
            prefs["n_named"] += 1
            prefs["claims_kwh"] += requested[i]
            if served[i]:
                prefs["n_served_first"] += 1
                if np.isfinite(capacity[i]):
                    served_kwh = capacity[i] * requested[i]
                    prefs["served_kwh"] += served_kwh
                    if side[i] == "seller":
                        seller_pair[s] += served_kwh
                continue
            prefs[self._reason(order[i], enabled[i], area[i], partner[i], s,
                               side[i], partner_of, where, side,
                               in_run)] += 1
        pair_kwh = slots.floats("pair_kwh")
        for s, kwh in seller_pair.items():
            self.pair_kwh_max_dev = max(self.pair_kwh_max_dev,
                                        abs(kwh - pair_kwh[s]))

    @staticmethod
    def _reason(order, enabled, own, partner, s, own_side, partner_of, where,
                side, in_run) -> str:
        if not enabled or order != "preferences_first":
            return "order_pro_rata_first"
        if partner not in in_run:
            return "partner_never_in_run"
        if partner_of.get(partner) != own:
            return "not_reciprocated"
        j = where.get((s, partner))
        if j is None:
            return "partner_not_trading"
        if side[j] == own_side:
            return "partner_same_side"
        return "other"

    # ---------------------------------------------------------- tables

    def fairness_table(self) -> list:
        out = []
        for (cell, side, role), acc in sorted(self.fair.items()):
            j_class = np.array(acc["j_class"], dtype=float)
            j_all = np.array(acc["j_all"], dtype=float)
            j_all = j_all[np.isfinite(j_all)]
            residual = np.array(acc["residual"], dtype=float)
            out.append({
                "cell": cell, "side": side, "side_role": role,
                "trivial": role == "short",
                "n_slots": len(acc["slots"]),
                "n_classes": acc["n_classes"],
                "n_singleton": acc["n_singleton"],
                "n_members_nonsingleton": acc["n_members_nonsingleton"],
                "j_class_min": float(np.nanmin(j_class))
                if len(j_class) else None,
                "j_class_median": float(np.nanmedian(j_class))
                if len(j_class) else None,
                "residual_n": len(residual),
                "residual_skipped": acc["residual_skipped"],
                "residual_max_abs_dev": float(residual.max())
                if len(residual) else None,
                "j_all_median": float(np.median(j_all)) if len(j_all) else None,
                "j_all_min": float(j_all.min()) if len(j_all) else None})
        return out

    def preferences_table(self) -> list:
        out = []
        for cell, acc in sorted(self.prefs.items()):
            n = acc["n_named"]
            out.append({
                "cell": cell, "n_named": int(n),
                "n_served_first": int(acc["n_served_first"]),
                "share_served_first": acc["n_served_first"] / n if n else None,
                "claims_kwh": acc["claims_kwh"],
                "served_kwh": acc["served_kwh"],
                "share_energy_served_first": acc["served_kwh"]
                / acc["claims_kwh"] if acc["claims_kwh"] else None,
                **{f"n_{reason}": int(acc[reason]) for reason in REASONS}})
        return out
