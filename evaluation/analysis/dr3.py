"""DR3: individual rationality per participant (A3).

Realised utility per participant-slot, against an outside option of zero in
margin terms:

    seller: u = (r - K_lower) * s - Phi - phi + c
    buyer:  u = (K_upper - r_b) * min(q, actual) - r_b * [q - actual]^+ - phi + c

with r the seller's final rate by energy type, r_b the buyer's final rate
and c the computed compensation (empty means 0). Block 1 does not execute:
its penalties are zero, `actual = q` for its buyers, and IR there is about
the levy.

A violation is u < -1e-9. Its cause is every adjustment whose removal
*alone* restores u >= 0; a violation no single removal restores is
`combined`.
"""
from collections import defaultdict

import numpy as np

import replay as R

IR_TOL = 1e-9
CAUSES = ("levy", "shortfall", "externality", "unneeded_energy", "combined")
#: The two levy levels Eq. (4.8) is read at: the configured one and the cap.
EXPOSURE_LEVIES = (0.10, 0.20)
#: The reference cells of the exposure count: the delta baseline on the
#: reference battery window, and the overlap window's own baseline (D-83).
EXPOSURE_CELLS = ("baseline_pro_rata", "mult_overlap_off")


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict.fromkeys(("min", "p10", "median", "mean", "p90", "max"))
    return {"min": float(values.min()),
            "p10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(values.max())}


def realised(data) -> dict:
    """Per area row: u, the adjustments, and the rates it was settled at."""
    run, areas, slots = data.run, data.areas, data.slots
    band = R.Band.from_dict(run.sigmoid)
    idx = data.slot_index
    seller = areas.strings("side") == "seller"
    grey = areas.strings("energy_type") == "grey"
    q = areas.floats("allocated_kwh")
    price = slots.floats("clearing_price_ct")[idx]

    def rate(column):
        values = data.slot_value(column) if slots.has(column) else price
        return np.where(np.isnan(values), price, values)

    r_seller = np.where(grey, rate("grey_final_ct"), rate("green_final_ct"))
    r_buyer = rate("buyer_final_ct")
    shortfall = areas.floats_or_zero("shortfall_penalty_ct")
    externality = areas.floats_or_zero("externality_penalty_ct")
    compensation = areas.floats_or_zero("compensation_ct")
    actual = areas.floats("actual_kwh")
    actual = np.where(np.isnan(actual), q, actual)

    used = np.minimum(q, actual)
    unneeded = np.maximum(q - actual, 0.0)
    u = np.where(
        seller,
        (r_seller - band.k_lower) * q - shortfall - externality + compensation,
        (band.k_upper - r_buyer) * used - r_buyer * unneeded - externality
        + compensation)
    levy_ct = np.where(seller & grey, np.maximum(price - r_seller, 0.0) * q, 0.0)
    unneeded_ct = np.where(~seller, r_buyer * unneeded, 0.0)
    deviators = set(run.deviators)
    named = np.array([a in deviators for a in areas.strings("area_uuid")])
    return {"u": u, "seller": seller, "grey": grey, "q": q, "price": price,
            "r_seller": r_seller, "r_buyer": r_buyer, "shortfall": shortfall,
            "externality": externality, "compensation": compensation,
            "levy_ct": levy_ct, "unneeded_ct": unneeded_ct, "named": named,
            "band": band}


def causes(rec) -> dict:
    """Boolean masks: violation, and each single removal that restores it."""
    u = rec["u"]
    violation = u < -IR_TOL
    restored = {
        "levy": violation & (rec["levy_ct"] > 0)
                & (u + rec["levy_ct"] >= -IR_TOL),
        "shortfall": violation & (rec["shortfall"] > 0)
                     & (u + rec["shortfall"] >= -IR_TOL),
        "externality": violation & (rec["externality"] > 0)
                       & (u + rec["externality"] >= -IR_TOL),
        "unneeded_energy": violation & (rec["unneeded_ct"] > 0)
                           & (u + rec["unneeded_ct"] >= -IR_TOL),
    }
    single = np.zeros_like(violation)
    for mask in restored.values():
        single |= mask
    restored["combined"] = violation & ~single
    restored["any"] = violation
    return restored


class DR3:
    def __init__(self):
        self.n = defaultdict(int)
        self.viol = defaultdict(lambda: [0, np.inf, 0.0])
        self.rates = defaultdict(list)
        self.eq48 = []
        self.exposure = defaultdict(list)
        self.bands = {}

    def add(self, data) -> None:
        run = data.run
        rec = realised(data)
        masks = causes(rec)
        for side, side_mask in (("seller", rec["seller"]),
                                ("buyer", ~rec["seller"])):
            for group, group_mask in (("named", rec["named"]),
                                      ("other", ~rec["named"])):
                base = side_mask & group_mask
                if not base.any():
                    continue
                self.n[(run.cell, side, group)] += int(base.sum())
                for cause, mask in masks.items():
                    hit = base & mask
                    if not hit.any():
                        continue
                    acc = self.viol[(run.cell, side, group, cause)]
                    acc[0] += int(hit.sum())
                    acc[1] = min(acc[1], float(rec["u"][hit].min()))
                    acc[2] += float(rec["u"][hit].sum())

        q = rec["q"]
        with np.errstate(divide="ignore", invalid="ignore"):
            seller_rate = (rec["r_seller"] * q - rec["shortfall"]
                           - rec["externality"] + rec["compensation"]) / q
            buyer_rate = (rec["r_buyer"] * q + rec["externality"]
                          - rec["compensation"]) / q
        traded = q > 0
        self.rates[(run.cell, "seller")].append(
            seller_rate[traded & rec["seller"]])
        self.rates[(run.cell, "buyer")].append(
            buyer_rate[traded & ~rec["seller"]])

        self._eq48(data, rec)

    def _eq48(self, data, rec) -> None:
        run, slots = data.run, data.slots
        band = rec["band"]
        self.bands[run.cell] = band
        if slots.has("mult_enabled"):
            enabled = slots.bools("mult_enabled")[data.slot_index]
            rows = enabled & rec["seller"] & rec["grey"] & (rec["q"] > 0)
            if rows.any():
                grey_final = rec["r_seller"][rows]
                lam = 1.0 - grey_final / rec["price"][rows]
                below = grey_final < band.k_lower
                self.eq48.append({
                    "cell": run.cell, "n": int(rows.sum()),
                    "n_below": int(below.sum()),
                    "lambda_min": float(lam.min()),
                    "lambda_max": float(lam.max()),
                    "min_grey_final_ct": float(grey_final.min()),
                    "n_slots_below": int(len(set(
                        data.slot_index[rows][below].tolist())))})
        if run.cell in EXPOSURE_CELLS:
            grey_rows = rec["seller"] & rec["grey"] & (rec["q"] > 0)
            grey_slots = np.unique(data.slot_index[grey_rows])
            self.exposure[run.cell].append(
                slots.floats("clearing_price_ct")[grey_slots])

    # ---------------------------------------------------------- tables

    def ir_table(self) -> list:
        out = []
        for (cell, side, group), n in sorted(self.n.items()):
            for cause in ("any",) + CAUSES:
                count, min_u, sum_u = self.viol.get(
                    (cell, side, group, cause), (0, np.inf, 0.0))
                out.append({"cell": cell, "side": side, "group": group,
                            "cause": cause, "n": n, "n_violations": count,
                            "share": count / n if n else None,
                            "min_u_ct": min_u if count else None,
                            "sum_negative_u_ct": sum_u if count else 0.0})
        return out

    def settlement_table(self) -> list:
        out = []
        for (cell, side), parts in sorted(self.rates.items()):
            values = np.concatenate(parts) if parts else np.array([])
            stats = _quantiles(values)
            out.append({"cell": cell, "side": side, "n": len(values),
                        **{f"rate_{k}_ct": v for k, v in stats.items()}})
        return out

    def eq48_table(self) -> list:
        out = []
        measured = defaultdict(list)
        for row in self.eq48:
            measured[row["cell"]].append(row)
        for cell, rows in sorted(measured.items()):
            n = sum(r["n"] for r in rows)
            below = sum(r["n_below"] for r in rows)
            out.append({"part": "measured", "cell": cell, "levy": None,
                        "n": n, "n_below": below,
                        "share_below": below / n if n else None,
                        "lambda_min": min(r["lambda_min"] for r in rows),
                        "lambda_max": max(r["lambda_max"] for r in rows),
                        "min_price_ct": None,
                        "min_grey_final_ct": min(r["min_grey_final_ct"]
                                                 for r in rows),
                        "price_threshold_ct": None, "ratio_threshold": None,
                        "n_slots_below": sum(r["n_slots_below"] for r in rows)})
        for cell, parts in sorted(self.exposure.items()):
            prices = np.concatenate(parts) if parts else np.array([])
            band = self.bands[cell]
            for levy in EXPOSURE_LEVIES:
                threshold = band.k_lower / (1.0 - levy)
                below = int((prices < threshold).sum())
                out.append({"part": "exposure", "cell": cell, "levy": levy,
                            "n": len(prices), "n_below": below,
                            "share_below": below / len(prices)
                            if len(prices) else None,
                            "lambda_min": None, "lambda_max": None,
                            "min_price_ct": float(prices.min())
                            if len(prices) else None,
                            "min_grey_final_ct": None,
                            "price_threshold_ct": threshold,
                            "ratio_threshold": band.ratio_at_price(threshold),
                            "n_slots_below": below})
        return out
