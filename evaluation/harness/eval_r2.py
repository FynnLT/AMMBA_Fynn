"""R2 - the multiplier side matrix against the supervisor's worked example.

The example: clearing price 50, green producers paid 10 ct extra per kWh,
funded by the grey market, green consumers untouched. Targets are
green producer 60, grey producer 48.75, grey consumer 51.25, green consumer 50.

The artifact has one global `sides` switch and does not differentiate buyers
by energy type. This run measures which cells it can reproduce.
"""
import asyncio, statistics, sys, time

import stack, scenario, runner, campaign
from campaign import OUT, write, write_manifest

# theta == the slot's actual ratio puts the sigmoid at the midpoint of
# [k_lower, k_upper], so the price is exactly 50.00.
SIGMOID_50 = {"k_upper": 100.0, "k_lower": 0.0, "theta": 1.25, "steepness": 2.5}

# 2.5 kWh green against 10.0 kWh grey - the 4:1 grey:green volume the
# example's funding condition requires. 10 kWh demand -> ratio 1.25.
SCEN = {
    "producers": [
        {"area_uuid": "gen-green", "name": "Green Gen", "energy": 2.5,
         "energy_type": "green"},
        {"area_uuid": "gen-grey", "name": "Grey Gen", "energy": 10.0,
         "energy_type": "grey"},
    ],
    "consumers": [
        {"area_uuid": "load-a", "name": "Load A", "energy": 4.0},
        {"area_uuid": "load-b", "name": "Load B", "energy": 3.5},
        {"area_uuid": "load-c", "name": "Load C", "energy": 2.5},
    ],
}

PREFS = {"enabled": True, "multipliers_enabled": True,
         "order": "pro_rata_first", "mode": "multiplicative",
         "sides": "seller", "green_multiplier": 0.2, "grey_levy": 0.05,
         "levy_cap": 0.2}

# The example's values. Used only as the `target_ct` column.
TARGET = {("producer", "green"): 60.00, ("producer", "grey"): 48.75,
          ("consumer", "grey"): 51.25, ("consumer", "green"): 50.00}

ARMS = [
    ("seller_multiplicative", "multiplicative", "seller"),
    ("both_multiplicative", "multiplicative", "both"),
    ("seller_additive", "additive", "seller"),
]


async def run_arm(st, arm, mode, sides):
    prefs = {**PREFS, "mode": mode, "sides": sides}
    r = await runner.run_slot(st, SCEN, preferences=prefs, execute=False,
                              sigmoid=SIGMOID_50)
    return r["clearing"], prefs


def party_rows(arm, mode, sides, clearing):
    """One row per party. The buyer side is uniform in the implementation, so
    both buyer rows carry the same measured rate against different targets -
    which is exactly the cell the example asks about."""
    m = clearing["preferences"]["multipliers"]
    rows = []
    for p in clearing["allocations"]["producers"]:
        etype = p["energy_type"]
        rows.append({"arm": arm, "mode": mode, "sides": sides,
                     "party": p["name"], "energy_type": etype,
                     "side": "producer",
                     "rate_ct": p["final_energy_rate"],
                     "target_ct": TARGET[("producer", etype)],
                     "delta_ct": round(p["final_energy_rate"]
                                       - TARGET[("producer", etype)], 6)})
    # The pipeline serves every buyer the same rate; the example distinguishes
    # green from grey consumers. Both comparisons are recorded against the one
    # measured buyer rate.
    buyer_rate = m["buyer_final_ct_per_kwh"]
    for etype in ("grey", "green"):
        rows.append({"arm": arm, "mode": mode, "sides": sides,
                     "party": f"all consumers (as {etype})",
                     "energy_type": etype, "side": "consumer",
                     "rate_ct": buyer_rate,
                     "target_ct": TARGET[("consumer", etype)],
                     "delta_ct": round(buyer_rate - TARGET[("consumer", etype)], 6)})
    return rows


def econ_row(arm, mode, sides, clearing):
    m = clearing["preferences"]["multipliers"]
    return {"arm": arm, "mode": mode, "sides": sides,
            "clearing_price_ct": clearing["clearing_price_ct_per_kwh"],
            "ratio": clearing["ratio"], "round_type": clearing["round_type"],
            "green_alloc_kwh": m["green_alloc_kwh"],
            "grey_alloc_kwh": m["grey_alloc_kwh"],
            "levy_collected_ct": m["levy_collected_ct"],
            "bonus_requested_ct": m["bonus_requested_ct"],
            "bonus_paid_ct": m["bonus_paid_ct"],
            "scale": m["scale"],
            "pool_surplus_ct": m["pool_surplus_ct"],
            "buyers_pay_ct": m["buyers_pay_ct"],
            "sellers_receive_ct": m["sellers_receive_ct"]}


def funding_rows(green_alloc, grey_alloc, price):
    """1.3 - the two numbers computed alongside, not from the pipeline.

    Funding condition of the example: bonus 10 ct on every green kWh must be
    paid out of 2.5 ct on every grey kWh, i.e. grey volume = 4x green volume,
    i.e. a green share of 1/5 by volume. The evaluation generator's default
    green share is a class attribute of scenario.make_scenario.
    """
    bonus_per_kwh = price * PREFS["green_multiplier"]
    levy_per_kwh = price * PREFS["grey_levy"]
    required_grey_per_green = bonus_per_kwh / levy_per_kwh
    green_share_required = 1.0 / (1.0 + required_grey_per_green)

    gen_green_share = scenario.make_scenario.__defaults__[
        scenario.make_scenario.__code__.co_varnames.index("green_share")
        - (scenario.make_scenario.__code__.co_argcount
           - len(scenario.make_scenario.__defaults__))]

    rows = [{
        "case": "funding_condition_of_the_example",
        "green_share_by_volume": round(green_share_required, 6),
        "grey_per_green_kwh": round(required_grey_per_green, 6),
        "bonus_per_kwh_ct": round(bonus_per_kwh, 6),
        "levy_per_kwh_ct": round(levy_per_kwh, 6),
        "implied_scale": 1.0,
        "note": "grey volume exactly funds the bonus",
    }]
    # Same parameters at the generator's green share: what would the subsidy
    # scale be? scale = min(1, levy_collected / bonus_requested).
    g = gen_green_share
    levy_collected = (1 - g) * price * PREFS["grey_levy"]
    bonus_requested = g * price * PREFS["green_multiplier"]
    rows.append({
        "case": "same_parameters_at_generator_green_share",
        "green_share_by_volume": g,
        "grey_per_green_kwh": round((1 - g) / g, 6),
        "bonus_per_kwh_ct": round(bonus_per_kwh, 6),
        "levy_per_kwh_ct": round(levy_per_kwh, 6),
        "implied_scale": round(min(1.0, levy_collected / bonus_requested), 6),
        "note": "levy cannot fund the bonus; green bonus is scaled down",
    })
    return rows, green_share_required, gen_green_share


# Control, not part of the matrix: at the example's parameters the levy funds
# the bonus exactly, the pool surplus is 0 and therefore
# buyer_final == sellers_receive / traded == price. That makes all three arms
# agree, which on its own is indistinguishable from `sides` not being wired at
# all. Breaking the funding balance separates the two readings.
WIRING_CHECK = {"grey_levy": 0.10}


async def run_wiring_check(st):
    rows = []
    for sides in ("seller", "both"):
        prefs = {**PREFS, **WIRING_CHECK, "sides": sides}
        r = await runner.run_slot(st, SCEN, preferences=prefs, execute=False,
                                  sigmoid=SIGMOID_50)
        c = r["clearing"]
        m = c["preferences"]["multipliers"]
        rows.append({"sides": sides, "mode": prefs["mode"],
                     "green_multiplier": prefs["green_multiplier"],
                     "grey_levy": prefs["grey_levy"],
                     "clearing_price_ct": c["clearing_price_ct_per_kwh"],
                     "green_final_ct": m["green_final_ct_per_kwh"],
                     "grey_final_ct": m["grey_final_ct_per_kwh"],
                     "buyer_final_ct": m["buyer_final_ct_per_kwh"],
                     "levy_collected_ct": m["levy_collected_ct"],
                     "bonus_paid_ct": m["bonus_paid_ct"],
                     "pool_surplus_ct": m["pool_surplus_ct"]})
    return rows


async def main():
    t0 = time.perf_counter()
    st = stack.Stack()
    try:
        print(f"repo SHA {campaign.SHA}")
        all_rows, econ = [], []
        alloc = None
        for arm, mode, sides in ARMS:
            clearing, prefs = await run_arm(st, arm, mode, sides)
            if clearing.get("status") != "cleared":
                raise RuntimeError(f"arm {arm} did not clear: {clearing}")
            all_rows += party_rows(arm, mode, sides, clearing)
            econ.append(econ_row(arm, mode, sides, clearing))
            if alloc is None:
                m = clearing["preferences"]["multipliers"]
                alloc = (m["green_alloc_kwh"], m["grey_alloc_kwh"],
                         clearing["clearing_price_ct_per_kwh"],
                         clearing["allocations"]["producers"])

        green_alloc, grey_alloc, price, prods = alloc
        fund, share_req, gen_share = funding_rows(green_alloc, grey_alloc, price)

        wiring = await run_wiring_check(st)

        write("R2_side_matrix.csv", all_rows)
        write("R2_sides_wiring_check.csv", wiring)
        write("R2_side_economics.csv", econ)
        write("R2_funding_condition.csv", fund)

        # --- printed tables ---
        print("\nprice / allocation check")
        print(f"  clearing price      {price} ct/kWh")
        print(f"  green allocated     {green_alloc} kWh")
        print(f"  grey allocated      {grey_alloc} kWh")
        print(f"  grey:green ratio    {grey_alloc / green_alloc:.6f}")
        for p in prods:
            print(f"    {p['name']:<10} requested {p['requested_kwh']:>6} "
                  f"allocated {p['allocated_kwh']:>6} "
                  f"fill {p['fill_rate']:.6f}")

        print("\nside matrix (rate vs the example's target)")
        for arm, _, _ in ARMS:
            print(f"  {arm}")
            for r in all_rows:
                if r["arm"] != arm:
                    continue
                flag = "ok" if abs(r["delta_ct"]) < 1e-6 else "MISS"
                print(f"    {r['party']:<22} {r['energy_type']:<6} "
                      f"{r['side']:<9} rate {r['rate_ct']:>9} "
                      f"target {r['target_ct']:>7} "
                      f"delta {r['delta_ct']:>+10.4f}  {flag}")

        print("\nround economics per arm")
        for e in econ:
            print(f"  {e['arm']:<22} levy {e['levy_collected_ct']:>8.4f}  "
                  f"bonus_req {e['bonus_requested_ct']:>8.4f}  "
                  f"bonus_paid {e['bonus_paid_ct']:>8.4f}  "
                  f"scale {e['scale']:>8.6f}  "
                  f"surplus {e['pool_surplus_ct']:>+9.4f}")

        print("\nsides wiring check (levy 0.10, deliberately over-collecting)")
        for w in wiring:
            print(f"  sides={w['sides']:<7} green {w['green_final_ct']:>9.4f}  "
                  f"grey {w['grey_final_ct']:>9.4f}  "
                  f"buyer {w['buyer_final_ct']:>9.4f}  "
                  f"surplus {w['pool_surplus_ct']:>+10.4f}")

        print("\nfunding condition (computed, not from the pipeline)")
        for f in fund:
            print(f"  {f['case']:<42} green share "
                  f"{f['green_share_by_volume']:.4f}  "
                  f"scale {f['implied_scale']:.6f}")

        wall = time.perf_counter() - t0
        write_manifest("R2_side_matrix",
                       params={"scenario": "hand-built 2.5 green / 10.0 grey "
                                           "vs 10.0 demand, ratio 1.25",
                               "sigmoid": SIGMOID_50,
                               "preferences_base": PREFS,
                               "arms": [{"arm": a, "mode": m, "sides": s}
                                        for a, m, s in ARMS],
                               "example_targets": {f"{k[1]}_{k[0]}": v
                                                   for k, v in TARGET.items()},
                               "generator_green_share": gen_share},
                       seeds=[None], wall_sec=wall,
                       output_files=["R2_side_matrix.csv",
                                     "R2_side_economics.csv",
                                     "R2_sides_wiring_check.csv",
                                     "R2_funding_condition.csv"])
        return 0
    finally:
        await st.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
