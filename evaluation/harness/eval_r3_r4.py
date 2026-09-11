"""R3 - oracle against meter through the pipeline.
R4 - the buyer externality, and the regime boundary.

Both use hand-built scenarios rather than the synthetic generator: every
quantity here has to be exact (a fixed ratio, a fixed capacity, a fixed
under-report), and the generator's log-normal draws would only add noise to
numbers that are meant to be read off directly.
"""
import asyncio, statistics, sys, time

import stack, scenario, runner, campaign
from campaign import OUT, write, write_manifest

SIG = dict(runner.SIGMOID)          # 28.5 / 8.0 / theta 1.0 / B 2.5
PREF_OFF = {"enabled": False, "multipliers_enabled": False}
GAMMA, ETA = 1.1, 0.0

sigmoid_price = stack.sigmoid_price


# =====================================================================
# R3 - oracle vs meter
# =====================================================================
# Supply-limited: 8.0 kWh supply against 10.0 kWh demand, ratio 0.8.
# The withholder can deliver 5.0 but offers only 3.0, and then delivers
# exactly the 3.0 it sold - a successful withholder, no shortfall.
R3_WITHHOLDER = "gen-withholder"
R3_CAPACITY = 5.0
R3_OFFERED = 3.0

R3_SCEN = {
    "producers": [
        {"area_uuid": R3_WITHHOLDER, "name": "Withholder", "energy": R3_OFFERED,
         "energy_type": "green"},
        {"area_uuid": "gen-1", "name": "Gen 1", "energy": 2.5,
         "energy_type": "green"},
        {"area_uuid": "gen-2", "name": "Gen 2", "energy": 1.5,
         "energy_type": "green"},
        {"area_uuid": "gen-3", "name": "Gen 3", "energy": 1.0,
         "energy_type": "grey"},
    ],
    "consumers": [
        {"area_uuid": "load-1", "name": "Load 1", "energy": 4.0},
        {"area_uuid": "load-2", "name": "Load 2", "energy": 3.5},
        {"area_uuid": "load-3", "name": "Load 3", "energy": 2.5},
    ],
}


async def run_r3(st):
    t0 = time.perf_counter()
    rows, totals = [], {}
    # Supply-limited -> every offer is filled in full, so the meter reading
    # that reproduces "delivered exactly what was sold" is the offer itself.
    measurements = {p["area_uuid"]: p["energy"] for p in R3_SCEN["producers"]}

    for variant, forecasts in (
            ("B2-meter", None),
            ("B2-oracle", {R3_WITHHOLDER: R3_CAPACITY})):
        r = await runner.run_slot(st, R3_SCEN, preferences=PREF_OFF,
                                  measurements=measurements,
                                  forecasts=forecasts, gamma=GAMMA, eta=ETA,
                                  sigmoid=SIG)
        ex = r["execution"]
        if ex is None or ex.get("status") != "executed":
            raise RuntimeError(f"{variant}: execution did not run: {ex}")
        for row in ex["results"]:
            rows.append({
                "variant": variant, "area_uuid": row["area_uuid"],
                "role": row["role"], "traded_kwh": row["traded_kwh"],
                "actual_kwh": row["actual_kwh"],
                "deliverable_kwh": row["deliverable_kwh"],
                "deliverable_source": row["deliverable_source"],
                "shortfall_penalty_ct": row["shortfall_penalty_ct"],
                "externality_kwh": row["externality_kwh"],
                "externality_penalty_ct": row["externality_penalty_ct"],
                "total_ct": row["total_penalty_ct"],
            })
        totals[variant] = {
            "round_type": ex["round_type"],
            "ratio": round(ex["total_supply_kwh"] / ex["total_demand_kwh"], 6),
            "clearing_price_ct": ex["clearing_price_ct_per_kwh"],
            "traded_quantity_kwh": ex["traded_quantity_kwh"],
            "market_total_penalties_ct": ex["total_penalties_ct"],
        }
        rows.append({
            "variant": variant, "area_uuid": "MARKET_TOTAL",
            "role": "-", "traded_kwh": ex["traded_quantity_kwh"],
            "actual_kwh": "", "deliverable_kwh": "",
            "deliverable_source": ex["round_type"],
            "shortfall_penalty_ct": round(
                sum(x["shortfall_penalty_ct"] for x in ex["results"]), 6),
            "externality_kwh": "",
            "externality_penalty_ct": round(
                sum(x["externality_penalty_ct"] for x in ex["results"]), 6),
            "total_ct": ex["total_penalties_ct"],
        })

    write("R3_oracle_vs_meter.csv", rows)
    wall = time.perf_counter() - t0
    write_manifest("R3_oracle_vs_meter",
                   params={"scenario": "hand-built 8.0 kWh supply / 10.0 kWh "
                                       "demand, ratio 0.8 (SUPPLY_LIMITED)",
                           "withholder": R3_WITHHOLDER,
                           "capacity_kwh": R3_CAPACITY,
                           "offered_kwh": R3_OFFERED,
                           "sigmoid": SIG, "gamma": GAMMA, "eta_kwh": ETA,
                           "preferences": PREF_OFF,
                           "variants": ["B2-meter (no forecasts)",
                                        "B2-oracle (forecast = capacity)"],
                           "totals": totals},
                   seeds=[None], wall_sec=wall,
                   output_files=["R3_oracle_vs_meter.csv"])
    return rows, totals


# =====================================================================
# R4a - buyer under-reporting
# =====================================================================
# Demand-limited: supply fixed at 12.5 kWh, the two honest buyers fixed at
# 6.0 kWh. The manipulator's true demand is 4.0; at honest reporting the
# ratio is 12.5 / 10.0 = 1.25. Reporting less raises the ratio further and
# pushes the price down, which is the manipulation Eq. 29 exists to deter.
R4_MANIPULATOR = "load-m"
R4_ACTUAL_DEMAND = 4.0
R4_LEVELS = [4.0, 3.5, 3.0, 2.5, 2.0]
R4_RETAIL_CT = 28.5                  # K_upper, the outside backstop price


def r4_scenario(reported):
    return {
        "producers": [
            {"area_uuid": "gen-a", "name": "Gen A", "energy": 7.5,
             "energy_type": "green"},
            {"area_uuid": "gen-b", "name": "Gen B", "energy": 5.0,
             "energy_type": "grey"},
        ],
        "consumers": [
            {"area_uuid": R4_MANIPULATOR, "name": "Manipulator",
             "energy": reported},
            {"area_uuid": "load-x", "name": "Load X", "energy": 3.5},
            {"area_uuid": "load-y", "name": "Load Y", "energy": 2.5},
        ],
    }


async def run_r4a(st):
    t0 = time.perf_counter()
    rows = []
    honest_price = None
    for reported in R4_LEVELS:
        scen = r4_scenario(reported)
        # The meter always reports the true 4.0; the honest buyers consume
        # exactly what they bid (demand-limited, so they are filled in full).
        measurements = {R4_MANIPULATOR: R4_ACTUAL_DEMAND,
                        "load-x": 3.5, "load-y": 2.5}
        r = await runner.run_slot(st, scen, preferences=PREF_OFF,
                                  measurements=measurements, gamma=GAMMA,
                                  eta=ETA, sigmoid=SIG)
        ex = r["execution"]
        if ex is None or ex.get("status") != "executed":
            raise RuntimeError(f"reported={reported}: no execution: {ex}")
        row = next(x for x in ex["results"]
                   if x["area_uuid"] == R4_MANIPULATOR)
        price = ex["clearing_price_ct_per_kwh"]
        if honest_price is None:          # first level is the honest one
            honest_price = price
        w_buy = row["externality_kwh"]
        pen = row["externality_penalty_ct"]

        # What the manipulation is worth. The artifact bills the buyer for the
        # quantity he *reported*; the extra consumption the meter records is
        # not charged anywhere, so "inside the market" he keeps it for free
        # and pays a lower price on the rest.
        cost_honest = R4_ACTUAL_DEMAND * honest_price
        cost_manip = row["traded_kwh"] * price
        gain_inside = cost_honest - cost_manip
        # Conservative reading: he still has to source the unreported kWh
        # somewhere, at the retail ceiling.
        gain_backstop = (cost_honest - cost_manip
                         - (R4_ACTUAL_DEMAND - row["traded_kwh"]) * R4_RETAIL_CT)
        rows.append({
            "reported_kwh": reported, "actual_kwh": row["actual_kwh"],
            "w_buy_kwh": w_buy,
            "price_ct": price,
            "price_cf_ct": row["counterfactual_price_ct_per_kwh"],
            "q_t_kwh": ex["traded_quantity_kwh"],
            "ratio": round(ex["total_supply_kwh"] / ex["total_demand_kwh"], 6),
            "round_type": ex["round_type"],
            "buyer_externality_ct": pen,
            "penalty_per_kwh_withheld": round(pen / w_buy, 6) if w_buy else None,
            "cost_honest_ct": round(cost_honest, 6),
            "cost_manipulated_ct": round(cost_manip, 6),
            "gain_inside_ct": round(gain_inside, 6),
            "net_inside_ct": round(gain_inside - pen, 6),
            "gain_retail_backstop_ct": round(gain_backstop, 6),
            "net_retail_backstop_ct": round(gain_backstop - pen, 6),
        })

    write("R4a_buyer_underreporting.csv", rows)
    wall = time.perf_counter() - t0
    write_manifest("R4a_buyer_underreporting",
                   params={"scenario": "hand-built 12.5 kWh supply, 6.0 kWh "
                                       "honest demand, manipulator true "
                                       "demand 4.0 (ratio 1.25 when honest)",
                           "reported_levels_kwh": R4_LEVELS,
                           "actual_demand_kwh": R4_ACTUAL_DEMAND,
                           "retail_backstop_ct": R4_RETAIL_CT,
                           "sigmoid": SIG, "gamma": GAMMA, "eta_kwh": ETA,
                           "preferences": PREF_OFF,
                           "generator_used": False},
                   seeds=[None], wall_sec=wall,
                   output_files=["R4a_buyer_underreporting.csv"])
    return rows


# =====================================================================
# R4b - the regime boundary
# =====================================================================
# Demand is held at exactly 10.0 kWh; supply is the probe.
#
# Two separate thresholds are in play and the probes are chosen to tell them
# apart. The Clearing Node calls `round_type` on the raw sums; the Execution
# Node re-derives the round type from `total_supply_kwh` / `total_demand_kwh`
# as written into the trade parameters, which `trade_builder` rounds to 6
# decimals. Both then apply the same 1e-9 epsilon. So probes with
# 1e-9 < |S - D| < 5e-7 should make the two nodes disagree.
R4B_DEMAND = [("load-p", 4.0), ("load-q", 3.5), ("load-r", 2.5)]   # 10.0
R4B_PROBES = [
    ("0.98  (S-D = -2e-1)", 9.8),
    ("0.999999  (S-D = -1e-5)", 9.99999),
    ("S-D = +5e-10  (inside 1e-9 epsilon)", 10.0 + 5e-10),
    ("1.00 exact  (S-D = 0)", 10.0),
    ("S-D = +1e-8  (below 6dp rounding)", 10.0 + 1e-8),
    ("S-D = +1e-7  (below 6dp rounding)", 10.0 + 1e-7),
    ("S-D = +1e-6  (at 6dp resolution)", 10.0 + 1e-6),
    ("1.000001  (S-D = +1e-5)", 10.00001),
    ("1.02  (S-D = +2e-1)", 10.2),
]
R4B_WITHHOLDER = "gen-w"
R4B_WITHHOLDER_OFFER = 3.0
R4B_WITHHOLDER_CAPACITY = 5.0
R4B_UNDERREPORTER = "load-p"          # bids 4.0, meter says 5.0


def r4b_scenario(total_supply):
    rest = round(total_supply - R4B_WITHHOLDER_OFFER, 12)
    return {
        "producers": [
            {"area_uuid": R4B_WITHHOLDER, "name": "Withholder",
             "energy": R4B_WITHHOLDER_OFFER, "energy_type": "green"},
            {"area_uuid": "gen-rest", "name": "Gen Rest", "energy": rest,
             "energy_type": "green"},
        ],
        "consumers": [{"area_uuid": a, "name": a, "energy": e}
                      for a, e in R4B_DEMAND],
    }


async def run_r4b(st):
    """Every probe carries both a potential seller withholder (forecast above
    its offer) and a potential buyer under-reporter (meter above its bid), so
    which externality branch actually fires is measured, not inferred."""
    t0 = time.perf_counter()
    rows = []
    for label, supply in R4B_PROBES:
        scen = r4b_scenario(supply)
        # Only the under-reporter gets a meter reading above its bid; every
        # other area falls back to "delivered as traded" -> shortfall 0.
        measurements = {R4B_UNDERREPORTER: 5.0}
        r = await runner.run_slot(st, scen, preferences=PREF_OFF,
                                  measurements=measurements,
                                  forecasts={R4B_WITHHOLDER:
                                             R4B_WITHHOLDER_CAPACITY},
                                  gamma=GAMMA, eta=ETA, sigmoid=SIG)
        ex = r["execution"]
        c = r["clearing"]
        if ex is None or ex.get("status") != "executed":
            raise RuntimeError(f"{label}: no execution: {ex}")
        seller_ext = round(sum(x["externality_penalty_ct"] for x in ex["results"]
                               if x["role"] == "seller"), 6)
        buyer_ext = round(sum(x["externality_penalty_ct"] for x in ex["results"]
                              if x["role"] == "buyer"), 6)
        shortfall = round(sum(x["shortfall_penalty_ct"]
                              for x in ex["results"]), 6)
        fired = [n for n, v in (("seller_externality", seller_ext),
                                ("buyer_externality", buyer_ext)) if v > 0]
        # What the Execution Node saw: the 6-decimal totals off the trades.
        s_exec, d_exec = ex["total_supply_kwh"], ex["total_demand_kwh"]
        # What the scenario actually put into the order book.
        s_raw = sum(p["energy"] for p in scen["producers"])
        d_raw = sum(x["energy"] for x in scen["consumers"])
        agree = c["round_type"] == ex["round_type"]
        rows.append({
            "ratio_target": label,
            "supply_raw_kwh": repr(s_raw), "demand_raw_kwh": repr(d_raw),
            "supply_minus_demand_raw_kwh": repr(s_raw - d_raw),
            "supply_exec_kwh": repr(s_exec), "demand_exec_kwh": repr(d_exec),
            "supply_minus_demand_exec_kwh": repr(s_exec - d_exec),
            "ratio_actual": c["ratio"],
            "round_type_clearing": c["round_type"],
            "round_type_execution": ex["round_type"],
            "round_types_agree": agree,
            "price_ct": ex["clearing_price_ct_per_kwh"],
            "penalty_branch": "+".join(fired) if fired else "none",
            "seller_externality_ct": seller_ext,
            "buyer_externality_ct": buyer_ext,
            "shortfall_ct": shortfall,
            "notes": ("clearing and execution disagree: the 6-decimal totals "
                      "in the trade parameters collapse a difference the "
                      "clearing saw" if not agree else
                      "both nodes agree"),
        })

    write("R4b_regime_boundary.csv", rows)
    wall = time.perf_counter() - t0
    write_manifest("R4b_regime_boundary",
                   params={"demand_kwh": 10.0,
                           "supply_probes": [p for _, p in R4B_PROBES],
                           "withholder_offer_kwh": R4B_WITHHOLDER_OFFER,
                           "withholder_forecast_kwh": R4B_WITHHOLDER_CAPACITY,
                           "underreporter_bid_kwh": 4.0,
                           "underreporter_meter_kwh": 5.0,
                           "sigmoid": SIG, "gamma": GAMMA, "eta_kwh": ETA,
                           "preferences": PREF_OFF},
                   seeds=[None], wall_sec=wall,
                   output_files=["R4b_regime_boundary.csv"])
    return rows


async def main():
    st = stack.Stack()
    try:
        print(f"repo SHA {campaign.SHA}")

        print("\nR3 oracle vs meter")
        r3, totals = await run_r3(st)
        for v, t in totals.items():
            print(f"  {v:<10} ratio {t['ratio']}  {t['round_type']}  "
                  f"price {t['clearing_price_ct']} ct  "
                  f"Q_t {t['traded_quantity_kwh']} kWh  "
                  f"market total {t['market_total_penalties_ct']} ct")
        print(f"  {'variant':<10} {'area':<16} {'traded':>7} {'actual':>7} "
              f"{'deliv':>7} {'source':>9} {'short ct':>9} {'ext kWh':>8} "
              f"{'ext ct':>10} {'total ct':>10}")
        for row in r3:
            print(f"  {row['variant']:<10} {row['area_uuid']:<16} "
                  f"{str(row['traded_kwh']):>7} {str(row['actual_kwh']):>7} "
                  f"{str(row['deliverable_kwh']):>7} "
                  f"{str(row['deliverable_source']):>9} "
                  f"{row['shortfall_penalty_ct']:>9} "
                  f"{str(row['externality_kwh']):>8} "
                  f"{row['externality_penalty_ct']:>10} "
                  f"{row['total_ct']:>10}")

        print("\nR4a buyer under-reporting")
        r4a = await run_r4a(st)
        print(f"  {'rep':>5} {'act':>5} {'W_buy':>6} {'price':>9} {'p_cf':>9} "
              f"{'Q_t':>6} {'penalty ct':>11} {'ct/kWh':>9} "
              f"{'gain in ct':>11} {'net in ct':>10} {'net retail':>11}")
        for r in r4a:
            print(f"  {r['reported_kwh']:>5} {r['actual_kwh']:>5} "
                  f"{r['w_buy_kwh']:>6} {r['price_ct']:>9} "
                  f"{str(r['price_cf_ct']):>9} {r['q_t_kwh']:>6} "
                  f"{r['buyer_externality_ct']:>11} "
                  f"{str(r['penalty_per_kwh_withheld']):>9} "
                  f"{r['gain_inside_ct']:>11} {r['net_inside_ct']:>10} "
                  f"{r['net_retail_backstop_ct']:>11}")

        print("\nR4b regime boundary")
        r4b = await run_r4b(st)
        print(f"  {'probe':<38} {'S-D raw':>10} {'S-D exec':>10} "
              f"{'clearing':>15} {'execution':>15} {'agree':>6} "
              f"{'branch':>19} {'sell ct':>9} {'buy ct':>9}")
        for r in r4b:
            print(f"  {r['ratio_target']:<38} "
                  f"{float(r['supply_minus_demand_raw_kwh']):>10.1e} "
                  f"{float(r['supply_minus_demand_exec_kwh']):>10.1e} "
                  f"{r['round_type_clearing']:>15} "
                  f"{r['round_type_execution']:>15} "
                  f"{str(r['round_types_agree']):>6} "
                  f"{r['penalty_branch']:>19} "
                  f"{r['seller_externality_ct']:>9} "
                  f"{r['buyer_externality_ct']:>9}")
        return 0
    finally:
        await st.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
