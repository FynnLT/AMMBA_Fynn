"""R5 - the three conditions under which the +/-0 compensation property fails.

All three are computed offline from the penalty rows the Execution Node
already returns. D-25 keeps redistribution at the analysis level and D-42's
`redistribution` block is not implemented, so no payout rule is added here -
the compensation is applied to the returned rows, not to the artifact.

Breaker 1  non-additive counterfactuals with several deviators
Breaker 2  the deviator is compensated for damage he caused
Breaker 3  rounding at the x10,000 on-chain scaling
"""
import asyncio, random, statistics, sys, time

import stack, scenario, runner, campaign, blockd
from campaign import OUT, write, write_manifest

SIG = dict(runner.SIGMOID)
PREF_OFF = {"enabled": False, "multipliers_enabled": False}
GAMMA, ETA = 1.1, 0.0

# The artifact's own on-chain scaling, not a re-implementation.
to_node_int = stack.CLR["sigmoid"].to_node_int
from_node_int = stack.CLR["sigmoid"].from_node_int
SCALING = stack.CLR["sigmoid"].NODE_FLOAT_SCALING_FACTOR

# Block D's supply-limited base market: 13.0 kWh supply, 16.0 kWh demand.
BASE_OFFERS = blockd.BASE_OFFERS
BASE_BIDS = blockd.BASE_BIDS
DEVIATOR_COUNTS = (2, 3, 5, 8)


def write_sections(name, sections):
    """One CSV holding all three breakers. The sections carry different
    columns, so the header is their union and a cell a section does not have
    stays empty - `campaign.write` takes its fieldnames from the first row
    only and would drop the others."""
    import csv
    keys, rows = ["section"], []
    for label, section in sections:
        for row in section:
            row = {"section": label, **row}
            for k in row:
                if k not in keys:
                    keys.append(k)
            rows.append(row)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {name}  ({len(rows)} rows, {len(keys)} columns)")


def base_scenario():
    return {
        "producers": [{"area_uuid": f"gen-{i}", "name": f"Gen {i}",
                       "energy": e, "energy_type": "green"}
                      for i, e in enumerate(BASE_OFFERS)],
        "consumers": [{"area_uuid": f"load-{i}", "name": f"Load {i}",
                       "energy": e} for i, e in enumerate(BASE_BIDS)],
    }


# =====================================================================
# Breaker 1 - non-additive counterfactuals
# =====================================================================
def breaker_1():
    """Reuses block D's `d4_multiagent` unchanged, filtered to 2/3/5/8."""
    rows = []
    for r in blockd.d4_multiagent():
        if r["n_deviators"] not in DEVIATOR_COUNTS:
            continue
        rows.append({"breaker": "1_non_additive_counterfactuals",
                     "n_deviators": r["n_deviators"],
                     "withheld_each_kwh": r["withheld_each_kwh"],
                     "sum_individual_ct": r["sum_individual_ct"],
                     "joint_ct": r["joint_ct"],
                     "gap_ct": r["over_collection_ct"],
                     "gap_pct_of_joint": r["over_collection_pct"]})
    return rows


# =====================================================================
# Breaker 2 - the deviator is compensated for damage he caused
# =====================================================================
async def run_deviator_slot(st, n_deviators, withheld_each=0.5):
    """One supply-limited slot in which the first `n_deviators` sellers each
    hold back `withheld_each` kWh, declared through the forecast channel."""
    scen = base_scenario()
    forecasts = {f"gen-{i}": BASE_OFFERS[i] + withheld_each
                 for i in range(n_deviators)}
    measurements = {f"gen-{i}": e for i, e in enumerate(BASE_OFFERS)}
    r = await runner.run_slot(st, scen, preferences=PREF_OFF,
                              measurements=measurements, forecasts=forecasts,
                              gamma=GAMMA, eta=ETA, sigmoid=SIG)
    ex = r["execution"]
    if ex is None or ex.get("status") != "executed":
        raise RuntimeError(f"slot did not execute: {ex}")
    return ex, set(forecasts)


def proportional_compensation(rows, recipients, collected):
    """Split `collected` over `recipients` in proportion to traded volume."""
    weight = sum(r["traded_kwh"] for r in rows if r["area_uuid"] in recipients)
    if weight <= 0:
        return {}
    return {r["area_uuid"]: collected * r["traded_kwh"] / weight
            for r in rows if r["area_uuid"] in recipients}


async def breaker_2(st, n_deviators=3, withheld_each=0.5):
    ex, deviators = await run_deviator_slot(st, n_deviators, withheld_each)
    rows = ex["results"]
    collected = ex["total_penalties_ct"]
    everyone = {r["area_uuid"] for r in rows}
    honest = everyone - deviators

    incl = proportional_compensation(rows, everyone, collected)
    excl = proportional_compensation(rows, honest, collected)

    out = []
    for r in sorted(rows, key=lambda x: x["area_uuid"]):
        a = r["area_uuid"]
        is_dev = a in deviators
        out.append({
            "breaker": "2_deviator_compensated",
            "area_uuid": a, "role": r["role"], "is_deviator": is_dev,
            "traded_kwh": r["traded_kwh"],
            "penalty_paid_ct": r["total_penalty_ct"],
            "compensation_incl_deviators_ct": round(incl.get(a, 0.0), 6),
            "compensation_excl_deviators_ct": round(excl.get(a, 0.0), 6),
            "difference_ct": round(excl.get(a, 0.0) - incl.get(a, 0.0), 6),
        })
    to_deviators = sum(incl.get(a, 0.0) for a in deviators)
    per_honest = [excl.get(a, 0.0) - incl.get(a, 0.0) for a in honest]
    summary = {
        "n_deviators": n_deviators, "withheld_each_kwh": withheld_each,
        "round_type": ex["round_type"],
        "collected_ct": collected,
        "paid_back_to_deviators_ct": round(to_deviators, 6),
        "paid_back_to_deviators_pct": round(100 * to_deviators / collected, 4)
        if collected else None,
        "mean_gain_per_honest_participant_ct": round(
            statistics.mean(per_honest), 6) if per_honest else None,
        "max_gain_per_honest_participant_ct": round(max(per_honest), 6)
        if per_honest else None,
        "n_honest": len(honest),
    }
    return out, summary


# =====================================================================
# Breaker 3 - rounding at the x10,000 scaling
# =====================================================================
async def breaker_3(st, n_slots=96, seed=11):
    """A 96-slot day, one deviator per slot, compensation paid to the honest
    participants and put through the artifact's integer scaling."""
    rng = random.Random(seed)
    rows = []
    total_collected = total_redistributed = 0.0
    for slot in range(n_slots):
        which = rng.randrange(len(BASE_OFFERS))
        withheld = round(rng.uniform(0.1, 1.0), 4)
        scen = base_scenario()
        forecasts = {f"gen-{which}": BASE_OFFERS[which] + withheld}
        measurements = {f"gen-{i}": e for i, e in enumerate(BASE_OFFERS)}
        r = await runner.run_slot(st, scen, preferences=PREF_OFF,
                                  measurements=measurements,
                                  forecasts=forecasts, gamma=GAMMA, eta=ETA,
                                  sigmoid=SIG)
        ex = r["execution"]
        collected = ex["total_penalties_ct"]
        deviators = {f"gen-{which}"}
        honest = {x["area_uuid"] for x in ex["results"]} - deviators
        comp = proportional_compensation(ex["results"], honest, collected)
        # Anything paid out on chain is an integer at the scaling factor.
        paid_int = {a: to_node_int(v) for a, v in comp.items()}
        paid = sum(from_node_int(i) for i in paid_int.values())
        residual = collected - paid
        total_collected += collected
        total_redistributed += paid
        rows.append({
            "breaker": "3_scaling_residual", "slot": slot,
            "deviator": f"gen-{which}", "withheld_kwh": withheld,
            "n_recipients": len(comp),
            "collected_ct": round(collected, 6),
            "redistributed_ct": round(paid, 6),
            "residual_ct": round(residual, 10),
        })
    summary = {
        "n_slots": n_slots, "seed": seed, "scaling_factor": SCALING,
        "total_collected_ct": round(total_collected, 6),
        "total_redistributed_ct": round(total_redistributed, 6),
        "residual_ct": round(total_collected - total_redistributed, 10),
        "residual_share_of_collected": (
            (total_collected - total_redistributed) / total_collected
            if total_collected else None),
        "max_abs_slot_residual_ct": round(
            max(abs(r["residual_ct"]) for r in rows), 10),
    }
    return rows, summary


async def main():
    t0 = time.perf_counter()
    st = stack.Stack()
    try:
        print(f"repo SHA {campaign.SHA}")

        print("\nbreaker 1 - non-additive counterfactuals")
        b1 = breaker_1()
        print(f"  {'k':>3} {'withheld each':>14} {'sum individual':>15} "
              f"{'joint':>10} {'gap ct':>10} {'gap % of joint':>15}")
        for r in b1:
            print(f"  {r['n_deviators']:>3} {r['withheld_each_kwh']:>14} "
                  f"{r['sum_individual_ct']:>15} {r['joint_ct']:>10} "
                  f"{r['gap_ct']:>10} {str(r['gap_pct_of_joint']):>15}")

        print("\nbreaker 2 - deviator compensated for his own damage")
        b2, b2s = await breaker_2(st)
        print(f"  slot: {b2s['n_deviators']} deviators, "
              f"{b2s['withheld_each_kwh']} kWh each, {b2s['round_type']}, "
              f"{b2s['collected_ct']} ct collected")
        print(f"  {'area':<10} {'dev':>5} {'traded':>7} {'penalty':>10} "
              f"{'comp incl':>11} {'comp excl':>11} {'diff':>10}")
        for r in b2:
            print(f"  {r['area_uuid']:<10} {str(r['is_deviator']):>5} "
                  f"{r['traded_kwh']:>7} {r['penalty_paid_ct']:>10} "
                  f"{r['compensation_incl_deviators_ct']:>11} "
                  f"{r['compensation_excl_deviators_ct']:>11} "
                  f"{r['difference_ct']:>+10}")
        print(f"  back to deviators {b2s['paid_back_to_deviators_ct']} ct "
              f"({b2s['paid_back_to_deviators_pct']} % of the collected sum); "
              f"mean gain per honest participant when they are excluded "
              f"{b2s['mean_gain_per_honest_participant_ct']} ct "
              f"(n={b2s['n_honest']})")

        print("\nbreaker 3 - rounding at the x10,000 scaling")
        b3, b3s = await breaker_3(st)
        print(f"  {b3s['n_slots']} slots, seed {b3s['seed']}, "
              f"scaling factor {b3s['scaling_factor']}")
        print(f"  collected      {b3s['total_collected_ct']} ct")
        print(f"  redistributed  {b3s['total_redistributed_ct']} ct")
        print(f"  residual       {b3s['residual_ct']} ct "
              f"({b3s['residual_share_of_collected']:.3e} of the collected sum)")
        print(f"  largest single-slot residual "
              f"{b3s['max_abs_slot_residual_ct']} ct")

        write_sections("R5_pm_zero_breakers.csv", [
            ("1_non_additive_counterfactuals",
             [{k: v for k, v in r.items() if k != "breaker"} for r in b1]),
            ("2_deviator_compensated",
             [{k: v for k, v in r.items() if k != "breaker"} for r in b2]),
            ("3_scaling_residual",
             [{k: v for k, v in r.items() if k != "breaker"} for r in b3]),
        ])
        write_sections("R5_breaker_summaries.csv", [
            ("2_deviator_compensated", [b2s]),
            ("3_scaling_residual", [b3s]),
        ])

        wall = time.perf_counter() - t0
        write_manifest("R5_pm_zero_breakers",
                       params={"base_market": "blockd BASE_OFFERS/BASE_BIDS "
                                              "(13.0 kWh supply / 16.0 kWh "
                                              "demand, SUPPLY_LIMITED)",
                               "breaker_1_deviator_counts": list(DEVIATOR_COUNTS),
                               "breaker_2": {"n_deviators": b2s["n_deviators"],
                                             "withheld_each_kwh":
                                                 b2s["withheld_each_kwh"],
                                             "rule": "proportional to traded "
                                                     "volume, recipient set "
                                                     "with and without the "
                                                     "deviators"},
                               "breaker_3": {"n_slots": b3s["n_slots"],
                                             "scaling_factor": SCALING,
                                             "withheld_kwh_range": [0.1, 1.0]},
                               "sigmoid": SIG, "gamma": GAMMA, "eta_kwh": ETA,
                               "redistribution_implemented_in_artifact": False},
                       seeds=[b3s["seed"]], wall_sec=wall,
                       output_files=["R5_pm_zero_breakers.csv",
                                     "R5_breaker_summaries.csv"])
        return 0
    finally:
        await st.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
