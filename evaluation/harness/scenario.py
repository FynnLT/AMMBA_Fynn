"""Synthetic scenario generator for the pilot campaign.

DELIBERATELY CRUDE. The distributions below are invented, not sourced.
Their only job is to produce a plausible order book so the mechanism can be
exercised; every distributional claim from these runs is illustrative only.
Honours the harness assumption "one order per area and slot".
"""
import random


def make_scenario(seed=0, n_prod=40, n_cons=60, sd_ratio=1.25,
                  green_share=0.75, pair_density=0.0, mean_demand=1.2):
    rng = random.Random(seed)
    cons = []
    for i in range(n_cons):
        # lognormal-ish household demand per 15-min slot, kWh
        d = max(0.05, rng.lognormvariate(0.0, 0.55) * mean_demand * 0.5)
        cons.append({"area_uuid": f"load-{i:03d}", "name": f"Load {i:03d}",
                     "energy": round(d, 4)})
    total_demand = sum(c["energy"] for c in cons)

    prod = []
    for i in range(n_prod):
        cap = max(0.05, rng.lognormvariate(0.0, 0.6))
        etype = "green" if rng.random() < green_share else "grey"
        prod.append({"area_uuid": f"gen-{i:03d}", "name": f"Gen {i:03d}",
                     "energy": cap, "energy_type": etype})
    scale = (sd_ratio * total_demand) / sum(p["energy"] for p in prod)
    for p in prod:
        p["energy"] = round(p["energy"] * scale, 4)

    # mutual preferred pairs
    n_pairs = int(round(pair_density * min(n_prod, n_cons)))
    ps = rng.sample(range(n_prod), n_pairs)
    cs = rng.sample(range(n_cons), n_pairs)
    for pi, ci in zip(ps, cs):
        prod[pi]["preferred_partner"] = cons[ci]["area_uuid"]
        cons[ci]["preferred_partner"] = prod[pi]["area_uuid"]
    return {"producers": prod, "consumers": cons, "seed": seed,
            "sd_ratio_target": sd_ratio, "pair_density": pair_density,
            "green_share": green_share}


GUIDE = {  # implementation-guide reference example, 12.5 vs 10 kWh
    "producers": [
        {"area_uuid": "area-pv-a", "name": "PV A", "energy": 5.0,
         "energy_type": "green", "preferred_partner": "area-house-1"},
        {"area_uuid": "area-pv-b", "name": "PV B", "energy": 3.5,
         "energy_type": "green"},
        {"area_uuid": "area-battery", "name": "Battery", "energy": 4.0,
         "energy_type": "grey"},
    ],
    "consumers": [
        {"area_uuid": "area-house-1", "name": "House 1", "energy": 4.5,
         "preferred_partner": "area-pv-a"},
        {"area_uuid": "area-house-2", "name": "House 2", "energy": 3.0},
        {"area_uuid": "area-bakery", "name": "Bakery", "energy": 2.5},
    ],
    "seed": None, "sd_ratio_target": 1.25, "pair_density": 1/3,
    "green_share": None,
}
