"""Scenario generators for the evaluation campaign.

`make_scenario` is the synthetic generator of the pilot campaign and is
DELIBERATELY CRUDE: the distributions are invented, not sourced. Their only
job is to produce a plausible order book so the mechanism can be exercised;
every distributional claim from those runs is illustrative only. It produced
the 25.08. and 02.09. figures and is not touched -- new behaviour lives in
`make_scenario_from_profiles` beside it.

Both honour the harness assumption "one order per area and slot", and
`_assert_one_order_per_area` checks it on every book rather than trusting it:
the artifact enforces nothing here, and a profile source that breaks the
assumption must fail loudly instead of being netted quietly.
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


# --------------------------------------------------------- the real profiles

NET_EPSILON = 1e-9


class ScenarioError(ValueError):
    """The generated order book breaks a harness assumption."""


def _assert_one_order_per_area(scen: dict) -> dict:
    """One order per area and slot, checked on the book that was built.

    The clearing node raises if an area holds both sides (D-58), but nothing
    stops an area posting *two* offers, and pro-rata would then weight it
    twice. Netting is exactly the step that could produce it, so the check
    sits on the generator's output.
    """
    seen = {}
    for side in ("producers", "consumers"):
        for order in scen[side]:
            area = order["area_uuid"]
            if area in seen:
                raise ScenarioError(
                    f"area {area} posts more than one order in slot "
                    f"{scen.get('slot_index')}: {seen[area]} and {side}")
            seen[area] = side
    return scen


def make_scenario_from_profiles(week, index, batteries=(), *,
                                epsilon: float = NET_EPSILON) -> dict:
    """One slot's order book from the real profiles, netted per player (D-61).

    Netting happens here, in the scenario builder, not in the runner: the
    runner posts whatever book it is handed, and what was netted away has to
    be visible in the slot record rather than inferred from it.

    Per player and slot, `net = load_kwh - pv_kwh`:

    * `net > 0`   -> a Bid for `net`
    * `net < 0`   -> an Offer for `-net`
    * `|net| <= epsilon` -> the player posts nothing in that slot

    `batteries` are separate market areas with their own offers (D-75); they
    are not netted against any household, which is the only reason a storage
    unit can sell at all. Their offers carry `energy_type: "grey"`.

    `week` is anything exposing `players`, `load_kwh` and `pv_kwh` as
    player -> per-slot kWh (`profiles.ProfileWeek` or `profiles.DayProfile`).
    """
    producers, consumers = [], []
    generation = consumption = 0.0

    for player in week.players:
        load = week.load_kwh[player][index]
        pv = week.pv_kwh[player][index]
        consumption += load
        generation += pv
        net = load - pv
        if net > epsilon:
            consumers.append({"area_uuid": f"player-{player:03d}",
                              "name": f"Player {player:03d}",
                              "energy": round(net, 6)})
        elif net < -epsilon:
            producers.append({"area_uuid": f"player-{player:03d}",
                              "name": f"Player {player:03d}",
                              "energy": round(-net, 6),
                              "energy_type": "green"})

    battery_kwh = 0.0
    for battery in batteries:
        energy = battery.slot_energy(index)
        if energy <= epsilon:
            continue
        battery_kwh += energy
        producers.append({"area_uuid": battery.area_uuid,
                          "name": battery.name,
                          "energy": round(energy, 6),
                          "energy_type": battery.energy_type})

    supply = sum(p["energy"] for p in producers)
    demand = sum(c["energy"] for c in consumers)
    scen = {
        "producers": producers, "consumers": consumers,
        "slot_index": index,
        # Gross, before netting: 5.1 has to be able to state what was netted
        # away behind the meter rather than what reached the market.
        "generation_kwh": round(generation, 6),
        "consumption_kwh": round(consumption, 6),
        "net_kwh": round(consumption - generation, 6),
        # What netting removed from the market: sum(min(load, pv)) per player.
        # Identically gross consumption minus the demand that reached the book
        # (batteries are producers, so `demand` is players only).
        "self_consumed_kwh": round(consumption - demand, 6),
        "battery_kwh": round(battery_kwh, 6),
        "posted_supply_kwh": round(supply, 6),
        "posted_demand_kwh": round(demand, 6),
        "source": "profiles",
    }
    return _assert_one_order_per_area(scen)


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
