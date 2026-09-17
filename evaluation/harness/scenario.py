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

`draw_preferences` is the profile path's equivalent of the pairing block in
`make_scenario`: it draws the partner relationships *once per run*, and
`make_scenario_from_profiles` posts them in whichever slots both sides
actually have an order (D-71/D-81).
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


def area_uuid_for(player) -> str:
    """The market area id of a household.

    One place, because `draw_preferences` and `make_scenario_from_profiles`
    have to agree on it exactly: a preference dict keyed on a differently
    formatted id would silently post no pairs at all, which is the D-71
    failure this section exists to close.
    """
    return player if isinstance(player, str) else f"player-{int(player):03d}"


def draw_preferences(players, *, named_share: float, mutual_share: float,
                     seed: int) -> dict:
    """Preferred-partner relationships for one run (D-71).

    Drawn **once per run and held over the week** (D-81): a household does
    not change its preferred neighbour every quarter hour, and holding the
    pairs is what makes "the partner had nothing to post in this slot" a
    countable result (`pairs_unpostable`) rather than a property of a
    per-slot redraw.

    `round(named_share * len(players))` households are drawn as namers; each
    names exactly one partner, so the returned dict has exactly that many
    entries. A fraction `mutual_share` of them are paired *with each other*
    and name back; the rest draw their partner uniformly from the other
    players. Accidental reciprocity among the non-mutual namers is redrawn,
    so `mutual_share` is the realised mutual rate and not merely its
    expectation.

    A household names a partner regardless of which side it ends up on in a
    given slot -- under netting (D-61) the same player is a buyer in one slot
    and a seller in the next, and the pair is a relationship, not a slot
    property.

    **Battery areas never name and are never named.** They are a modelling
    assumption (D-75), and a preference for a storage unit would put that
    assumption on both sides of the RQ1 comparison; `players` carries
    households only, so this holds by construction rather than by filtering.

    Deterministic in `seed` through its own `random.Random` -- never the
    module-level RNG, which a campaign worker shares with everything else it
    imports.

    Returns `{area_uuid: partner_area_uuid}`.
    """
    if not 0.0 <= named_share <= 1.0:
        raise ScenarioError(
            f"named_share must be in [0, 1], got {named_share}")
    if not 0.0 <= mutual_share <= 1.0:
        raise ScenarioError(
            f"mutual_share must be in [0, 1], got {mutual_share}")

    areas = [area_uuid_for(player) for player in players]
    if len(set(areas)) != len(areas):
        raise ScenarioError("players do not map to distinct area ids")
    n_named = int(round(named_share * len(areas)))
    if n_named < 1 or len(areas) < 2:
        return {}

    rng = random.Random(seed)
    namers = rng.sample(areas, n_named)

    # A reciprocal relationship takes two households, so the mutual block is
    # rounded down to an even count: `mutual_share` names a share of the
    # *pairs*, and half a pair cannot name back.
    n_mutual = min(n_named, int(round(mutual_share * n_named)))
    n_mutual -= n_mutual % 2

    preferences = {}
    for first, second in zip(namers[:n_mutual:2], namers[1:n_mutual:2]):
        preferences[first] = second
        preferences[second] = first

    solo = namers[n_mutual:]
    for area in solo:
        candidates = [other for other in areas if other != area]
        preferences[area] = rng.choice(candidates)

    # Repair accidental reciprocity: two solo namers that happened to draw
    # each other would raise the realised mutual rate above `mutual_share`,
    # and the `mutual_share = 0` cell has to mean exactly zero mutual pairs.
    solo_set = set(solo)
    for area in solo:
        partner = preferences[area]
        if partner in solo_set and preferences.get(partner) == area:
            candidates = [other for other in areas
                          if other != area and other != partner]
            if candidates:
                preferences[area] = rng.choice(candidates)
    return preferences


def _apply_preferences(producers: list, consumers: list,
                       preferences: dict | None) -> dict:
    """Post the held pairs into one slot's book, and count what could not be.

    A named partner is only postable where it holds an order **on the
    opposite side in this same slot**: the clearing node drops a
    `preferred_partner` that is not a counterparty in the market
    (`preferences.parse_preferred_partner`), so posting one anyway would
    trade a countable harness result for a warning in a log nobody reads
    over 672 slots.
    """
    counts = {"pairs_posted": 0, "pairs_unpostable": 0, "pairs_mutual": 0}
    if not preferences:
        return counts

    side_of = {}
    for side, orders in (("producers", producers), ("consumers", consumers)):
        for order in orders:
            side_of[order["area_uuid"]] = side

    posted = {}
    for orders in (producers, consumers):
        for order in orders:
            area = order["area_uuid"]
            partner = preferences.get(area)
            if partner is None:
                continue
            if side_of.get(partner) not in (None, side_of[area]):
                order["preferred_partner"] = partner
                posted[area] = partner
            else:
                # The partner nets to zero in this slot, or is on the same
                # side of it. No pair here, and that is a 5.2 result of its
                # own rather than a silent gap.
                counts["pairs_unpostable"] += 1

    counts["pairs_posted"] = len(posted)
    counts["pairs_mutual"] = sum(
        1 for area, partner in posted.items()
        if posted.get(partner) == area) // 2
    return counts


def make_scenario_from_profiles(week, index, batteries=(), *,
                                epsilon: float = NET_EPSILON,
                                preferences: dict = None) -> dict:
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

    `preferences` is the run's held `{area_uuid: partner_area_uuid}` draw
    (`draw_preferences`). It is applied **after** the book is built: the
    netting and battery logic above is the D-61/D-75 implementation and does
    not change because a household expressed a preference. Where both sides
    of a named pair hold an order on opposite sides of this slot the
    `preferred_partner` requirement is posted; where they do not, the pair is
    counted under `pairs_unpostable` and nothing is posted.
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

    pairs = _apply_preferences(producers, consumers, preferences)

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
        # Always present, also at zero: a CSV column that appears only in the
        # preference cells cannot be compared against the baseline's.
        **pairs,
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
