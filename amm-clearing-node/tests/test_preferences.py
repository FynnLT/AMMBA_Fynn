"""User preference mechanisms (guide §4.4 step 6, §6).

Two mechanisms, tested against the guide's reference market (12.5 kWh supply
vs 10 kWh demand -> ratio 1.25 -> p ≈ 15.1472 ct/kWh):

* preferred trading partner priority — mutual pairs first, residual pro-rata
* energy-type multipliers — green bonus funded by a grey levy

The first test is the golden backward-compatibility case: orders without
`requirements`/`attributes` must clear exactly as they did before Phase 2.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.clearing import run_clearing
from src.config import (Config, PreferenceConfig, PreferenceConfigError,
                        load_config, resolve_preferences)
from src.contract import MockContractClient
from src.preferences import (apply_preference_allocation, parse_energy_type,
                             parse_preferred_partner)

from .test_clearing import MARKET, SLOT, FakeOffchainDB, trigger

# sigmoid(1.25) for K=28.5/8.0, theta=1.0, B=2.5, rounded like the trades.
GUIDE_PRICE = 15.147225


# --------------------------------------------------------------- order books

def make_order(idx: int, order_type: str, name: str, area: str, energy: float,
               *, energy_type: str | None = None,
               partner: str | None = None, **extra) -> dict:
    order = {
        "order_id": f"0x{idx:064x}", "order_type": order_type,
        "status": "Open", "created_by": name, "area_uuid": area,
        "market_id": MARKET, "time_slot": SLOT, "creation_time": 100 + idx,
        "energy": energy,
        "energy_rate": 28.5 if order_type == "Bid" else 8.0,
    }
    if energy_type is not None:
        order["attributes"] = {"energy_type": energy_type}
    if partner is not None:
        order["requirements"] = {"preferred_partner": partner}
    order.update(extra)
    return order


def reference_book(*, mutual: bool = False, one_sided: bool = False,
                   typed: bool = True) -> list[dict]:
    """Guide reference market. `mutual` makes PV A <-> Household 1 a pair,
    `one_sided` lets only the seller name a partner, `typed` marks the
    battery as grey (everything else green)."""
    pv_a_partner = "area-house-1" if (mutual or one_sided) else None
    return [
        make_order(1, "Offer", "PV A", "area-pv-a", 5.0,
                   energy_type="green" if typed else None,
                   partner=pv_a_partner),
        make_order(2, "Offer", "PV B", "area-pv-b", 3.5,
                   energy_type="green" if typed else None),
        make_order(3, "Offer", "Battery", "area-battery", 4.0,
                   energy_type="grey" if typed else None),
        make_order(4, "Bid", "Household 1", "area-house-1", 4.5,
                   partner="area-pv-a" if mutual else None),
        make_order(5, "Bid", "Household 2", "area-house-2", 3.0),
        make_order(6, "Bid", "Bakery", "area-bakery", 2.5),
    ]


async def clear(orders: list[dict], **preference_params) -> dict:
    payload = trigger()
    if preference_params:
        payload["preference_params"] = preference_params
    return await run_clearing(payload, Config(), FakeOffchainDB(orders),
                              MockContractClient())


def by_name(summary: dict, side: str) -> dict:
    return {row["name"]: row for row in summary["allocations"][side]}


# ------------------------------------------------- §6.1 golden compatibility

@pytest.mark.anyio
@pytest.mark.parametrize("enabled", [True, False])
async def test_orders_without_preference_data_clear_exactly_as_before(enabled):
    """Golden test: no `requirements`, no `attributes` anywhere — the result
    must be the pre-Phase-2 uniform-price pro-rata clearing."""
    result = await clear(reference_book(typed=False), enabled=enabled)

    assert result["clearing_price_ct_per_kwh"] == pytest.approx(15.1472,
                                                               abs=1e-3)
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)
        assert producer["preference_matched"] is False
        assert producer["final_energy_rate"] == pytest.approx(
            result["clearing_price_ct_per_kwh"])
    for consumer in result["allocations"]["consumers"]:
        assert consumer["fill_rate"] == pytest.approx(1.0)

    preferences = result["preferences"]
    assert preferences["mutual_pairs"] == []
    assert preferences["pairs_rationed"] is False
    # Untyped orders default to green: no grey volume, so no levy revenue and
    # therefore no bonus — the pool is exactly balanced.
    multipliers = preferences["multipliers"]
    assert multipliers["grey_alloc_kwh"] == 0.0
    assert multipliers["bonus_paid_ct"] == 0.0
    assert multipliers["pool_surplus_ct"] == 0.0


# --------------------------------------------- §6.2 reference case with pair

@pytest.mark.anyio
async def test_mutual_pair_is_served_first():
    """PV A <-> Household 1 mutual, `preferences_first`.

    The pair takes 4.5 kWh off the top; the remaining 5.5 kWh are split
    pro-rata over the *unfilled* volumes (PV A 0.5, PV B 3.5, Battery 4.0),
    which lifts PV A from the 80 % pro-rata baseline to 96.875 %."""
    result = await clear(reference_book(mutual=True))
    producers = by_name(result, "producers")
    consumers = by_name(result, "consumers")

    assert producers["PV A"]["allocated_kwh"] == pytest.approx(4.843750,
                                                              abs=1e-6)
    assert producers["PV B"]["allocated_kwh"] == pytest.approx(2.406250,
                                                              abs=1e-6)
    assert producers["Battery"]["allocated_kwh"] == pytest.approx(2.750000,
                                                                  abs=1e-6)
    assert producers["PV A"]["fill_rate"] == pytest.approx(0.968750, abs=1e-6)
    assert producers["PV B"]["fill_rate"] == pytest.approx(0.687500, abs=1e-6)
    assert producers["Battery"]["fill_rate"] == pytest.approx(0.687500,
                                                              abs=1e-6)
    for name in ("Household 1", "Household 2", "Bakery"):
        assert consumers[name]["fill_rate"] == pytest.approx(1.0)

    assert sum(p["allocated_kwh"] for p in producers.values()) == \
        pytest.approx(10.0, abs=1e-9)
    assert sum(c["allocated_kwh"] for c in consumers.values()) == \
        pytest.approx(10.0, abs=1e-9)

    assert producers["PV A"]["preference_matched"] is True
    assert consumers["Household 1"]["preference_matched"] is True
    assert producers["PV B"]["preference_matched"] is False
    assert result["preferences"]["mutual_pairs"] == [
        {"bid_area": "area-house-1", "offer_area": "area-pv-a",
         "energy_kwh": 4.5}]
    assert result["preferences"]["pairs_rationed"] is False


@pytest.mark.anyio
async def test_pro_rata_baseline_for_the_same_market():
    """Comparison baseline for §6.2: without the pair every seller is at 80 %."""
    result = await clear(reference_book(mutual=False))
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)


# ------------------------------------------------------- §6.3 multiplicative

@pytest.mark.anyio
async def test_multipliers_on_the_reference_case():
    result = await clear(reference_book(mutual=True))
    multipliers = result["preferences"]["multipliers"]

    assert multipliers["green_alloc_kwh"] == pytest.approx(7.25, abs=1e-6)
    assert multipliers["grey_alloc_kwh"] == pytest.approx(2.75, abs=1e-6)
    assert multipliers["levy_collected_ct"] == pytest.approx(4.16548, abs=1e-4)
    assert multipliers["bonus_requested_ct"] == pytest.approx(10.98172,
                                                              abs=1e-4)
    assert multipliers["scale"] == pytest.approx(0.37931, abs=1e-5)
    assert multipliers["green_final_ct_per_kwh"] == pytest.approx(15.721775,
                                                                  abs=1e-5)
    assert multipliers["grey_final_ct_per_kwh"] == pytest.approx(13.632503,
                                                                 abs=1e-5)
    assert multipliers["buyers_pay_ct"] == pytest.approx(151.47225, abs=1e-4)
    assert multipliers["sellers_receive_ct"] == pytest.approx(151.47225,
                                                              abs=1e-4)
    # Zero-sum while the bonus is being scaled; the residual is rounding dust
    # from settling rates at 6 decimals.
    assert multipliers["pool_surplus_ct"] == pytest.approx(0.0, abs=1e-4)

    producers = by_name(result, "producers")
    assert producers["Battery"]["final_energy_rate"] == pytest.approx(
        13.632503, abs=1e-5)
    assert producers["PV A"]["final_energy_rate"] == pytest.approx(15.721775,
                                                                   abs=1e-5)


# ------------------------------------------- §6.4 over-collection -> surplus

@pytest.mark.anyio
async def test_over_collected_levy_becomes_a_reported_pool_surplus():
    """The B1 finding: the multiplicative formulation is NOT zero-sum.

    With 7.25 kWh green and 2.75 kWh grey the levy exactly funds the bonus at
    green_multiplier = 0.275/7.25 = 0.037931. Anything below that over-collects:
    the bonus is paid in full (scale == 1), buyers still pay p, and the pool
    keeps the difference. This is an intended, measured property of the
    formulation — it is reported as `pool_surplus_ct`, never absorbed.

    NOTE: the task sheet names 0.04 here, which is just *above* the crossover
    and therefore under-funds instead (asserted below); 0.03 is the smallest
    round value that actually over-collects.
    """
    result = await clear(reference_book(mutual=True),
                         green_multiplier=0.03, grey_levy=0.10)
    multipliers = result["preferences"]["multipliers"]

    assert multipliers["scale"] == 1.0  # exactly: the ratio is clamped
    assert multipliers["pool_surplus_ct"] > 0.0
    assert multipliers["pool_surplus_ct"] == pytest.approx(
        multipliers["levy_collected_ct"] - multipliers["bonus_paid_ct"],
        abs=1e-4)
    assert multipliers["bonus_paid_ct"] == pytest.approx(
        multipliers["bonus_requested_ct"], abs=1e-4)

    # …and just above the crossover the bonus is scaled down instead.
    under = await clear(reference_book(mutual=True),
                        green_multiplier=0.04, grey_levy=0.10)
    assert under["preferences"]["multipliers"]["scale"] < 1.0
    assert under["preferences"]["multipliers"]["pool_surplus_ct"] == \
        pytest.approx(0.0, abs=1e-4)


# -------------------------------------------------------- §6.5 further cases

@pytest.mark.anyio
async def test_one_sided_preference_does_not_create_a_pair():
    result = await clear(reference_book(one_sided=True))
    assert result["preferences"]["mutual_pairs"] == []
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)
        assert producer["preference_matched"] is False


@pytest.mark.anyio
async def test_unknown_partner_area_degrades_to_no_preference(caplog):
    orders = reference_book(mutual=True)
    orders[0]["requirements"] = {"preferred_partner": "area-does-not-exist"}
    with caplog.at_level("WARNING"):
        result = await clear(orders)
    assert result["preferences"]["mutual_pairs"] == []
    assert any("not a counterparty" in record.message
               for record in caplog.records)
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)


@pytest.mark.anyio
async def test_unequal_pair_takes_the_smaller_side():
    """Bid 10 kWh vs offer 2 kWh: the pair moves 2 kWh, the bid's remainder
    comes out of the pool like everyone else's."""
    orders = [
        make_order(1, "Offer", "Small PV", "area-pv-a", 2.0,
                   partner="area-house-1"),
        make_order(2, "Offer", "Big PV", "area-pv-b", 8.0),
        make_order(3, "Bid", "Big Load", "area-house-1", 10.0,
                   partner="area-pv-a"),
        make_order(4, "Bid", "Small Load", "area-house-2", 2.0),
    ]
    result = await clear(orders)
    pairs = result["preferences"]["mutual_pairs"]
    assert pairs == [{"bid_area": "area-house-1", "offer_area": "area-pv-a",
                      "energy_kwh": 2.0}]

    producers = by_name(result, "producers")
    consumers = by_name(result, "consumers")
    # supply 10, demand 12 -> traded 10; pair 2, residual 8 over the rest
    assert producers["Small PV"]["allocated_kwh"] == pytest.approx(2.0)
    assert producers["Big PV"]["allocated_kwh"] == pytest.approx(8.0)
    assert consumers["Big Load"]["allocated_kwh"] == pytest.approx(2.0 + 8.0
                                                                   * 8.0 / 10.0)
    assert consumers["Small Load"]["allocated_kwh"] == pytest.approx(
        8.0 * 2.0 / 10.0)


@pytest.mark.anyio
async def test_all_participants_paired_leaves_no_residual():
    orders = [
        make_order(1, "Offer", "PV A", "area-pv-a", 3.0,
                   partner="area-house-1"),
        make_order(2, "Offer", "PV B", "area-pv-b", 2.0,
                   partner="area-house-2"),
        make_order(3, "Bid", "House 1", "area-house-1", 3.0,
                   partner="area-pv-a"),
        make_order(4, "Bid", "House 2", "area-house-2", 2.0,
                   partner="area-pv-b"),
    ]
    result = await clear(orders)
    assert len(result["preferences"]["mutual_pairs"]) == 2
    for row in (result["allocations"]["producers"]
                + result["allocations"]["consumers"]):
        assert row["fill_rate"] == pytest.approx(1.0)
        assert row["preference_matched"] is True


@pytest.mark.anyio
async def test_saturating_pairs_leave_no_residual_and_are_not_rationed():
    """Pairs that consume the entire cleared quantity.

    Supply 6 kWh, demand 16 kWh -> 6 kWh clears, and the two pairs want
    exactly those 6 kWh: the residual step gets R == 0 with S_rest == 0 and
    must skip rather than divide by zero.

    This is also the tightest the rationing branch of §3.3 step 3 can ever
    get — see `test_pairs_can_never_oversubscribe_the_cleared_quantity`.
    """
    orders = [
        make_order(1, "Offer", "PV A", "area-pv-a", 3.0,
                   partner="area-house-1"),
        make_order(2, "Offer", "PV B", "area-pv-b", 3.0,
                   partner="area-house-2"),
        make_order(3, "Bid", "House 1", "area-house-1", 6.0,
                   partner="area-pv-a"),
        make_order(4, "Bid", "House 2", "area-house-2", 6.0,
                   partner="area-pv-b"),
        make_order(5, "Bid", "Bakery", "area-bakery", 4.0),
    ]
    result = await clear(orders)

    assert result["preferences"]["pairs_rationed"] is False
    assert sum(pair["energy_kwh"]
               for pair in result["preferences"]["mutual_pairs"]) == \
        pytest.approx(6.0)
    assert sum(p["allocated_kwh"]
               for p in result["allocations"]["producers"]) == \
        pytest.approx(6.0, abs=1e-9)
    # Bakery named nobody and everything was taken by the pairs.
    assert by_name(result, "consumers")["Bakery"]["allocated_kwh"] == 0.0


def test_pairs_can_never_oversubscribe_the_cleared_quantity():
    """The rationing branch of §3.3 step 3 is unreachable by construction.

    With *exactly one* preferred partner per order every order belongs to at
    most one pair, so `Σ min(b, o) ≤ Σ b ≤ demand` and likewise `≤ supply`,
    i.e. never more than `Q = min(supply, demand)`. The branch stays in the
    code as cheap insurance for the day the "one partner" rule is relaxed
    (multiple partners / bipartite matching are explicitly out of scope here),
    and this test pins the property that makes it dead code today.
    """
    rng = random.Random(4711)
    cfg = PreferenceConfig(order="preferences_first")
    for _ in range(300):
        bids, offers = _random_book(rng)
        traded = min(sum(o["energy"] for o in offers),
                     sum(b["energy"] for b in bids))
        result = apply_preference_allocation(
            bids, offers, traded_quantity=traded,
            total_supply_kwh=sum(o["energy"] for o in offers),
            total_demand_kwh=sum(b["energy"] for b in bids), cfg=cfg)
        assert result.preferential_kwh <= traded + 1e-9
        assert result.pairs_rationed is False


@pytest.mark.anyio
async def test_pro_rata_first_keeps_quantities_but_reports_pairs():
    preferences_first = await clear(reference_book(mutual=True))
    pro_rata_first = await clear(reference_book(mutual=True),
                                 order="pro_rata_first")
    baseline = await clear(reference_book(mutual=False))

    assert (by_name(pro_rata_first, "producers")["PV A"]["allocated_kwh"]
            == pytest.approx(
                by_name(baseline, "producers")["PV A"]["allocated_kwh"]))
    for producer in pro_rata_first["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)
    # …but the pair is still detected and routed
    assert by_name(pro_rata_first, "producers")["PV A"][
        "preference_matched"] is True
    assert pro_rata_first["preferences"]["mutual_pairs"] == [
        {"bid_area": "area-house-1", "offer_area": "area-pv-a",
         "energy_kwh": 0.0}]
    assert (by_name(preferences_first, "producers")["PV A"]["allocated_kwh"]
            > by_name(pro_rata_first, "producers")["PV A"]["allocated_kwh"])


@pytest.mark.anyio
@pytest.mark.parametrize("green_multiplier,grey_levy,levy_cap",
                         [(0.10, 0.10, 0.20), (0.04, 0.10, 0.20),
                          (0.50, 0.10, 0.20), (0.02, 0.30, 0.05),
                          (0.0, 0.10, 0.20)])
async def test_additive_mode_is_zero_sum(green_multiplier, grey_levy,
                                         levy_cap):
    result = await clear(reference_book(mutual=True), mode="additive",
                         green_multiplier=green_multiplier,
                         grey_levy=grey_levy, levy_cap=levy_cap)
    multipliers = result["preferences"]["multipliers"]
    assert multipliers["grey_alloc_kwh"] > 0
    assert multipliers["pool_surplus_ct"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.anyio
async def test_additive_mode_reproduces_the_infopaper_shape():
    """Bonus-driven: the levy per kWh follows from funding the bonus."""
    result = await clear(reference_book(mutual=True), mode="additive",
                         green_multiplier=0.10, levy_cap=1.0)
    multipliers = result["preferences"]["multipliers"]
    price = result["clearing_price_ct_per_kwh"]
    bonus_per_kwh = price * 0.10
    levy_per_kwh = bonus_per_kwh * 7.25 / 2.75
    assert multipliers["green_final_ct_per_kwh"] == pytest.approx(
        price + bonus_per_kwh, abs=1e-5)
    assert multipliers["grey_final_ct_per_kwh"] == pytest.approx(
        price - levy_per_kwh, abs=1e-5)
    assert multipliers["scale"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.anyio
async def test_additive_levy_cap_scales_the_bonus_down():
    result = await clear(reference_book(mutual=True), mode="additive",
                         green_multiplier=0.10, levy_cap=0.05)
    multipliers = result["preferences"]["multipliers"]
    price = result["clearing_price_ct_per_kwh"]
    assert multipliers["grey_final_ct_per_kwh"] == pytest.approx(
        price * 0.95, abs=1e-5)
    assert multipliers["scale"] < 1.0
    assert multipliers["pool_surplus_ct"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.anyio
async def test_no_grey_volume_means_no_funding_and_a_warning(caplog):
    orders = reference_book(mutual=True)
    orders[2]["attributes"] = {"energy_type": "green"}
    with caplog.at_level("WARNING"):
        result = await clear(orders, green_multiplier=0.10)
    multipliers = result["preferences"]["multipliers"]
    assert multipliers["grey_alloc_kwh"] == 0.0
    assert multipliers["bonus_paid_ct"] == 0.0
    assert multipliers["pool_surplus_ct"] == pytest.approx(0.0, abs=1e-6)
    assert any("no funding source" in record.message
               for record in caplog.records)


@pytest.mark.anyio
@pytest.mark.parametrize("attributes", ["Green", 123, {"energy_type": "Green"},
                                        {"energy_type": None},
                                        {"energy_type": 123}])
async def test_malformed_attributes_degrade_to_green(attributes, caplog):
    orders = reference_book(mutual=True)
    orders[2]["attributes"] = attributes
    with caplog.at_level("WARNING"):
        result = await clear(orders)
    battery = by_name(result, "producers")["Battery"]
    assert battery["energy_type"] == "green"
    assert result["preferences"]["multipliers"]["grey_alloc_kwh"] == 0.0
    assert caplog.records  # degraded, but loudly


@pytest.mark.anyio
@pytest.mark.parametrize("requirements", ["area-pv-a", 42,
                                          {"preferred_partner": 42},
                                          {"preferred_partner": ""},
                                          {"something_else": "x"}])
async def test_malformed_requirements_degrade_to_no_preference(requirements):
    orders = reference_book(mutual=True)
    orders[0]["requirements"] = requirements
    result = await clear(orders)
    assert result["preferences"]["mutual_pairs"] == []
    for producer in result["allocations"]["producers"]:
        assert producer["fill_rate"] == pytest.approx(0.8)


def test_parsers_are_defensive():
    assert parse_energy_type({}) == "green"
    assert parse_energy_type({"attributes": None}) == "green"
    assert parse_energy_type({"attributes": {"energy_type": "grey"}}) == "grey"
    assert parse_energy_type({"attributes": "grey"}) == "green"

    areas = {"area-a"}
    assert parse_preferred_partner({}, areas) is None
    assert parse_preferred_partner({"requirements": None}, areas) is None
    assert parse_preferred_partner(
        {"requirements": {"preferred_partner": None}}, areas) is None
    assert parse_preferred_partner(
        {"requirements": {"preferred_partner": "area-a"}}, areas) == "area-a"
    assert parse_preferred_partner(
        {"requirements": {"preferred_partner": "area-b"}}, areas) is None


# ------------------------------------------- §6.6 property-based balance test

def _random_book(rng: random.Random) -> tuple[list[dict], list[dict]]:
    n_offers = rng.randint(1, 6)
    n_bids = rng.randint(1, 6)
    offers = [make_order(i, "Offer", f"S{i}", f"area-s{i}",
                         round(rng.uniform(0.1, 12.0), 3),
                         energy_type=rng.choice(["green", "grey", None]))
              for i in range(n_offers)]
    bids = [make_order(100 + i, "Bid", f"B{i}", f"area-b{i}",
                       round(rng.uniform(0.1, 12.0), 3))
            for i in range(n_bids)]
    # preference density: roughly half the orders nominate a counterparty
    for offer in offers:
        if rng.random() < 0.5:
            offer["requirements"] = {
                "preferred_partner": rng.choice(bids)["area_uuid"]}
    for bid in bids:
        if rng.random() < 0.5:
            bid["requirements"] = {
                "preferred_partner": rng.choice(offers)["area_uuid"]}
    return bids, offers


@pytest.mark.parametrize("order", ["preferences_first", "pro_rata_first"])
@pytest.mark.parametrize("enabled", [True, False])
def test_allocation_always_balances(order, enabled):
    """Randomised order books (fixed seed): §3.5 must hold everywhere."""
    rng = random.Random(20250810)
    cfg = PreferenceConfig(enabled=enabled, order=order)
    for _ in range(200):
        bids, offers = _random_book(rng)
        supply = sum(o["energy"] for o in offers)
        demand = sum(b["energy"] for b in bids)
        traded = min(supply, demand)
        apply_preference_allocation(
            bids, offers, traded_quantity=traded, total_supply_kwh=supply,
            total_demand_kwh=demand, cfg=cfg)

        assert sum(b["allocated_energy"] for b in bids) == pytest.approx(
            traded, abs=1e-9 * max(1.0, traded))
        assert sum(o["allocated_energy"] for o in offers) == pytest.approx(
            traded, abs=1e-9 * max(1.0, traded))
        for entry in bids + offers:
            assert 0.0 <= entry["allocated_energy"] <= entry["energy"] + 1e-12


# ------------------------------------------------------ configuration surface

def test_preference_config_validates_its_inputs():
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(order="whatever")
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(mode="exponential")
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(sides="buyer")
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(green_multiplier=-0.1)
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(grey_levy=-0.1)
    with pytest.raises(PreferenceConfigError):
        PreferenceConfig(levy_cap=-1.0)


def test_configuration_yaml_ships_a_valid_preference_block():
    preferences = load_config().preferences
    assert preferences.enabled is True
    assert preferences.order == "preferences_first"
    assert preferences.mode == "multiplicative"
    assert preferences.sides == "seller"
    assert preferences.levy_cap == pytest.approx(0.20)


def test_environment_variables_override_the_configuration_file(monkeypatch):
    monkeypatch.setenv("PREFERENCE_ORDER", "pro_rata_first")
    monkeypatch.setenv("MULTIPLIER_MODE", "additive")
    monkeypatch.setenv("MULTIPLIER_SIDES", "both")
    monkeypatch.setenv("GREEN_MULTIPLIER", "0.25")
    monkeypatch.setenv("PREFERENCES_ENABLED", "false")

    preferences = load_config().preferences
    assert preferences.order == "pro_rata_first"
    assert preferences.mode == "additive"
    assert preferences.sides == "both"
    assert preferences.green_multiplier == pytest.approx(0.25)
    assert preferences.enabled is False


def test_an_unusable_environment_override_fails_loudly(monkeypatch):
    monkeypatch.setenv("MULTIPLIER_MODE", "exponential")
    with pytest.raises(PreferenceConfigError):
        load_config()


def test_trigger_overrides_win_over_configuration():
    cfg = Config(preferences=PreferenceConfig(mode="multiplicative"))
    resolved = resolve_preferences(cfg, {"mode": "additive",
                                         "green_multiplier": 0.25})
    assert resolved.mode == "additive"
    assert resolved.green_multiplier == 0.25
    assert resolved.grey_levy == cfg.preferences.grey_levy  # untouched


# ---------------------------------------------------------- §4.3 / sides=both

@pytest.mark.anyio
async def test_uniform_energy_rate_is_never_overwritten():
    """The §4.3 trap: the Execution Node reads `parameters.energy_rate`."""
    result = await clear(reference_book(mutual=True))
    price = result["clearing_price_ct_per_kwh"]
    for trade in result["trades"]:
        assert trade["parameters"]["energy_rate"] == pytest.approx(price)
    seller_trades = [t for t in result["trades"]
                     if t["buyer"].startswith("AMM_POOL")]
    assert any(t["parameters"]["final_energy_rate"]
               != t["parameters"]["energy_rate"] for t in seller_trades)
    assert all("multiplier_applied" in t["parameters"]
               for t in result["trades"])


@pytest.mark.anyio
async def test_sides_both_passes_the_pool_position_to_the_buyers():
    seller_only = await clear(reference_book(mutual=True), sides="seller")
    both = await clear(reference_book(mutual=True), sides="both",
                       green_multiplier=0.03)
    price = both["clearing_price_ct_per_kwh"]

    assert seller_only["preferences"]["multipliers"][
        "buyer_final_ct_per_kwh"] == pytest.approx(price)
    # over-collecting levy: with sides="both" the surplus goes back to buyers
    assert both["preferences"]["multipliers"][
        "buyer_final_ct_per_kwh"] < price
    assert both["preferences"]["multipliers"]["pool_surplus_ct"] == \
        pytest.approx(0.0, abs=1e-3)


@pytest.mark.anyio
@pytest.mark.parametrize("preference_params", [
    {},
    {"order": "pro_rata_first"},
    {"mode": "additive"},
    {"sides": "both"},
    {"mode": "additive", "sides": "both", "green_multiplier": 0.25},
    {"multipliers_enabled": False},
    {"enabled": False},
])
async def test_idempotent_retrigger_rebuilds_the_same_preferences_block(
        preference_params):
    """The re-trigger path reads the block back off the stored trades, so
    every configuration must survive the round trip unchanged."""
    orders = reference_book(mutual=True)
    db = FakeOffchainDB(orders)
    chain = MockContractClient()
    payload = trigger(preference_params=preference_params or None)
    first = await run_clearing(payload, Config(), db, chain)
    second = await run_clearing(payload, Config(), db, chain)

    assert second["status"] == "already_cleared"
    assert second["preferences"] == first["preferences"]
    assert (by_name(second, "producers")
            == by_name(first, "producers"))
    producers = by_name(second, "producers")
    assert producers["PV A"]["preference_matched"] is (
        preference_params.get("enabled", True))
    assert producers["Battery"]["energy_type"] == "grey"


# --------------------------------------------- §6.7 execution-node regression

_EXECUTION_DIR = Path(__file__).resolve().parents[2] / "amm-execution-node"

# The Execution Node is a separate service with its own top-level `src`
# package, so it cannot be imported into this process alongside the Clearing
# Node's. Running it in a subprocess keeps the regression honest: these are
# the real penalty results, computed by the real service code.
_RUN_EXECUTION = """
import asyncio, json, sys
from src.config import Config
from src.execution import run_execution

payload = json.load(sys.stdin)


class FakeDB:
    async def get_trades(self, market_id):
        return payload["trades"]

    async def get_measurements(self, community_uuid, time_slot=None):
        return payload["measurements"]

    async def update_trade(self, trade_uuid, *, status=None, parameters=None):
        return {}


result = asyncio.run(run_execution(payload["trigger"], Config(), FakeDB()))
json.dump(result["results"], sys.stdout)
"""


def _run_execution_node(trades: list[dict]) -> list[dict]:
    measurements = [
        {"community_uuid": "communityid_1", "area_uuid": area,
         "time_slot": SLOT, "energy_kwh": kwh}
        for area, kwh in {"area-pv-a": 3.0, "area-pv-b": 2.4,
                          "area-battery": 2.75, "area-house-1": 4.5,
                          "area-house-2": 3.0, "area-bakery": 4.5}.items()]
    payload = {"trades": trades, "measurements": measurements,
               "trigger": {"market_id": MARKET,
                           "community_uuid": "communityid_1",
                           "time_slot": SLOT}}
    completed = subprocess.run(
        [sys.executable, "-c", _RUN_EXECUTION], input=json.dumps(payload),
        capture_output=True, text=True, cwd=_EXECUTION_DIR)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.anyio
@pytest.mark.skipif(not _EXECUTION_DIR.exists(),
                    reason="sibling service not present in this checkout")
async def test_multipliers_do_not_change_execution_node_penalties():
    """§4.3 guard: penalties are computed from the *uniform* clearing price,
    so switching the multipliers on must not move a single cent."""
    with_multipliers = await clear(reference_book(mutual=True),
                                   multipliers_enabled=True)
    without = await clear(reference_book(mutual=True),
                          multipliers_enabled=False)

    # Same allocation either way — only the rates differ.
    def allocated(run: dict) -> list[float]:
        return [p["allocated_kwh"] for p in run["allocations"]["producers"]]

    assert allocated(with_multipliers) == allocated(without)

    penalties_with = _run_execution_node(with_multipliers["trades"])
    penalties_without = _run_execution_node(without["trades"])

    def comparable(rows: list[dict]) -> dict:
        return {row["name"]: {k: v for k, v in row.items()
                              if k != "trade_uuid"} for row in rows}

    assert comparable(penalties_with) == comparable(penalties_without)
    assert any(row["total_penalty_ct"] > 0 for row in penalties_with)
