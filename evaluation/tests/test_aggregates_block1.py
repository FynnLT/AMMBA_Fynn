"""Section 2: the block-1 quantities reach disk (Code To-Do 3.11).

The first campaign wrote eight quantities, every one of them invariant under
preference order and under the multipliers, so `baseline_pro_rata`,
`preferences_first`, `pro_rata_first` and `multipliers_on` produced
byte-identical slot CSVs. The regression guard below is the test that would
have caught that: it clears the *same* profile slot twice, once with the
multipliers on and once with preferences off, and asserts the rows differ.
"""
import csv

import pytest

import aggregates
import campaign
import profiles
import runner
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    return stack.Stack()


def _preferences_block(**over):
    """A `clearing["preferences"]` block shaped like the artifact's."""
    multipliers = {
        "enabled": True, "mode": "multiplicative", "sides": "seller",
        "green_alloc_kwh": 8.5, "grey_alloc_kwh": 4.0,
        "levy_collected_ct": 6.0, "bonus_requested_ct": 12.75,
        "bonus_paid_ct": 6.0, "scale": 0.470588,
        "green_final_ct_per_kwh": 15.705882, "grey_final_ct_per_kwh": 13.5,
        "buyer_final_ct_per_kwh": 15.0, "buyers_pay_ct": 150.0,
        "sellers_receive_ct": 187.5, "pool_surplus_ct": -37.5,
        "warnings": [],
    }
    multipliers.update(over.pop("multipliers", {}))
    block = {
        "enabled": True, "order": "preferences_first",
        "mutual_pairs": [{"bid_area": "player-001", "offer_area": "player-002",
                          "energy_kwh": 1.25},
                         {"bid_area": "player-003", "offer_area": "player-004",
                          "energy_kwh": 0.75}],
        "pairs_rationed": False, "multipliers": multipliers,
    }
    block.update(over)
    return block


def _record(*, preferences=None, scen=None, price=15.0):
    clearing = {
        "status": "cleared", "total_supply_kwh": 12.5,
        "total_demand_kwh": 10.0, "ratio": 1.25,
        "clearing_price_ct_per_kwh": price, "traded_quantity_kwh": 10.0,
        "round_type": aggregates.DEMAND_LIMITED,
        "allocations": {
            "producers": [
                {"area_uuid": "player-002", "requested_kwh": 5.0,
                 "allocated_kwh": 4.0, "fill_rate": 0.8,
                 "energy_type": "green", "preference_matched": True},
                {"area_uuid": "battery-007", "requested_kwh": 5.0,
                 "allocated_kwh": 4.0, "fill_rate": 0.8,
                 "energy_type": "grey", "preference_matched": False}],
            "consumers": [
                {"area_uuid": "player-001", "requested_kwh": 3.0,
                 "allocated_kwh": 3.0, "fill_rate": 1.0,
                 "energy_type": "mixed", "preference_matched": True}]},
    }
    if preferences is not None:
        clearing["preferences"] = preferences
    return {"time_slot": BASE_SLOT, "market_id": "0xabc", "community": "c",
            "clearing": clearing, "execution": None,
            "scenario": scen if scen is not None
            else {"pairs_posted": 2, "pairs_unpostable": 1, "pairs_mutual": 1}}


# ------------------------------------------------------ the round-trip

def test_the_preference_block_round_trips_field_by_field():
    """Copied out of the response, not recomputed beside it."""
    row = aggregates.slot_row(_record(preferences=_preferences_block()))

    assert row["pref_enabled"] is True
    assert row["pref_order"] == "preferences_first"
    assert row["n_pairs"] == 2
    assert row["pair_kwh"] == 2.0
    assert row["pairs_rationed"] is False

    assert row["mult_enabled"] is True
    assert row["mult_mode"] == "multiplicative"
    assert row["mult_sides"] == "seller"
    assert row["green_alloc_kwh"] == 8.5
    assert row["grey_alloc_kwh"] == 4.0
    assert row["levy_collected_ct"] == 6.0
    assert row["bonus_requested_ct"] == 12.75
    assert row["bonus_paid_ct"] == 6.0
    assert row["bonus_scale"] == 0.470588        # response calls it `scale`
    assert row["green_final_ct"] == 15.705882
    assert row["grey_final_ct"] == 13.5
    assert row["buyer_final_ct"] == 15.0
    assert row["buyers_pay_ct"] == 150.0
    assert row["sellers_receive_ct"] == 187.5
    assert row["pool_surplus_ct"] == -37.5

    assert row["pairs_posted"] == 2
    assert row["pairs_unpostable"] == 1

    # The one derived field: levy / (p * green_alloc) = 6 / (15 * 8.5).
    assert row["crossover_mult"] == pytest.approx(0.047059, abs=1e-6)
    # and it is the run plan's grey_levy * grey/green at the effective levy:
    # grey_final = p * (1 - levy) -> levy = 0.1, 0.1 * 4.0 / 8.5.
    assert row["crossover_mult"] == pytest.approx(0.1 * 4.0 / 8.5, abs=1e-6)

    # The eight original fields are untouched.
    assert [row[f] for f in aggregates.BASE_FIELDS] == \
        [BASE_SLOT, 12.5, 10.0, 1.25, 15.0, 10.0, 2, 1]


def test_a_record_without_the_block_yields_none_in_every_new_field():
    """The no-trade path returns no `preferences` block at all. A gap is
    written as a gap; a measured zero is written as a zero."""
    row = aggregates.slot_row(_record(preferences=None,
                                      scen={"pairs_posted": 0,
                                            "pairs_unpostable": 0,
                                            "pairs_mutual": 0}))
    for field in (aggregates.PREFERENCE_FIELDS + aggregates.MULTIPLIER_FIELDS
                  + aggregates.DERIVED_FIELDS):
        assert row[field] is None, field
    assert [row[f] for f in aggregates.BASE_FIELDS] == \
        [BASE_SLOT, 12.5, 10.0, 1.25, 15.0, 10.0, 2, 1]

    # And with preferences disabled the block *is* present: `enabled: False`
    # plus the round's green/grey split is a measurement, not a gap.
    off = aggregates.slot_row(_record(preferences=_preferences_block(
        enabled=False, order="preferences_first", mutual_pairs=[],
        multipliers={"enabled": False, "levy_collected_ct": 0.0,
                     "bonus_requested_ct": 0.0, "bonus_paid_ct": 0.0,
                     "green_final_ct_per_kwh": 15.0,
                     "grey_final_ct_per_kwh": 15.0})))
    assert off["pref_enabled"] is False
    assert off["n_pairs"] == 0
    assert off["pair_kwh"] == 0.0
    assert off["mult_enabled"] is False
    assert off["green_alloc_kwh"] == 8.5          # measured, not None
    assert off["levy_collected_ct"] == 0.0
    assert off["crossover_mult"] is None          # no levy: no crossover


# ------------------------------------------------------ the regression guard

@pytest.mark.anyio
async def test_multipliers_change_the_row_on_a_real_profile_slot(st):
    """**The test that would have failed on 16e696a.**

    One profile slot with a grey battery offer, cleared twice: with the
    multipliers on and with preferences off. On the old `FIELDS` the two rows
    were identical in all eight columns, which is exactly how four mechanism
    cells came to write the same CSV.
    """
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    batteries = profiles.make_battery_areas(community.bess, players,
                                            participation=0.25, seed=7)
    week = profiles.extend_to_week(community.day_kwh(players), seed=4242,
                                   days=1)

    # An evening index: the battery rule discharges into periods 73..88, so
    # this is where the grey offers are.
    index = next(i for i in range(72, 88)
                 if (scenario.make_scenario_from_profiles(week, i, batteries)
                     ["battery_kwh"] > 0))
    scen = scenario.make_scenario_from_profiles(week, index, batteries)
    assert any(p.get("energy_type") == "grey" for p in scen["producers"])

    try:
        on = await runner.run_slot(
            st, scen, community="s2-mult-on", slot=BASE_SLOT,
            market_id=runner.market_id_for("s2-mult-on", BASE_SLOT),
            preferences=campaign.prefs(multipliers_enabled=True),
            execute=False)
        off = await runner.run_slot(
            st, scen, community="s2-mult-off", slot=BASE_SLOT,
            market_id=runner.market_id_for("s2-mult-off", BASE_SLOT),
            preferences=campaign.PREF_OFF, execute=False)
    finally:
        await st.close()

    row_on = aggregates.slot_row(on)
    row_off = aggregates.slot_row(off)

    # The eight original fields agree -- which is the whole problem.
    assert [row_on[f] for f in aggregates.BASE_FIELDS] == \
        [row_off[f] for f in aggregates.BASE_FIELDS]

    assert row_on["levy_collected_ct"] > 0
    assert row_off["levy_collected_ct"] == 0.0
    assert row_on["mult_enabled"] is True
    assert row_off["mult_enabled"] is False
    assert row_on["sellers_receive_ct"] != row_off["sellers_receive_ct"]
    assert row_on["pool_surplus_ct"] != row_off["pool_surplus_ct"]

    # `crossover_mult` is None here, and that is a result rather than a gap:
    # on this week the battery rule discharges into 18:15..22:00 and PV runs
    # 08:00..17:45, so a slot carries either green supply or grey supply and
    # never both. `green_alloc_kwh` is 0 in every slot that has a levy to
    # collect, so the levy funds no bonus at any `green_multiplier` -- see
    # the section-5 note on D-79.
    assert row_on["green_alloc_kwh"] == 0.0
    assert row_on["grey_alloc_kwh"] > 0
    assert row_on["crossover_mult"] is None
    assert row_off["crossover_mult"] is None


# ---------------------------------------------------------- the area CSV

def test_write_area_csv_writes_one_row_per_allocation_and_carries_the_draw(
        tmp_path):
    preferences = {"player-001": "player-002", "player-002": "player-001"}
    records = [_record(preferences=_preferences_block()),
               {**_record(preferences=_preferences_block()),
                "time_slot": BASE_SLOT + 900},
               # A no-trade slot writes nothing here.
               {"time_slot": BASE_SLOT + 1800, "market_id": "0xdef",
                "community": "c", "execution": None,
                "clearing": {"status": "no_trade", "total_supply_kwh": 0.0,
                             "total_demand_kwh": 4.0, "trades": []}}]

    path = aggregates.write_area_csv(tmp_path / "r_areas.csv", records,
                                     preferences)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6                       # 3 allocations x 2 slots
    assert list(rows[0]) == list(aggregates.AREA_FIELDS)
    assert {r["slot"] for r in rows} == {str(BASE_SLOT), str(BASE_SLOT + 900)}

    by_area = {r["area_uuid"]: r for r in rows if r["slot"] == str(BASE_SLOT)}
    assert by_area["player-001"]["side"] == "buyer"
    assert by_area["player-001"]["named"] == "True"
    assert by_area["player-001"]["partner"] == "player-002"
    assert by_area["player-002"]["side"] == "seller"
    assert by_area["player-002"]["partner"] == "player-001"
    assert by_area["player-002"]["energy_type"] == "green"
    assert by_area["player-002"]["fill_rate"] == "0.8"
    assert by_area["player-002"]["preference_matched"] == "True"
    # The battery names nobody and is named by nobody (D-75).
    assert by_area["battery-007"]["named"] == "False"
    assert by_area["battery-007"]["partner"] == ""
    assert by_area["battery-007"]["energy_type"] == "grey"


# ==================================================================
# Section 4: the `redistribution` block reaches disk (Code To-Do 3.4)
# ==================================================================

def _execution(**over):
    """An execution response shaped like `execution.run_execution`'s."""
    block = {
        "status": "executed",
        "round_type": aggregates.SUPPLY_LIMITED,
        "clearing_price_ct_per_kwh": 15.0,
        "total_penalties_ct": 9.5,
        "redistribution": {
            "rule": "proportional", "computed_only": True,
            "harmed_side": "buyer", "harmed_side_kwh": 10.0,
            "penalty_pool_ct": 8.0, "compensated_ct": 8.0,
            "budget_balance_ct": 0.0,
            "excluded_deviators": ["player-002"],
            "rows": [{"area_uuid": "player-001", "damage_ct": 3.0,
                      "compensation_ct": 8.0}]},
        "results": [
            {"area_uuid": "player-002", "role": "seller",
             "actual_kwh": 4.0, "deliverable_kwh": 5.0,
             "deliverable_source": "forecast", "shortfall_kwh": 0.0,
             "shortfall_penalty_ct": 0.0, "externality_kwh": 1.0,
             "externality_penalty_ct": 8.0, "total_penalty_ct": 8.0},
            {"area_uuid": "battery-007", "role": "seller",
             "actual_kwh": 3.8, "deliverable_kwh": 4.0,
             "deliverable_source": "forecast", "shortfall_kwh": 0.1,
             "shortfall_penalty_ct": 1.5, "externality_kwh": 0.0,
             "externality_penalty_ct": 0.0, "total_penalty_ct": 1.5},
            {"area_uuid": "player-001", "role": "buyer",
             "actual_kwh": 3.0, "deliverable_kwh": 3.0,
             "deliverable_source": "meter", "shortfall_kwh": 0.0,
             "shortfall_penalty_ct": 0.0, "externality_kwh": 0.0,
             "externality_penalty_ct": 0.0, "total_penalty_ct": 0.0}],
    }
    block.update(over)
    return block


def test_the_execution_block_round_trips_field_by_field():
    """Copied, not recomputed: `budget_balance_ct` is the node's own
    statement of whether the rule closed, and the +-0 test in 5.3 reads this
    column."""
    record = {**_record(preferences=_preferences_block()),
              "execution": _execution()}
    row = aggregates.slot_row(record)

    assert row["round_type_exec"] == aggregates.SUPPLY_LIMITED
    assert row["total_penalties_ct"] == 9.5
    assert row["penalty_pool_ct"] == 8.0
    assert row["compensated_ct"] == 8.0
    assert row["budget_balance_ct"] == 0.0
    assert row["n_excluded"] == 1
    assert row["n_harmed"] == 1
    assert row["harmed_side"] == "buyer"
    assert row["n_deviators"] == 1
    assert row["n_shortfall"] == 1
    assert row["withheld_kwh"] == 1.0            # sellers only
    assert row["underreported_kwh"] == 0.0       # buyers only
    assert row["shortfall_penalty_ct"] == 1.5
    assert row["externality_penalty_ct"] == 8.0

    # The clearing-side round type is kept apart from the execution one: two
    # services compute it from the same totals, and a merged column would
    # hide a disagreement rather than show it.
    assert "round_type" not in row


def test_a_record_without_an_execution_block_yields_none_in_every_field():
    """A slot that did not execute is not a slot with zero penalties: the
    first is a gap and the second is a measurement."""
    row = aggregates.slot_row(_record(preferences=_preferences_block()))
    for field in aggregates.EXECUTION_FIELDS:
        assert row[field] is None, field

    # The section-2 fields are untouched by the execution projection.
    with_exec = aggregates.slot_row(
        {**_record(preferences=_preferences_block()),
         "execution": _execution()})
    for field in (aggregates.BASE_FIELDS + aggregates.PREFERENCE_FIELDS
                  + aggregates.MULTIPLIER_FIELDS + aggregates.SCENARIO_FIELDS
                  + aggregates.DERIVED_FIELDS):
        assert row[field] == with_exec[field], field

    # A no-trades execution response is a gap as well.
    empty = aggregates.slot_row(
        {**_record(preferences=_preferences_block()),
         "execution": {"status": "no_trades", "results": []}})
    for field in aggregates.EXECUTION_FIELDS:
        assert empty[field] is None, field


def test_the_area_csv_carries_the_per_area_execution_columns(tmp_path):
    records = [{**_record(preferences=_preferences_block()),
                "execution": _execution()}]
    path = aggregates.write_area_csv(tmp_path / "e_areas.csv", records,
                                     {"player-001": "player-002"})
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {r["area_uuid"]: r for r in csv.DictReader(handle)}

    assert list(next(iter(rows.values()))) == list(aggregates.AREA_FIELDS)

    deviator = rows["player-002"]
    assert deviator["actual_kwh"] == "4.0"
    assert deviator["deliverable_kwh"] == "5.0"
    assert deviator["deliverable_source"] == "forecast"
    assert deviator["externality_penalty_ct"] == "8.0"
    assert deviator["total_penalty_ct"] == "8.0"
    # A deviator is excluded from the harmed set (D-43), so it has no row in
    # `redistribution.rows` and no compensation.
    assert deviator["compensation_ct"] == ""

    harmed = rows["player-001"]
    assert harmed["compensation_ct"] == "8.0"
    assert harmed["externality_penalty_ct"] == "0.0"
    assert rows["battery-007"]["shortfall_penalty_ct"] == "1.5"

    # Without an execution block the same seven columns are empty.
    plain = aggregates.write_area_csv(tmp_path / "p_areas.csv",
                                      [_record(preferences=None)], {})
    with open(plain, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in aggregates._AREA_EXECUTION_KEYS + ("compensation_ct",):
                assert row[key] == "", key
