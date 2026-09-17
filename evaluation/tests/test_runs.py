"""Section 3: per-slot aggregates, the manifest and the pre-flight guards."""
import json

import pytest

import aggregates
import profiles
import runner
import runs
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


def _record(slot, *, supply, demand, price=15.0, status="cleared",
            n_producers=3, n_consumers=4):
    """A clearing response shaped like the artifact's, for the projections."""
    ratio = (supply / demand) if demand else None
    round_type = (aggregates.SUPPLY_LIMITED if supply < demand
                  else aggregates.DEMAND_LIMITED if supply > demand
                  else aggregates.BALANCED)
    clearing = {"status": status, "total_supply_kwh": supply,
                "total_demand_kwh": demand, "ratio": ratio,
                "clearing_price_ct_per_kwh": price,
                "traded_quantity_kwh": min(supply, demand),
                "round_type": round_type,
                "allocations": {"producers": [{}] * n_producers,
                                "consumers": [{}] * n_consumers}}
    if status != "cleared":
        clearing = {"status": status, "total_supply_kwh": supply,
                    "total_demand_kwh": demand, "trades": []}
    return {"time_slot": slot, "market_id": f"0x{slot:064x}",
            "community": "c", "clearing": clearing, "execution": None}


# ------------------------------------------------------------- aggregates

def test_the_csv_is_a_projection_of_the_clearing_response(tmp_path):
    """Not a recomputation: the calibration has to measure the artifact, not
    the generator that fed it."""
    records = [_record(BASE_SLOT, supply=12.5, demand=10.0),
               _record(BASE_SLOT + 900, supply=4.0, demand=10.0)]
    path = aggregates.write_slot_csv(tmp_path / "run_slots.csv", records)

    import csv
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [r["slot"] for r in rows] == [str(BASE_SLOT), str(BASE_SLOT + 900)]
    assert tuple(rows[0]) == aggregates.FIELDS
    assert float(rows[0]["ratio"]) == 1.25
    assert float(rows[0]["traded_kwh"]) == 10.0
    assert int(rows[0]["n_producers"]) == 3
    assert int(rows[0]["n_consumers"]) == 4
    # Read off the response, not divided out of the scenario.
    assert float(rows[1]["ratio"]) == pytest.approx(0.4)


def test_round_type_census_counts_both_types_and_keeps_no_trade_apart():
    records = [_record(BASE_SLOT, supply=12.0, demand=10.0),
               _record(BASE_SLOT + 900, supply=4.0, demand=10.0),
               _record(BASE_SLOT + 1800, supply=4.0, demand=10.0),
               _record(BASE_SLOT + 2700, supply=0.0, demand=10.0,
                       status="no_trade")]
    census = aggregates.round_type_census(records)
    assert census[aggregates.DEMAND_LIMITED] == 1
    assert census[aggregates.SUPPLY_LIMITED] == 2
    # A slot with no supply is not a supply-limited round; it is a slot the
    # mechanism never ran on, and 5.1 must not sum the two.
    assert census["no_trade"] == 1
    assert sum(census.values()) == len(records)


# ------------------------------------------------------------- the manifest

def _manifest_kwargs(**overrides):
    spec = runs.RunSpec(run_id="r-001", seed=3, cell_seeds=(0, 1, 2, 3, 4),
                        cell="theta_1.0")
    kwargs = dict(config=spec.config(), seed=spec.seed,
                  dataset_extension_seed=spec.dataset_extension_seed,
                  player_ids=[1, 5, 9], cell_seeds=spec.cell_seeds,
                  output_files=["r-001_slots.csv"], sha="deadbeef")
    kwargs.update(overrides)
    return kwargs


def test_the_manifest_records_every_required_field_for_every_run(tmp_path):
    path = tmp_path / "manifest.json"
    runs.write_manifest(path, "r-001", **_manifest_kwargs())
    runs.write_manifest(path, "r-002", **_manifest_kwargs(seed=4))

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert [e["run_id"] for e in entries] == ["r-001", "r-002"]
    for entry in entries:
        for required in runs.REQUIRED_MANIFEST_FIELDS:
            assert entry.get(required) is not None, required
        assert entry["repo_sha"] == "deadbeef"
        assert entry["dataset_extension_seed"] == 4242
        assert entry["player_ids"] == [1, 5, 9]
        assert entry["dataset_sha256"] == runs.DATASET_SHA256
        # Individually, not "seeds 0-4": one deviating cell must be
        # re-runnable on its own (D-59).
        assert entry["cell_seeds"] == [0, 1, 2, 3, 4]
        # The delta baseline is named once and not re-derived per table.
        assert entry["baseline"] == runs.BASELINE


def test_a_manifest_missing_a_required_field_is_refused(tmp_path):
    with pytest.raises(runs.PreflightError, match="dataset_extension_seed"):
        runs.write_manifest(tmp_path / "manifest.json", "r-003",
                            **_manifest_kwargs(dataset_extension_seed=None))
    assert not (tmp_path / "manifest.json").exists()


# ----------------------------------------------------------- the guards

@pytest.fixture
def st():
    return stack.Stack()


@pytest.mark.anyio
async def test_check_ratio_spread_raises_on_a_fixed_ratio_sequence(st):
    """`make_scenario` scales generation so the ratio hits `sd_ratio`
    exactly, so a sweep at one fixed ratio gives 672 identical ratios and a
    fit that looks fitted with nothing reporting an error."""
    slots = [BASE_SLOT + i * runner.SLOT_SEC for i in range(12)]
    records = await runner.run_sequence(
        st, "fixed-ratio", slots,
        lambda index, slot: scenario.make_scenario(
            seed=index, n_prod=8, n_cons=12, sd_ratio=1.25),
        execute=False)

    assert len(aggregates.ratios(records)) == 12
    # Not *exactly* fixed -- `make_scenario` rounds the scaled energies to
    # four decimals, so the realised ratio wobbles in the fifth. That is the
    # realistic version of the failure: it moves just enough to look alive.
    assert aggregates.ratio_span(records) < 1e-3
    with pytest.raises(runs.PreflightError, match="ratio span"):
        runs.check_ratio_spread(records)


@pytest.mark.anyio
async def test_check_ratio_spread_passes_on_the_real_week(st):
    """The same guard on the profile slots, through the real services."""
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    week = profiles.extend_to_week(community.day_kwh(players), seed=4242,
                                   days=1)
    batteries = profiles.make_battery_areas(community.bess, players, seed=7)
    scenarios = [scenario.make_scenario_from_profiles(week, i, batteries)
                 for i in range(week.n_slots)]

    # Midday into the evening: PV-rich slots and battery-only slots in one
    # slice, which is where the ratio actually moves.
    window = range(40, 80)
    slots = [BASE_SLOT + i * runner.SLOT_SEC for i in window]
    records = await runner.run_sequence(
        st, "real-week", slots, lambda index, slot: scenarios[window[index]],
        areas=runs.run_areas(players, batteries), execute=False)

    span = runs.check_ratio_spread(records)
    assert span > 0.25
    census = runs.check_round_type_census(records)
    assert census[aggregates.SUPPLY_LIMITED] > 0
    assert census[aggregates.DEMAND_LIMITED] > 0


def test_check_ratio_spread_raises_when_nothing_cleared():
    records = [_record(BASE_SLOT, supply=0.0, demand=10.0, status="no_trade")]
    with pytest.raises(runs.PreflightError, match="no slot cleared"):
        runs.check_ratio_spread(records)


def test_preflight_requires_both_round_types():
    records = [_record(BASE_SLOT + i * 900, supply=4.0 + i, demand=10.0)
               for i in range(6)]          # supply-limited only
    with pytest.raises(runs.PreflightError, match="DEMAND_LIMITED"):
        runs.preflight(records, minimum=0.0)


@pytest.mark.anyio
async def test_a_measurement_that_is_written_can_be_read_back(st):
    """25.08.: the route had been renamed, the harness took the 404, and two
    blocks returned nulls that read like findings."""
    await runs.assert_measurement_round_trip(st, "preflight-c", BASE_SLOT)


# ------------------------------------------- the effective sigmoid (§9)

def test_sigmoid_effective_is_read_off_the_response_not_the_constant():
    """Finding C. `sigmoid: null` in `config` means "whatever `runner.SIGMOID`
    held that day", and it has already meant two different bands: the
    calibration weeks cleared at steepness 2.5 under 77726be, the same config
    re-run at HEAD clears at 0.6. Same ratios, different prices.

    The manifest therefore records what the *node* applied. Read off the
    response's own echo rather than resolved from `runner.SIGMOID` here --
    answering from the harness's constant would reproduce exactly the gap
    this closes.
    """
    applied = {"k_upper": 40.0, "k_lower": 8.0, "theta": 0.9986,
               "steepness": 0.5991}
    record = _record(BASE_SLOT, supply=12.5, demand=10.0)
    record["clearing"]["sigmoid_params"] = applied
    assert runs.sigmoid_effective([record]) == applied
    # A copy, not the response's own dict.
    assert runs.sigmoid_effective([record]) is not applied

    # The first *cleared* slot answers; a no-trade slot carries no block.
    no_trade = _record(BASE_SLOT - 900, supply=0.0, demand=10.0,
                       status="no_trade")
    assert runs.sigmoid_effective([no_trade, record]) == applied

    # Nothing cleared: None, and `preflight` says so rather than leaving a
    # null to be read as "the node reported no parameters".
    assert runs.sigmoid_effective([no_trade]) is None


def test_preflight_reports_whether_the_effective_sigmoid_was_readable():
    records = [_record(BASE_SLOT, supply=12.5, demand=10.0),
               _record(BASE_SLOT + 900, supply=4.0, demand=10.0)]
    for record in records:
        record["clearing"]["sigmoid_params"] = dict(runner.SIGMOID)
    checks = runs.preflight(records, minimum=0.0)
    assert checks["sigmoid_effective_read"] is True

    for record in records:
        record["clearing"].pop("sigmoid_params")
    assert runs.preflight(records, minimum=0.0)["sigmoid_effective_read"] \
        is False


@pytest.mark.anyio
async def test_a_run_records_the_band_the_node_actually_applied(tmp_path):
    """End to end, on a short week: the default `sigmoid=None` records the
    harness's current constant, an explicit dict records that dict -- and in
    both cases the value came back out of the clearing response."""
    fitted = {"k_upper": 40.0, "k_lower": 8.0, "theta": 1.2, "steepness": 0.4}

    for name, sigmoid, expected in (("default", None, dict(runner.SIGMOID)),
                                    ("explicit", fitted, fitted)):
        spec = runs.RunSpec(run_id=f"sig-{name}", seed=0, days=1,
                            n_players=30, sigmoid=sigmoid,
                            out_dir=str(tmp_path / name))
        result = await runs.execute_run(spec, sha="0" * 40)
        manifest = result["manifest"]
        assert set(manifest["sigmoid_effective"]) == {
            "k_upper", "k_lower", "theta", "steepness"}
        assert manifest["sigmoid_effective"] == expected, name
        assert manifest["checks"]["sigmoid_effective_read"] is True
        # The declared window travels with it (D-83).
        assert manifest["battery_window"] == [73, 88]
