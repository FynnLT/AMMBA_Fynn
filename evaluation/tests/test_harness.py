"""The four harness tests. Each one corresponds to a defect that happened.

They are deliberately end-to-end rather than unit-sized: every one of these
failures got past unit tests and was caught, late, by a number that looked
plausible. The other test modules cover the same components more finely; this
module is the list of things that must never silently pass again.
"""
import json

import pytest

import aggregates
import profiles
import runner
import runs
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    return stack.Stack()


@pytest.fixture(scope="module")
def week():
    community = profiles.load_community()
    players = profiles.select_players(community.flags, n=100, seed=20260916)
    batteries = profiles.make_battery_areas(community.bess, players, seed=7)
    day = community.day_kwh(players)
    return profiles.extend_to_week(day, seed=4242, days=1), players, batteries


# ---------------------------------------------------------------------- 1

def test_the_generators_honour_one_order_per_area_and_slot(week):
    """D-58. The clearing node raises if an area holds *both* sides, but
    nothing stops an area posting two offers, and pro-rata would then weight
    it twice. Both generators are checked, on every slot of a day."""
    profile_week, _players, batteries = week

    for index in range(profile_week.n_slots):
        scen = scenario.make_scenario_from_profiles(profile_week, index,
                                                    batteries)
        areas = [o["area_uuid"] for o in scen["producers"] + scen["consumers"]]
        assert len(areas) == len(set(areas)), f"slot {index}"

    for seed in range(5):
        scen = scenario.make_scenario(seed=seed, n_prod=40, n_cons=60,
                                      sd_ratio=1.25, pair_density=0.5)
        areas = [o["area_uuid"] for o in scen["producers"] + scen["consumers"]]
        assert len(areas) == len(set(areas)), f"seed {seed}"


@pytest.mark.anyio
async def test_an_area_on_both_sides_is_refused_by_the_artifact(st):
    """The other half of the same assumption, checked where it is enforced:
    an area posting both sides would have one meter reading read twice and
    would pass the mutuality check against itself."""
    both_sides = {
        "producers": [{"area_uuid": "area-x", "name": "X", "energy": 2.0,
                       "energy_type": "green"}],
        "consumers": [{"area_uuid": "area-x", "name": "X", "energy": 1.0}],
    }
    with pytest.raises(Exception, match="only one side"):
        await runner.run_slot(st, both_sides, community="both-sides",
                              slot=BASE_SLOT, execute=False)


# ---------------------------------------------------------------------- 2

def test_the_manifest_records_config_seed_and_sha_for_every_run(tmp_path):
    """A figure that cannot be traced back to `config + seed + SHA` is not
    evidence. Every run writes one entry, and every entry carries all three."""
    path = tmp_path / "manifest.json"
    specs = [runs.RunSpec(run_id=f"cell-a-seed-{seed}", seed=seed,
                          cell_seeds=(0, 1, 2, 3, 4), cell="cell-a")
             for seed in range(5)]
    for spec in specs:
        runs.write_manifest(
            path, spec.run_id, config=spec.config(), seed=spec.seed,
            dataset_extension_seed=spec.dataset_extension_seed,
            player_ids=[1, 2, 3], cell_seeds=spec.cell_seeds,
            output_files=[f"{spec.run_id}_slots.csv"], sha="0f0f0f0")

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 5
    assert [e["seed"] for e in entries] == [0, 1, 2, 3, 4]
    for entry in entries:
        assert entry["config"]
        assert entry["seed"] is not None
        assert entry["repo_sha"] == "0f0f0f0"
        # The whole cell, named seed by seed, so one deviating run can be
        # repeated on its own without guessing what the cell contained.
        assert entry["cell_seeds"] == [0, 1, 2, 3, 4]
        assert entry["config"]["cell"] == "cell-a"


def test_the_real_sha_is_the_repository_head():
    """`sha=` is only for tests; a campaign reads the working tree's HEAD."""
    sha = runs.repo_sha()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------------- 3

@pytest.mark.anyio
async def test_run_sequence_produces_exactly_the_requested_slots(st, week):
    """Over the real profiles: the requested number of slots, in order,
    strictly increasing, one market each. Without a time axis the calibration
    has no distribution to fit."""
    profile_week, players, batteries = week
    requested = 24
    window = range(48, 48 + requested)
    slots = [BASE_SLOT + i * runner.SLOT_SEC for i in window]
    scenarios = [scenario.make_scenario_from_profiles(profile_week, i,
                                                      batteries)
                 for i in window]

    records = await runner.run_sequence(
        st, "harness-week", slots,
        lambda index, slot: scenarios[index],
        areas=runs.run_areas(players, batteries), execute=False)

    assert len(records) == requested
    times = [r["time_slot"] for r in records]
    assert times == slots
    assert all(b > a for a, b in zip(times, times[1:]))
    assert len(set(r["market_id"] for r in records)) == requested

    rows = aggregates.slot_rows(records)
    assert len(rows) == requested
    assert [r["slot"] for r in rows] == slots


# ---------------------------------------------------------------------- 4

@pytest.mark.anyio
async def test_the_pre_campaign_smoke_test_asserts_a_measurement_was_stored(st):
    """25.08.: `POST /asset_measurements` had been renamed in the same commit.
    The harness took the 404, two blocks returned null results, and the nulls
    looked like findings.

    This is not a status-code check. The measurement is posted, read back off
    the channel the Execution Node actually queries, and then confirmed to
    have been *used*: `measurement_found` is what tells a real meter reading
    apart from the node's own fallback to the traded quantity.
    """
    community = "measurement-community"
    await runs.assert_measurement_round_trip(st, community, BASE_SLOT)

    # And through the pipeline: a delivered quantity below what was traded has
    # to produce a shortfall. If the write vanished, the node falls back to
    # "delivered exactly what was traded" and the penalty is silently zero.
    scen = scenario.make_scenario(seed=2, n_prod=4, n_cons=6, sd_ratio=1.0)
    slot = BASE_SLOT + runner.SLOT_SEC

    priced = await runner.run_slot(st, scen, community=community, slot=slot,
                                   execute=False)
    allocated = {p["area_uuid"]: p["allocated_kwh"]
                 for p in priced["clearing"]["allocations"]["producers"]}
    assert allocated, "the reference slot must allocate something to measure"

    slot2 = slot + runner.SLOT_SEC
    result = await runner.run_slot(
        st, scen, community=community, slot=slot2,
        market_id=runner.market_id_for(community, slot2),
        measurements={area: kwh * 0.5 for area, kwh in allocated.items()},
        execute=True)

    execution = result["execution"]
    assert execution is not None, "the slot must have executed"
    sellers = [r for r in execution["results"] if r["role"] == "seller"]
    assert sellers
    assert all(r["measurement_found"] for r in sellers), (
        "the execution node did not find the measurements the harness wrote; "
        "it fell back to the traded quantity and every penalty is a null "
        "result that looks like a finding")
    assert any(r["shortfall_kwh"] > 0 for r in sellers)

    # Explicitly: absence must fail, not pass quietly.
    stored = await st.get("/measurements", community_uuid=community,
                          start_time=slot2,
                          end_time=slot2 + runner.SLOT_SEC - 1)
    assert {row["area_uuid"] for row in stored} == set(allocated)


@pytest.mark.anyio
async def test_a_missing_measurement_channel_is_detected_not_tolerated(st):
    """The guard has to fail when the write does not land. Simulated by
    asking for a slot nothing was written to."""
    with pytest.raises(runs.PreflightError, match="not readable back"):
        await runs.assert_measurement_round_trip(
            _SilentlyDroppingStack(st), "dropped-community", BASE_SLOT)


class _SilentlyDroppingStack:
    """A stack whose measurement writes are accepted and then lost -- the
    25.08. failure mode, minus the 404."""

    def __init__(self, inner):
        self._inner = inner

    async def post(self, path, json):
        if path == "/measurements":
            return []
        return await self._inner.post(path, json)

    async def get(self, path, **params):
        return await self._inner.get(path, **params)
