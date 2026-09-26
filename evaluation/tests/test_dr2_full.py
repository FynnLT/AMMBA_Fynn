"""The DR2 check of the full artifact (`analysis/dr2_full.py`).

The worked examples are the two of the task specification (25.09.), whose
numbers were computed with the artifact's own `apply_preference_allocation`
and rate functions. The run fixture is cleared through the real stack
(`runner.run_slot`, preferences and multipliers on) and written with the
harness's own `aggregates`, so R4 is held to what the artifact recorded and
not to anything this module computed. Nothing reads `evaluation/out`.
"""
import csv
import json

import numpy as np
import pytest

import aggregates
import campaign
import discovery
import dr2
import dr2_full as F
import dr_analysis
import replay as R
import runner
import stack
from test_dr_analysis import TWO_SLOTS, write_run

BAND = R.Band(k_upper=40.0, k_lower=8.0, theta=1.0, steepness=0.6)
GAMMA, ETA = 1.1, 0.10
BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


def book(offers, bids, slot=0):
    """offers: (area, energy_type, kwh, partner); bids: (area, kwh,
    partner). Partner "" for none."""
    rows = ([(a, R.SELLER, t, e, p) for a, t, e, p in offers]
            + [(a, R.BUYER, "mixed", e, p) for a, e, p in bids])
    return F.build_book(slot, *zip(*rows))


def population(the_book, **cfg):
    return F.assemble([F.SlotInput(the_book, F.PreferenceConfig(**cfg),
                                   BAND)], GAMMA, ETA)


# ------------------------------------------------ worked example A (F7.1)

EXAMPLE_A = book(
    offers=[("S1", "green", 2.0, "B1"), ("S2", "green", 3.0, ""),
            ("S3", "grey", 1.0, "")],
    bids=[("B1", 1.5, "S1"), ("B2", 2.5, "")])
#: S1 S2 S3 B1 B2, truthful and with S1 reporting 2.5 (+25 %).
TABLE_A = {
    ("preferences_first", 0.0): [1.777778, 1.666667, 0.555556, 1.5, 2.5],
    ("pro_rata_first", 0.0): [1.333333, 2.0, 0.666667, 1.5, 2.5],
    ("preferences_first", 0.25): [2.0, 1.5, 0.5, 1.5, 2.5],
    ("pro_rata_first", 0.25): [1.538462, 1.846154, 0.615385, 1.5, 2.5],
}


@pytest.mark.parametrize("order, rel", sorted(TABLE_A))
def test_worked_example_a_allocations(order, rel):
    """A matched long-side seller gains priority: E = 6, D_hat = 4, Q = 4.
    The closed form reproduces all four rows, every order of the slot."""
    pop = population(EXAMPLE_A, order=order, multipliers_enabled=False)
    assert pop["area"].tolist() == ["S1", "S2", "S3", "B1", "B2"]
    expected = TABLE_A[(order, rel)]
    members = F.member_allocations(pop, [0], [rel], +1)[0, 0]
    assert members == pytest.approx(expected, abs=1e-6)
    own = F.evaluate(pop, [0], [rel], +1)["allocation"][0]
    assert own == pytest.approx(expected[0], abs=1e-6)
    if rel == 0.0:
        every = F.evaluate(pop, np.arange(5), np.zeros(5), +1)["allocation"]
        assert every == pytest.approx(expected, abs=1e-6)
    # ... and so does the artifact itself.
    ref = F.replay_full(EXAMPLE_A, F.PreferenceConfig(order=order), BAND,
                        GAMMA, ETA, "S1", 2.0 * (1 + rel), 2.0)
    assert list(ref.allocations) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("mode, green, grey", [
    ("multiplicative", 20.322581, 18.0), ("additive", 20.4, 17.52)])
def test_worked_example_a_rates(mode, green, grey):
    """At p = 20, green multiplier 0.02, levy 0.10, cap 0.20, on the
    preferences_first allocation (G = 31/9, Y = 5/9)."""
    cfg = F.PreferenceConfig(mode=mode, green_multiplier=0.02, grey_levy=0.10,
                             levy_cap=0.20)
    g, y = 1.7777777777777777 + 1.6666666666666667, 0.5555555555555556
    rates = F.rates_vec(np.array(g), np.array(y), np.array(20.0),
                        active=True, additive=mode == "additive",
                        green_multiplier=0.02, grey_levy=0.10, levy_cap=0.20)
    assert rates[:2] == pytest.approx((green, grey), abs=1e-9)
    rate_function = (F._PREFS._additive_rates if mode == "additive"
                     else F._PREFS._multiplicative_rates)
    assert [round(v, 6) for v in rate_function(g, y, 20.0, cfg)] == \
        pytest.approx([green, grey], abs=1e-9)

    # `run_clearing` settles on the trades' `selected_energy`, which
    # `trade_builder` rounds to six decimals: G = 3.444445, Y = 0.555556.
    # The engine follows the pipeline; in the additive grey rate that is
    # one unit in the sixth decimal.
    _result, _trades, mult = F.clear_book(EXAMPLE_A.bids, EXAMPLE_A.offers,
                                          cfg, 20.0)
    assert (mult.green_alloc_kwh, mult.grey_alloc_kwh) == pytest.approx(
        (3.444445, 0.555556), abs=1e-12)
    pipeline_grey = 17.520002 if mode == "additive" else grey
    assert (mult.green_final_ct_per_kwh, mult.grey_final_ct_per_kwh) == \
        pytest.approx((green, pipeline_grey), abs=1e-9)
    rates = F.rates_vec(np.array(3.444445), np.array(0.555556),
                        np.array(20.0), active=True,
                        additive=mode == "additive", green_multiplier=0.02,
                        grey_levy=0.10, levy_cap=0.20)
    assert rates[:2] == pytest.approx((green, pipeline_grey), abs=1e-9)


# ------------------------------------------------ worked example B (F7.2)

EXAMPLE_B = book(
    offers=[("S1", "green", 1.0, "B1"), ("S2", "green", 3.0, ""),
            ("S3", "grey", 1.0, "")],
    bids=[("B1", 2.0, "S1"), ("B2", 1.0, "")])


def test_worked_example_b_preferences_can_remove_a_profitable_overreport():
    """Q = 3. Under preferences_first S1 (A = 1.0) is served fully by its
    pair; over-reporting to 1.25 is allocated 1.25 and the 0.125 kWh
    beyond the deadband costs gamma K_upper 0.125 = 5.5 ct. Under
    pro_rata_first the same report moves S1 from 0.6 to 0.714286 kWh,
    within A, and costs nothing."""
    pop = population(EXAMPLE_B, order="preferences_first",
                     multipliers_enabled=False)
    truthful = F.evaluate(pop, [0], [0.0], +1)
    over = F.evaluate(pop, [0], [0.25], +1)
    assert truthful["allocation"][0] == pytest.approx(1.0, abs=1e-12)
    assert over["allocation"][0] == pytest.approx(1.25, abs=1e-12)
    assert F.member_allocations(pop, [0], [0.25], +1)[0, 0][:3] == \
        pytest.approx([1.25, 1.3125, 0.4375], abs=1e-12)
    assert over["shortfall_ct"][0] == pytest.approx(5.5, abs=1e-9)
    assert over["externality_ct"][0] == 0.0
    ref = F.replay_full(EXAMPLE_B,
                        F.PreferenceConfig(multipliers_enabled=False), BAND,
                        GAMMA, ETA, "S1", 1.25, 1.0)
    assert ref.allocation == pytest.approx(1.25, abs=1e-12)
    assert ref.shortfall_ct == pytest.approx(5.5, abs=1e-9)
    assert ref.utility == pytest.approx(over["utility"][0], abs=1e-9)

    pop = population(EXAMPLE_B, order="pro_rata_first",
                     multipliers_enabled=False)
    assert F.evaluate(pop, [0], [0.0], +1)["allocation"][0] == \
        pytest.approx(0.6, abs=1e-12)
    over = F.evaluate(pop, [0], [0.25], +1)
    assert over["allocation"][0] == pytest.approx(0.714286, abs=1e-6)
    assert over["penalty_ct"][0] == 0.0


# ---------------------------------------------------- F1: the reconstruction

def test_the_reconstruction_drops_a_partner_that_cannot_be_matched():
    """A nomination reaches the book only if the partner posts on the
    opposite side in the same slot."""
    the_book = book(
        offers=[("S1", "green", 1.0, "B1"),   # posted, opposite side: kept
                ("S2", "grey", 1.0, "S1"),    # same side: dropped
                ("S3", "green", 1.0, "X9")],  # not in the slot: dropped
        bids=[("B1", 1.0, "S1"), ("B2", 1.0, "")])
    offers = {o["area_uuid"]: o for o in the_book.offers}
    assert offers["S1"]["requirements"] == {"preferred_partner": "B1"}
    assert "requirements" not in offers["S2"]
    assert "requirements" not in offers["S3"]
    assert offers["S2"]["attributes"] == {"energy_type": "grey"}
    bids = {b["area_uuid"]: b for b in the_book.bids}
    assert bids["B1"]["requirements"] == {"preferred_partner": "S1"}
    assert "attributes" not in bids["B1"]
    pop = population(the_book)
    assert pop["matched"].tolist() == [True, False, False, True, False]


def test_round6_is_pythons_round():
    """Including the ties the grey levy lands on: p * 0.9 of a six-decimal
    price, where numpy's rounding parts from Python's."""
    rng = np.random.default_rng(3)
    values = np.concatenate([
        rng.uniform(0, 40, 20_000),
        np.round(rng.uniform(8, 40, 20_000), 6) * 0.9,
        np.round(rng.uniform(8, 40, 20_000), 6) * 1.02])
    got = F.round6(values)
    assert all(g == round(float(v), 6) for g, v in zip(got, values))
    # The point of it: numpy alone does not agree.
    assert (np.round(values, 6) != got).any()
    assert F.round6(26.277805 * 0.9) == round(26.277805 * 0.9, 6)


# ------------------------------------------------- random books (C1, R5)

def random_slot(rng, slot, n_sellers=7, n_buyers=9):
    sellers = [f"player-{slot}{k:02d}" if k % 3 else f"battery-{slot}{k:02d}"
               for k in range(n_sellers)]
    buyers = [f"player-{slot}{50 + k:02d}" for k in range(n_buyers)]
    partner = {}
    for s, b in zip(rng.permutation(sellers)[:3], rng.permutation(buyers)[:3]):
        partner[s], partner[b] = b, s          # mutual
    partner[sellers[-1]] = buyers[-1]          # one-sided
    partner[buyers[0]] = "player-999"          # never posted
    offers = [(a, "grey" if a.startswith("battery") else "green",
               float(rng.uniform(0.05, 3.0)), partner.get(a, ""))
              for a in sellers]
    bids = [(a, float(rng.uniform(0.05, 3.0)), partner.get(a, ""))
            for a in buyers]
    return book(offers, bids, slot=slot)


def random_population(seed, n_slots=12, **cfg):
    rng = np.random.default_rng(seed)
    slots = []
    for s in range(n_slots):
        # Alternate the regime: more sellers' energy, or more buyers'.
        n_sellers, n_buyers = (9, 6) if s % 2 else (5, 10)
        slots.append(F.SlotInput(random_slot(rng, s, n_sellers, n_buyers),
                                 F.PreferenceConfig(**cfg), BAND))
    return F.assemble(slots, GAMMA, ETA)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c1_pro_rata_first_equals_the_baseline(seed):
    """Pairs are detected and flagged, and get no priority: the full
    engine is the proportional one, at every grid point, both directions."""
    pop = random_population(seed, order="pro_rata_first",
                            multipliers_enabled=False)
    assert pop["matched"].any() and not pop["pq"].any()
    idx = np.arange(pop["n"])
    for direction in (dr2.WITHHOLD, dr2.OVERREPORT):
        res = F.sweep(pop, idx, direction)
        assert res["c1_max"] <= F.C1_TOL
        assert res["c2_allocation_max"] <= F.C2_ALLOCATION_TOL
        assert res["c2_utility_max"] <= F.C2_UTILITY_TOL


@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
def test_the_closed_form_is_the_artifact_on_random_books(mode):
    """R5 on synthetic books with pairs, green and grey sellers in the same
    slots and the multipliers on."""
    pop = random_population(7, mode=mode, multipliers_enabled=True,
                            green_multiplier=0.02)
    result = F.r5(pop, points=150)
    assert result["passed"], result
    counts = result["counts"]
    assert counts["hits_pair_priority"] > 0
    assert counts["hits_rate_adjusted"] > 0
    assert counts["hits_shortfall"] > 0
    assert counts["hits_seller_externality"] > 0
    assert counts["hits_buyer_externality"] > 0


def test_preconditions_are_asserted():
    class Run:
        run_id = "x--seed-0"

        def __init__(self, prefs):
            self.config = {"preferences": prefs}

    with pytest.raises(F.PopulationError, match="sides"):
        F.preference_config(Run(campaign.prefs(sides="both")))
    with pytest.raises(F.PopulationError, match="order"):
        F.preference_config(Run({"sides": "seller"}))
    assert F.preference_config(Run(campaign.prefs())).order == \
        "preferences_first"


# ------------------------------------------------ R4 on a cleared run

#: Slot 0: worked example A (demand-limited, one pair, a grey seller).
#: Slot 1: supply-limited, the same pair, green and grey sellers. player-005
#: names an area that never posts; the reconstruction has to drop it.
FIXTURE_SLOTS = [
    {"producers": [("player-001", 2.0, "green"), ("player-002", 3.0, "green"),
                   ("battery-003", 1.0, "grey")],
     "consumers": [("player-004", 1.5), ("player-005", 2.5)]},
    {"producers": [("player-001", 1.2, "green"), ("battery-003", 0.8, "grey")],
     "consumers": [("player-004", 1.0), ("player-005", 2.0),
                   ("player-002", 1.5)]},
]
FIXTURE_NAMED = {"player-001": "player-004", "player-004": "player-001",
                 "player-005": "player-099"}


async def clear_run(root, cell, prefs, seed=0):
    """One block-1 run of FIXTURE_SLOTS, through the real pipeline, in the
    harness's own CSV shape."""
    st = stack.Stack()
    records = []
    try:
        for i, spec in enumerate(FIXTURE_SLOTS):
            slot = BASE_SLOT + runner.SLOT_SEC * i
            posted = {a for a, *_ in spec["producers"]} | {
                a for a, _ in spec["consumers"]}
            scen = {
                "producers": [{"name": a, "area_uuid": a, "energy": e,
                               "energy_type": t,
                               "preferred_partner": FIXTURE_NAMED.get(a)
                               if FIXTURE_NAMED.get(a) in posted else None}
                              for a, e, t in spec["producers"]],
                "consumers": [{"name": a, "area_uuid": a, "energy": e,
                               "preferred_partner": FIXTURE_NAMED.get(a)
                               if FIXTURE_NAMED.get(a) in posted else None}
                              for a, e in spec["consumers"]]}
            records.append(await runner.run_slot(
                st, scen, community=cell, slot=slot,
                market_id=runner.market_id_for(cell, slot),
                preferences=prefs, execute=False))
    finally:
        await st.close()
    run_id = f"{cell}--seed-{seed}"
    directory = root / run_id
    aggregates.write_slot_csv(directory / f"{run_id}_slots.csv", records)
    aggregates.write_area_csv(directory / f"{run_id}_areas.csv", records,
                              FIXTURE_NAMED)
    entry = {"run_id": run_id, "cell": cell, "seed": seed,
             "repo_sha": "0" * 40,
             "config": {"execute": False, "sigmoid": runner.SIGMOID,
                        "preferences": prefs, "cell_seeds": [seed],
                        "gamma": None, "eta_relative": None},
             "sigmoid_effective": runner.SIGMOID,
             "checks": {"round_type_census":
                        aggregates.round_type_census(records)}}
    (directory / "manifest.json").write_text(json.dumps([entry]),
                                             encoding="utf-8")
    return directory


def load_population(root):
    runs = discovery.discover([root]).runs
    return F.build_population(
        [F.extract(discovery.RunData.load(r)) for r in runs], GAMMA, ETA)


def _edit_area_csv(directory, change):
    path = next(directory.glob("*_areas.csv"))
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    change(rows)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=aggregates.AREA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
async def test_r4_reproduces_a_run_the_artifact_cleared(tmp_path, mode):
    prefs = campaign.prefs(multipliers_enabled=True, mode=mode,
                           green_multiplier=0.02)
    directory = await clear_run(tmp_path, "multipliers_on", prefs)
    pop = load_population(tmp_path)
    assert pop["n"] == 10
    # The pair is served in both slots; player-005's nominee never posts.
    assert sorted(pop["area"][pop["matched"]].tolist()) == [
        "player-001", "player-001", "player-004", "player-004"]
    assert set(pop["named_partner"][pop["area"] == "player-005"]) == {
        "player-099"}
    result = F.r4(pop)
    assert result["passed"], result
    worst = result["max_deviation_by_quantity"]
    for name in ("green_final_ct", "grey_final_ct", "buyer_final_ct",
                 "vectorised_rate_ct"):
        assert worst[name] == 0.0, name
    # The slot with both types really moved the rates off the price.
    slots = discovery.RunData.load(
        discovery.discover([tmp_path]).runs[0]).slots
    assert (slots.floats("grey_final_ct")
            != slots.floats("clearing_price_ct")).any()

    def nudge(rows):
        rows[0]["allocated_kwh"] = str(float(rows[0]["allocated_kwh"]) + 1e-3)
    _edit_area_csv(directory, nudge)
    assert not F.r4(load_population(tmp_path))["passed"]


@pytest.mark.anyio
async def test_r4_catches_a_preference_flag_it_cannot_reproduce(tmp_path):
    directory = await clear_run(tmp_path, "prefs_first_d050",
                                campaign.prefs())

    def flip(rows):
        row = next(r for r in rows if r["area_uuid"] == "player-005")
        row["preference_matched"] = "True"
    _edit_area_csv(directory, flip)
    result = F.r4(load_population(tmp_path))
    assert not result["passed"]
    assert result["n_preference_matched_mismatch"] == 1


# ------------------------------------------- dr_analysis.py, end to end

def test_penalty_parameters_come_from_the_reference_cell(tmp_path):
    write_run(tmp_path, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    data = discovery.RunData.load(discovery.discover([tmp_path]).runs[0])
    params = F.penalty_parameters([data])
    assert (params["gamma"], params["eta_relative"]) == (1.1, 0.10)
    assert "b2_noise_off gamma_eff" in params["source"]
    assert F.penalty_parameters([])["source"].startswith("constants")


@pytest.mark.anyio
async def test_the_stage_writes_its_tables_and_leaves_the_others_alone(
        tmp_path, capsys, monkeypatch):
    # Ten participant-slots; 5,000 points per draw would only repeat them.
    monkeypatch.setattr(F, "R5_POINTS", 200)
    runs = tmp_path / "runs"
    write_run(runs, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    await clear_run(runs, "multipliers_on",
                    campaign.prefs(multipliers_enabled=True,
                                   green_multiplier=0.02))
    first = tmp_path / "first"
    dr_analysis.main(["--runs", str(runs), "--dest", str(first),
                      "--full-cells", "multipliers_on"])
    written = {p.name for p in first.iterdir()}
    for table in F.TABLES:
        assert f"{table}.csv" in written
    result = json.loads((first / "checks.json").read_text(encoding="utf-8"))
    for name in ("dr2_full_population", "R4", "R5", "C1", "C2", "C3"):
        assert result["checks"][name]["passed"], (name,
                                                  result["checks"][name])
    assert result["checks"]["C1"]["skipped"]       # no control cell here
    assert result["checks"]["C3"]["cells"]["multipliers_on"][
        "identical"] is False
    findings = result["findings"]["dr2_full"]
    assert set(findings["n_additional_by_cell"]) == {"multipliers_on"}
    assert findings["penalty_parameters"]["gamma"] == 1.1
    manifest = json.loads((first / "manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["full_cells"] == ["multipliers_on"]
    assert "b2_noise_off" in manifest["dr2_full_penalty_parameters"][
        "source"]
    cases = list(csv.DictReader(open(first / "dr2_full_cases.csv",
                                     encoding="utf-8")))
    assert {r["group"] for r in cases} >= {"all", "matched", "unmatched"}

    # The same runs again: every pre-existing table byte for byte.
    capsys.readouterr()
    second = tmp_path / "second"
    dr_analysis.main(["--runs", str(runs), "--dest", str(second),
                      "--full-cells", "multipliers_on",
                      "--compare-to", str(first)])
    out = capsys.readouterr().out
    for table in ("census", "dr2_cases", "coalitions"):
        assert f"  {table}.csv: identical" in out
    assert out.count(": identical") == 14
    manifest = json.loads((second / "manifest.json").read_text(
        encoding="utf-8"))
    assert set(manifest["compare_to"]["tables"].values()) == {"identical"}


def test_compare_to_names_the_first_differing_line(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "t.csv").write_bytes(b"x,y\r\n1,2\r\n3,4\r\n")
    (b / "t.csv").write_bytes(b"x,y\r\n1,2\r\n3,5\r\n")
    (a / "u.csv").write_bytes(b"x\r\n")
    (b / "u.csv").write_bytes(b"x\r\n")
    (a / "v.csv").write_bytes(b"x\r\n")
    got = dr_analysis.compare_tables(a, b, ["t", "u", "v"])
    assert got["t"] == "line 3: '3,4' != '3,5'"
    assert got["u"] == "identical"
    assert got["v"].startswith("missing in")


def test_a_missing_cell_stops_the_stage_and_nothing_else(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_run(runs, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    dest = tmp_path / "dest"
    code = dr_analysis.main(["--runs", str(runs), "--dest", str(dest)])
    assert code == 1
    err = capsys.readouterr().err
    assert "DR2 full artifact: missing prefs_first_d050 (no run)" in err
    written = {p.name for p in dest.iterdir()}
    assert "dr2_cases.csv" in written and "census.csv" in written
    assert not any(name.startswith("dr2_full") for name in written)
    result = json.loads((dest / "checks.json").read_text(encoding="utf-8"))
    population = result["checks"]["dr2_full_population"]
    assert not population["passed"]
    assert "mult_overlap_additive (no run)" in population["reason"]


def test_missing_seeds_are_named():
    class Run:
        def __init__(self, cell, seed):
            self.cell, self.seed = cell, seed
            self.config = {"cell_seeds": [0, 1, 2]}

    runs = [Run("a", 0), Run("a", 2), Run("b", 0)]
    assert F.missing_runs(runs, ("a", "c")) == ["a seed 1", "c (no run)"]
