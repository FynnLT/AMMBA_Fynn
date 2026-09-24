"""The DR2-DR5 analysis (`analysis/dr_analysis.py`) on synthetic runs.

Every fixture here is a handful of slots written in the harness's own CSV
shape (`aggregates.FIELDS` / `AREA_FIELDS`), priced and penalised with the
artifact's functions. Nothing reads `evaluation/out`.
"""
import csv
import hashlib
import json

import numpy as np
import pytest

import aggregates
import checks
import discovery
import dr2
import dr3
import dr4
import dr5
import dr_analysis
import replay as R

CALIBRATED = {"k_upper": 40.0, "k_lower": 8.0, "theta": 1.0, "steepness": 0.6}
#: The illustrative band of Section 4.6.1, which the worked examples use.
ILLUSTRATIVE = R.Band(k_upper=40.0, k_lower=8.0, theta=1.0, steepness=2.5)
GAMMA, ETA = 1.1, 0.10
BASE_SLOT = 1_757_000_000 // 900 * 900


# ------------------------------------------------- the worked examples (A1)

def state(supply, demand, band=ILLUSTRATIVE):
    return R.SlotState(supply, demand, band, GAMMA, ETA)


def test_worked_example_1_price_and_margin_elasticity():
    """E = 12.5, D_hat = 10: p = 19.157, epsilon = 2.036 (Section 4.3.3)."""
    assert ILLUSTRATIVE.price(1.25) == pytest.approx(19.157, abs=1e-3)
    assert R.margin_elasticity(1.25, ILLUSTRATIVE, R.SELLER) == pytest.approx(
        2.036, abs=1e-3)


def test_worked_example_2_withholding_pays_on_the_long_side():
    """Seller A = 5: withholding 0.75 kWh moves p 19.157 -> 20.555 and the
    payoff up by 0.784 ct, with no penalty (demand-limited)."""
    st = state(12.5, 10.0)
    truthful = R.replay(st, R.SELLER, 5.0, 5.0, 5.0)
    withheld = R.replay(st, R.SELLER, 5.0, 4.25, 5.0)
    assert truthful.price == pytest.approx(19.157, abs=1e-3)
    assert withheld.price == pytest.approx(20.555, abs=1e-3)
    assert withheld.round_type == R.DEMAND_LIMITED
    assert withheld.penalty_ct == 0.0
    assert withheld.utility - truthful.utility == pytest.approx(0.784, abs=1e-3)


def test_worked_example_3_overreporting_pays_on_the_long_side():
    """Seller A = 1.25: a report of 1.61 kWh lifts the allocation 1.000 ->
    1.252 at p = 18.512, no penalty, payoff +17.7 .. 18.1 % (17.96 % on the
    full allocation; the text's 17.8 % counts 1.25 kWh)."""
    st = state(12.5, 10.0)
    truthful = R.replay(st, R.SELLER, 1.25, 1.25, 1.25)
    over = R.replay(st, R.SELLER, 1.25, 1.61, 1.25)
    assert truthful.allocation == pytest.approx(1.000, abs=1e-3)
    assert over.allocation == pytest.approx(1.252, abs=1e-3)
    assert over.price == pytest.approx(18.512, abs=1e-3)
    assert over.penalty_ct == 0.0
    gain = over.utility / truthful.utility - 1.0
    assert 0.177 <= gain <= 0.181
    assert gain == pytest.approx(0.1796, abs=1e-3)
    text = (over.price - 8.0) * 1.25 / truthful.utility - 1.0
    assert text == pytest.approx(0.178, abs=1e-3)


def test_worked_example_4_overreporting_on_the_short_side():
    """E = 8, D_hat = 10, A = 2: eps * sigma = 0.189; +0.22 kWh gains 3.458
    ct gross with no penalty; the net gain crosses zero between 0.3636 kWh
    (+0.011 ct) and 0.37 kWh (-0.148 ct)."""
    st = state(8.0, 10.0)
    eps = R.margin_elasticity(0.8, ILLUSTRATIVE, R.SELLER)
    assert eps * 2.0 / 8.0 == pytest.approx(0.189, abs=1e-3)
    truthful = R.replay(st, R.SELLER, 2.0, 2.0, 2.0)
    over = R.replay(st, R.SELLER, 2.0, 2.22, 2.0)
    assert over.gross_utility - truthful.gross_utility == pytest.approx(
        3.458, abs=1e-3)
    assert over.penalty_ct == 0.0
    before = R.replay(st, R.SELLER, 2.0, 2.3636, 2.0)
    after = R.replay(st, R.SELLER, 2.0, 2.37, 2.0)
    assert before.utility - truthful.utility == pytest.approx(0.011, abs=1e-3)
    assert after.utility - truthful.utility == pytest.approx(-0.148, abs=1e-3)


@pytest.mark.parametrize("band", [ILLUSTRATIVE, R.Band.from_dict(CALIBRATED),
                                  R.Band(40.0, 8.0, 1.2, 1.0)])
def test_price_derivative_matches_a_central_difference(band):
    h = 1e-6
    for x in (0.05, 0.4, 0.8, 1.0, 1.25, 2.0, 6.0):
        numeric = (band.price(x + h) - band.price(x - h)) / (2 * h)
        assert R.price_derivative(x, band) == pytest.approx(numeric,
                                                            rel=1e-6, abs=1e-9)


def test_vectorised_engine_matches_the_artifact_functions():
    """R3 in miniature: both externality paths and the shortfall are hit."""
    rng = np.random.default_rng(1)
    n = 400
    is_seller = rng.random(n) < 0.5
    supply = rng.uniform(2, 20, n)
    demand = rng.uniform(2, 20, n)
    own = np.where(is_seller, supply, demand) * rng.uniform(0.01, 0.4, n)
    new = own * (1 + rng.uniform(-0.5, 0.5, n))
    truth = own * (1 + rng.uniform(-0.3, 0.3, n))
    vec = R.replay_vec(is_seller=is_seller, supply=supply, demand=demand,
                       own_report=own, new_report=new, truth=truth,
                       k_upper=40.0, k_lower=8.0, theta=1.0, steepness=0.6,
                       gamma=GAMMA, eta_relative=ETA)
    band = R.Band.from_dict(CALIBRATED)
    hits = {"shortfall": 0, "externality": 0}
    for i in range(n):
        ref = R.replay(R.SlotState(supply[i], demand[i], band, GAMMA, ETA),
                       R.SELLER if is_seller[i] else R.BUYER,
                       own[i], new[i], truth[i])
        assert vec["price"][i] == pytest.approx(ref.price, abs=1e-9)
        assert vec["shortfall_ct"][i] == pytest.approx(ref.shortfall_ct,
                                                       abs=1e-9)
        assert vec["externality_ct"][i] == pytest.approx(ref.externality_ct,
                                                         abs=1e-9)
        assert vec["utility"][i] == pytest.approx(ref.utility, abs=1e-9)
        hits["shortfall"] += ref.shortfall_ct > 0
        hits["externality"] += ref.externality_ct > 0
    assert hits["shortfall"] and hits["externality"]


# ------------------------------------------------------ synthetic runs

def _execute_slot(sellers, buyers, supply, demand, price, sigmoid):
    """The execution node's arithmetic for one slot, row by row
    (`execution.run_execution` without the database)."""
    traded_q = min(supply, demand)
    kind = R.determine_round_type(supply, demand)
    results = []
    for area, row in sellers.items():
        traded = row["allocated_kwh"]
        eta = ETA * traded
        shortfall = R.seller_shortfall_penalty(
            traded, row["actual_kwh"], sigmoid["k_upper"], GAMMA, eta)
        result = {"area_uuid": area, "role": "seller", "traded_kwh": traded,
                  "shortfall_penalty_ct": shortfall["penalty_ct"],
                  "externality_penalty_ct": 0.0}
        if kind == R.SUPPLY_LIMITED:
            ext = R.seller_externality_penalty(
                traded, row["deliverable_kwh"], eta, supply, demand,
                traded_q, price, sigmoid)
            if ext:
                result["externality_penalty_ct"] = ext["penalty_ct"]
                result["counterfactual_price_ct_per_kwh"] = \
                    ext["counterfactual_price_ct_per_kwh"]
        results.append(result)
    for area, row in buyers.items():
        result = {"area_uuid": area, "role": "buyer",
                  "traded_kwh": row["allocated_kwh"],
                  "shortfall_penalty_ct": 0.0, "externality_penalty_ct": 0.0}
        if kind == R.DEMAND_LIMITED:
            ext = R.buyer_externality_penalty(
                row["requested_kwh"], row["actual_kwh"], supply, demand,
                traded_q, price, sigmoid)
            if ext:
                result["externality_penalty_ct"] = ext["penalty_ct"]
                result["counterfactual_price_ct_per_kwh"] = \
                    ext["counterfactual_price_ct_per_kwh"]
        results.append(result)
    return kind, results, R._PENALTIES.redistribution(results, price, traded_q)


def write_run(root, cell, seed, slots, *, execute=False, deviators=(),
              arm="none", entries=1, sigmoid=CALIBRATED, config_params=True):
    """One run directory in the harness's shape.

    `slots` is a list of `{"sellers": {area: kwh}, "buyers": {area: kwh}}`,
    optionally with `"actual"` / `"deliverable"` maps per area.
    """
    run_id = f"{cell}--seed-{seed}"
    directory = root / run_id
    directory.mkdir(parents=True)
    band = R.Band.from_dict(sigmoid)
    slot_rows, area_rows = [], []
    census = {R.SUPPLY_LIMITED: 0, R.DEMAND_LIMITED: 0, R.BALANCED: 0,
              "no_trade": 0}
    for i, spec in enumerate(slots):
        t = BASE_SLOT + 900 * i
        supply = sum(spec["sellers"].values())
        demand = sum(spec["buyers"].values())
        price = round(band.price(supply / demand), 6)
        traded = min(supply, demand)
        census[R.determine_round_type(supply, demand)] += 1
        actual = spec.get("actual", {})
        deliverable = spec.get("deliverable", {})
        rows = {"seller": {}, "buyer": {}}
        for side, key, total in (("seller", "sellers", supply),
                                 ("buyer", "buyers", demand)):
            for area, kwh in spec[key].items():
                alloc = round(kwh / total * traded, 6)
                row = {"slot": t, "area_uuid": area, "side": side,
                       "energy_type": "green" if side == "seller" else "mixed",
                       "requested_kwh": kwh, "allocated_kwh": alloc,
                       "fill_rate": round(alloc / kwh, 6),
                       "preference_matched": False, "named": False,
                       "partner": None}
                if execute:
                    row["actual_kwh"] = actual.get(area, alloc)
                    row["deliverable_kwh"] = deliverable.get(
                        area, row["actual_kwh"])
                    row["deliverable_source"] = "forecast"
                rows[side][area] = row
        slot = {"slot": t, "total_supply_kwh": supply,
                "total_demand_kwh": demand,
                "ratio": round(supply / demand, 6), "clearing_price_ct": price,
                "traded_kwh": traded, "n_producers": len(rows["seller"]),
                "n_consumers": len(rows["buyer"]), "pref_enabled": False,
                "pref_order": "preferences_first", "n_pairs": 0,
                "pair_kwh": 0.0, "mult_enabled": False,
                "mult_mode": "multiplicative", "mult_sides": "seller",
                "green_final_ct": price, "grey_final_ct": price,
                "buyer_final_ct": price}
        if execute:
            kind, results, red = _execute_slot(
                rows["seller"], rows["buyer"], supply, demand, price, sigmoid)
            compensation = {r["area_uuid"]: r["compensation_ct"]
                            for r in red["rows"]}
            by_area = {r["area_uuid"]: r for r in results}
            for side_rows in rows.values():
                for area, row in side_rows.items():
                    result = by_area[area]
                    row["shortfall_penalty_ct"] = result["shortfall_penalty_ct"]
                    row["externality_penalty_ct"] = \
                        result["externality_penalty_ct"]
                    row["total_penalty_ct"] = round(
                        result["shortfall_penalty_ct"]
                        + result["externality_penalty_ct"], 6)
                    row["compensation_ct"] = compensation.get(area)
            slot.update({
                "round_type_exec": kind,
                "penalty_pool_ct": red["penalty_pool_ct"],
                "compensated_ct": red["compensated_ct"],
                "budget_balance_ct": red["budget_balance_ct"],
                "n_deviators": sum(1 for r in results
                                   if r["externality_penalty_ct"] > 0),
                "shortfall_penalty_ct": round(sum(
                    r["shortfall_penalty_ct"] for r in results), 6),
                "gamma_eff": GAMMA, "eta_relative_eff": ETA,
                "eta_mode": "relative",
                "k_sho_ct_per_kwh": round(GAMMA * sigmoid["k_upper"], 6)})
        slot_rows.append(slot)
        area_rows += list(rows["seller"].values()) + list(rows["buyer"].values())

    for name, fields, rows in ((f"{run_id}_slots.csv", aggregates.FIELDS,
                                slot_rows),
                               (f"{run_id}_areas.csv", aggregates.AREA_FIELDS,
                                area_rows)):
        with open(directory / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    entry = {"run_id": run_id, "cell": cell, "seed": seed,
             "repo_sha": "0" * 40,
             "config": {"execute": execute, "sigmoid": sigmoid,
                        "gamma": GAMMA if config_params else None,
                        "eta_relative": ETA if config_params else None},
             "sigmoid_effective": sigmoid,
             "deviation": {"arm": arm, "deviators": list(deviators),
                           "share": 0.25, "k": len(deviators),
                           "slots_with_absent_deviator": 0},
             "checks": {"round_type_census": census}}
    (directory / "manifest.json").write_text(
        json.dumps([entry] * entries), encoding="utf-8")
    return directory


#: Hand-computed pro rata. Slot 0: supply-limited (E = 5 < D = 8), sellers
#: filled, buyers 4 -> 2.5 each. Slot 1: demand-limited (E = 10 > D = 5),
#: buyers filled, sellers 6 -> 3.0 and 4 -> 2.0.
TWO_SLOTS = [
    {"sellers": {"player-001": 2.0, "player-002": 3.0},
     "buyers": {"player-003": 4.0, "player-004": 4.0}},
    {"sellers": {"player-001": 6.0, "battery-002": 4.0},
     "buyers": {"player-003": 3.0, "player-004": 2.0}},
]
HAND = {(0, "player-001"): 2.0, (0, "player-002"): 3.0,
        (0, "player-003"): 2.5, (0, "player-004"): 2.5,
        (1, "player-001"): 3.0, (1, "battery-002"): 2.0,
        (1, "player-003"): 3.0, (1, "player-004"): 2.0}


def test_r1_reproduces_a_hand_computed_pro_rata_allocation(tmp_path):
    write_run(tmp_path, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    found = discovery.discover([tmp_path])
    data = discovery.RunData.load(found.runs[0])
    slot_of = {BASE_SLOT: 0, BASE_SLOT + 900: 1}
    for slot, area, alloc in zip(data.areas.ints("slot"),
                                 data.areas.strings("area_uuid"),
                                 data.areas.floats("allocated_kwh")):
        assert alloc == HAND[(slot_of[int(slot)], area)]
    pop = dr2.build_population([data])
    result = checks.r1(pop)
    assert result["passed"], result
    assert result["n_rows"] == 8 and result["n_slots"] == 2

    pop["allocated"] = pop["allocated"].copy()
    pop["allocated"][0] += 1e-3
    assert not checks.r1(pop)["passed"]


def test_one_deviator_is_covered_exactly(tmp_path):
    """C_pen = 1: with one deviator the node's counterfactual is the joint
    one, so the pool is exactly the damage it caused."""
    slot = {"sellers": {"player-001": 2.0, "player-002": 3.0},
            "buyers": {"player-003": 4.0, "player-004": 4.0},
            "deliverable": {"player-001": 2.5}}
    write_run(tmp_path, "b2_sell_s25", 0, [slot], execute=True,
              deviators=("player-001",), arm="sellers_withhold")
    data = discovery.RunData.load(discovery.discover([tmp_path]).runs[0])
    assert data.slots.floats("penalty_pool_ct")[0] > 0
    acc = dr4.DR4()
    acc.add(data)
    check = acc.single_deviator_check()
    assert check["passed"], check
    assert check["n_slots"] == 1
    row, = [r for r in acc.coverage_table() if r["n_deviators"] == "1"]
    assert row["deviator_class"] == "all_named"
    assert row["c_pen_median"] == pytest.approx(1.0, abs=1e-6)
    # Without the deadband the damage is larger than what was penalised.
    assert row["c_tot_median"] < 1.0


def test_jain_index_bounds():
    assert dr5.jain([0.3, 0.3, 0.3, 0.3]) == pytest.approx(1.0)
    assert dr5.jain([1.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(1 / 5)
    assert dr5.jain([2.0, 0.0]) == pytest.approx(0.5)


def test_discovery_refuses_a_duplicate_cell_and_seed(tmp_path):
    write_run(tmp_path / "a", "b2_noise_off", 0, TWO_SLOTS)
    write_run(tmp_path / "b", "b2_noise_off", 0, TWO_SLOTS)
    with pytest.raises(discovery.DiscoveryError) as err:
        discovery.discover([tmp_path])
    assert str(tmp_path / "a") in str(err.value)
    assert str(tmp_path / "b") in str(err.value)


def test_discovery_skips_what_is_not_a_run_and_reads_the_last_entry(tmp_path):
    write_run(tmp_path, "b2_noise_off", 0, TWO_SLOTS, entries=2)
    write_run(tmp_path, "calibration", 0, TWO_SLOTS)
    (tmp_path / "ncurve").mkdir()
    (tmp_path / "ncurve" / "manifest.json").write_text("{}", encoding="utf-8")
    found = discovery.discover([tmp_path])
    assert [r.cell for r in found.runs] == ["b2_noise_off"]
    assert found.runs[0].n_entries == 2
    assert found.multi_entry == [str(tmp_path / "b2_noise_off--seed-0")]
    reasons = dict(found.skipped)
    assert reasons[str(tmp_path / "calibration--seed-0")] == "calibration run"
    assert reasons[str(tmp_path / "ncurve")] == "no area CSV"


def test_refuses_a_dest_that_already_holds_a_manifest(tmp_path, capsys):
    write_run(tmp_path / "runs", "b2_noise_off", 0, TWO_SLOTS, execute=True)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "manifest.json").write_text("{}", encoding="utf-8")
    code = dr_analysis.main(["--runs", str(tmp_path / "runs"),
                             "--dest", str(dest)])
    assert code == 2
    assert "refusing to start" in capsys.readouterr().err
    assert sorted(p.name for p in dest.iterdir()) == ["manifest.json"]


PARAM_COLUMNS = ("gamma_eff", "eta_relative_eff", "eta_mode",
                 "k_sho_ct_per_kwh")


def _drop_param_columns(directory):
    """A slot CSV as the harness wrote it before 91790e9 (18.09.)."""
    path = next(directory.glob("*_slots.csv"))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fields = [f for f in aggregates.FIELDS if f not in PARAM_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_old_slot_csvs_take_the_penalty_parameters_from_the_manifest(
        tmp_path):
    """The block-2 cells ran at 3422529, before `gamma_eff` existed; their
    manifests carry gamma and eta explicitly, and R2 has to pass on them."""
    runs = tmp_path / "runs"
    _drop_param_columns(write_run(runs, "b2_noise_off", 0, TWO_SLOTS,
                                  execute=True))
    withhold = [dict(TWO_SLOTS[0], deliverable={"player-001": 2.5}),
                TWO_SLOTS[1]]
    _drop_param_columns(write_run(runs, "b2_sell_s25", 0, withhold,
                                  execute=True, deviators=("player-001",),
                                  arm="sellers_withhold"))
    data = discovery.RunData.load(discovery.discover([runs]).runs[1])
    assert data.param_source == "manifest_config"
    assert set(data.slots.floats("gamma_eff")) == {GAMMA}
    assert set(data.slots.floats("k_sho_ct_per_kwh")) == {
        round(GAMMA * CALIBRATED["k_upper"], 6)}

    dest = tmp_path / "dest"
    dr_analysis.main(["--runs", str(runs), "--dest", str(dest)])
    result = json.loads((dest / "checks.json").read_text(encoding="utf-8"))
    r2 = result["checks"]["R2"]
    assert r2["passed"] and r2["n_rows"] > 0
    assert r2["by_param_source"]["manifest_config"]["passed"]
    assert r2["by_param_source"]["manifest_config"]["n_rows"] == r2["n_rows"]
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert {r["param_source"] for r in manifest["runs"]} == {"manifest_config"}


def test_without_either_parameter_source_the_analysis_stops(tmp_path,
                                                            capsys):
    """`config.gamma = None` means "whatever the node held that day", which
    is not a record of what it applied."""
    directory = write_run(tmp_path / "runs", "b2_noise_off", 0, TWO_SLOTS,
                          execute=True, config_params=False)
    _drop_param_columns(directory)
    dest = tmp_path / "dest"
    code = dr_analysis.main(["--runs", str(tmp_path / "runs"),
                             "--dest", str(dest)])
    assert code == 2
    assert "b2_noise_off--seed-0_slots.csv:gamma_eff" in capsys.readouterr().err
    assert not dest.exists()


def test_archive_directories_are_not_read(tmp_path):
    """`out/_archive/...` holds earlier runs of the same cells; reading it
    would be a duplicate (cell, seed)."""
    write_run(tmp_path, "b2_noise_off", 0, TWO_SLOTS)
    write_run(tmp_path / "_archive", "b2_noise_off", 0, TWO_SLOTS)
    found = discovery.discover([tmp_path])
    assert [str(r.path) for r in found.runs] == [
        str(tmp_path / "b2_noise_off--seed-0")]
    assert (str(tmp_path / "_archive"),
            "archive directory (leading _)") in found.skipped


def test_short_side_externality_activates_below_the_deadband_share(tmp_path):
    """delta_active / A = eta / (1 + eta), not eta: the deadband is relative
    to the *sold* quantity."""
    write_run(tmp_path, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    pop = dr2.build_population(
        [discovery.RunData.load(discovery.discover([tmp_path]).runs[0])])
    idx = np.where(pop["is_seller"] & (pop["regime"] == R.SL))[0]
    res = dr2.sweep(pop, idx, dr2.WITHHOLD)
    thr = dr2.thresholds(pop, idx, dr2.WITHHOLD, res)
    assert thr["delta_active"] == pytest.approx(ETA / (1 + ETA), abs=1e-9)
    # Withholding on the short side never pays here, and beyond activation
    # nothing beats net(delta_active) (Proposition 2).
    assert (thr["net_max"] <= dr2.NET_TOL).all()
    assert not dr2.proposition2_violations(res, thr).any()


def test_ir_causes_are_single_removals():
    rec = {"u": np.array([1.0, -2.0, -2.0, -2.0, -5.0]),
           "levy_ct": np.array([0.0, 3.0, 0.0, 0.0, 1.0]),
           "shortfall": np.array([0.0, 0.0, 2.5, 0.0, 1.0]),
           "externality": np.array([0.0, 0.0, 2.5, 1.0, 1.0]),
           "unneeded_ct": np.zeros(5)}
    masks = dr3.causes(rec)
    assert masks["any"].tolist() == [False, True, True, True, True]
    assert masks["levy"].tolist() == [False, True, False, False, False]
    # Either removal alone restores row 2, so it counts under both.
    assert masks["shortfall"].tolist() == [False, False, True, False, False]
    assert masks["externality"].tolist() == [False, False, True, False, False]
    # No single removal restores rows 3 and 4.
    assert masks["combined"].tolist() == [False, False, False, True, True]


def _digest(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_full_run_writes_every_table_and_touches_no_run_data(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    withhold = [dict(TWO_SLOTS[0], deliverable={"player-001": 2.5}),
                TWO_SLOTS[1]]
    write_run(runs, "b2_sell_s25", 0, withhold, execute=True,
              deviators=("player-001",), arm="sellers_withhold")
    write_run(runs, "baseline_pro_rata", 0, TWO_SLOTS)
    before = _digest(runs)
    dest = tmp_path / "dest"
    code = dr_analysis.main(["--runs", str(runs), "--dest", str(dest)])
    assert _digest(runs) == before

    result = json.loads((dest / "checks.json").read_text(encoding="utf-8"))
    for name in ("R1", "R2", "R3", "dr4_single_deviator_coverage",
                 "dr2_band_calibrated_reproduces_main",
                 "census_agrees_with_manifests"):
        assert result["checks"][name]["passed"], (name, result["checks"][name])
    # The coalition cells are absent from this fixture, so A6 cannot pass
    # and the exit code says so.
    assert code == 1
    assert not result["checks"]["a6_buyers_underreport"]["passed"]

    written = {p.name for p in dest.iterdir()}
    for table in ("census", "dr2_cases", "dr2_curves", "dr2_conditions",
                  "dr2_band_sensitivity", "dr2_named", "dr3_ir",
                  "dr3_settlement", "dr3_eq48", "dr4_layers", "dr4_coverage",
                  "dr5_fairness", "dr5_preferences", "coalitions"):
        assert f"{table}.csv" in written
    assert {"checks.json", "columns.md", "manifest.json"} <= written
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert {r["cell"] for r in manifest["runs"]} == {
        "b2_noise_off", "b2_sell_s25", "baseline_pro_rata"}
    assert all(r["sigmoid_source"] == "sigmoid_effective"
               for r in manifest["runs"])

    cases = list(csv.DictReader(open(dest / "dr2_cases.csv",
                                     encoding="utf-8")))
    order = [dr2.case_name(*case) for case in dr2.CASES]
    for row in cases:
        assert order[int(row["table_4_2_row"]) - 1] == row["case"]

    named = list(csv.DictReader(open(dest / "dr2_named.csv",
                                     encoding="utf-8")))
    assert [r["deviator"] for r in named] == ["player-001"]
    assert int(named[0]["n_replayed"]) == 1
    assert int(named[0]["n_not_replayed"]) == 1
    assert float(named[0]["forgone_margin_ct"]) > 0
