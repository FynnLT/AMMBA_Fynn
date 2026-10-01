"""The combined IR cells' analysis stage (`analysis/ir_combined.py`).

Synthetic runs only, in the harness's own CSV shape: `write_run` from
`test_dr_analysis.py` for whole runs, `raw_run` below where a worked example
needs exact rates and penalties. Nothing reads `evaluation/out`.
"""
import csv
import functools
import json

import numpy as np
import pytest

import aggregates
import campaign
import discovery
import dr3
import dr_analysis
import ir_combined
import replay as R
import test_dr_analysis as tda
from test_dr_analysis import BASE_SLOT, CALIBRATED, TWO_SLOTS, write_run

FAKE_SHA = "0" * 40


# ------------------------------------------------------------- fixtures

def raw_run(root, cell, slots, areas, *, seed=0):
    """One executed run written row by row; every column not given is
    empty, as the harness writes a column a slot does not carry."""
    run_id = f"{cell}--seed-{seed}"
    directory = root / run_id
    directory.mkdir(parents=True)
    for name, fields, rows in ((f"{run_id}_slots.csv", aggregates.FIELDS,
                                slots),
                               (f"{run_id}_areas.csv", aggregates.AREA_FIELDS,
                                areas)):
        with open(directory / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
    entry = {"run_id": run_id, "cell": cell, "seed": seed,
             "repo_sha": FAKE_SHA,
             "config": {"execute": True, "sigmoid": CALIBRATED,
                        "gamma": 1.1, "eta_relative": 0.1},
             "sigmoid_effective": CALIBRATED,
             "deviation": {"arm": "none", "sigma": 0.05, "deviators": []}}
    (directory / "manifest.json").write_text(json.dumps([entry]),
                                             encoding="utf-8")
    run, = [r for r in discovery.discover([root]).runs if r.run_id == run_id]
    return discovery.RunData.load(run)


def slot_row(t, price, *, grey=None, green=None):
    return {"slot": t, "total_supply_kwh": 1.0, "total_demand_kwh": 2.0,
            "clearing_price_ct": price, "traded_kwh": 1.0,
            "green_final_ct": price if green is None else green,
            "grey_final_ct": price if grey is None else grey,
            "buyer_final_ct": price, "round_type_exec": "SUPPLY_LIMITED",
            "penalty_pool_ct": 0.0, "compensated_ct": 0.0,
            "budget_balance_ct": 0.0, "n_deviators": 0,
            "shortfall_penalty_ct": 0.0, "gamma_eff": 1.1,
            "eta_relative_eff": 0.1, "eta_mode": "relative",
            "k_sho_ct_per_kwh": 44.0}


def area_row(t, area, side, energy, *, q=1.0, shortfall=0.0,
             externality=0.0):
    return {"slot": t, "area_uuid": area, "side": side, "energy_type": energy,
            "requested_kwh": q, "allocated_kwh": q, "fill_rate": 1.0,
            "preference_matched": False, "named": False, "partner": "",
            "actual_kwh": q, "deliverable_kwh": q,
            "shortfall_penalty_ct": shortfall,
            "externality_penalty_ct": externality,
            "total_penalty_ct": shortfall + externality, "compensation_ct": ""}


# --------------------------------------------- the worked examples (A5.1)

T0, T1 = BASE_SLOT, BASE_SLOT + 900
#: K_lower = 8, K_upper = 40, q = 1 kWh. area -> (u, u_no_adj, u_no_exec,
#: class); p = 12.0 in both slots.
WORKED = {
    # gray seller, lambda = 0.10 -> r = 10.8
    "battery-001": (0.8, 2.0, 2.8, None),
    "battery-002": (-0.2, 1.0, 2.8, "interaction"),
    "battery-003": (-1.7, -0.5, 2.8, "exec_only"),
    # gray seller at the cap, lambda = 0.20 -> r = 9.6
    "battery-004": (-0.1, 2.3, 1.6, "interaction"),
    # green seller, G = 0.02 -> r = 12.24
    "player-005": (0.14, -0.1, 4.24, "rescued"),
    # a buyer at the price whose externality exceeds its margin
    "player-006": (-12.0, -12.0, 28.0, "exec_only"),
}
NAMES = ("exec_only", "adj_only", "both_alone", "interaction", "rescued")


def worked_run(root):
    slots = [slot_row(T0, 12.0, grey=10.8, green=12.24),
             slot_row(T1, 12.0, grey=9.6)]
    areas = [area_row(T0, "battery-001", "seller", "grey", shortfall=2.0),
             area_row(T0, "battery-002", "seller", "grey", shortfall=3.0),
             area_row(T0, "battery-003", "seller", "grey", shortfall=4.5),
             area_row(T0, "player-005", "seller", "green", shortfall=4.1),
             area_row(T0, "player-006", "buyer", "mixed", externality=40.0),
             area_row(T1, "battery-004", "seller", "grey", shortfall=1.7)]
    return raw_run(root, "ir_ref_mult", slots, areas)


def test_the_worked_examples(tmp_path):
    """Through `dr3.realised` and the stage's own decomposition and
    classification, so u is the one `dr3` defines."""
    data = worked_run(tmp_path)
    rec = dr3.realised(data)
    parts = ir_combined.decompose(rec)
    masks = ir_combined.classify(parts)
    ids = list(data.areas.strings("area_uuid"))
    for area, (u, no_adj, no_exec, klass) in WORKED.items():
        i = ids.index(area)
        assert parts["u"][i] == pytest.approx(u, abs=1e-9), area
        assert parts["u_no_adj"][i] == pytest.approx(no_adj, abs=1e-9), area
        assert parts["u_no_exec"][i] == pytest.approx(no_exec, abs=1e-9), area
        expected = [klass] if klass else []
        assert [n for n in NAMES if masks[n][i]] == expected, area
        assert masks["violation"][i] == (klass not in (None, "rescued")), area


def test_the_worked_examples_in_the_decomposition_table(tmp_path):
    acc = ir_combined.IRCombined()
    acc.add(worked_run(tmp_path))
    rows = {r["group"]: r for r in acc.decomposition_table()}
    grey, green, buyer = (rows["seller_grey"], rows["seller_green"],
                          rows["buyer"])
    assert (grey["n"], grey["n_levy"], grey["n_penalised"],
            grey["n_adj_and_penalty"]) == (4, 4, 4, 4)
    assert (grey["n_violations"], grey["n_interaction"], grey["n_exec_only"],
            grey["n_rescued"]) == (3, 2, 1, 0)
    assert (green["n_bonus"], green["n_violations"], green["n_rescued"]) == \
        (1, 0, 1)
    assert (buyer["n_violations"], buyer["n_exec_only"],
            buyer["n_adj_and_penalty"]) == (1, 1, 0)
    assert buyer["n_net_rate_below_k_lower"] is None
    # Net seller rates r - Phi: 8.8, 7.8, 6.3, 7.9 (grey), 8.14 (green).
    assert grey["net_rate_min_ct"] == pytest.approx(6.3)
    assert grey["n_net_rate_below_k_lower"] == 3
    assert rows["all"]["n_violations"] == 4
    assert rows["all"]["n_net_rate_below_k_lower"] == 3
    assert rows["all"]["net_rate_min_ct"] is None
    assert rows["all"]["min_u_ct"] == pytest.approx(-12.0)
    assert rows["all"]["sum_negative_u_ct"] == pytest.approx(-14.0)


def test_buyers_never_land_in_a_class_the_adjustment_decides():
    """Their adj is 0, so u_no_adj = u, and their margin without execution
    is never negative: a buyer violation is exec_only, always."""
    rng = np.random.default_rng(20261001)
    n = 5000
    price = rng.uniform(8.0, 40.0, n)
    rec = {"seller": np.zeros(n, dtype=bool), "q": rng.uniform(0, 3, n),
           "price": price, "r_buyer": price,
           "r_seller": price * rng.uniform(0.8, 1.2, n),
           "u": rng.normal(0.0, 5.0, n), "band": R.Band.from_dict(CALIBRATED)}
    parts = ir_combined.decompose(rec)
    masks = ir_combined.classify(parts)
    assert (parts["adj"] == 0).all()
    assert masks["violation"].any()
    for name in ("adj_only", "both_alone", "interaction", "rescued"):
        assert not masks[name].any(), name
    assert (masks["exec_only"] == masks["violation"]).all()


def test_a_violating_slot_in_a_positive_week_is_not_a_negative_week(
        tmp_path):
    """IR at participant level: U = sum of u over the week."""
    slots = [slot_row(T0, 12.0), slot_row(T1, 12.0)]
    areas = [
        # u = 4 - 4.5 = -0.5, then 4: a violating slot, U = 3.5
        area_row(T0, "player-001", "seller", "green", shortfall=4.5),
        area_row(T1, "player-001", "seller", "green"),
        # u = -1 twice: U = -2
        area_row(T0, "player-002", "seller", "green", shortfall=5.0),
        area_row(T1, "player-002", "seller", "green", shortfall=5.0),
        # u = 4 twice
        area_row(T0, "battery-003", "seller", "grey"),
        area_row(T1, "battery-003", "seller", "grey"),
    ]
    acc = ir_combined.IRCombined()
    acc.add(raw_run(tmp_path, "ir_ref_mult", slots, areas))
    rows = {r["type"]: r for r in acc.participants_table()}
    household, battery = rows["household"], rows["battery"]
    assert household["n_participant_weeks"] == 2
    assert household["n_with_violating_slot"] == 2
    assert household["n_negative_week"] == 1
    assert household["min_week_u_ct"] == pytest.approx(-2.0)
    assert household["median_week_u_ct"] == pytest.approx(0.75)
    assert (battery["n_participant_weeks"], battery["n_with_violating_slot"],
            battery["n_negative_week"]) == (1, 0, 0)
    assert battery["min_week_u_ct"] == pytest.approx(8.0)


# ------------------------------------------- the analysis and the grid

@functools.lru_cache(maxsize=1)
def ir_grid():
    return campaign.ir_grid()


def test_the_stage_expects_what_the_grid_builds():
    """`ir_combined.CELLS` restates the grid for the checks; this is what
    keeps the two from drifting apart."""
    grid, block1 = ir_grid(), campaign.block1_grid()
    assert set(grid) == set(ir_combined.CELLS)
    assert all(ir_combined.belongs(cell) for cell in grid)
    for name, spec in ir_combined.CELLS.items():
        cell = grid[name]
        assert spec.source in block1, name
        assert cell["preferences"]["mode"] == spec.mode, name
        assert cell["preferences"]["grey_levy"] == spec.grey_levy, name
        assert cell["gamma"] == spec.gamma, name
        assert cell["eta_relative"] == spec.eta_relative, name
        assert cell["deviation"]["sigma"] == ir_combined.SIGMA, name
        # The reference cells leave the window at the `RunSpec` default.
        window = cell.get("evening_periods")
        assert window == block1[spec.source].get("evening_periods"), name
        assert spec.bonus == (window == campaign.OVERLAP_EVENING_PERIODS), name
        # G5's counterpart clears the same book: the block-1 source, or the
        # unmoved combined cell, which differs only in what execution and
        # the levy read.
        if spec.counterpart == spec.source:
            assert not spec.rates_move
        else:
            other = grid[spec.counterpart]
            moved = sorted(k for k in cell if cell[k] != other[k])
            assert moved in (["gamma"], ["eta_relative"], ["preferences"]), \
                name
            assert spec.rates_move == (moved == ["preferences"]), name
    assert ir_combined.REFERENCE_GAMMA == campaign.execution_gamma()
    assert ir_combined.REFERENCE_ETA_RELATIVE == campaign.ETA_RELATIVE
    assert ir_combined.SIGMA == campaign.SIGMA
    assert ir_combined.SEEDS == campaign.SEEDS


def test_g1_names_what_is_missing():
    runs = [type("Run", (), {"cell": cell, "seed": seed})
            for cell in ir_combined.CELLS for seed in ir_combined.SEEDS
            if (cell, seed) != ("ir_overlap_add", 3)]
    runs.append(type("Run", (), {"cell": "ir_other", "seed": 0}))
    result = ir_combined.completeness(runs, list(ir_combined.CELLS),
                                      ir_combined.SEEDS)
    assert not result["passed"]
    assert result["missing"] == ["ir_overlap_add seed 3"]
    assert result["unexpected"] == ["ir_other seed 0"]


def _run(sha, n_entries=1):
    return type("Run", (), {"repo_sha": sha, "run_id": f"r-{sha[:3]}",
                            "n_entries": n_entries})


def test_g3_reads_the_grid_at_the_recorded_commit():
    def git(*args):
        if args[0] == "cat-file":
            if args[2] != "a" * 40 + "^{commit}":
                raise ir_combined.subprocess.CalledProcessError(1, "git")
            return ""
        return "import x\n\ndef ir_grid():\n    return {}\n"

    assert ir_combined.provenance([_run("a" * 40)] * 3, git=git)["passed"]
    two = ir_combined.provenance([_run("a" * 40), _run("b" * 40)], git=git)
    assert not two["passed"] and "2 different repo_sha" in two["problems"][0]
    unknown = ir_combined.provenance([_run("b" * 40)], git=git)
    assert unknown["problems"] == [f"{'b' * 40} is not a commit of this "
                                   f"checkout"]
    appended = ir_combined.provenance([_run("a" * 40, n_entries=2)], git=git)
    assert not appended["passed"]
    assert not ir_combined.defines_grid("def block1_grid():\n    pass\n")
    assert not ir_combined.defines_grid("# def ir_grid():\n")


def test_g3_on_a_real_commit_without_the_grid():
    """eb2640d is the commit this group was specified against; its
    `campaign.py` has no `ir_grid`, so runs claiming it fail G3."""
    sha = "eb2640dad04c2d5798017828b5db49cf7aaa0e50"
    try:
        ir_combined._git("cat-file", "-e", f"{sha}^{{commit}}")
    except (OSError, ir_combined.subprocess.CalledProcessError):
        pytest.skip("eb2640d is not in this clone")
    result = ir_combined.provenance([_run(sha)])
    assert result["problems"] == [f"{ir_combined.GRID_FILE} at {sha} does "
                                  f"not define ir_grid"]


# ---------------------------------------------------- the whole group

#: Slot 0 supply-limited, player-001 withholds 0.5 kWh: a penalty pool.
#: Slot 1 demand-limited with green and grey supply, battery-002 delivers
#: 1.0 of 2.0 kWh: a levy (and, on the overlap window, a bonus) and a
#: shortfall penalty on the same grey row.
GROUP_BOOK = [
    {"sellers": {"player-001": 2.0, "player-002": 3.0},
     "buyers": {"player-003": 4.0, "player-004": 4.0},
     "deliverable": {"player-001": 2.5}},
    {"sellers": {"player-001": 6.0, "battery-002": 4.0},
     "buyers": {"player-003": 3.0, "player-004": 2.0},
     "actual": {"battery-002": 1.0}},
]


def _rewrite(path, change):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        change(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _settle(directory, prefs, bonus):
    """The origin settlement the clearing node would record: battery areas
    sell grey, the levy on them, the bonus where green and grey meet."""
    run_id = directory.name
    green, grey = {}, {}

    def energy(row):
        if row["side"] == "seller" and row["area_uuid"].startswith("battery-"):
            row["energy_type"] = "grey"
        if row["side"] == "seller":
            into = grey if row["energy_type"] == "grey" else green
            into[row["slot"]] = into.get(row["slot"], 0.0) + float(
                row["allocated_kwh"])

    def rates(row):
        p = float(row["clearing_price_ct"])
        levy = paid = 0.0
        if grey.get(row["slot"]):
            row["grey_final_ct"] = round(p * (1 - prefs["grey_levy"]), 6)
            levy = round(prefs["grey_levy"] * p * grey[row["slot"]], 6)
            if bonus and green.get(row["slot"]):
                g = prefs["green_multiplier"]
                row["green_final_ct"] = round(p * (1 + g), 6)
                paid = round(g * p * green[row["slot"]], 6)
        row.update({"pref_enabled": True, "pref_order": prefs["order"],
                    "mult_enabled": True, "mult_mode": prefs["mode"],
                    "mult_sides": prefs["sides"], "levy_collected_ct": levy,
                    "bonus_paid_ct": paid,
                    "pool_surplus_ct": round(levy - paid, 6)})

    _rewrite(directory / f"{run_id}_areas.csv", energy)
    _rewrite(directory / f"{run_id}_slots.csv", rates)


def _manifest(directory, spec, deviation):
    path = directory / "manifest.json"
    entry, = json.loads(path.read_text(encoding="utf-8"))
    entry["config"] = json.loads(json.dumps(spec.config()))
    entry["deviation"] = deviation
    entry["battery_window"] = [min(spec.evening_periods),
                               max(spec.evening_periods)]
    path.write_text(json.dumps([entry]), encoding="utf-8")


def write_group(group_root, source_root, monkeypatch, *, skip=()):
    """The six combined cells at every seed under `group_root`, their three
    block-1 sources under `source_root`, configured as `campaign.ir_grid()`
    and `block1_grid()` build them."""
    grid, block1 = ir_grid(), campaign.block1_grid()
    for cell, config in grid.items():
        for seed in ir_combined.SEEDS:
            if (cell, seed) in skip:
                continue
            # Executed at the cell's own gamma and eta, so the penalties,
            # the pool and `gamma_eff` / `eta_relative_eff` agree.
            with monkeypatch.context() as patch:
                patch.setattr(tda, "GAMMA", config["gamma"])
                patch.setattr(tda, "ETA", config["eta_relative"])
                directory = write_run(group_root, cell, seed, GROUP_BOOK,
                                      execute=True)
            _settle(directory, config["preferences"],
                    ir_combined.CELLS[cell].bonus)
            _manifest(directory, campaign.spec_for(cell, seed, **config),
                      {**config["deviation"], "deviators": [],
                       "slots_with_absent_deviator": 0})
    for source in sorted({c.source for c in ir_combined.CELLS.values()}):
        config = block1[source]
        for seed in ir_combined.SEEDS:
            directory = write_run(source_root, source, seed, GROUP_BOOK)
            _settle(directory, config["preferences"],
                    config.get("evening_periods")
                    == campaign.OVERLAP_EVENING_PERIODS)
            _manifest(directory, campaign.spec_for(source, seed, **config),
                      None)


def fake_git(*args):
    """The recorded commit exists and its `campaign.py` defines the grid."""
    if args[0] == "show":
        return "def ir_grid():\n    return {}\n"
    return ""


def analyse(roots, dest, *extra):
    code = dr_analysis.main(["--runs", *map(str, roots), "--dest", str(dest),
                             *extra])
    result = json.loads((dest / "checks.json").read_text(encoding="utf-8"))
    return code, result["checks"], result["findings"]


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def other_tables(dest) -> dict:
    """Every CSV the analysis wrote that is not one of the group's."""
    return {p.name: p.read_bytes() for p in sorted(dest.glob("*.csv"))
            if not ir_combined.is_table(p.stem)}


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """(ordinary, sources, group): the honest reference cell, so R1-R3
    pass; the three block-1 sources; the six combined cells."""
    monkeypatch.setattr(ir_combined, "_git", fake_git)
    ordinary = tmp_path / "ordinary"
    write_run(ordinary, "b2_noise_off", 0, TWO_SLOTS, execute=True)
    return ordinary, tmp_path / "sources", tmp_path / "group"


def test_a_complete_group_passes_and_writes_its_tables(roots, tmp_path,
                                                       monkeypatch):
    ordinary, sources, group = roots
    write_group(group, sources, monkeypatch)
    dest = tmp_path / "dest"
    code, checks, findings = analyse([ordinary, sources, group], dest)
    for name in ir_combined.HARD + (ir_combined.G7, "ir_combined"):
        assert checks[name]["passed"], (name, checks[name])
    # The coalition cells are absent, so A6 fails and the exit code is 1
    # for a reason that has nothing to do with this stage.
    assert code == 1
    assert not checks["a6_buyers_underreport"]["passed"]
    assert checks[ir_combined.G5]["max_deviation"] == 0.0
    assert len(checks[ir_combined.G5]["pairs"]) == 6
    assert checks[ir_combined.G4]["n_adj_and_penalty_by_cell"] == \
        dict.fromkeys(sorted(ir_combined.CELLS), 5)
    assert checks[ir_combined.G2]["gamma_eff"]["ir_overlap_mult_g150"] == [1.5]
    assert checks[ir_combined.G2]["eta_relative_eff"][
        "ir_overlap_mult_eta005"] == [0.05]
    for table in ir_combined.TABLES:
        assert (dest / f"{table}.csv").exists(), table

    census = read(dest / "ir_combined_census.csv")
    assert len(census) == 30
    assert sorted({r["cell"] for r in census}) == sorted(ir_combined.CELLS)
    assert not {r["cell"] for r in read(dest / "census.csv")} \
        & set(ir_combined.CELLS)
    assert not {r["cell"] for r in read(dest / "dr3_ir.csv")} \
        & set(ir_combined.CELLS)

    decomposition = read(dest / "ir_combined_decomposition.csv")
    assert len(decomposition) == 6 * len(ir_combined.GROUPS)
    # The net rates are dr3's: the lowest seller rate of the settlement
    # table is the lowest of the two seller groups.
    settlement = {(r["cell"], r["side"]): r
                  for r in read(dest / "ir_combined_settlement.csv")}
    for cell in ir_combined.CELLS:
        lowest = min(float(r["net_rate_min_ct"]) for r in decomposition
                     if r["cell"] == cell and r["group"].startswith("seller"))
        assert lowest == pytest.approx(
            float(settlement[(cell, "seller")]["rate_min_ct"]), abs=1e-12)
    participants = read(dest / "ir_combined_participants.csv")
    assert len(participants) == 6 * 2

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ir_combined"]["cells"] == sorted(ir_combined.CELLS)
    assert manifest["ir_combined"]["repo_sha"] == FAKE_SHA
    assert manifest["ir_combined"]["tables_written"] is True
    assert set(findings["ir_combined"]["cells"]) == set(ir_combined.CELLS)


def test_one_combined_run_moves_no_other_table(roots, tmp_path):
    """A5.4: one `ir_` cell beside one ordinary cell. Once with and once
    without the `ir_` run directory, every table that is not the group's is
    byte-identical -- and `--compare-to` says so."""
    ordinary, _sources, group = roots
    write_run(group, "ir_ref_mult", 0, TWO_SLOTS, execute=True)
    with_group, without = tmp_path / "with", tmp_path / "without"
    code, checks, _ = analyse([ordinary, group], with_group)
    # One cell of six: G1 stops the stage, and nothing of it is written.
    assert not checks[ir_combined.G1]["passed"]
    assert not list(with_group.glob("ir_combined*"))
    assert code == 1

    _code, checks, _ = analyse([ordinary], without, "--compare-to",
                               str(with_group))
    assert checks["ir_combined"]["skipped"]
    assert other_tables(with_group) == other_tables(without)
    manifest = json.loads((without / "manifest.json").read_text(
        encoding="utf-8"))
    verdicts = manifest["compare_to"]["tables"]
    assert set(verdicts) == {name[:-4] for name in other_tables(without)}
    assert set(verdicts.values()) == {"identical"}


def test_the_whole_group_moves_no_other_table(roots, tmp_path, monkeypatch):
    """The same with the stage passing and writing its tables."""
    ordinary, sources, group = roots
    write_group(group, sources, monkeypatch)
    with_group, without = tmp_path / "with", tmp_path / "without"
    analyse([ordinary, sources, group], with_group)
    assert (with_group / "ir_combined_decomposition.csv").exists()
    analyse([ordinary, sources], without)
    assert other_tables(with_group) == other_tables(without)


def test_without_a_combined_run_the_stage_is_skipped(roots, tmp_path):
    """A5.5: passed, named, and nothing written."""
    ordinary, _sources, _group = roots
    dest = tmp_path / "dest"
    _code, checks, findings = analyse([ordinary], dest)
    assert checks["ir_combined"] == {"passed": True, "skipped": True,
                                     "reason": "no runs"}
    assert not [name for name in checks
                if name.startswith("ir_combined_")]
    assert not list(dest.glob("ir_combined*"))
    assert findings["ir_combined"] == {}
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ir_combined"]["cells"] == []
    assert manifest["ir_combined"]["tables_written"] is False


def test_g5_catches_a_changed_allocation(roots, tmp_path, monkeypatch,
                                        capsys):
    """A5.6: one allocation of one combined run moved by 0.1 kWh. G5 stops
    the stage; the other tables are written; the exit code is 1."""
    ordinary, sources, group = roots
    write_group(group, sources, monkeypatch)
    directory = group / "ir_overlap_mult--seed-2"

    def move(row):
        if row["area_uuid"] == "player-003" and row["slot"] == str(BASE_SLOT):
            row["allocated_kwh"] = str(float(row["allocated_kwh"]) - 0.1)

    _rewrite(directory / "ir_overlap_mult--seed-2_areas.csv", move)
    dest = tmp_path / "dest"
    code, checks, _ = analyse([ordinary, sources, group], dest)
    assert code == 1
    g5 = checks[ir_combined.G5]
    assert not g5["passed"]
    assert g5["max_deviation"] == pytest.approx(0.1)
    failed = sorted(pair for pair, result in g5["pairs"].items()
                    if not result["passed"])
    # The run is compared with its block-1 source, and three cells are
    # compared with it.
    assert failed == ["ir_overlap_mult <-> mult_overlap_multiplicative",
                      "ir_overlap_mult_eta005 <-> ir_overlap_mult",
                      "ir_overlap_mult_g150 <-> ir_overlap_mult",
                      "ir_overlap_mult_levy020 <-> ir_overlap_mult"]
    assert g5["pairs"]["ir_overlap_mult <-> mult_overlap_multiplicative"][
        "max_area_deviation"]["allocated_kwh"] == pytest.approx(0.1)
    assert checks["ir_combined"]["stopped_by"] == [ir_combined.G5]
    assert not list(dest.glob("ir_combined*"))
    assert (dest / "census.csv").exists() and (dest / "dr3_ir.csv").exists()
    assert ir_combined.G5 in capsys.readouterr().err


def test_g1_names_a_missing_seed(roots, tmp_path, monkeypatch, capsys):
    """A5.7, through the analysis: the partial group stops the stage and
    the missing run is named."""
    ordinary, sources, group = roots
    write_group(group, sources, monkeypatch,
                skip={("ir_overlap_add", 3)})
    dest = tmp_path / "dest"
    code, checks, _ = analyse([ordinary, sources, group], dest)
    assert code == 1
    assert checks[ir_combined.G1]["missing"] == ["ir_overlap_add seed 3"]
    assert checks["ir_combined"]["stopped_by"] == [ir_combined.G1]
    assert ir_combined.G2 not in checks
    assert "ir_overlap_add seed 3" in capsys.readouterr().err
    assert not list(dest.glob("ir_combined*"))
    assert (dest / "census.csv").exists()


