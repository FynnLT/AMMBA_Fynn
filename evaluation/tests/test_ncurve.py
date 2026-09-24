"""Section 3: the N-curve runner.

All four tests run at small N -- the curve's own large points are a
measurement, not a test, and a test suite that took ten minutes to tell you
`n_split` is off by one would not be run.
"""
import asyncio
import csv
import json

import pytest

import ncurve
import scenario


def read_rows(directory):
    with open(directory / "ncurve.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_split_sums_to_n_and_leaves_neither_side_empty():
    """N is the number of orders in the slot (D-85), so the two sides have to
    add up to it exactly -- at odd N a second rounding would quietly move the
    x axis off the N the point is named after."""
    for n in (5, 6, 7, 100, 1001):
        n_prod, n_cons = ncurve.n_split(n)
        assert n_prod + n_cons == n, n
        assert n_prod >= 1 and n_cons >= 1, n

    assert ncurve.n_split(5) == (2, 3)
    with pytest.raises(ValueError):
        ncurve.n_split(1)


def test_two_points_write_every_timing_column(tmp_path):
    out = tmp_path / "ncurve"
    code = asyncio.run(ncurve.main(
        ["--points", "5,10", "--repeats", "1", "--out", str(out)]))
    assert code == 0

    rows = read_rows(out)
    assert [int(r["n"]) for r in rows] == [5, 10]
    for row in rows:
        assert row["status"] == "ok"
        assert int(row["repeats"]) == 1
        setup = float(row["t_setup_s"])
        clear = float(row["t_clear_s"])
        store = float(row["t_store_in_clear_s"])
        assert setup > 0 and clear > 0
        # The store time is accumulated *inside* the clearing call, so it
        # cannot exceed it. If it ever did, the counters would be catching
        # requests from outside the timed section and the component split
        # D-86 asks for would be meaningless.
        assert 0 < store <= clear
        # And the cycle really does talk to the store: the clearing node's
        # own queries go through the same transport as the harness's posts.
        assert int(row["n_store_calls_in_clear"]) > 0
        assert float(row["slot_share_pct"]) == pytest.approx(
            100 * clear / ncurve.SLOT_SEC, rel=1e-3)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stopped_at_n"] is None
    assert manifest["n_points"] == [5, 10]
    assert manifest["repo_sha"] and manifest["wall_sec"] >= 0
    # No fit, no extrapolation, no capacity (D-87).
    assert "saturation_n" not in manifest
    assert not any("saturation" in key for key in manifest)

    # A second invocation into the same directory is refused, not appended to.
    assert asyncio.run(ncurve.main(
        ["--points", "5", "--repeats", "1", "--out", str(out)])) == 2


def test_one_order_per_area_holds_at_the_largest_tested_point():
    """D-85. Asserted on the book that was built, not read off the
    generator: N orders means N market areas, and the whole x axis is that
    identity."""
    n = 1001
    scen = ncurve.make_book(n, seed=0)
    scenario._assert_one_order_per_area(scen)

    areas = [order["area_uuid"]
             for side in ("producers", "consumers")
             for order in scen[side]]
    assert len(areas) == n
    assert len(set(areas)) == n


def test_the_stopping_rule_stops_the_curve_and_not_the_script(
        tmp_path, monkeypatch):
    """A stopping rule that raises loses every point already measured."""
    out = tmp_path / "ncurve"
    monkeypatch.setattr(ncurve, "SLOT_SEC", 1e-9)
    code = asyncio.run(ncurve.main(
        ["--points", "5,10", "--repeats", "1", "--out", str(out)]))
    assert code == 0

    rows = read_rows(out)
    assert [int(r["n"]) for r in rows] == [5]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "slot_exceeded"
    assert manifest["stopped_at_n"] == 5
    # The point that triggered it is in the data, not merely in the message.
    assert manifest["n_points_measured"] == [5]
