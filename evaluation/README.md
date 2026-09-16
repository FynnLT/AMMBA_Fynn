# Evaluation harness

**`evaluation/` is not part of the artifact.** It is the simulation harness for
the thesis evaluation: it drives the three services in process, records what
they answer, and writes figures. Nothing here is deployed, and nothing in the
three services depends on it — the boundary is the same one `ui/` sits outside
of (D-13).

## Setup

```bash
pip install -r evaluation/requirements.txt
```

The harness loads the three services' `src` packages side by side out of the
working tree, so their own dependencies (`fastapi`, `httpx`, `pydantic`,
`pyyaml`) must be installed as well. No ports, no docker, no database.

## Smoke test

One slot through the real pipeline — run it after moving the harness, after
changing a service, and before starting a campaign:

```bash
cd evaluation/harness
python smoke.py
```

It proves the three services load side by side, a clearing round completes,
and the execution node answers with a redistribution block. The expected
output is fixed and is the regression check for the runner:

```
clearing price : 15.147274 ct/kWh
trades         : 100
round type     : DEMAND_LIMITED
redistribution : present

SMOKE TEST PASSED
```

If that price moves, something changed that should not have.

## Tests

```bash
cd evaluation
python -m pytest -q
```

`conftest.py` puts `harness/` (and `calibration/`, once it exists) on
`sys.path`, so the flat imports the scripts use work under pytest as well.
`tests/test_harness.py` holds the four tests that each stand for a defect that
actually reached a figure; do not weaken them.

## Running a campaign

```bash
cd evaluation/harness
python campaign.py
```

A campaign runs one process per run (`multiprocessing`, spawn), each worker
building its own `Stack` and therefore its own empty store, with five seeds per
cell recorded individually. `runs.RunSpec` carries everything a run depends on;
`runs.execute_run` builds the stack, walks the week, writes the CSVs, writes
the manifest and runs the pre-flight checks.

## Where output lands

Everything goes to `evaluation/out/`, one subdirectory per run, and `out/` is
git-ignored: results are not part of the artifact either.

```
out/<run_id>/<run_id>_slots.csv    one row per slot
out/<run_id>/manifest.json         one provenance entry per run
out/figures/                       figures from plots.py
```

**Everything the thesis cites comes out of `out/` and is identified by the
manifest, not by a file path.** Each manifest entry carries `config + seed +
SHA`, the dataset extension seed, the selected player ids and the five seeds of
its cell individually, so a single deviating cell can be re-run alone.

## Data

`data/raw/` holds the Faia et al. energy-community workbook and is **read-only
for every script here** — a run that writes into `raw/` is a bug.
`data/PROVENANCE.md` carries the source, DOI, licence, SHA-256 and the
properties measured off the file; `data/derived/` is git-ignored and is
regenerated from `raw/` plus the extension seed recorded in the manifest.

## Frozen scripts: `harness/legacy/`

`eval_r0_r1.py`, `eval_r2.py`, `eval_r3_r4.py`, `eval_r5.py`, `blockd.py`,
`blockd2.py` and `blocke.py` are frozen at the 02.09.2026 test run and run
against `git checkout fe52d63`, not against the current runner signature — it
now requires an explicit community and slot, and these scripts call `run_slot`
positionally. They produced the 02.09. test-run figures; no Chapter 5 result
depends on them, so they are kept as a record rather than adapted.
