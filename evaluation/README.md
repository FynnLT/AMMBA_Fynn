# Evaluation harness

**`evaluation/` is not part of the artifact.** It holds the simulation harness and
the analysis behind Chapter 5 of the thesis. It loads the three services' `src`
packages into one process, drives them through simulated market weeks, records
their responses and builds the tables and figures. The services do not depend on
anything here, and nothing here is deployed.

## Setup

From the repository root:

```bash
pip install -r amm-execution-node/requirements.txt -r evaluation/requirements.txt
```

The first file installs the services' own dependencies (`fastapi`, `httpx`,
`pydantic`, `pyyaml`), the second the harness's (`openpyxl`, `matplotlib`, `numpy`,
`pytest`). No ports, no Docker and no blockchain are needed. The Clearing Node runs
in mock mode.

The input data is the energy-community dataset of Faia et al. (2023) in
`data/raw/`. Its source, licence (CC BY 4.0) and SHA-256 are in
[`data/PROVENANCE.md`](data/PROVENANCE.md). Every script treats `data/raw/` as
read-only.

## Smoke test

One slot through the real pipeline. Run it after changing a service and before a
campaign:

```bash
cd evaluation/harness
python smoke.py
```

Expected output. The first four lines use the illustrative parameters of the
reference example (`K_upper` = 28.5, `B` = 2.5), the fifth the configured band:

```
clearing price : 15.147274 ct/kWh
trades         : 100
round type     : DEMAND_LIMITED
redistribution : present
configured band: 8.0 .. 40.0 ct/kWh -> 22.802265 ct/kWh

SMOKE TEST PASSED
```

## Tests

```bash
cd evaluation
python -m pytest -q          # 175 passed, 1 skipped at 63043bb
```

The skipped test needs the manifest of a calibration week, which lives in the
git-ignored `out/`. `tests/test_harness.py` holds seven tests, each standing for a
defect that once reached a figure.

## The campaign

```bash
cd evaluation/harness
python campaign.py --block 1          # one group; see the table
python campaign.py --block ir --resume
python campaign.py                    # every group, including the calibration weeks
```

| Group | Flag | Cells × seeds | Content | Ran at (thesis, Appendix C.3) |
|---|---|---|---|---|
| Preference block | `--block 1` | 19 × 5 | allocation orders, partner density and the origin adjustment; not executed (no penalties) | `f83b57a` |
| Penalty block | `--block 2` | 14 × 5 | forecast noise, deliberate deviations and coalitions; executed | `3422529` |
| γ/η sweep | `--block sweep` | 12 × 5 | the penalty block's reference with one of γ and η moved | `3422529`, `91790e9` |
| Combined IR cells | `--block ir` | 6 × 5 | preference-block configurations executed under the penalty block's noise | `3680010` |
| Calibration weeks | `--calibration` | 6 × 1 | the five fitting weeks (extension seeds 20260917–20260921) and the out-of-sample week 4242 | — |

The four groups are 51 cells and 255 runs. Without a flag the campaign runs all of
them plus the six calibration weeks.

One process per run (`multiprocessing`, spawn), each with its own in-memory store.
A run is one simulated week of 672 slots of a community of 100 households.
`--resume` skips runs whose manifest exists. `--out DIR` writes elsewhere than
`out/`. Preferences and pricing parameters reach the Clearing Node through the
trigger (`preference_params`, `sigmoid_params`), as recorded in each manifest.

The services' executable code is the same in all commits of the table and at
`63043bb`; the commits differ in the evaluation code and in comments.

## Calibration of the pricing function

```bash
cd evaluation/calibration
python fit.py ../out/<run-id>/<run-id>_slots.csv --run-id <name>
```

A bounded coarse-to-fine grid search for `Θ` and `B` over the loss of Appendix C.2
(α = 0.5, band 8–40 ct/kWh by default). The thesis uses the medians over the five
fitting weeks, rounded to `Θ` = 1.0 and `B` = 0.6.

## Analysis

```bash
cd evaluation/analysis
python dr_analysis.py                 # all DR2–DR5 tables
python dr5_cost.py                    # the cost of priority (Section 5.2.2)
```

`dr_analysis.py` reads the runs under `out/` (or `--runs DIR …`) and writes into a
fresh `out/analysis/<date>-<sha>/` (or `--dest DIR`). It never writes into a run
directory. Before any table it checks that it reproduces what the artifact
recorded (R1–R3: prices, allocations and penalties) and exits with 1 naming the
failure otherwise. `--compare-to DIR` compares the new tables byte for byte with
an earlier analysis. `dr5_cost.py` reads the preference block's runs and writes one
JSON and one CSV.

## Scalability

```bash
cd amm-smart-contract
npx hardhat compile                    # first, so that no compiler message lands in the CSV
mkdir -p ../evaluation/out/gas-<date>
npx hardhat run scripts/gas_series.js > ../evaluation/out/gas-<date>/gas_series.csv

cd ../evaluation/harness
python ncurve.py --out ../out/ncurve-<date>          # N = 5 … 100,000 orders per slot
```

The gas series runs on the in-process Hardhat network: 20 clearings at recorded
slot aggregates and six at the aggregates of 6 to 3,200 participants. Each
`clearMarket` costs 216,843 gas, minus 12 gas per zero byte of calldata, and the
cost does not grow with the number of participants. `ncurve.py` times the
off-chain clearing cycle in process and reports points only (no fit, no capacity).

## Figures

```bash
cd evaluation/harness
python figures.py            # the figures of Chapter 5 and Appendix C
python figures.py --list     # what each figure shows
```

The figures are read from the newest matching directory under `out/`. `--analysis`,
`--ncurve` and `--gas` pin a directory. Module numbers and figures in the thesis:

| `figures.py` | Thesis |
|---|---|
| 1 | Figure 3, partner priority on the buying side |
| 10 | Figure 4, median net gain over the deviation size |
| 9 | Figure 5, gas and off-chain runtime |
| 3 | Figure C1, the two origin formulations |

Figure 2 of the thesis, the pricing function, is computed with the Clearing Node's
`sigmoid_price` and is not produced by `figures.py`. The other six modules
(`--all`) draw figures that the thesis does not use.

## Where output lands

Everything goes to `evaluation/out/`, which is git-ignored:

```
out/<cell>--seed-<n>/manifest.json     configuration, seeds, commit, input digest
out/<cell>--seed-<n>/*_slots.csv       one row per slot
out/<cell>--seed-<n>/*_areas.csv       one row per participant and slot
out/analysis/<date>-<sha>/             tables of dr_analysis.py and dr5_cost.py
out/gas-*/, out/ncurve-*/              scalability measurements
out/figures/                           figures.py
```

A figure in the thesis is traced through the manifest of the run behind it
(configuration, seeds, commit), not through a file path. The run data of the thesis
are not in the repository and can be regenerated with the commands above.

## Legacy scripts

`harness/legacy/` (`eval_r0_r1.py`, `eval_r2.py`, `eval_r3_r4.py`, `eval_r5.py`,
`blockd.py`, `blockd2.py`, `blocke.py`) is frozen at the test run of 02.09.2026 and
runs against `git checkout fe52d63`, not against the current runner.
`harness/pilot.py` and `plots.py` belong to the same test run. No result of the
thesis depends on any of them; they are kept as a record.

## A note on references in the code

Comments in `evaluation/` cite tables, figures and sections of thesis drafts from
September and October 2026 (for example *Table 4.2*, *Figure 5.1*). The submitted
thesis numbers them differently: Table 4.2 of the drafts is Table 5, Table 4.3 is
Table 6, and the figures are mapped above. `D-xx` and `#xx` refer to the thesis'
own decision log and issue list (see the root README).
