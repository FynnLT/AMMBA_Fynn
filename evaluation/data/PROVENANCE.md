# Evaluation data — provenance

This directory holds the base dataset of the thesis evaluation campaign and the
record that identifies it. Nothing here is part of the artifact; `evaluation/`
sits outside the artifact boundary, like `ui` (D-13).

## Source

| Field | Value |
|---|---|
| Title | *A complete energy community dataset with photovoltaic generation, battery energy storage systems and electric vehicles* |
| Authors | Ricardo Faia, Calvin Gonçalves, Luis Gomes, Zita Vale |
| Dataset DOI | [10.5281/zenodo.11351017](https://doi.org/10.5281/zenodo.11351017) |
| Version | **v1.5, published 27 May 2024** |
| Accompanying article | Faia, R., Gonçalves, C., Gomes, L., Vale, Z. (2023): *Dataset of an energy community with prosumer consumption, photovoltaic generation, battery storage, and electric vehicles.* **Data in Brief 48, 109218.** [10.1016/j.dib.2023.109218](https://doi.org/10.1016/j.dib.2023.109218) |
| Licence | **CC BY 4.0** — see `LICENSE-CC-BY-4.0.txt` |
| Retrieved | 16 September 2026 |

## The file

| Field | Value |
|---|---|
| Path | `raw/EC_EV_dataset (fixed error).xlsx` |
| Size | 1,403,559 bytes |
| SHA-256 | `f96ef125d1c0f49cf3288b17598c166062570c7baa6c4d2d25ac545813531c3a` |

The filename is kept exactly as published. The `(fixed error)` suffix is the
only version marker carried by the file itself; renaming it would lose the
distinction from the earlier Zenodo version.

Verify before a campaign run:

```bash
sha256sum "raw/EC_EV_dataset (fixed error).xlsx"
```

A mismatch means the file is not the one the reported figures were produced
from, and the run is not the run the manifest names.

## What the file contains

19 sheets. Seven are used by this evaluation; the twelve electric-vehicle
sheets are out of scope and are not read.

| Sheet | Shape | Used for |
|---|---|---|
| `General Information` | 254 × 43 | Per-player PV and ESS flags, totals, storage parameters |
| `Load` | 97 × 251 | Household load, 96 periods × 250 players |
| `PV` | 97 × 251 | PV generation, same layout |
| `BESS` | 8 × 251 | Storage parameters per player (no time series) |
| `Buy Price` | 97 × 251 | Retail price per period and player |
| `Sell Price` | 97 × 251 | Feed-in tariff per period and player |
| `Limits` | 10 × 251 | Contracted power, fixed costs |

Layout: row 1 is the player id (1…250), column 1 is the period index (1…96).

**Time mapping.** Period `n` covers `[(n−1)·900 s, n·900 s)` and is labelled by
its end. Read off the EV departure table, which pairs clock times with period
numbers: period 33 = 08:15, period 36 = 09:00, period 76 = 19:00. The
15-minute resolution matches the artifact's 900 s market slot exactly.

## Units — the sheets are not consistent, and one label is wrong

**`Load` and `PV` are powers in kW.** Energy per period is `value × 0.25 h`.
Grounds: the sheet headers use kW throughout, the resulting mean household load
is 1.02 kW, and reading the values as kWh would put a household at 98 kWh/day.
To be confirmed against the *Data in Brief* article before Chapter 5 states a
unit.

**`BESS` mixes the two, and its `(kW)` labels are partly wrong.** The sheet
labels every row `(kW)`, but three of them are energies:

| Row | Label in the sheet | Read as | Why |
|---|---|---|---|
| Capacity | `Capacity (kW)` | **kWh** | a storage capacity is an energy |
| Initial | `Initial (kW)` | **kWh** | it is exactly `Capacity / 2` for **all 150 units** (6.4/3.2, 11.7/5.85, 6.6/3.3, 12.9/6.45 …), which is a state of charge, not a power |
| Final | `Final (kW)` | **kWh** | equal to `Initial` for all 150 units — the optimisation's end-of-horizon constraint |
| Charge / Discharge | `(kW)` | kW | genuine powers; energy per period is `value × 0.25 h` |

Verified against the file on 16.09.2026: 150 of 150 units satisfy both
relations exactly. `harness/profiles.py` implements this reading and documents
it in `BatterySpec`.

## Measured on 16.09.2026

Computed from the file, not quoted from a description of it.

| Quantity | Value |
|---|---|
| Players / with PV / with ESS | 250 / 200 / 150 |
| Players with ESS but no PV | **0** |
| Players with neither | 50 |
| Total load over the day | 24,541 kW-periods ≈ 6,135 kWh |
| Total PV over the day | 9,563 kW-periods ≈ 2,391 kWh |
| Load per period, min / mean / max | 102.6 / 255.6 / 453.9 kW |
| PV per period, min / mean / max | 0 / 99.6 / 561.3 kW |
| Periods with any PV | 40 of 96 (periods 32–71, 07:45–17:45) |
| Supply-limited periods (PV > load) | 15 of 96 |
| Supply/demand ratio, periods with PV | 0.000523 … 2.1184 |
| Supply/demand ratio, all 96 periods | **0 … 2.1184** — it is exactly zero in the 56 periods without generation |
| Retail price | 0.1073 / 0.1710 / 0.2336 €/kWh, three levels per household |
| Feed-in tariff | 0.045 €/kWh, constant |

The two ratio rows are stated separately on purpose: a minimum of 0.000523
describes only the periods in which anything is generated, and quoting it for
the day would hide that more than half of it has no supply at all.

Two properties determine how the campaign is built:

1. **Every storage system belongs to a PV household, and none has a time
   series.** `BESS` holds design parameters only. A grey selling side
   therefore has to be modelled explicitly; it cannot be read out of the data.
   The rule this repository applies is declared in
   `harness/profiles.make_battery_areas` and is a modelling assumption of this
   work (D-75).
2. **56 of 96 periods carry no PV at all.** Without a non-PV seller, more than
   half of the campaign's slots have no supply.

## Derived data

`derived/` is git-ignored. It holds the 672-slot week built from this file by
the synthetic extension, and is regenerated from `raw/` plus the extension seed
recorded in the run manifest. The manifest carries `config + seed + SHA` and
the extension seed; those five values, with this record, are what makes a
reported figure reproducible.

## Attribution

When a figure derived from this dataset is published, cite Faia et al. (2023)
and state that the data were extended synthetically for this work. The licence
requires attribution and an indication of changes; `LICENSE-CC-BY-4.0.txt`
carries both.
