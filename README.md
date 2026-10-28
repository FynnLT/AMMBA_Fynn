# AMM Execution Engine — post-price clearing for local energy markets

This repository holds the artifact of the master's thesis *Design and
Implementation of a Hybrid Post-Price Clearing Mechanism for Decentralized
Energy Markets* (Fynn Teutenberg, University of Cologne, 2026). It implements
the automated-market-maker-based batch auction (AMMBA) as a clearing and
settlement engine for the data model of the GSY Decentralized Exchange (GSY DEX).

Each 15-minute delivery slot is cleared as one batch:

- one uniform price, set by a sigmoid of the supply-to-demand ratio and bounded
  by the feed-in tariff and the retail price;
- the cleared quantity is the smaller of total supply and total demand, and the
  long side is rationed pro rata;
- two preference mechanisms act **after** price formation: mutually nominated
  trading partners are served first, and an energy-origin adjustment lets a gray
  levy fund a green bonus;
- the aggregates and the price are anchored in a smart contract on an EVM chain,
  while orders, identities and allocations stay off-chain;
- after delivery, shortfall and externality penalties are computed against meter
  readings and forecasts, and their redistribution is computed but not paid out.

The thesis describes the code at commit `63043bb` (Chapter 4, Appendix A).
Later commits change documentation and code comments only.

---

## Repository layout

| Directory | What it is | Part of the artifact | Port |
|---|---|---|---|
| [`amm-smart-contract/`](amm-smart-contract/) | Pricing Smart Contract `AMMBA.sol` (Solidity 0.8.20, Hardhat 2.19.5) with its tests and deployment scripts | yes | — |
| [`amm-clearing-node/`](amm-clearing-node/) | Clearing Node (Python, FastAPI): price, allocation, preferences, trade objects, anchoring | yes | 8081 |
| [`amm-execution-node/`](amm-execution-node/) | Execution and Penalty Node (Python, FastAPI): reconciliation after delivery, penalties, redistribution | yes | 8082 |
| [`mock-offchain-db/`](mock-offchain-db/) | In-memory stand-in for the GSY DEX off-chain storage, with the same routes | no | 8080 |
| [`ui/`](ui/) | Browser demo for walking through one market cycle | no | 3000 |
| [`evaluation/`](evaluation/) | Simulation harness and analysis behind Chapter 5 of the thesis → [`evaluation/README.md`](evaluation/README.md) | no | — |
| [`scripts/`](scripts/) | Local start and stop scripts (Windows), UI server, end-to-end check `e2e_demo.py` | no | — |

The mock, the demo interface and the evaluation harness sit in the same repository
but outside the artifact boundary. Nothing in the three artifact components depends
on them; the two nodes talk to any service that offers the off-chain storage routes,
configured through `OFFCHAIN_DB_URL`.

---

## One market cycle

1. **Orders.** Buyers (`Bid`) and sellers (`Offer`) post orders in kWh to the
   off-chain storage. Two optional fields carry preferences:
   `attributes.energy_type` (`"green"` or `"grey"`, default green) and
   `requirements.preferred_partner` (the `area_uuid` of a participant on the other
   side).
2. **Clearing** — `POST /trigger-clearing` on the Clearing Node after market
   closure. The caller is responsible for the timing; in the GSY DEX this is the
   Market Orchestrator. The node
   - returns the stored result if the market already has trades (idempotent);
   - fetches the open orders of the slot and rejects the slot if an area appears
     on both sides;
   - expires all orders if one side is empty (no trade, no transaction);
   - computes `p = K_upper − (K_upper − K_lower) / (1 + exp(−B · (x − Θ)))` with
     `x = supply / demand` and clamps it to `[K_lower, K_upper]`;
   - anchors supply, demand and price (scaled by 10,000) with `clearMarket`;
     trades are written only after the anchor exists;
   - serves mutual pairs first and splits the remaining quantity pro rata;
   - writes one trade per order against the community pool (identifier:
     BLAKE2b-256 of the content), with the uniform price in `energy_rate` and the
     origin-adjusted rate in `final_energy_rate`;
   - marks the orders as executed.
3. **Reconciliation** — `POST /trigger-execution` on the Execution and Penalty Node
   after delivery. The node reads meter readings (`/measurements`) and forecasts
   (`/forecasts`; without a forecast it falls back to the meter reading) and
   computes
   - the seller shortfall penalty `Φ = γ · K_upper · max(0, sold − delivered − η)`;
   - externality penalties on the short side of the market, priced with a
     counterfactual clearing price;
   - the proportional redistribution of the externality penalties to the harmed
     side.

   The results are written back into the trade `parameters`. No payment is made.

The contract stores one result per market and checks that the community exists, the
caller is a whitelisted Clearing Node, the market has not been cleared before and
the price lies within the registered bounds. It does not recompute the price.

---

## Requirements

| Tool | Version |
|---|---|
| Python | 3.11 or 3.12 |
| Node.js and npm | Node.js 20 LTS recommended for Hardhat 2.19.5; newer versions print a warning |
| Docker with Compose | optional, for the container stack |

Tested on October 28, 2026, at `63043bb` with fastapi 0.162.0, starlette 1.10.0,
pydantic 2.14.1, httpx 0.28.1, uvicorn 0.54.0 and, for live mode, web3 7.16.0. The
requirement files give version ranges, so a later install may resolve newer
versions.

---

## Quick start (mock mode)

The default stack needs no blockchain: the Clearing Node simulates the anchor and
labels the transaction hash as simulated.

**Docker**

```bash
docker compose up --build
```

**Windows, without Docker**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
powershell -ExecutionPolicy Bypass -File scripts\stop-local.ps1     # to stop
```

The start script creates `.venv` on its first run.

**Any OS, without Docker**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r amm-execution-node/requirements.txt    # covers all three services in mock mode

(cd mock-offchain-db   && python -m uvicorn src.main:app --port 8080) &
(cd amm-clearing-node  && python -m uvicorn src.main:app --port 8081) &
(cd amm-execution-node && python -m uvicorn src.main:app --port 8082) &
python scripts/serve_ui.py 3000
```

Then open **http://localhost:3000**. To check the running stack from the command
line, run `python scripts/e2e_demo.py`. It drives one full cycle, including a
preference round and a one-sided slot, and must end with
`ALL CHECKS PASSED`.

---

## Demo interface

The interface is a demonstration aid, not part of the artifact. Every number it
shows comes from the backend: the browser creates the market and the orders,
triggers the nodes and renders their responses.

| Panel | What it does |
|---|---|
| **A — Market Configuration** | price bounds `K_upper` and `K_lower`, midpoint `Θ` and steepness `B`; the price curve updates live |
| **B — Participant Input** | producers and consumers, each producer's energy type, each participant's preferred partner |
| **C — Preferences & Energy-Type Multipliers** | green bonus, gray levy and levy cap. Allocation order, multiplier mode and side come from the Clearing Node's configuration; after each run the panel shows the rules the node reports it applied |
| **D — Run Clearing** | creates the market and the orders in the off-chain storage and triggers the Clearing Node |
| **Clearing Results** | price, allocations with fill rates and final rates, the served pairs, levy and bonus, the trade objects, the (simulated) anchor transaction |
| **E — Post-Delivery Simulation** | enter delivered or consumed energy per participant, trigger the Execution and Penalty Node, inspect the penalties and counterfactual prices |

The interface sends its pricing and preference values with every trigger. This is a
convenience of the proof of concept: values sent with a trigger are not checked
against the contract.

The interface starts with the calibrated parameters of the thesis: `K_upper` = 40.0,
`K_lower` = 8.0, `Θ` = 1.0 and `B` = 0.6. The default market has three producers
(12.5 kWh: PV A 5.0, PV B 3.5, battery 4.0, gray) and three consumers (10 kWh).
PV A and Household 1 nominate each other.

| Scenario | How | What you see |
|---|---|---|
| Demand-limited round | defaults | ratio 1.25, price **22.80 ct/kWh**, 10 kWh traded; without the pair every producer is filled to 80 % |
| Mutual preferred pair | defaults | the pair's 4.5 kWh is served first: PV A is filled to **96.9 %**, PV B and the battery to 68.75 %; the market still clears 10 kWh |
| One-sided nomination | clear Household 1's partner | no pair, since priority needs both sides; pure pro-rata allocation |
| Allocation-order baseline | restart the Clearing Node with `PREFERENCE_ORDER=pro_rata_first` | the pair is still detected and flagged, but the quantities are pro rata |
| Origin adjustment, multiplicative | defaults (bonus 0.10, levy 0.10) | the levy funds 37.9 % of the requested bonus; green **23.67**, gray **20.52 ct/kWh**; buyers pay what sellers receive (228.02 ct) |
| Pool surplus | set the green multiplier to 0.03 | the bonus is paid in full and the pool keeps **1.31 ct**: the multiplicative mode is not zero-sum, and the surplus is reported |
| Origin adjustment, additive | restart with `MULTIPLIER_MODE=additive` | the levy follows from the bonus and hits its 20 % cap here; green **24.53**, gray **18.24 ct/kWh**, zero-sum |
| Supply-limited round | lower producer energy below total demand | consumers are rationed, the price moves toward `K_upper` |
| Seller shortfall | panel E: a producer delivers less than allocated | `Φ = γ · K_upper` = 44 ct per missing kWh (shipped configuration: γ = 1.1, no deadband) |
| Seller withholding | supply-limited round, a producer delivers more than allocated | seller externality penalty with its counterfactual price |
| Buyer under-reporting | demand-limited round, a consumer uses more than allocated | buyer externality penalty |
| No-trade slot | remove all producers or all consumers | orders expired, no transaction |

Numbers computed with the services at `63043bb`.

---

## Configuration

Each node reads `configuration.yaml` in its directory. Environment variables
override it.

**Clearing Node** ([`amm-clearing-node/configuration.yaml`](amm-clearing-node/configuration.yaml))

| Variable | Default | Meaning |
|---|---|---|
| `OFFCHAIN_DB_URL` | `http://localhost:8080` | off-chain storage |
| `BLOCKCHAIN_MODE` | `mock` | `mock` simulates the anchor, `live` sends `clearMarket` transactions |
| `RPC_URL`, `CONTRACT_ADDRESS`, `CLEARING_NODE_PRIVATE_KEY` | — | live mode only; the key of a whitelisted account |
| `MIN_PRIORITY_FEE_WEI` | `1000000000` | floor for the EIP-1559 priority fee, needed on mostly empty proof-of-authority chains such as Volta |
| `TIME_SLOT_SEC` | `900` | slot length |
| `PREFERENCES_ENABLED`, `PREFERENCE_ORDER` | `true`, `preferences_first` | partner priority; `pro_rata_first` is the comparison baseline |
| `MULTIPLIERS_ENABLED`, `MULTIPLIER_MODE`, `MULTIPLIER_SIDES` | `true`, `multiplicative`, `seller` | origin adjustment; `additive`, and `both` for adjusting buyer rates too |
| `GREEN_MULTIPLIER`, `GREY_LEVY`, `LEVY_CAP` | `0.10`, `0.10`, `0.20` | bonus rate, levy rate, cap of the levy rate |

The pricing parameters are set per community in `configuration.yaml` (`communities`,
with `default_community` for any other id): `K_upper` 40.0 and `K_lower` 8.0 ct/kWh,
`Θ` 1.0, `B` 0.6.

**Execution and Penalty Node** ([`amm-execution-node/configuration.yaml`](amm-execution-node/configuration.yaml))

| Variable | Default | Meaning |
|---|---|---|
| `OFFCHAIN_DB_URL` | `http://localhost:8080` | off-chain storage |
| `PENALTY_GAMMA` | `1.1` | γ in `K_sho = γ · K_upper` |
| `PENALTY_ETA_RELATIVE` | unset | deadband relative to the traded quantity; the thesis uses **0.10** |
| `PENALTY_TOLERANCE_KWH` | `0.0` | absolute deadband, used while no relative one is set |
| `TIME_SLOT_SEC`, `EXECUTION_OFFSET_MIN` | `900`, `-120` | slot length; offset for the polling mode |
| `POLLING_ENABLED`, `POLLING_INTERVAL_SEC` | `false`, `300` | polling loop, off by default (the communities are listed in `poll_communities` in the YAML) |

The shipped configuration has **no deadband**. To run the reference setting of the
thesis, set `PENALTY_ETA_RELATIVE=0.10`.

**Order fields** — partners are identified by `area_uuid`, never by the free-text
`created_by`:

```jsonc
{
  "attributes":   { "energy_type": "green" },            // "green" | "grey"
  "requirements": { "preferred_partner": "area-house-1" }
}
```

Orders without these fields clear as green orders without a partner.

---

## Live blockchain mode

Live mode runs from a local Python environment. The Docker image of the Clearing
Node installs `requirements.txt` only, without `web3`, so it cannot start in live
mode.

**1 · Contract** (local Hardhat network)

```bash
cd amm-smart-contract
npm ci                                   # once
npx hardhat test                         # 15 tests
npx hardhat node                         # terminal 1: local chain with funded accounts
npx hardhat run scripts/deploy.js --network localhost      # terminal 2
```

`deploy.js` deploys `AMMBA`, whitelists `CLEARING_NODE_ADDRESS` (default: the
deployer) and registers `COMMUNITY_UUID` (default `communityid_1`) with the
parameters of `configuration.yaml`, scaled by 10,000. It prints the address and the
transaction hashes. Community ids that are not 32-byte hex strings are stored as
`keccak256` of the string, the same convention as the Clearing Node.

**2 · Clearing Node**

```bash
pip install -r amm-clearing-node/requirements-live.txt     # adds web3
export BLOCKCHAIN_MODE=live
export RPC_URL=http://127.0.0.1:8545
export CONTRACT_ADDRESS=<address printed by deploy.js>
export CLEARING_NODE_PRIVATE_KEY=<key of the whitelisted account, by default account #0 of the Hardhat node>
cd amm-clearing-node && python -m uvicorn src.main:app --port 8081
```

At start-up the node compares the parameters of every community in its
configuration with those registered on-chain and refuses to start on a mismatch.
Parameters sent with a trigger are not checked.

**3 · Run a cycle**

```bash
python scripts/e2e_demo.py --community communityid_1 --expect-mode live
```

Increase `--slot-offset` for every further run in the same 15-minute window. The
market id is derived from the slot, and the contract accepts each market only once.
The script sends its own pricing parameters with each trigger (`K_upper` = 28.5,
`B` = 2.5, the reference example it checks against).

The demo interface derives its community id from the community name
(`Community 1` → `community-community-1`). Register that id on-chain with
`scripts/register_community.js` before using the interface in live mode.

**Volta (Energy Web test network)** — Hardhat does not read a `.env` file, so set the
keys in the shell:

- `DEPLOYER_PRIVATE_KEY` for `--network volta`;
- `VOLTA_RPC_URL` if not `https://volta-rpc.energyweb.org`;
- `RPC_URL=https://volta-rpc.energyweb.org` for the node, and `--timeout 120` for
  `e2e_demo.py`.

Further scripts in `amm-smart-contract/scripts/`:

| Script | Purpose |
|---|---|
| `register_community.js` | register another community |
| `volta_evidence.js` | exercises the contract's checks by script: a no-trade call, a clearing, an unauthorized caller, a second clearing of the same market, a price out of bounds |
| `read_anchor.js` | read anchored results back next to the off-chain trades |
| `gas_series.js` | the gas series of the thesis, on the in-process Hardhat network |

### Volta deployment cited in the thesis

| | |
|---|---|
| Contract | [`0xA532e4280F990F28845300cFF1798054b949A938`](https://volta-explorer.energyweb.org/address/0xA532e4280F990F28845300cFF1798054b949A938) on Volta (chain ID 73799) |
| Deployed | October 16, 2026, from commit `63043bb`; source verified on the block explorer |
| Transactions | 16, listed in Table A1 of the thesis: deployment and set-up, the contract checks, six clearings sent by the Clearing Node in live mode, and the closing calls |
| State | the Clearing Node was revoked and ownership moved to an address without a known key, so the contract accepts no further results |

---

## Tests

```bash
cd amm-clearing-node  && python -m pytest -q
cd amm-execution-node && python -m pytest -q
cd mock-offchain-db   && python -m pytest -q
cd evaluation         && python -m pytest -q    # needs evaluation/requirements.txt
cd amm-smart-contract && npx hardhat test
```

State on October 28, 2026, at `63043bb` (Python 3.11 and 3.12):

| Suite | Result |
|---|---|
| `amm-clearing-node` | 127 passed, 8 skipped (the live-mode tests are skipped without `web3`); 135 passed with `requirements-live.txt` |
| `amm-execution-node` | 48 passed |
| `mock-offchain-db` | 21 passed |
| `evaluation` | 175 passed, 1 skipped (needs a run directory that is not in the repository) |
| `amm-smart-contract` | 15 passing |

---

## Reading the code comments

The comments record where a design choice came from. Three kinds of reference
point outside the repository:

| In a comment | Refers to |
|---|---|
| `guide §x.y` | an integration guide for the GSY DEX side that was available to the author during development but is not published. The published reference for the interfaces is the GSY DEX source at commit `aa99ea2`, which the comments cite where it matters |
| `D-xx` | an entry in the decision log kept while writing the thesis; the decisions that shape the design are argued in Chapter 4 of the thesis |
| `#xx`, `issue #xx` | an entry in the thesis' own issue list, not a GitHub issue |

Naming: the code and its API use `grey`, the thesis uses the American spelling
*gray*. `Market Orchestrator` is the GSY DEX component that opens and closes markets
and would call the two triggers.

---

## Limitations of the prototype

These are declared in the thesis (Sections 4.4–4.7 and 6.5) and not fixed here:

- The triggers are unauthenticated, and pricing and preference parameters sent with
  a trigger are not checked against the contract. The start-up check covers only the
  communities in the configuration, and only in live mode.
- Neither node checks the clock: clearing after market closure and reconciliation
  after delivery are up to the caller, and a repeated reconciliation overwrites the
  stored results.
- Defaults for missing data are lenient: a missing meter reading counts as delivery
  as allocated, and an order without an energy type counts as green. The energy type
  is self-declared.
- A slot in which one area posts on both sides is rejected as a whole.
- Anchoring is not monitored, and the anchor holds aggregates, not a hash of the
  records.
- The two nodes classify a window as supply- or demand-limited at different
  precision (unrounded totals versus totals rounded to six decimals).
- Payments are computed, not executed. A buyer is not compensated for energy that is
  not delivered.
- The published GSY DEX storage has no routes for the order status after clearing or
  for the reconciliation results of a trade. The mock offers both, and against a
  storage without them the nodes skip these writes.

---

## Background and citation

- Mechanism: Celig, T., Talari, S., Ketter, W., & Schoder, D. (2024). Automated
  market maker-based batch auction for decentralized customer-centric local energy
  sharing. *2024 IEEE PES ISGT Europe*. https://doi.org/10.1109/ISGTEUROPE62998.2024.10863799
- Penalty rules: Saber, H., Celig, T., Talari, S., Ketter, W., & Schoder, D. (2026).
  An incentivized automated market maker-based batch auction mechanism for
  network-aware local energy markets with loss allocation. *Applied Energy, 416*,
  127957. https://doi.org/10.1016/j.apenergy.2026.127957
- Data model and interfaces: Grid Singularity, *gsy-decentralized-exchange*,
  commit `aa99ea2`. https://github.com/gridsingularity/gsy-decentralized-exchange
- Evaluation data: Faia, R., Gonçalves, C., Gomes, L., & Vale, Z. (2023). *Data in
  Brief, 48*, 109218. https://doi.org/10.1016/j.dib.2023.109218 (CC BY 4.0, see
  [`evaluation/data/PROVENANCE.md`](evaluation/data/PROVENANCE.md))

To cite this artifact:

> Teutenberg, F. (2026). *Design and implementation of a hybrid post-price clearing
> mechanism for decentralized energy markets* [Master's thesis, University of
> Cologne]. Source code: https://github.com/FynnLT/AMMBA_Fynn
