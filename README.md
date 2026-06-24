# AMMBA — Automated Market Maker for Batch Auctions (PoC)

Proof-of-concept implementation of a P2P energy market clearing mechanism for
a Master's thesis. Each 15-minute market slot collects energy **bids**
(buyers) and **offers** (sellers); at market close a uniform clearing price is
derived from the supply/demand ratio via a **sigmoid function**, quantities
are allocated **pro-rata**, and the aggregated result is anchored on an EVM
chain for auditability. After delivery, deviations between traded and metered
energy are penalized (shortfall + VCG-style externality penalties).


| Component | Tech | Port | Status |
|---|---|---|---|
| [mock-offchain-db](mock-offchain-db/) | Python · FastAPI | 8080 | **Mock** of the GSY-DEX off-chain DB |
| [amm-clearing-node](amm-clearing-node/) | Python · FastAPI | 8081 | Implemented |
| [amm-execution-node](amm-execution-node/) | Python · FastAPI | 8082 | Implemented |
| [amm-smart-contract](amm-smart-contract/) | Solidity · Hardhat | — | Implemented + tested; *simulated* in the default stack |
| [ui](ui/) | HTML/JS/CSS (no build step) | 3000 | Simulation & visualization for thesis demos |

---

## Quick start

```bash
docker-compose up --build
```

Then open **http://localhost:3000** and:

1. **A — Market Configuration**: adjust `K_upper`, `K_lower`, `θ`, `B`; the
   sigmoid price curve updates live.
2. **B — Participants**: add/remove producers and consumers. The current
   supply/demand ratio and the resulting clearing price preview update live.
3. **C — Run Clearing**: creates the market + orders in the off-chain DB and
   triggers the Clearing Node (the calls the Market Orchestrator would make).
4. **D — Results**: clearing price, sigmoid intersection, pro-rata
   allocations with fill rates and revenue/cost, the generated trade objects
   (blake2b-256 ids) and the simulated on-chain anchor tx hash.
5. **E — Post-Delivery Simulation**: edit "actual delivered/consumed" per
   participant (pre-filled with the allocation), run the Execution Node, and
   inspect shortfall/externality penalties incl. counterfactual prices.

Suggested demo scenarios:

| Scenario | How | What you see |
|---|---|---|
| Demand-limited round | default participants (12.5 kWh supply vs 10 kWh demand) | producers filled 80 %, price ≈ 15.15 ct/kWh |
| Supply-limited round | lower a producer's energy below total demand | consumers rationed, price rises toward `K_upper` |
| Seller shortfall | in panel E, set a producer's *actual* below its allocation | `Φ = γ·K_upper·shortfall` penalty |
| Seller withholding | supply-limited round + producer's *actual* above allocation | externality penalty + counterfactual price chart |
| Buyer underreporting | demand-limited round + consumer's *actual* above allocation | buyer externality penalty |
| No-trade slot | remove all producers (or all consumers) | orders expired, no-trade result |

Each "Run Clearing" creates a fresh market slot, so you can iterate freely.

### Running without Docker

**Windows one-command start/stop** (creates the venv on first run, opens each
service in a minimized console window, waits for health):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1   # start everything
powershell -ExecutionPolicy Bypass -File scripts\stop-local.ps1    # stop everything
```

All state is in-memory, so **restarting is a full reset**. To wipe data
without restarting: `curl.exe -X POST http://127.0.0.1:8080/reset`
(mock-only endpoint).

Manual equivalent (any OS):

```bash
# terminal 1 — mock off-chain DB
cd mock-offchain-db && pip install -r requirements.txt
uvicorn src.main:app --port 8080

# terminal 2 — clearing node
cd amm-clearing-node && pip install -r requirements.txt
uvicorn src.main:app --port 8081

# terminal 3 — execution node
cd amm-execution-node && pip install -r requirements.txt
uvicorn src.main:app --port 8082

# terminal 4 — UI (any static server)
cd ui && python -m http.server 3000
```

---

## ⚠️ What is mocked / not yet implemented and why

| What | Why | Where |
|---|---|---|
| **GSY-DEX off-chain database** | The real service does not exist yet. Replaced by `mock-offchain-db`, an in-memory FastAPI service implementing the exact REST interface from §2 (`/health_check`, `/market`, `/community-market`, `/orders`, `/orders-normalized`, `/trades`, `/trades-normalized`, `/asset_measurements`). The Clearing/Execution Nodes only know `OFFCHAIN_DB_URL`, so swapping in the real DB requires **zero core-logic changes**. | [mock-offchain-db/src/main.py](mock-offchain-db/src/main.py) |
| **On-chain anchoring (default stack)** | The compose stack runs without a blockchain node, so `BLOCKCHAIN_MODE=mock` simulates the `clearMarket` call and returns a simulated tx hash (labelled *simulated* in the UI). The Solidity contract itself **is** implemented and fully tested with Hardhat; switch to a real chain via [Live blockchain mode](#live-blockchain-mode). | [amm-clearing-node/src/contract.py](amm-clearing-node/src/contract.py), [amm-smart-contract/](amm-smart-contract/) |
| **User preferences (`requirements`/`attributes`)** | Not yet provided by GSY DEX (§7.3). Orders tolerate their absence (`None`); preference matching + energy-type multipliers are stubs marked `TODO(phase2)`. | [amm-clearing-node/src/preferences.py](amm-clearing-node/src/preferences.py) |
| **Order status updates (`PATCH /orders/{id}`)** | Unknown whether the production orderbook exposes this (§7.2) — in GSY-DEX statuses are updated by a blockchain event listener, which doesn't fit the AMM's single `MarketCleared` event. Implemented in the mock DB; marked `TODO(confirm-with-supervisor)` for production. | [amm-clearing-node/src/offchain_db.py](amm-clearing-node/src/offchain_db.py) |
| **Penalty output schema** | Unresolved (§7.5). PoC extends the trade `parameters` field via a mock-only `PATCH /trades/{trade_uuid}`; marked `TODO(confirm-with-supervisor)`. | [amm-execution-node/src/execution.py](amm-execution-node/src/execution.py) |
| **Market Orchestrator** | Separate system; the demo UI plays its role (creates markets, picks `time_slot`, triggers clearing). The mock DB generates `market_id = blake2b("spot" + timestamp)` on its behalf when omitted. | [ui/app.js](ui/app.js), [mock-offchain-db/src/store.py](mock-offchain-db/src/store.py) |
| **Energy Web Digital Spine** | Out of scope entirely (per constraints; pending the EW integration meeting, §7.6–7.7). | — |
| **Execution polling loop** | Implemented (§5.2 pattern) but **disabled by default** (`POLLING_ENABLED=false`): the demo simulates delivery time, so the UI triggers `POST /trigger-execution` explicitly. | [amm-execution-node/src/main.py](amm-execution-node/src/main.py) |

All open questions from §7 are marked in code as `TODO(phase2)` or
`TODO(confirm-with-supervisor)`:

```bash
grep -rn "TODO(phase2)\|TODO(confirm-with-supervisor)" --include="*.py" .
```

---

## Architecture

```
                         ┌──────────────────────────┐
                         │  Demo UI (localhost:3000)│   green = implemented
                         │  = Market Orchestrator + │   grey  = mocked
                         │    participants          │
                         └────┬──────────┬──────────┘
        POST /trigger-clearing│          │POST /trigger-execution
                              │          │      + POST /market /orders /asset_measurements
                  ┌───────────▼───┐  ┌───▼───────────────┐
                  │ AMM Clearing  │  │ AMM Execution     │
                  │ Node  :8081   │  │ Node  :8082       │
                  └───┬───────┬───┘  └───────┬───────────┘
       clearMarket()  │       │ GET /orders  │ GET /trades, /asset_measurements
       (simulated in  │       │ POST /trades │ PATCH /trades (penalties)
        mock mode)    │       │              │
                  ┌───▼───┐ ┌─▼──────────────▼──┐
                  │ AMMBA │ │ Mock Off-Chain DB │
                  │  .sol │ │ :8080 (in-memory) │
                  └───────┘ └───────────────────┘
```

**Clearing cycle** (§4.4): fetch `Open` orders for the slot → aggregate →
`price = K_upper − (K_upper − K_lower)/(1 + e^(−B·(ratio−θ)))`, clamped to
`[K_lower, K_upper]` → anchor `(supply, demand, price)` on-chain → pro-rata
allocation → two-legged pool trades (buyer→pool, pool→seller; one trade per
participant) with blake2b-256 ids → `POST /trades-normalized` → orders
`Executed`. Clearing is idempotent per `market_id` + `time_slot` (§4.7).

**Execution cycle** (§5): recover the clearing context from trade
`parameters` (self-contained, §5.4) → fetch measurements → penalties:

- seller shortfall `Φ = γ·K_upper · max(0, traded − delivered − η)`
- seller externality (supply-limited rounds): counterfactual price with the
  withheld energy added to supply
- buyer externality (demand-limited rounds): counterfactual price with the
  underreported demand added

**Scaling**: all on-chain values are integers × `10000`
(`NODE_FLOAT_SCALING_FACTOR`, §3.2), e.g. `28.5 ct/kWh → 285000`.

---

## Smart contract

The sigmoid runs **off-chain** (recommended approach, §3.3): the contract
verifies `K_lower ≤ price ≤ K_upper` against per-community parameters, stores
the anonymized aggregates immutably and emits `MarketCleared`. Only
whitelisted clearing-node addresses may call `clearMarket`; a market id can
only be cleared once.

```bash
cd amm-smart-contract
npm install
npm test                  # 15 Hardhat tests
```

> Dependency pins are Node 16-compatible (individual Hardhat plugins instead
> of `@nomicfoundation/hardhat-toolbox`, whose floating peers require
> Node ≥ 18). Works on Node 16–20.

### Live blockchain mode

```bash
# 1. local chain
cd amm-smart-contract && npm run node          # hardhat node on :8545

# 2. deploy + authorize the clearing node (prints CONTRACT_ADDRESS)
npm run deploy:local

# 3. point the stack at it — create .env (see .env.example):
#    BLOCKCHAIN_MODE=live
#    RPC_URL=http://host.docker.internal:8545
#    CONTRACT_ADDRESS=0x...           (from step 2)
#    CLEARING_NODE_PRIVATE_KEY=0x...  (a hardhat dev key / the whitelisted account)
docker-compose up --build
```

For the Energy Web **Volta** testnet: `npm run deploy:volta` with
`DEPLOYER_PRIVATE_KEY` set, then `RPC_URL=https://volta-rpc.energyweb.org`
(§3.6, §6). The community-uuid → `bytes32` conversion (keccak256 of the
UTF-8 string) is shared between `scripts/deploy.js` and
`amm-clearing-node/src/contract.py`.

---

## Tests

```bash
# Python services (62 tests)
cd mock-offchain-db   && pip install -r requirements.txt httpx pytest && pytest
cd amm-clearing-node  && pip install -r requirements.txt pytest       && pytest
cd amm-execution-node && pip install -r requirements.txt pytest       && pytest

# Solidity (15 tests)
cd amm-smart-contract && npm install && npm test
```

With the stack running (`docker-compose up` or the three uvicorn services),
[scripts/e2e_demo.py](scripts/e2e_demo.py) drives a complete
clearing + execution cycle over REST and asserts the guide's reference
numbers (price 15.1472 ct/kWh, shortfall 31.35 ct, externality 25.69 ct):

```bash
pip install httpx && python scripts/e2e_demo.py
```

The clearing/execution suites run the full cycle against in-memory fakes of
the off-chain DB (§9 step 5) and cover: the guide's reference scenario
(12.5/10 kWh → 15.147 ct/kWh), supply-/demand-limited rounds, no-trade slots,
idempotent re-triggers, residual handling, blake2b hashing, scaling, penalty
formulas with hand-computed reference values, and round-type gating of
externality penalties.

---

## Configuration

Both nodes load `configuration.yaml` with **environment variables taking
precedence** (§4.3). Key variables:

| Variable | Service | Default | Meaning |
|---|---|---|---|
| `OFFCHAIN_DB_URL` | both | `http://localhost:8080` | off-chain DB base URL |
| `BLOCKCHAIN_MODE` | clearing | `mock` | `mock` (simulated anchor) or `live` |
| `RPC_URL`, `CONTRACT_ADDRESS`, `CLEARING_NODE_PRIVATE_KEY` | clearing | — | live mode only |
| `TIME_SLOT_SEC` | both | `900` | delivery slot length (§8) |
| `EXECUTION_OFFSET_MIN` | execution | `-120` | delay before a slot is executed (§5.2) |
| `POLLING_ENABLED`, `POLLING_INTERVAL_SEC` | execution | `false`, `300` | production-style polling loop |
| `PENALTY_GAMMA` | execution | `1.1` | `K_sho = γ·K_upper` |
| `PENALTY_TOLERANCE_KWH` | execution | `0.0` | shortfall tolerance `η` |

Per-community sigmoid parameters: `amm-clearing-node/configuration.yaml`
(`communities:`); unknown communities fall back to `default_community`, and
the demo UI passes its panel-A parameters with the trigger (PoC convenience,
`TODO(confirm-with-supervisor)` for production governance).

---

## PoC design decisions (deviations worth knowing)

- **`/trigger-clearing` returns `200` + the full result** instead of `202
  Accepted`: §4.2 allows synchronous processing, and the demo UI renders the
  response directly. Production would switch to 202 + background task.
- **`k_upper`/`k_lower` are added to trade `parameters`** (not in the §2
  schema): §5.4 requires them for the Execution Node's counterfactuals; this
  keeps trades self-contained. Same for `pool_id` (lets the Execution Node
  identify the pool side without configuration).
- **Matched orders are marked `Executed` even on partial fills**, with the
  residual recorded on the trade object — §7.8 is open; revisit with Spyros.
- **Re-running execution recomputes penalties** (overwrites the previous
  result) — deliberate for demos; production idempotency policy TBD.
- **`market_id` timestamp encoding** assumed `u64 little-endian` in
  `blake2b("spot" + timestamp_bytes)` — `TODO(confirm-with-supervisor)`.

## Open questions (§7) → code markers

| § | Item | Marker location |
|---|---|---|
| 7.1 | pool naming / registration as area | `amm-clearing-node/src/config.py` |
| 7.2 | order-status PATCH endpoint | `amm-clearing-node/src/offchain_db.py`, mock `main.py` |
| 7.3 | user preferences API | `amm-clearing-node/src/preferences.py` |
| 7.4 | energy-type multiplier storage | `amm-clearing-node/src/preferences.py` |
| 7.5 | penalty output schema | `amm-execution-node/src/execution.py`, `src/offchain_db.py` |
| 7.6/7.7 | EW Digital Spine / ontology | out of scope (README only) |
| 7.8 | partial-order residual policy | `amm-clearing-node/src/trade_builder.py`, `src/clearing.py` |

## Repository layout

```
├── docker-compose.yml          # 4 services, one shared network
├── .env.example                # live-blockchain settings (optional)
├── docs/IMPLEMENTATION_GUIDE.md
├── mock-offchain-db/           # in-memory GSY-DEX off-chain DB stand-in
├── amm-smart-contract/         # Solidity AMMBA.sol + Hardhat tests + deploy
├── amm-clearing-node/          # sigmoid clearing, pro-rata, trade builder
├── amm-execution-node/         # penalties: shortfall + externalities
└── ui/                         # single-page demo (panels A–F)
```
