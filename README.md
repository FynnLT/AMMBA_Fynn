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
powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
```

Then open **http://localhost:3000** and:

1. **A — Market Configuration**: adjust `K_upper`, `K_lower`, `θ`, `B`; the
   sigmoid price curve updates live.
2. **B — Participants**: add/remove producers and consumers and their preferences. The current
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


To stop

```bash
powershell -ExecutionPolicy Bypass -File scripts\stop-local.ps1
```