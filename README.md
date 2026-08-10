# AMMBA — Automated Market Maker for Batch Auctions (PoC)

Proof-of-concept implementation of a P2P energy market clearing mechanism for
a Master's thesis. Each 15-minute market slot collects energy **bids**
(buyers) and **offers** (sellers); at market close a uniform clearing price is
derived from the supply/demand ratio via a **sigmoid function**, quantities
are allocated **pro-rata**, and the aggregated result is anchored on an EVM
chain for auditability. Two user-preference mechanisms sit on top of the
allocation: **preferred trading partners** (mutually nominated pairs are
served before the pro-rata residual) and **energy-type multipliers** (a green
bonus funded by a grey levy). After delivery, deviations between traded and
metered energy are penalized (shortfall + VCG-style externality penalties).


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
2. **B — Participants**: add/remove producers and consumers, pick each
   producer's energy source (green/grey) and each side's preferred trading
   partner. The supply/demand ratio and the clearing price preview update live.
3. **C — Preferences & Energy-Type Multipliers**: the green bonus, grey levy
   and levy cap are adjustable and travel with the trigger. The *rules* —
   allocation order, multiplier formulation, and which side the multiplier is
   applied to — are fixed for the demo at `preferences_first` /
   `multiplicative` / `seller` and are not selectable in the UI; the panel
   shows a read-only line with what the Clearing Node reported it applied.
   The Clearing Node computes both mechanisms; nothing is calculated in the
   browser.
4. **D — Run Clearing**: creates the market + orders in the off-chain DB and
   triggers the Clearing Node (the calls the Market Orchestrator would make).
5. **Results**: clearing price, sigmoid intersection, allocations with fill
   rates and final per-kWh rates, which pairs were served preferentially, the
   levy/bonus balance, the generated trade objects (blake2b-256 ids) and the
   simulated on-chain anchor tx hash.
6. **E — Post-Delivery Simulation**: edit "actual delivered/consumed" per
   participant (pre-filled with the allocation), run the Execution Node, and
   inspect shortfall/externality penalties incl. counterfactual prices.

Suggested demo scenarios:

| Scenario | How | What you see |
|---|---|---|
| Demand-limited round | default participants (12.5 kWh supply vs 10 kWh demand) | producers filled 80 %, price ≈ 15.15 ct/kWh |
| Supply-limited round | lower a producer's energy below total demand | consumers rationed, price rises toward `K_upper` |
| Mutual preferred pair | default (PV A ↔ Household 1 select each other) | the pair's 4.5 kWh is served first: PV A fills to **96.9 %** instead of the 80 % pro-rata baseline, the market still balances at 10 kWh |
| One-sided preference | clear Household 1's partner, keep PV A's | no pair — a match needs both sides; allocation returns to pure pro-rata |
| Allocation-order comparison | restart with `PREFERENCE_ORDER=pro_rata_first` | the pair is still detected and flagged, but quantities are the pro-rata ones — the comparison baseline for the evaluation; panel C's status line reports the change |
| Green bonus, self-funded | default multipliers (0.10 / 0.10) | grey levy funds 37.9 % of the requested bonus; green 15.72, grey 13.63 ct/kWh; buyers pay = sellers receive |
| Pool surplus (levy over-collects) | set the green multiplier to 0.03 | bonus paid in full, buyers still pay the uniform price, and the pool **keeps** the difference — the multiplicative formulation is not zero-sum, and the surplus is reported rather than hidden |
| Zero-sum alternative | restart with `MULTIPLIER_MODE=additive` | the levy follows from funding the bonus; no pool surplus for any parameter combination |
| Seller shortfall | in panel E, set a producer's *actual* below its allocation | `Φ = γ·K_upper·shortfall` penalty |
| Seller withholding | supply-limited round + producer's *actual* above allocation | externality penalty + counterfactual price chart |
| Buyer underreporting | demand-limited round + consumer's *actual* above allocation | buyer externality penalty |
| No-trade slot | remove all producers (or all consumers) | orders expired, no-trade result |

### Preference configuration

Defaults live in [amm-clearing-node/configuration.yaml](amm-clearing-node/configuration.yaml)
and can be overridden by environment variables (`PREFERENCE_ORDER`,
`MULTIPLIER_MODE`, `MULTIPLIER_SIDES`, `GREEN_MULTIPLIER`, `GREY_LEVY`,
`LEVY_CAP`, …) or per request via `preference_params` in the
`/trigger-clearing` payload — the same PoC convenience as `sigmoid_params`:

```yaml
preferences:
  enabled: true
  order: "preferences_first"        # "preferences_first" | "pro_rata_first"
  multipliers:
    enabled: true
    mode: "multiplicative"          # "multiplicative" (Guide) | "additive" (InfoPaper)
    sides: "seller"                 # "seller" | "both"
    green_multiplier: 0.10
    grey_levy: 0.10
    levy_cap: 0.20
```

The demo UI sends only `green_multiplier`, `grey_levy` and `levy_cap`; `order`,
`mode` and `sides` come from the configuration alone, so switching variant is a
config change rather than a code change (supervisor question **B-04** is still
open, and evaluation block 1 compares exactly these variants):

```bash
PREFERENCE_ORDER=pro_rata_first docker-compose up      # pro-rata baseline
MULTIPLIER_MODE=additive docker-compose up             # InfoPaper formulation
```

Panel C echoes back what the Clearing Node reported it applied, so a changed
configuration is visible in the UI instead of silently diverging from it.

Orders express their preferences with the two optional GSY-DEX order fields;
**partners are identified by `area_uuid`**, never by the free-text
`created_by` name:

```jsonc
{
  "attributes":   { "energy_type": "green" },            // "green" | "grey"
  "requirements": { "preferred_partner": "area-house-1" }
}
```

Orders without them clear exactly as before. The uniform clearing price stays
in `parameters.energy_rate` (the Execution Node computes every counterfactual
penalty from it); the multiplier result is added alongside as
`parameters.final_energy_rate`.


To stop

```bash
powershell -ExecutionPolicy Bypass -File scripts\stop-local.ps1
```