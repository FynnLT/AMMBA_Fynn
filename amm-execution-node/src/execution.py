"""AMM Execution Node cycle (guide §5).

After the delivery period, fetch settled trades plus the two measurement
channels — meter readings (`/measurements`) for what was delivered and
forecasts (`/forecasts`) for what could have been delivered — compute
deviation penalties and write the results back to the off-chain DB.

The clearing result (totals, price, sigmoid parameters) is recovered entirely
from the trade objects' `parameters` field, so the Execution Node is
self-contained from trades + measurements alone (author's decision).
"""

import asyncio
import logging
import time

from src.config import Config
from src.offchain_db import OffchainDBClient, OffchainDBError
from src.penalties import (BALANCED, DEMAND_LIMITED, SUPPLY_LIMITED,
                           buyer_externality_penalty, determine_round_type,
                           redistribution, seller_externality_penalty,
                           seller_shortfall_penalty)

logger = logging.getLogger("amm-execution-node.execution")

_POOL_PREFIX = "AMM_POOL_"

# Result-row fields written back into trade `parameters`, a declared extension
# of the published GSY interface (D-48, see `OffchainDBClient.update_trade`).
_PENALTY_PARAM_KEYS = ("actual_kwh", "measurement_found", "deliverable_kwh",
                       "deliverable_source", "shortfall_kwh",
                       "shortfall_penalty_ct", "externality_kwh",
                       "externality_penalty_ct", "total_penalty_ct")


def compute_previous_timeslot(time_slot_sec: int, execution_offset_min: int,
                              now: float | None = None) -> int:
    """Most recently completed slot older than |EXECUTION_OFFSET_MIN| minutes
    (guide §5.2) — delivery is over and measurements should be available."""
    now = time.time() if now is None else now
    cutoff = now - abs(execution_offset_min) * 60
    return int(cutoff // time_slot_sec) * time_slot_sec


def _participant_from_trade(trade: dict) -> dict | None:
    """Identify the non-pool side of a trade.

    Every AMM trade has the community pool on exactly one side; `pool_id` is
    embedded in trade `parameters` by the Clearing Node (prefix fallback for
    robustness against foreign trades).
    """
    params = trade.get("parameters") or {}
    pool_id = params.get("pool_id", "")
    buyer, seller = trade.get("buyer", ""), trade.get("seller", "")

    def is_pool(name: str) -> bool:
        return (name == pool_id and bool(pool_id)) or name.startswith(_POOL_PREFIX)

    traded = float(params.get("selected_energy", 0.0))
    if is_pool(seller) and not is_pool(buyer):
        component = (trade.get("bid") or {}).get("bid_component") or {}
        residual = (trade.get("residual_bid") or {}).get("energy", 0.0)
        return {"name": buyer, "role": "buyer",
                "area_uuid": component.get("area_uuid"),
                "traded_kwh": traded,
                # original bid volume = allocation + residual
                "reported_kwh": traded + float(residual)}
    if is_pool(buyer) and not is_pool(seller):
        component = (trade.get("offer") or {}).get("offer_component") or {}
        residual = (trade.get("residual_offer") or {}).get("energy", 0.0)
        return {"name": seller, "role": "seller",
                "area_uuid": component.get("area_uuid"),
                "traded_kwh": traded,
                "reported_kwh": traded + float(residual)}
    logger.warning("trade %s has no identifiable pool side — skipped",
                   trade.get("trade_uuid"))
    return None


def _apply_externality(row: dict, externality: dict | None,
                       kwh_key: str) -> None:
    """Copy an externality-penalty result into the participant row.

    `kwh_key` names the deviation field in the penalty result
    ("withheld_kwh" for sellers, "underreported_kwh" for buyers)."""
    if externality is None:
        return
    row["externality_kwh"] = externality[kwh_key]
    row["externality_penalty_ct"] = externality["penalty_ct"]
    row["counterfactual_price_ct_per_kwh"] = \
        externality["counterfactual_price_ct_per_kwh"]
    row["counterfactual_ratio"] = externality["counterfactual_ratio"]


async def run_execution(trigger: dict, cfg: Config,
                        db: OffchainDBClient) -> dict:
    market_id = trigger["market_id"]
    community_uuid = trigger["community_uuid"]
    time_slot = int(trigger["time_slot"])

    logger.info("execution triggered: market=%s community=%s slot=%s",
                market_id, community_uuid, time_slot)

    trades = [t for t in await db.get_trades(market_id)
              if t.get("time_slot") == time_slot]
    if not trades:
        return {"status": "no_trades",
                "message": "no settled trades found for this market/slot",
                "market_id": market_id, "time_slot": time_slot, "results": []}

    # Clearing context, recovered from any trade's parameters (guide §5.4).
    params = trades[0].get("parameters") or {}
    sigmoid = {key: float(params.get(key)) for key in
               ("k_upper", "k_lower", "theta", "steepness")}
    total_supply = float(params.get("total_supply_kwh", 0.0))
    total_demand = float(params.get("total_demand_kwh", 0.0))
    clearing_price = float(params.get("energy_rate", 0.0))
    traded_quantity = min(total_supply, total_demand)
    round_kind = determine_round_type(total_supply, total_demand)

    measurements = await db.get_measurements(community_uuid, time_slot,
                                             cfg.time_slot_sec)
    actual_by_area = {m["area_uuid"]: float(m["energy_kwh"]) for m in measurements}

    # Second channel, and an optional one: a slot without forecasts is the
    # normal case, and a DB that does not serve the route at all must not
    # fail the run. Either way the penalties fall back to the meter reading.
    try:
        forecasts = await db.get_forecasts(community_uuid, time_slot,
                                           cfg.time_slot_sec)
    except OffchainDBError as exc:
        logger.warning("forecast channel unavailable (%s) — falling back to "
                       "meter readings for deliverable capacity", exc)
        forecasts = []
    # GSY's ForecastSchema.energy_kwh is signed — the community client writes a
    # seller's forecast as -energy — while this artifact keeps kWh positive on
    # both channels. Reading the magnitude is correct under either convention;
    # the warning makes the ambiguity visible instead of silent (issue #32).
    forecast_rows = [f for f in forecasts if f.get("energy_kwh") is not None]
    negative = [f["area_uuid"] for f in forecast_rows
                if float(f["energy_kwh"]) < 0.0]
    if negative:
        logger.warning("forecast channel returned %d negative value(s) for "
                       "slot %s (areas: %s) — reading the magnitude; GSY's "
                       "ForecastSchema is signed, this artifact is not",
                       len(negative), time_slot, ", ".join(map(str, negative)))
    deliverable_by_area = {f["area_uuid"]: abs(float(f["energy_kwh"]))
                           for f in forecast_rows}

    results = []
    total_penalties = 0.0
    writes = []
    for trade in trades:
        participant = _participant_from_trade(trade)
        if participant is None:
            continue

        actual = actual_by_area.get(participant["area_uuid"])
        measurement_found = actual is not None
        if not measurement_found:
            # No meter data: assume delivery as traded (no penalty) but flag
            # it — production behavior TBD with the measurement pipeline.
            logger.warning("no measurement for area %s @ %s — assuming "
                           "delivered == traded", participant["area_uuid"],
                           time_slot)
            actual = participant["traded_kwh"]

        # The externality penalty asks what the seller *could* have delivered.
        # With only a meter reading, delivered == deliverable by construction
        # and a successful withholder is invisible (issue #13). A separate
        # forecast channel is what makes the penalty measurable — and it is
        # manipulable by the party it is used to penalise, which is stated as
        # a limitation rather than fixed. Recorded on every row for the
        # harness; only the seller path reads it.
        forecast = deliverable_by_area.get(participant["area_uuid"])
        deliverable = actual if forecast is None else forecast

        row = {**participant, "actual_kwh": round(actual, 6),
               "measurement_found": measurement_found,
               "deliverable_kwh": round(deliverable, 6),
               "deliverable_source": "meter" if forecast is None else "forecast",
               "trade_uuid": trade.get("trade_uuid"),
               "shortfall_kwh": 0.0, "shortfall_penalty_ct": 0.0,
               "externality_kwh": 0.0, "externality_penalty_ct": 0.0,
               "counterfactual_price_ct_per_kwh": None}

        # η is configured relative to the trade's own quantity (D-26/D-44): an
        # absolute deadband makes the penalty a function of installation size —
        # 0.5 kWh is 50 % of a 1 kWh trade and 2.5 % of a 20 kWh one. The absolute
        # key stays as a fallback so runs recorded before 09/2026 reproduce.
        eta = (cfg.penalty_eta_relative * participant["traded_kwh"]
               if cfg.penalty_eta_relative is not None else cfg.penalty_eta_kwh)

        if participant["role"] == "seller":
            shortfall = seller_shortfall_penalty(
                participant["traded_kwh"], actual, sigmoid["k_upper"],
                cfg.penalty_gamma, eta)
            row["shortfall_kwh"] = shortfall["shortfall_kwh"]
            row["shortfall_penalty_ct"] = shortfall["penalty_ct"]

            if round_kind == SUPPLY_LIMITED:
                _apply_externality(row, seller_externality_penalty(
                    participant["traded_kwh"], deliverable,
                    eta, total_supply, total_demand,
                    traded_quantity, clearing_price, sigmoid), "withheld_kwh")
        elif round_kind == DEMAND_LIMITED:  # buyer
            _apply_externality(row, buyer_externality_penalty(
                participant["reported_kwh"], actual, total_supply,
                total_demand, traded_quantity, clearing_price, sigmoid),
                "underreported_kwh")

        row["total_penalty_ct"] = round(
            row["shortfall_penalty_ct"] + row["externality_penalty_ct"], 6)
        total_penalties += row["total_penalty_ct"]
        results.append(row)

        # Write penalties back: trade `parameters` is extended and the trade
        # is marked Executed ("energy delivery verified").
        # NOTE: re-triggering recomputes and overwrites — handy for demos;
        # production idempotency policy TBD.
        writes.append(db.update_trade(
            trade["trade_uuid"], status="Executed",
            parameters={**{key: row[key] for key in _PENALTY_PARAM_KEYS},
                        "round_type": round_kind,
                        "execution_time": int(time.time())}))

    await asyncio.gather(*writes)

    logger.info("execution done: market=%s, %d participants, %.4f ct total "
                "penalties (%s round)", market_id, len(results),
                total_penalties, round_kind)

    return {
        "status": "executed",
        "market_id": market_id,
        "community_uuid": community_uuid,
        "time_slot": time_slot,
        "round_type": round_kind,
        "clearing_price_ct_per_kwh": clearing_price,
        "total_supply_kwh": total_supply,
        "total_demand_kwh": total_demand,
        "traded_quantity_kwh": round(traded_quantity, 6),
        "sigmoid_params": sigmoid,
        # Both η keys are reported, so a run is identifiable from its own
        # output rather than from the configuration it was started with.
        "penalty_params": {"gamma": cfg.penalty_gamma,
                           "eta_kwh": cfg.penalty_eta_kwh,
                           "eta_relative": cfg.penalty_eta_relative,
                           "eta_mode": ("relative"
                                        if cfg.penalty_eta_relative is not None
                                        else "absolute"),
                           "k_sho_ct_per_kwh": round(
                               cfg.penalty_gamma * sigmoid["k_upper"], 6)},
        "total_penalties_ct": round(total_penalties, 6),
        # Per round, not per trade: the redistribution is a round-level
        # property and deliberately stays out of the trade `parameters`
        # written above (D-42). Computed only — nothing is paid out (D-25).
        "redistribution": redistribution(results, clearing_price,
                                         traded_quantity),
        "results": results,
    }


async def run_polling_cycle(cfg: Config, db: OffchainDBClient) -> list[dict]:
    """One pass of the production-style polling loop (guide §5.2): find
    markets whose delivery window closed >= |offset| minutes ago and execute
    them. Returns the execution summaries."""
    previous_slot = compute_previous_timeslot(cfg.time_slot_sec,
                                              cfg.execution_offset_min)
    summaries = []
    for community_uuid in cfg.poll_communities:
        markets = await db.get_community_markets(community_uuid)
        for market in markets:
            if market.get("time_slot") != previous_slot:
                continue
            summary = await run_execution(
                {"market_id": market["market_id"],
                 "community_uuid": community_uuid,
                 "time_slot": previous_slot}, cfg, db)
            summaries.append(summary)
    return summaries
