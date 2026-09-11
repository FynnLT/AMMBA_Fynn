"""One market slot through the real pipeline, in process."""
import itertools, time

_slot_counter = itertools.count(1)
SLOT_SEC = 900
SIGMOID = {"k_upper": 28.5, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}


async def run_slot(st, scen, preferences=None, measurements=None,
                   gamma=None, eta=None, community=None, execute=True,
                   sigmoid=None, forecasts=None):
    """Create market + orders, clear, optionally measure and execute.

    `measurements`: dict area_uuid -> delivered kWh. Areas not listed are
    assumed to have delivered exactly what they traded (execution node's own
    fallback), so only deviators need to be named.

    `forecasts`: dict area_uuid -> deliverable kWh, posted to `/forecasts`.
    This is the second measurement channel: what an area *could* have
    delivered, as against the meter's what it *did*. Areas not listed carry
    no forecast and fall back to the meter reading in `run_execution`.
    Issue #27: `community_uuid` is optional on the POST route but the
    Execution Node queries the channel with a community filter, so a forecast
    stored without one is invisible - it is always sent here.

    `sigmoid`: per-call override of the module-level SIGMOID constant, passed
    through the trigger's `sigmoid_params` (the clearing node's
    `resolve_community` merges it onto the configured community parameters).
    Defaults to SIGMOID, so existing call sites are unaffected.
    """
    community = community or f"c-{next(_slot_counter)}"
    slot = (int(time.time()) // SLOT_SEC + 8) * SLOT_SEC + next(_slot_counter) * SLOT_SEC

    areas = ([{"area_uuid": p["area_uuid"], "name": p["name"], "area_type": "PV"}
              for p in scen["producers"]] +
             [{"area_uuid": c["area_uuid"], "name": c["name"], "area_type": "Load"}
              for c in scen["consumers"]])
    market = await st.post("/market", {"community_uuid": community,
                                       "community_name": "Pilot",
                                       "time_slot": slot,
                                       "community_areas": areas})
    mid = market["market_id"]

    orders = []
    for p in scen["producers"]:
        o = {"order_type": "Offer", "created_by": p["name"],
             "area_uuid": p["area_uuid"], "market_id": mid, "time_slot": slot,
             "energy": p["energy"], "energy_rate": 8.0,
             "attributes": {"energy_type": p.get("energy_type", "green")}}
        if p.get("preferred_partner"):
            o["requirements"] = {"preferred_partner": p["preferred_partner"]}
        orders.append(o)
    for c in scen["consumers"]:
        o = {"order_type": "Bid", "created_by": c["name"],
             "area_uuid": c["area_uuid"], "market_id": mid, "time_slot": slot,
             "energy": c["energy"], "energy_rate": 28.5}
        if c.get("preferred_partner"):
            o["requirements"] = {"preferred_partner": c["preferred_partner"]}
        orders.append(o)
    await st.post("/orders-normalized", orders)

    trigger = {"market_id": mid, "community_uuid": community,
               "time_slot": slot, "community_name": "Pilot",
               "sigmoid_params": dict(sigmoid or SIGMOID)}
    if preferences is not None:
        trigger["preference_params"] = dict(preferences)
    clearing = await st.clear(trigger)

    execution = None
    if execute and clearing.get("status") == "cleared":
        if measurements:
            rows = [{"community_uuid": community, "area_uuid": a,
                     "time_slot": slot, "energy_kwh": kwh}
                    for a, kwh in measurements.items()]
            # Route renamed /asset_measurements -> /measurements in 6090516
            # (the target commit itself); the pilot harness predates that.
            await st.post("/measurements", rows)
        if forecasts:
            rows = [{"community_uuid": community, "area_uuid": a,
                     "time_slot": slot, "energy_kwh": kwh, "confidence": 1.0}
                    for a, kwh in forecasts.items()]
            await st.post("/forecasts", rows)
            # Read back before executing: a silent write failure would
            # otherwise be indistinguishable from a genuine null result.
            stored = await st.get("/forecasts", community_uuid=community,
                                  start_time=slot,
                                  end_time=slot + SLOT_SEC - 1)
            got = {f["area_uuid"]: f["energy_kwh"] for f in stored}
            missing = {a: kwh for a, kwh in forecasts.items()
                       if abs(got.get(a, float("nan")) - kwh) > 1e-9}
            if missing:
                raise RuntimeError(
                    f"forecast write-back verification failed for {missing}; "
                    f"read back {got}")
        execution = await st.execute(
            {"market_id": mid, "community_uuid": community, "time_slot": slot},
            gamma=gamma, eta=eta)
    return {"market_id": mid, "community": community, "time_slot": slot,
            "clearing": clearing, "execution": execution}
