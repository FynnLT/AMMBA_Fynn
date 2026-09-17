"""One market slot through the real pipeline, in process.

`run_slot` clears a single slot; `run_sequence` walks one community over a
contiguous list of slots and returns one record per slot.

Community and slot are *arguments*, not something the runner invents. The
pilot harness held a module-level `itertools.count()` and put every call into
its own community and its own slot, which meant no two calls ever shared a
time axis. The calibration fits theta and steepness against the distribution
of the supply/demand ratio *over slots*, so it needs that axis to exist.
"""
import hashlib
import struct

SLOT_SEC = 900
SIGMOID = {"k_upper": 40.0, "k_lower": 8.0, "theta": 1.0, "steepness": 2.5}


def market_id_for(community: str, slot: int) -> str:
    """Deterministic market id namespaced by community.

    `mock-offchain-db/src/store.py:22` derives the id from the delivery slot
    alone (blake2b-256 of b"Spot" + u64_BE(time_slot)), and `query_orders`
    filters on `market_id` plus the time window, *not* on `community_uuid`.
    Two communities clearing the same slot would therefore share one order
    book. The segmented baseline runs exactly that shape, so the id carries
    the community as well. Same digest size and prefix as the mock's own, so
    it is indistinguishable from a natively generated one downstream.
    """
    payload = b"Spot" + struct.pack(">Q", int(slot)) + community.encode("utf-8")
    return "0x" + hashlib.blake2b(payload, digest_size=32).hexdigest()


def areas_for(scen: dict) -> list[dict]:
    """The community area list of a scenario.

    Split out so a sequence builds it once instead of once per slot: at 100
    players and 672 slots the repeated list comprehension is a measurable
    share of the run. The market is still created per slot -- its id changes
    with the slot -- but it is created from this same list.
    """
    return ([{"area_uuid": p["area_uuid"], "name": p["name"], "area_type": "PV"}
             for p in scen["producers"]] +
            [{"area_uuid": c["area_uuid"], "name": c["name"], "area_type": "Load"}
             for c in scen["consumers"]])


async def run_slot(st, scen, *, community, slot, market_id=None,
                   preferences=None, measurements=None, gamma=None, eta=None,
                   eta_relative=None, execute=True, sigmoid=None,
                   forecasts=None, areas=None, community_name="Pilot"):
    """Create market + orders, clear, optionally measure and execute.

    `community` and `slot` are required: they place the call on a time axis
    the caller controls. There is deliberately no fallback default for
    either -- an implicit slot is what produced the market-id collision
    hazard that `market_id_for` now closes.

    `market_id`: explicit id for `POST /market`. Defaults to None, which lets
    the mock derive it from the slot the way the production orchestrator
    does. Pass one when more than one community clears the same slot.

    `areas`: prebuilt community area list (see `areas_for`). Defaults to
    building it from `scen`.

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

    `eta` / `eta_relative`: absolute and relative penalty deadband, passed
    through to `Stack.execute`. Chapter 5 reports a deadband relative to the
    traded quantity (D-26/D-44), which is `eta_relative`.

    `sigmoid`: per-call override of the module-level SIGMOID constant, passed
    through the trigger's `sigmoid_params` (the clearing node's
    `resolve_community` merges it onto the configured community parameters).
    Defaults to SIGMOID, so existing call sites are unaffected.
    """
    slot = int(slot)
    if areas is None:
        areas = areas_for(scen)
    market_payload = {"community_uuid": community,
                      "community_name": community_name,
                      "time_slot": slot,
                      "community_areas": areas}
    if market_id is not None:
        market_payload["market_id"] = market_id
    market = await st.post("/market", market_payload)
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
             "energy": c["energy"], "energy_rate": 40.0}
        if c.get("preferred_partner"):
            o["requirements"] = {"preferred_partner": c["preferred_partner"]}
        orders.append(o)
    await st.post("/orders-normalized", orders)

    trigger = {"market_id": mid, "community_uuid": community,
               "time_slot": slot, "community_name": community_name,
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
            gamma=gamma, eta=eta, eta_relative=eta_relative)
    return {"market_id": mid, "community": community, "time_slot": slot,
            "clearing": clearing, "execution": execution}


async def run_sequence(st, community, slots, scenario_for_slot,
                       **kw) -> list[dict]:
    """One community over a contiguous list of slots, one record per slot.

    `scenario_for_slot(index, slot)` returns the scenario for position
    `index` of `slots`. The position is what a profile week is indexed by;
    the slot is the delivery timestamp the artifact sees.

    The area list is built once from the first scenario and reused: community
    membership is a property of the run, not of the slot, and rebuilding it
    672 times is most of the runtime otherwise. A market is still created per
    slot because its id changes with the slot; unless the caller supplies
    `market_id`, that id is namespaced by community so a second community
    over the same slots cannot share the order book.
    """
    slots = [int(s) for s in slots]
    areas = kw.pop("areas", None)
    explicit_market_id = kw.pop("market_id", None)
    records = []
    for index, slot in enumerate(slots):
        scen = scenario_for_slot(index, slot)
        if areas is None:
            areas = areas_for(scen)
        records.append(await run_slot(
            st, scen, community=community, slot=slot,
            market_id=explicit_market_id or market_id_for(community, slot),
            areas=areas, **kw))
    return records
