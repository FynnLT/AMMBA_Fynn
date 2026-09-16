"""Section 1: slot history in the runner, relative eta in the stack."""
import pytest

import runner
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    instance = stack.Stack()
    yield instance
    # `Stack.close()` is a coroutine; the event loop is gone by the time a
    # sync fixture tears down, so the transport is dropped with the object.
    # Nothing outside the process holds it -- there are no real sockets.


def test_run_slot_requires_community_and_slot():
    """The counter is gone, not demoted to a default. An implicit slot is
    what let every call land in its own market."""
    import inspect

    sig = inspect.signature(runner.run_slot)
    for name in ("community", "slot"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
    assert not hasattr(runner, "_slot_counter")


def test_market_id_is_namespaced_by_community():
    """`query_orders` filters on market_id and the time window, not on the
    community -- two communities in one slot would share an order book."""
    a = runner.market_id_for("community-a", BASE_SLOT)
    b = runner.market_id_for("community-b", BASE_SLOT)
    assert a != b
    assert a == runner.market_id_for("community-a", BASE_SLOT)
    assert a.startswith("0x") and len(a) == 66


@pytest.mark.anyio
async def test_run_sequence_walks_one_community_over_contiguous_slots(st):
    """12 slots, 12 records, strictly increasing slot times, one market per
    slot. Without this the ratio has no time axis to be distributed over."""
    slots = [BASE_SLOT + i * runner.SLOT_SEC for i in range(12)]
    scen = scenario.make_scenario(seed=0, n_prod=8, n_cons=12, sd_ratio=1.2)

    records = await runner.run_sequence(
        st, "seq-community", slots, lambda index, slot: scen, execute=False)

    assert len(records) == 12
    times = [r["time_slot"] for r in records]
    assert times == slots
    assert all(b > a for a, b in zip(times, times[1:]))

    markets = [r["market_id"] for r in records]
    assert len(set(markets)) == 12
    assert all(r["community"] == "seq-community" for r in records)
    assert all(r["clearing"]["status"] == "cleared" for r in records)


@pytest.mark.anyio
async def test_execute_sets_the_relative_eta(st):
    """`eta_relative` has to reach `cfg.penalty_eta_relative`; with only the
    absolute key reachable, every harness run is an absolute-kWh run while
    Chapter 5 reports a deadband relative to the traded quantity (D-26/D-44).

    The deadband is not reported per row, but it is recoverable from one:
    `shortfall_kwh = max(0, traded - actual - eta)`, so a penalised row gives
    back exactly the eta the node applied.
    """
    scen = scenario.make_scenario(seed=1, n_prod=4, n_cons=6, sd_ratio=1.0)
    slot = BASE_SLOT

    # Every producer delivers 50 % of what it traded: well past any deadband
    # below 0.5, so every seller row carries a shortfall.
    first = await runner.run_slot(st, scen, community="eta-community",
                                  slot=slot, execute=False)
    allocated = {p["area_uuid"]: p["allocated_kwh"]
                 for p in first["clearing"]["allocations"]["producers"]}

    result = await runner.run_slot(
        st, scen, community="eta-community", slot=slot + runner.SLOT_SEC,
        market_id=runner.market_id_for("eta-community", slot + runner.SLOT_SEC),
        measurements={area: kwh * 0.5 for area, kwh in allocated.items()},
        eta_relative=0.05, execute=True)

    execution = result["execution"]
    assert execution["penalty_params"]["eta_relative"] == 0.05
    assert execution["penalty_params"]["eta_mode"] == "relative"

    sellers = [r for r in execution["results"]
               if r["role"] == "seller" and r["shortfall_kwh"] > 0]
    assert sellers, "expected every seller to be short by 50 %"
    applied = [row["traded_kwh"] - row["actual_kwh"] - row["shortfall_kwh"]
               for row in sellers]
    for row, eta in zip(sellers, applied):
        assert eta == pytest.approx(0.05 * row["traded_kwh"], abs=1e-6)
    # It scales: the deadband is not one constant shared by all rows.
    assert max(applied) > min(applied)


@pytest.mark.anyio
async def test_execute_rejects_an_out_of_range_relative_eta(st):
    """The service validates this at load time; `replace()` bypasses that,
    so the stack calls the service's own check rather than restating it."""
    with pytest.raises(ValueError, match="eta_relative"):
        await st.execute({"market_id": "0x00", "community_uuid": "c",
                          "time_slot": BASE_SLOT}, eta_relative=1.5)
