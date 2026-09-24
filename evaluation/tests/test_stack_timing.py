"""Section 2: the transport that says where a slot's time went.

D-86 requires each N-point of the curve to report clearing-cycle time
separately from off-chain store time. `Stack` owns one `httpx.AsyncClient`
over an `ASGITransport`, and both DB clients *and* the clearing node's own
queries go through it -- so one request on that client is one off-chain store
call, and counting them at the transport is the whole measurement. The third
test is the one that proves the seam is the right one.
"""
import httpx
import pytest

import runner
import scenario
import stack

BASE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


@pytest.fixture
def st():
    instance = stack.Stack()
    yield instance
    # Same reason as `tests/test_runner.py`: `close()` is a coroutine and the
    # loop is gone at teardown. Nothing outside the process holds the
    # transport; there are no real sockets.


def test_a_fresh_stack_has_counted_nothing():
    instance = stack.Stack()
    assert isinstance(instance.transport, stack.TimedTransport)
    assert instance.transport.n_requests == 0
    assert instance.transport.elapsed == 0.0
    # Wrapped around the ASGI transport, not replacing it: the client still
    # talks to the in-process app.
    assert isinstance(instance.transport._inner, httpx.ASGITransport)


@pytest.mark.anyio
async def test_one_post_is_one_request_and_reset_clears_the_counters(st):
    await st.post("/market", {"community_uuid": "timing",
                              "community_name": "Timing",
                              "time_slot": BASE_SLOT,
                              "community_areas": []})
    assert st.transport.n_requests == 1
    assert st.transport.elapsed > 0

    # `Stack.reset()` wipes the store and deliberately leaves the counters
    # alone: the N-curve resets them explicitly around the section it times,
    # and a reset hidden inside another method is how a measurement quietly
    # starts including its own setup.
    await st.reset()
    assert st.transport.n_requests == 2

    st.transport.reset()
    assert st.transport.n_requests == 0
    assert st.transport.elapsed == 0.0


@pytest.mark.anyio
async def test_the_counter_sees_the_clearing_node_s_own_queries(st):
    """Not only the harness's posts. `run_slot` posts twice (`/market`,
    `/orders-normalized`) and `run_clearing` queries the store itself, so a
    counter that stopped at two would be sitting on the wrong seam and the
    N-curve's store column would be harness setup rather than the cycle."""
    scen = {"producers": [{"area_uuid": "gen-000", "name": "Gen 000",
                           "energy": 4.0, "energy_type": "green"}],
            "consumers": [{"area_uuid": "load-000", "name": "Load 000",
                           "energy": 3.0}]}

    st.transport.reset()
    await runner.run_slot(st, scen, community="timing", slot=BASE_SLOT,
                          execute=False)
    posts_by_the_harness = 2
    assert st.transport.n_requests > posts_by_the_harness
    assert st.transport.elapsed > 0

    # And the book it cleared really was one order per area (D-85), which is
    # what makes "N orders" and "N areas" the same axis.
    scenario._assert_one_order_per_area(scen)
