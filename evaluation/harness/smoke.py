"""One slot through the real pipeline — the check to run after moving the harness.

Run it from the harness directory:

    python smoke.py

It proves three things at once: the three services load side by side, a clearing
round completes, and the execution node answers with a redistribution block. If
this passes, the harness is wired to the artifact correctly.
"""
import asyncio
import logging

import runner
import scenario
import stack

logging.disable(logging.WARNING)  # the execution node logs one line per area without a measurement

EXPECTED_PRICE = 15.147274
EXPECTED_TRADES = 100

# A fixed test input, not a configuration value. The golden book below has
# been cleared at this band since the pilot, so the price it produces is a
# constant the pipeline can be checked against: if 15.147274 ever moves,
# something in the three services changed. Pinning it here keeps that
# meaning across every change to the community parameters -- K_upper on
# 17.09.2026 (D-77), theta and B when the calibration lands (T-19). The
# configured band is checked separately below.
GOLDEN_SIGMOID = {"k_upper": 28.5, "k_lower": 8.0,
                  "theta": 1.0, "steepness": 2.5}

# Fixed, not derived from the clock or from a counter: the smoke test is a
# regression check, so the slot it clears has to be the same one every time.
SMOKE_COMMUNITY = "smoke-community"
SMOKE_SLOT = 1_757_000_000 // runner.SLOT_SEC * runner.SLOT_SEC


async def main() -> None:
    print("REPO resolved to:", stack.REPO)

    st = stack.Stack()
    scen = scenario.make_scenario(seed=0, n_prod=40, n_cons=60, sd_ratio=1.25)
    res = await runner.run_slot(st, scen, community=SMOKE_COMMUNITY,
                                slot=SMOKE_SLOT, execute=True,
                                sigmoid=GOLDEN_SIGMOID)

    clearing = res["clearing"]
    execution = res["execution"]
    price = clearing["clearing_price_ct_per_kwh"]
    trades = clearing["num_trades"]

    print(f"clearing price : {price} ct/kWh")
    print(f"trades         : {trades}")
    print(f"round type     : {execution['round_type']}")
    print(f"redistribution : {'present' if 'redistribution' in execution else 'MISSING'}")

    ok = True
    if abs(price - EXPECTED_PRICE) > 1e-6:
        print(f"  ! price differs from the reference run ({EXPECTED_PRICE})")
        ok = False
    if trades != EXPECTED_TRADES:
        print(f"  ! trade count differs from the reference run ({EXPECTED_TRADES})")
        ok = False
    if "redistribution" not in execution:
        print("  ! no redistribution block — the execution node is older than 3af25cd")
        ok = False

    # Second slot, no override: the Clearing Node uses its own
    # `configuration.yaml`. This is what checks that the *configured* band is
    # loaded and applied -- the golden slot above deliberately cannot.
    band = st.cfg_clr.default_community
    configured = await runner.run_slot(st, scen, community=SMOKE_COMMUNITY,
                                       slot=SMOKE_SLOT + runner.SLOT_SEC,
                                       execute=False, sigmoid=False)
    price_cfg = configured["clearing"]["clearing_price_ct_per_kwh"]
    print(f"configured band: {band.k_lower} .. {band.k_upper} ct/kWh "
          f"-> {price_cfg} ct/kWh")
    if not band.k_lower <= price_cfg <= band.k_upper:
        print(f"  ! the configured run cleared outside its own band")
        ok = False
    if configured["clearing"]["sigmoid_params"]["k_upper"] != band.k_upper:
        print("  ! the node did not use its configured k_upper")
        ok = False

    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")


if __name__ == "__main__":
    asyncio.run(main())
