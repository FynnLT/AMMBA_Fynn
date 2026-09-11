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


async def main() -> None:
    print("REPO resolved to:", stack.REPO)

    st = stack.Stack()
    scen = scenario.make_scenario(seed=0, n_prod=40, n_cons=60, sd_ratio=1.25)
    res = await runner.run_slot(st, scen, execute=True)

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

    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")


if __name__ == "__main__":
    asyncio.run(main())
