"""End-to-end smoke test / API walkthrough for the AMMBA PoC stack.

Drives the same calls the demo UI makes, against a running stack
(docker-compose up, or the three uvicorn services locally), and asserts the
numbers from the implementation guide's reference example.

Usage:
    python scripts/e2e_demo.py [--db URL] [--clearing URL] [--execution URL]
                               [--community UUID] [--slot-offset N]

    Mock mode (default stack): no arguments needed.
    Live mode (BLOCKCHAIN_MODE=live): pass a community that deploy.js has
    registered on-chain, e.g. --community communityid_1, and increment
    --slot-offset for every additional run in the same 15-minute window.
"""

import argparse
import sys
import time

import httpx

GUIDE_PRICE = 15.1472  # sigmoid(1.25) for K=28.5/8.0, theta=1.0, B=2.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="http://localhost:8080")
    parser.add_argument("--clearing", default="http://localhost:8081")
    parser.add_argument("--execution", default="http://localhost:8082")
    # In live mode the community must already have on-chain parameters:
    # AMMBA.sol::clearMarket reverts with "community params not set" for any
    # community that scripts/deploy.js never registered. The generated default
    # is therefore usable in mock mode only.
    parser.add_argument("--community", default=None,
                        help="community_uuid to clear for; must be registered "
                             "on-chain in live mode (default: generated)")
    # market_id is blake2b("Spot" + time_slot), so the delivery slot alone
    # determines it: two runs in the same 15-minute window collide. Vary this
    # to get a distinct market_id per run.
    parser.add_argument("--slot-offset", type=int, default=4,
                        help="delivery slot in 15-min steps ahead of now "
                             "(default: 4); vary it for a distinct market_id")
    # /trigger-clearing is synchronous and, in live mode, waits for the
    # clearMarket receipt inside the request. On a public chain that is block
    # time plus confirmation, not milliseconds, so the client timeout has to
    # cover it: 20 s is fine against a local node and too tight for Volta.
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="HTTP client timeout in seconds (default: 20; "
                             "use 120 against a public chain)")
    # Without this the run is indistinguishable from a live one: every check
    # below passes identically in mock mode, and the "tx hash" it asserts is a
    # blake2b simulation. Three full runs were mistaken for on-chain evidence
    # on 2026-08-29 for exactly this reason.
    parser.add_argument("--expect-mode", choices=("mock", "live"),
                        default=None,
                        help="abort unless the Clearing Node reports this "
                             "blockchain_mode; use --expect-mode live when "
                             "the run is meant to produce on-chain evidence")
    args = parser.parse_args()

    http = httpx.Client(timeout=args.timeout)

    def call(method: str, base: str, path: str, **kwargs):
        resp = http.request(method, base + path, **kwargs)
        if resp.status_code >= 400:
            raise SystemExit(f"FAIL {method} {path}: {resp.status_code} {resp.text}")
        return resp.json()

    def check(label: str, condition: bool, detail: str = ""):
        print(f"  {'OK ' if condition else 'FAIL'} {label}" +
              (f" ({detail})" if detail else ""))
        if not condition:
            raise SystemExit(1)

    print("== health ==")
    check("mock-offchain-db", call("GET", args.db, "/health_check")["status"] == "ok")
    clearing_health = call("GET", args.clearing, "/health")
    check("clearing node", clearing_health["status"] == "ok")
    check("execution node", call("GET", args.execution, "/health")["status"] == "ok")

    # The anchor mode decides what this run is evidence of, so it is stated
    # once, loudly, and optionally enforced.
    mode = clearing_health.get("blockchain_mode", "unknown")
    print(f"  >> blockchain_mode: {mode}")
    if args.expect_mode:
        check(f"blockchain_mode is {args.expect_mode}", mode == args.expect_mode,
              f"got {mode}")

    community = args.community or f"community-e2e-{int(time.time())}"
    time_slot = (int(time.time()) // 900 + args.slot_offset) * 900

    print("== create market + orders (guide example: 12.5 vs 10 kWh) ==")
    market = call("POST", args.db, "/market", json={
        "community_uuid": community, "community_name": "E2E Community",
        "time_slot": time_slot,
        "community_areas": [
            {"area_uuid": "area-pv-a", "name": "PV A", "area_type": "PV"},
            {"area_uuid": "area-pv-b", "name": "PV B", "area_type": "PV"},
            {"area_uuid": "area-battery", "name": "Battery", "area_type": "PV"},
            {"area_uuid": "area-house-1", "name": "House 1", "area_type": "Load"},
            {"area_uuid": "area-house-2", "name": "House 2", "area_type": "Load"},
            {"area_uuid": "area-bakery", "name": "Bakery", "area_type": "Load"},
        ]})
    market_id = market["market_id"]
    check("market_id generated (blake2b)", market_id.startswith("0x")
          and len(market_id) == 66, market_id[:18] + "…")

    def order(order_type, name, area, energy, rate, **extra):
        return {"order_type": order_type, "created_by": name,
                "area_uuid": area, "market_id": market_id,
                "time_slot": time_slot, "energy": energy, "energy_rate": rate,
                **extra}

    orders = call("POST", args.db, "/orders-normalized", json=[
        order("Offer", "PV A", "area-pv-a", 5.0, 8.0),
        order("Offer", "PV B", "area-pv-b", 3.5, 8.0),
        order("Offer", "Battery", "area-battery", 4.0, 8.0),
        order("Bid", "House 1", "area-house-1", 4.5, 28.5),
        order("Bid", "House 2", "area-house-2", 3.0, 28.5),
        order("Bid", "Bakery", "area-bakery", 2.5, 28.5),
    ])
    check("6 orders stored as Open", len(orders) == 6
          and all(o["status"] == "Open" for o in orders))

    print("== trigger clearing ==")
    result = call("POST", args.clearing, "/trigger-clearing", json={
        "market_id": market_id, "community_uuid": community,
        "time_slot": time_slot, "community_name": "E2E Community",
        "sigmoid_params": {"k_upper": 28.5, "k_lower": 8.0,
                           "theta": 1.0, "steepness": 2.5}})
    check("status cleared", result["status"] == "cleared")
    check("guide reference price ~15.147 ct/kWh",
          abs(result["clearing_price_ct_per_kwh"] - GUIDE_PRICE) < 1e-3,
          f"{result['clearing_price_ct_per_kwh']:.4f}")
    check("ratio 1.25", abs(result["ratio"] - 1.25) < 1e-9)
    check("round DEMAND_LIMITED", result["round_type"] == "DEMAND_LIMITED")
    check("6 trades (one per participant)", result["num_trades"] == 6)
    check(f"{'on-chain' if mode == 'live' else 'simulated'} tx hash",
          result["tx_hash"].startswith("0x"), result["tx_hash"])
    producers = {p["name"]: p for p in result["allocations"]["producers"]}
    check("producers filled pro-rata at 80%",
          all(abs(p["fill_rate"] - 0.8) < 1e-9 for p in producers.values()))

    trades = call("GET", args.db, f"/trades?market_id={market_id}")
    check("trades persisted in off-chain DB", len(trades) == 6)
    check("orders marked Executed",
          all(o["status"] == "Executed" for o in
              call("GET", args.db, f"/orders?market_id={market_id}")))

    print("== idempotent re-trigger ==")
    again = call("POST", args.clearing, "/trigger-clearing", json={
        "market_id": market_id, "community_uuid": community,
        "time_slot": time_slot})
    check("already_cleared", again["status"] == "already_cleared")
    check("no duplicate trades",
          len(call("GET", args.db, f"/trades?market_id={market_id}")) == 6)

    print("== post-delivery: measurements + execution ==")
    # PV A under-delivers by 1 kWh (traded 4.0); Bakery consumes 2 kWh more
    # than reported (2.5 -> 4.5) in a demand-limited round.
    perfect = {"area-pv-b": 2.8, "area-battery": 3.2, "area-house-1": 4.5,
               "area-house-2": 3.0}
    measurements = [{"community_uuid": community, "area_uuid": area,
                     "time_slot": time_slot, "energy_kwh": kwh}
                    for area, kwh in {**perfect, "area-pv-a": 3.0,
                                      "area-bakery": 4.5}.items()]
    call("POST", args.db, "/measurements", json=measurements)

    execution = call("POST", args.execution, "/trigger-execution", json={
        "market_id": market_id, "community_uuid": community,
        "time_slot": time_slot})
    check("status executed", execution["status"] == "executed")
    rows = {r["name"]: r for r in execution["results"]}
    check("PV A shortfall penalty = 1.1 * 28.5 * 1.0 = 31.35 ct",
          abs(rows["PV A"]["shortfall_penalty_ct"] - 31.35) < 1e-6,
          f"{rows['PV A']['shortfall_penalty_ct']:.4f}")
    check("Bakery buyer externality ~25.69 ct",
          abs(rows["Bakery"]["externality_penalty_ct"] - 25.694) < 1e-2,
          f"{rows['Bakery']['externality_penalty_ct']:.4f}")
    check("honest participants unpenalized",
          all(rows[n]["total_penalty_ct"] == 0
              for n in ("PV B", "Battery", "House 1", "House 2")))

    executed_trades = call("GET", args.db, f"/trades?market_id={market_id}")
    check("trades marked Executed with penalties in parameters",
          all(t["status"] == "Executed" and
              "total_penalty_ct" in t["parameters"] for t in executed_trades))

    print("== preferences + energy-type multipliers ==")
    # Same market as above, but PV A and House 1 name each other as preferred
    # partner and the battery declares itself grey. Expected (task §6.2/§6.3):
    # the pair moves 4.5 kWh off the top, which lifts PV A from the 80 %
    # pro-rata baseline to 96.875 %, and the grey levy funds 37.931 % of the
    # requested green bonus.
    pref_slot = time_slot + 1800
    pref_market = call("POST", args.db, "/market", json={
        "community_uuid": community, "time_slot": pref_slot})
    pref_market_id = pref_market["market_id"]

    def pref_order(order_type, name, area, energy, rate, **extra):
        return order(order_type, name, area, energy, rate, **extra) | {
            "market_id": pref_market_id, "time_slot": pref_slot}

    call("POST", args.db, "/orders-normalized", json=[
        pref_order("Offer", "PV A", "area-pv-a", 5.0, 8.0,
                   attributes={"energy_type": "green"},
                   requirements={"preferred_partner": "area-house-1"}),
        pref_order("Offer", "PV B", "area-pv-b", 3.5, 8.0,
                   attributes={"energy_type": "green"}),
        pref_order("Offer", "Battery", "area-battery", 4.0, 8.0,
                   attributes={"energy_type": "grey"}),
        pref_order("Bid", "House 1", "area-house-1", 4.5, 28.5,
                   requirements={"preferred_partner": "area-pv-a"}),
        pref_order("Bid", "House 2", "area-house-2", 3.0, 28.5),
        pref_order("Bid", "Bakery", "area-bakery", 2.5, 28.5),
    ])

    pref = call("POST", args.clearing, "/trigger-clearing", json={
        "market_id": pref_market_id, "community_uuid": community,
        "time_slot": pref_slot, "community_name": "E2E Community",
        "sigmoid_params": {"k_upper": 28.5, "k_lower": 8.0,
                           "theta": 1.0, "steepness": 2.5},
        "preference_params": {"enabled": True, "order": "preferences_first",
                              "multipliers_enabled": True,
                              "mode": "multiplicative", "sides": "seller",
                              "green_multiplier": 0.10, "grey_levy": 0.10,
                              "levy_cap": 0.20}})
    check("status cleared", pref["status"] == "cleared")
    prefs = pref["preferences"]
    check("one mutual pair (PV A <-> House 1) over 4.5 kWh",
          prefs["mutual_pairs"] == [{"bid_area": "area-house-1",
                                     "offer_area": "area-pv-a",
                                     "energy_kwh": 4.5}])
    check("pairs not rationed", prefs["pairs_rationed"] is False)

    pref_producers = {p["name"]: p for p in pref["allocations"]["producers"]}
    for name, allocated in (("PV A", 4.843750), ("PV B", 2.406250),
                            ("Battery", 2.750000)):
        check(f"{name} allocated {allocated:.6f} kWh",
              abs(pref_producers[name]["allocated_kwh"] - allocated) < 1e-6,
              f"{pref_producers[name]['allocated_kwh']:.6f}")
    check("PV A preferentially filled to 96.875 % (pro-rata baseline: 80 %)",
          abs(pref_producers["PV A"]["fill_rate"] - 0.968750) < 1e-6)
    check("allocation balances at 10 kWh on both sides",
          abs(sum(p["allocated_kwh"] for p in pref["allocations"]["producers"])
              - 10.0) < 1e-6
          and abs(sum(c["allocated_kwh"]
                      for c in pref["allocations"]["consumers"]) - 10.0) < 1e-6)

    mult = prefs["multipliers"]
    for label, key, expected, tol in (
            ("green volume 7.25 kWh", "green_alloc_kwh", 7.25, 1e-6),
            ("grey volume 2.75 kWh", "grey_alloc_kwh", 2.75, 1e-6),
            ("levy collected ~4.1655 ct", "levy_collected_ct", 4.16548, 1e-4),
            ("bonus requested ~10.9817 ct", "bonus_requested_ct", 10.98172,
             1e-4),
            ("subsidy scaling ~0.37931", "scale", 0.37931, 1e-5),
            ("green final ~15.7218 ct/kWh", "green_final_ct_per_kwh",
             15.721775, 1e-5),
            ("grey final ~13.6325 ct/kWh", "grey_final_ct_per_kwh", 13.632503,
             1e-5),
            ("buyers pay == sellers receive (~151.4723 ct)", "buyers_pay_ct",
             151.47225, 1e-3),
            ("no pool surplus while the bonus is scaled", "pool_surplus_ct",
             0.0, 1e-3)):
        check(label, abs(mult[key] - expected) < tol, f"{mult[key]}")

    pref_trades = call("GET", args.db, f"/trades?market_id={pref_market_id}")
    check("uniform energy_rate untouched in every trade (execution node "
          "reads it for counterfactuals)",
          all(abs(t["parameters"]["energy_rate"]
                  - pref["clearing_price_ct_per_kwh"]) < 1e-9
              for t in pref_trades))
    check("final_energy_rate added alongside it",
          all("final_energy_rate" in t["parameters"] for t in pref_trades))

    print("== no-trade slot (bids only) ==")
    nt_slot = time_slot + 900
    nt_market = call("POST", args.db, "/market", json={
        "community_uuid": community, "time_slot": nt_slot})
    call("POST", args.db, "/orders-normalized", json=[
        order("Bid", "House 1", "area-house-1", 5.0, 28.5) | {
            "market_id": nt_market["market_id"], "time_slot": nt_slot}])
    nt = call("POST", args.clearing, "/trigger-clearing", json={
        "market_id": nt_market["market_id"], "community_uuid": community,
        "time_slot": nt_slot})
    check("no_trade", nt["status"] == "no_trade")
    check("orders expired", all(
        o["status"] == "Expired" for o in
        call("GET", args.db, f"/orders?market_id={nt_market['market_id']}")))

    print("\nALL CHECKS PASSED — full clearing + execution cycle verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
