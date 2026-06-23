"""End-to-end smoke test / API walkthrough for the AMMBA PoC stack.

Drives the same calls the demo UI makes, against a running stack
(docker-compose up, or the three uvicorn services locally), and asserts the
numbers from the implementation guide's reference example.

Usage:
    python scripts/e2e_demo.py [--db URL] [--clearing URL] [--execution URL]
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
    args = parser.parse_args()

    http = httpx.Client(timeout=20)

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
    check("clearing node", call("GET", args.clearing, "/health")["status"] == "ok")
    check("execution node", call("GET", args.execution, "/health")["status"] == "ok")

    community = f"community-e2e-{int(time.time())}"
    time_slot = (int(time.time()) // 900 + 4) * 900

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

    def order(order_type, name, area, energy, rate):
        return {"order_type": order_type, "created_by": name,
                "area_uuid": area, "market_id": market_id,
                "time_slot": time_slot, "energy": energy, "energy_rate": rate}

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
    check("simulated tx hash", result["tx_hash"].startswith("0x"))
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
    call("POST", args.db, "/asset_measurements", json=measurements)

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
