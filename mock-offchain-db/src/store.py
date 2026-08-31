"""In-memory data store for the mock GSY-DEX off-chain database.

Everything is volatile by design: restarting the service wipes all data.
The store mirrors the collections the real off-chain DB exposes (markets,
orders, trades, measurements, forecasts) keyed the same way its REST API
queries them (gsy-decentralized-exchange main @ aa99ea2,
gsy-offchain-storage/src/startup.rs).
"""

import hashlib
import json
import struct
import time


def blake2b_hash(data: dict) -> str:
    """blake2b-256 hex hash of a JSON-serialised dict (GSY-DEX convention)."""
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return "0x" + hashlib.blake2b(encoded, digest_size=32).hexdigest()


def generate_market_id(time_slot: int) -> str:
    """blake2b-256 of the ASCII market type + the delivery timestamp.

    In production the Market Orchestrator generates this id; the mock does it
    as a stand-in convenience when POST /market omits `market_id`. The
    preimage follows the orchestrator byte-for-byte
    (gsy-decentralized-exchange main @ aa99ea2,
    gsy-market-orchestrator/src/orchestrator.rs:85): the market type as ASCII
    ("Spot", capitalised) followed by the timestamp as an unsigned 64-bit
    big-endian integer.
    """
    payload = b"Spot" + struct.pack(">Q", int(time_slot))
    return "0x" + hashlib.blake2b(payload, digest_size=32).hexdigest()


def now_ts() -> int:
    return int(time.time())


class InMemoryStore:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.markets: dict[str, dict] = {}            # market_id -> market
        self.orders: dict[str, dict] = {}             # order_id -> order
        self.trades: dict[str, dict] = {}             # trade_uuid -> trade
        # (community_uuid, area_uuid, time_slot) -> measurement
        self.measurements: dict[tuple, dict] = {}
        # Same key, second channel: what an area *expected* to deliver or
        # consume, as opposed to what its meter recorded.
        self.forecasts: dict[tuple, dict] = {}

    # ------------------------------------------------------------- markets

    def upsert_market(self, market: dict) -> dict:
        market.setdefault("creation_time", now_ts())
        market.setdefault("community_areas", [])
        if not market.get("market_id"):
            market["market_id"] = generate_market_id(market["time_slot"])
        existing = self.markets.get(market["market_id"])
        if existing is not None:
            return existing  # idempotent create
        self.markets[market["market_id"]] = market
        return market

    def markets_for_community(self, community_uuid: str) -> list[dict]:
        found = [m for m in self.markets.values()
                 if m.get("community_uuid") == community_uuid]
        return sorted(found, key=lambda m: m.get("time_slot", 0))

    # -------------------------------------------------------------- orders

    def add_order(self, order: dict) -> dict:
        order.setdefault("status", "Open")
        order.setdefault("creation_time", now_ts())
        if not order.get("order_id"):
            order_id = blake2b_hash(order)
            salt = 0
            # Two byte-identical orders would hash identically; salt to keep
            # every submitted order addressable in the mock.
            while order_id in self.orders:
                salt += 1
                order_id = blake2b_hash({**order, "_salt": salt})
            order["order_id"] = order_id
        self.orders[order["order_id"]] = order
        return order

    def query_orders(self, market_id: str | None,
                     start_time: int | None, end_time: int | None) -> list[dict]:
        result = []
        for order in self.orders.values():
            if market_id is not None and order.get("market_id") != market_id:
                continue
            ts = order.get("time_slot", 0)
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue
            result.append(order)
        return sorted(result, key=lambda o: o.get("creation_time", 0))

    # -------------------------------------------------------------- trades

    def add_trade(self, trade: dict) -> dict:
        trade.setdefault("creation_time", now_ts())
        if not trade.get("_id"):
            trade["_id"] = blake2b_hash(trade)
        key = trade.get("trade_uuid") or trade["_id"]
        trade.setdefault("trade_uuid", key)
        self.trades[key] = trade
        return trade

    def query_trades(self, market_id: str | None) -> list[dict]:
        result = [t for t in self.trades.values()
                  if market_id is None or t.get("market_id") == market_id]
        return sorted(result, key=lambda t: t.get("creation_time", 0))

    # ------------------------------------------- measurements / forecasts

    # Both channels carry one row per (community, area, slot) and are queried
    # identically, so they share the store pattern below.

    @staticmethod
    def _upsert_slot_row(collection: dict[tuple, dict], row: dict) -> dict:
        row.setdefault("creation_time", now_ts())
        key = (row.get("community_uuid"), row.get("area_uuid"),
               row.get("time_slot"))
        collection[key] = row
        return row

    @staticmethod
    def _query_slot_rows(collection: dict[tuple, dict],
                         community_uuid: str | None, area_uuid: str | None,
                         start_time: int | None,
                         end_time: int | None) -> list[dict]:
        """Rows whose slot lies in [start_time, end_time], both ends
        inclusive — the same convention `query_orders` uses, so a caller
        asking for [t, t + Δ - 1] sees exactly one slot on every channel."""
        result = []
        for (c_uuid, a_uuid, slot), row in collection.items():
            if community_uuid is not None and c_uuid != community_uuid:
                continue
            if area_uuid is not None and a_uuid != area_uuid:
                continue
            slot = slot or 0
            if start_time is not None and slot < start_time:
                continue
            if end_time is not None and slot > end_time:
                continue
            result.append(row)
        return sorted(result, key=lambda r: (r.get("area_uuid") or "",
                                             r.get("time_slot") or 0))

    def upsert_measurement(self, m: dict) -> dict:
        return self._upsert_slot_row(self.measurements, m)

    def query_measurements(self, community_uuid: str | None = None,
                           area_uuid: str | None = None,
                           start_time: int | None = None,
                           end_time: int | None = None) -> list[dict]:
        return self._query_slot_rows(self.measurements, community_uuid,
                                     area_uuid, start_time, end_time)

    def upsert_forecast(self, f: dict) -> dict:
        return self._upsert_slot_row(self.forecasts, f)

    def query_forecasts(self, community_uuid: str | None = None,
                        area_uuid: str | None = None,
                        start_time: int | None = None,
                        end_time: int | None = None) -> list[dict]:
        return self._query_slot_rows(self.forecasts, community_uuid,
                                     area_uuid, start_time, end_time)
