"""Async HTTP client for the GSY-DEX off-chain database REST API (guide §2).

The client only knows the documented REST interface; in the PoC it talks to
the mock-offchain-db service, in production it points at the real DB via
`OFFCHAIN_DB_URL` with no code changes.
"""

import logging

import httpx

logger = logging.getLogger("amm-clearing-node.offchain_db")


class OffchainDBError(RuntimeError):
    """Raised when the off-chain DB is unreachable or rejects a request."""


class OffchainDBClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs):
        # Off-chain DB unreachable -> fail fast (guide §4.7).
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OffchainDBError(
                f"off-chain DB unreachable: {method} {self.base_url}{path}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise OffchainDBError(
                f"off-chain DB error {response.status_code} on "
                f"{method} {path}: {response.text[:500]}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def health_check(self) -> dict:
        return await self._request("GET", "/health_check")

    async def get_orders(self, market_id: str,
                         start_time: int | None = None,
                         end_time: int | None = None) -> list[dict]:
        params: dict = {"market_id": market_id}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return await self._request("GET", "/orders", params=params)

    async def get_trades(self, market_id: str) -> list[dict]:
        return await self._request("GET", "/trades",
                                   params={"market_id": market_id})

    async def post_trades(self, trades: list[dict]) -> list[dict]:
        # JSON variant of the trades endpoint (plain /trades is SCALE-encoded).
        return await self._request("POST", "/trades-normalized", json=trades)

    async def update_order(self, order_id: str, *, status: str | None = None,
                           energy: float | None = None) -> dict:
        # TODO(confirm-with-supervisor): does PATCH /orders/{id} exist in the
        # production orderbook service? In GSY-DEX, status updates are driven
        # by the blockchain event listener; since the AMM contract emits a
        # single MarketCleared event (not per-order events), the Clearing Node
        # must update statuses directly (guide §4.4 step 8, §7.2). The mock
        # DB implements this endpoint.
        patch: dict = {}
        if status is not None:
            patch["status"] = status
        if energy is not None:
            patch["energy"] = energy
        return await self._request("PATCH", f"/orders/{order_id}", json=patch)
