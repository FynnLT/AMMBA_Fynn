"""Async HTTP client for the GSY-DEX off-chain database REST API.

The client only knows the routes the production service registers
(gsy-decentralized-exchange main @ aa99ea2,
gsy-offchain-storage/src/startup.rs); in the PoC it talks to the
mock-offchain-db service, in production it points at the real DB via
`OFFCHAIN_DB_URL` with no code changes.
"""

import logging

import httpx

logger = logging.getLogger("amm-clearing-node.offchain_db")


class OffchainDBError(RuntimeError):
    """Raised when the off-chain DB is unreachable or rejects a request."""


class OffchainDBClient:
    def __init__(self, base_url: str, timeout: float = 10.0,
                 client: httpx.AsyncClient | None = None) -> None:
        """`client` lets a caller inject its own transport — the simulation
        harness passes an `httpx.ASGITransport` bound to the mock DB app so a
        campaign runs in-process instead of over real ports. An injected client
        is not owned by this instance and is therefore not closed by `close()`.
        """
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self.base_url,
                                                   timeout=timeout)

    async def close(self) -> None:
        # Only close what we created; an injected client belongs to the caller
        # and may still be shared with the other service's DB client.
        if self._owns_client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, *,
                       tolerate_status: tuple[int, ...] = (), **kwargs):
        # Off-chain DB unreachable -> fail fast (guide §4.7).
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OffchainDBError(
                f"off-chain DB unreachable: {method} {self.base_url}{path}: {exc}"
            ) from exc
        if response.status_code in tolerate_status:
            # A route the production API does not register: log and carry on
            # (see `update_order`).
            logger.debug("off-chain DB answered %s on %s %s — treated as a "
                         "no-op", response.status_code, method, path)
            return None
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
                           energy: float | None = None) -> dict | None:
        """Mark an order Executed/Expired after the clearing.

        The production service registers no PATCH route for orders
        (gsy-offchain-storage/src/startup.rs @ aa99ea2): there the blockchain
        event listener drives order status. The AMM contract emits a single
        MarketCleared event rather than per-order events, so the Clearing Node
        updates statuses directly against the mock, and a 404/405 from a DB
        without the route is a no-op — a missing status update must never
        abort a clearing run. Returns None in that case.
        """
        patch: dict = {}
        if status is not None:
            patch["status"] = status
        if energy is not None:
            patch["energy"] = energy
        return await self._request("PATCH", f"/orders/{order_id}", json=patch,
                                   tolerate_status=(404, 405))
