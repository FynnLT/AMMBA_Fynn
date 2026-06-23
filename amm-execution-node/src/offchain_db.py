"""Async HTTP client for the off-chain DB endpoints used by the Execution Node."""

import logging

import httpx

logger = logging.getLogger("amm-execution-node.offchain_db")


class OffchainDBError(RuntimeError):
    """Raised when the off-chain DB is unreachable or rejects a request."""


class OffchainDBClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs):
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OffchainDBError(
                f"off-chain DB unreachable: {method} {self.base_url}{path}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise OffchainDBError(
                f"off-chain DB error {response.status_code} on "
                f"{method} {path}: {response.text[:500]}")
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def health_check(self) -> dict:
        return await self._request("GET", "/health_check")

    async def get_trades(self, market_id: str) -> list[dict]:
        return await self._request("GET", "/trades",
                                   params={"market_id": market_id})

    async def get_measurements(self, community_uuid: str,
                               area_uuid: str) -> list[dict]:
        """Post-delivery smart-meter data for one asset (guide §2)."""
        return await self._request(
            "GET", "/asset_measurements",
            params={"community_uuid": community_uuid, "area_uuid": area_uuid})

    async def get_community_markets(self, community_uuid: str) -> list[dict]:
        return await self._request(
            "GET", "/community-market",
            params={"community_uuid": community_uuid})

    async def update_trade(self, trade_uuid: str, *, status: str | None = None,
                           parameters: dict | None = None) -> dict:
        # TODO(confirm-with-supervisor): penalty output schema (guide §7.5) —
        # extend trade `parameters` (current PoC approach, via a mock-only
        # PATCH endpoint) or a separate /penalties endpoint?
        patch: dict = {}
        if status is not None:
            patch["status"] = status
        if parameters is not None:
            patch["parameters"] = parameters
        return await self._request("PATCH", f"/trades/{trade_uuid}", json=patch)
