"""Async HTTP client for the off-chain DB endpoints used by the Execution Node.

Route names and filter semantics follow the production service
(gsy-decentralized-exchange main @ aa99ea2,
gsy-offchain-storage/src/startup.rs), so pointing `OFFCHAIN_DB_URL` at the
real DB needs no code change here.
"""

import logging

import httpx

logger = logging.getLogger("amm-execution-node.offchain_db")


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
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OffchainDBError(
                f"off-chain DB unreachable: {method} {self.base_url}{path}: {exc}"
            ) from exc
        if response.status_code in tolerate_status:
            # A route the production API does not register: log and carry on
            # (see `update_trade`).
            logger.debug("off-chain DB answered %s on %s %s — treated as a "
                         "no-op", response.status_code, method, path)
            return None
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
        """Trades of one market.

        The production route accepts `market_id` but its server-side filter is
        commented out (gsy-offchain-storage @ aa99ea2), so a live DB answers
        with the whole time window. Filtering again here makes the Execution
        Node behave identically against the mock and the real service.
        """
        trades = await self._request("GET", "/trades",
                                     params={"market_id": market_id})
        return [t for t in (trades or []) if t.get("market_id") == market_id]

    @staticmethod
    def _slot_window(time_slot: int, time_slot_sec: int) -> dict:
        """One delivery slot as the inclusive bounds the DB filters on.

        `time_slot + time_slot_sec` is already the *next* slot, so the upper
        bound is one second below it: half-open [t, t + Δ) in effect. Same
        convention as the Clearing Node's order window, and the same on both
        measurement channels."""
        return {"start_time": int(time_slot),
                "end_time": int(time_slot) + int(time_slot_sec) - 1}

    async def get_measurements(self, community_uuid: str, time_slot: int,
                               time_slot_sec: int,
                               area_uuid: str | None = None) -> list[dict]:
        """Post-delivery smart-meter data for one slot. Omit `area_uuid` to
        fetch the whole community in one request."""
        params: dict = {"community_uuid": community_uuid,
                        **self._slot_window(time_slot, time_slot_sec)}
        if area_uuid is not None:
            params["area_uuid"] = area_uuid
        return await self._request("GET", "/measurements", params=params)

    async def get_forecasts(self, community_uuid: str, time_slot: int,
                            time_slot_sec: int,
                            area_uuid: str | None = None) -> list[dict]:
        """Forecast data for one slot — what an area expected to deliver or
        consume, queried exactly like `get_measurements`."""
        params: dict = {"community_uuid": community_uuid,
                        **self._slot_window(time_slot, time_slot_sec)}
        if area_uuid is not None:
            params["area_uuid"] = area_uuid
        return await self._request("GET", "/forecasts", params=params)

    async def get_community_markets(self, community_uuid: str) -> list[dict]:
        return await self._request(
            "GET", "/community-market",
            params={"community_uuid": community_uuid})

    async def update_trade(self, trade_uuid: str, *, status: str | None = None,
                           parameters: dict | None = None) -> dict | None:
        """Write the penalty result back onto the trade.

        The penalty result extends the trade `parameters`, an extension of
        the published GSY interface declared as such (D-48). The published
        off-chain storage has no route for it, hence the mock-only
        `PATCH /trades`.

        The production service registers no PATCH route
        (gsy-offchain-storage/src/startup.rs @ aa99ea2), so a 404/405 is a
        no-op and returns None rather than aborting the execution run.
        """
        patch: dict = {}
        if status is not None:
            patch["status"] = status
        if parameters is not None:
            patch["parameters"] = parameters
        return await self._request("PATCH", f"/trades/{trade_uuid}", json=patch,
                                   tolerate_status=(404, 405))
