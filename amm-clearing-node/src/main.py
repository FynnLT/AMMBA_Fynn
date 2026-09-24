"""AMM Clearing Node — FastAPI service (guide §4).

Exposes:
    GET  /health            liveness probe
    POST /trigger-clearing  called by the Market Orchestrator at market close

PoC note on the trigger response: the guide (§4.2) allows synchronous
processing "if the clearing is fast enough", which it is for community-sized
markets. The PoC therefore runs the cycle synchronously and returns 200 with
the full clearing result so the demo UI can render it directly. A production
deployment would return 202 Accepted and process in a background task.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.clearing import run_clearing
from src.config import Config, PreferenceConfigError, load_config
from src.contract import BaseContractClient, ContractError, build_contract_client
from src.offchain_db import OffchainDBClient, OffchainDBError
from src.preferences import AllocationError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("amm-clearing-node")


class SigmoidParams(BaseModel):
    """PoC convenience: the demo UI passes the community's sigmoid parameters
    with the trigger. NOTE(poc-scope): a PoC convenience for the demo UI and
    the evaluation harness. In production the parameters belong to the
    configuration and the contract owner; what the artifact enforces today
    is the start-up check against the on-chain parameters
    (`verify_community_params`)."""
    k_upper: float | None = None
    k_lower: float | None = None
    theta: float | None = None
    steepness: float | None = None


class PreferenceParams(BaseModel):
    """PoC convenience, same rationale as `SigmoidParams`: the demo UI passes
    the preference settings with the trigger so both allocation orders and
    both multiplier modes are runnable without a redeploy.
    NOTE(poc-scope): a PoC convenience for the demo UI and the evaluation
    harness. In production the parameters belong to the configuration and
    the contract owner; what the artifact enforces today is the start-up
    check against the on-chain parameters (`verify_community_params`)."""
    enabled: bool | None = None
    order: str | None = None
    multipliers_enabled: bool | None = None
    mode: str | None = None
    sides: str | None = None
    green_multiplier: float | None = None
    grey_levy: float | None = None
    levy_cap: float | None = None


class TriggerClearing(BaseModel):
    market_id: str = Field(min_length=1)
    community_uuid: str = Field(min_length=1)
    time_slot: int
    community_name: str | None = None
    sigmoid_params: SigmoidParams | None = None
    preference_params: PreferenceParams | None = None


def create_app(cfg: Config | None = None,
               db: OffchainDBClient | None = None,
               chain: BaseContractClient | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or OffchainDBClient(cfg.offchain_db_url)
    chain = chain or build_contract_client(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("AMM Clearing Node up: db=%s blockchain_mode=%s "
                    "time_slot_sec=%s", cfg.offchain_db_url,
                    cfg.blockchain_mode, cfg.time_slot_sec)
        # Verified once per start, not per clearing: one check is the
        # governance statement, while a check per clearing would add an RPC
        # call to every slot and distort the evaluation's runtime figures.
        # A ContractError here is deliberately not caught — the service must
        # not come up with parameters that differ from the contract's.
        for community_uuid, params in cfg.communities.items():
            await chain.verify_community_params(community_uuid, params)
        if cfg.communities:
            logger.info("verified on-chain community parameters for %s "
                        "(blockchain_mode=%s)",
                        ", ".join(cfg.communities), cfg.blockchain_mode)
        yield
        await db.close()

    app = FastAPI(title="AMM Clearing Node",
                  description=__doc__, version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.db = db
    app.state.chain = chain

    # The demo UI calls this service directly from the browser.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # Upstream failures map to 502. On a contract failure no trades were
    # written; the orchestrator may re-trigger (guide §4.7).
    @app.exception_handler(OffchainDBError)
    async def offchain_db_error(_request: Request, exc: OffchainDBError):
        logger.error("off-chain DB failure: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ContractError)
    async def contract_error(_request: Request, exc: ContractError):
        logger.error("contract failure: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    # The conservation invariant failing is a bug in this service, not a bad
    # request and not an upstream outage — but it answers with a defined
    # error rather than an unhandled AssertionError.
    @app.exception_handler(AllocationError)
    async def allocation_error(_request: Request, exc: AllocationError):
        logger.error("allocation invariant violated: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # An unusable preference override is a client error, not an outage.
    @app.exception_handler(PreferenceConfigError)
    async def preference_config_error(_request: Request,
                                      exc: PreferenceConfigError):
        logger.error("invalid preference parameters: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "amm-clearing-node",
                "blockchain_mode": chain.mode}

    @app.post("/trigger-clearing")
    async def trigger_clearing(trigger: TriggerClearing) -> dict:
        payload = trigger.model_dump()
        for key in ("sigmoid_params", "preference_params"):
            if payload.get(key) is not None:
                payload[key] = {k: v for k, v in payload[key].items()
                                if v is not None}
        return await run_clearing(payload, cfg, db, chain)

    return app


app = create_app()
