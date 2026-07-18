"""AMM Execution Node — FastAPI service (guide §5).

Exposes:
    GET  /health             liveness probe
    POST /trigger-execution  explicit trigger (used by the demo UI)

Production trigger is a polling loop (guide §5.2): every
POLLING_INTERVAL_SEC it executes all markets whose delivery window closed
|EXECUTION_OFFSET_MIN| minutes ago. The loop ships here too (set
POLLING_ENABLED=true and poll_communities) but is disabled by default in the
PoC, where delivery time is simulated and the UI triggers explicitly.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import Config, load_config
from src.execution import run_execution, run_polling_cycle
from src.offchain_db import OffchainDBClient, OffchainDBError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("amm-execution-node")


class TriggerExecution(BaseModel):
    market_id: str = Field(min_length=1)
    community_uuid: str = Field(min_length=1)
    time_slot: int


async def _polling_loop(cfg: Config, db: OffchainDBClient) -> None:
    logger.info("polling loop enabled: every %ss, offset %s min, communities=%s",
                cfg.polling_interval_sec, cfg.execution_offset_min,
                list(cfg.poll_communities))
    while True:
        try:
            await run_polling_cycle(cfg, db)
        except OffchainDBError as exc:
            logger.error("polling cycle failed (off-chain DB): %s", exc)
        except Exception:  # noqa: BLE001 - keep the loop alive
            logger.exception("polling cycle failed")
        await asyncio.sleep(cfg.polling_interval_sec)


def create_app(cfg: Config | None = None,
               db: OffchainDBClient | None = None) -> FastAPI:
    cfg = cfg or load_config()
    db = db or OffchainDBClient(cfg.offchain_db_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if cfg.polling_enabled:
            task = asyncio.create_task(_polling_loop(cfg, db))
        logger.info("AMM Execution Node up: db=%s polling=%s gamma=%s eta=%s",
                    cfg.offchain_db_url, cfg.polling_enabled,
                    cfg.penalty_gamma, cfg.penalty_eta_kwh)
        yield
        if task is not None:
            task.cancel()
        await db.close()

    app = FastAPI(title="AMM Execution Node",
                  description=__doc__, version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.db = db

    # The demo UI calls this service directly from the browser.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(OffchainDBError)
    async def offchain_db_error(_request: Request, exc: OffchainDBError):
        logger.error("off-chain DB failure: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "amm-execution-node",
                "polling_enabled": cfg.polling_enabled}

    @app.post("/trigger-execution")
    async def trigger_execution(trigger: TriggerExecution) -> dict:
        return await run_execution(trigger.model_dump(), cfg, db)

    return app


app = create_app()
