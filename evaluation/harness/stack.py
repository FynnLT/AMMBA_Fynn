"""In-process AMMBA stack for the pilot campaign.

Loads the three services' `src` packages under distinct names (they all use
`from src.x import y`, so they cannot be imported naively side by side) and
wires the two off-chain DB clients to the mock DB's FastAPI app through an
httpx ASGITransport. No real ports, no docker.

NOTE: the transport is injected through the `client=` parameter of
`OffchainDBClient.__init__`, which landed in the repository (both nodes guard
`close()` with an ownership flag, so the shared client is not closed twice).
The pilot's `db._client = ...` monkey-patch is therefore no longer needed.

`REPO` is derived from this file's own location: the harness directory and the
clone sit side by side, so the harness is path-portable (the pilot ran on
Linux with a hard-coded /tmp path; this runs on Windows).
"""
import asyncio, importlib, sys, os, copy
from dataclasses import replace
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent


def _find_repo(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "mock-offchain-db").is_dir():
            return candidate
    raise RuntimeError(f"no AMMBA repository root above {start}")


REPO = _find_repo(HARNESS_DIR)

def _load(root, modules, tag):
    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(sys.modules):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]
    sys.path.insert(0, root)
    try:
        mods = {m: importlib.import_module("src." + m) for m in modules}
    finally:
        sys.path.pop(0)
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        sys.modules.update(saved)
    return mods

MOCK = _load(str(REPO / "mock-offchain-db"), ["main", "store"], "db")
CLR  = _load(str(REPO / "amm-clearing-node"),
             ["main", "clearing", "config", "contract", "offchain_db",
              "preferences", "sigmoid", "trade_builder"], "clr")
EXE  = _load(str(REPO / "amm-execution-node"),
             ["main", "execution", "config", "offchain_db", "penalties",
              "sigmoid"], "exe")

import httpx


class Stack:
    def __init__(self):
        self.app = MOCK["main"].create_app()
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://mockdb", timeout=30)
        # Real dependency injection: the client is owned by this Stack and
        # closed in `close()`, not by either DB client.
        self.db_clr = CLR["offchain_db"].OffchainDBClient("http://mockdb",
                                                          client=self.http)
        self.db_exe = EXE["offchain_db"].OffchainDBClient("http://mockdb",
                                                          client=self.http)
        self.chain = CLR["contract"].MockContractClient()
        self.cfg_clr = CLR["config"].load_config()
        self.cfg_exe = EXE["config"].load_config()

    async def close(self):
        await self.http.aclose()

    # --- raw REST helpers against the mock DB ---
    async def post(self, path, json):
        r = await self.http.post(path, json=json)
        r.raise_for_status()
        return r.json()

    async def get(self, path, **params):
        r = await self.http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def reset(self):
        await self.http.post("/reset")
        self.chain = CLR["contract"].MockContractClient()

    # --- pipeline ---
    async def clear(self, trigger, preferences=None):
        cfg = self.cfg_clr
        if preferences is not None:
            cfg = replace(cfg, preferences=replace(cfg.preferences, **preferences))
        return await CLR["clearing"].run_clearing(trigger, cfg, self.db_clr, self.chain)

    async def execute(self, trigger, gamma=None, eta=None):
        cfg = self.cfg_exe
        kw = {}
        if gamma is not None: kw["penalty_gamma"] = gamma
        if eta is not None: kw["penalty_eta_kwh"] = eta
        if kw: cfg = replace(cfg, **kw)
        return await EXE["execution"].run_execution(trigger, cfg, self.db_exe)


def sigmoid_price(ratio, k_upper=28.5, k_lower=8.0, theta=1.0, steepness=2.5):
    return CLR["sigmoid"].sigmoid_price(ratio, k_upper, k_lower, theta, steepness)

penalties = EXE["penalties"]
