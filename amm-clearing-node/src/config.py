"""Configuration loading: configuration.yaml with environment-variable overrides.

Environment variables always win (same pattern as the GSY-DEX services,
guide §4.3). Recognized overrides:

    OFFCHAIN_DB_URL, RPC_URL, CONTRACT_ADDRESS, CLEARING_NODE_PRIVATE_KEY,
    TIME_SLOT_SEC, BLOCKCHAIN_MODE, HOST, PORT, CONFIG_FILE
"""

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

logger = logging.getLogger("amm-clearing-node.config")

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configuration.yaml"


@dataclass(frozen=True)
class CommunityParams:
    k_upper: float = 28.5   # retail buy price, ct/kWh
    k_lower: float = 8.0    # feed-in tariff, ct/kWh
    theta: float = 1.0      # sigmoid midpoint (optimized offline)
    steepness: float = 2.5  # sigmoid steepness B (optimized offline)
    pool_id: str | None = None

    def pool_for(self, community_uuid: str) -> str:
        # TODO(confirm-with-supervisor): pool naming convention and whether
        # the pool must be registered as an area in `community_areas`
        # (guide §7.1). Current plan: AMM_POOL_{community_uuid}.
        return self.pool_id or f"AMM_POOL_{community_uuid}"


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8081
    offchain_db_url: str = "http://localhost:8080"
    blockchain_mode: str = "mock"          # "mock" | "live"
    rpc_url: str = ""
    contract_address: str = ""
    private_key: str = ""
    time_slot_sec: int = 900
    communities: dict = field(default_factory=dict)  # community_uuid -> CommunityParams
    default_community: CommunityParams = field(default_factory=CommunityParams)


def _community_from_yaml(raw: dict) -> CommunityParams:
    return CommunityParams(
        k_upper=float(raw.get("k_upper_ct_per_kwh", 28.5)),
        k_lower=float(raw.get("k_lower_ct_per_kwh", 8.0)),
        theta=float(raw.get("theta", 1.0)),
        steepness=float(raw.get("steepness", 2.5)),
        pool_id=raw.get("pool_id"),
    )


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(os.environ.get("CONFIG_FILE", path or DEFAULT_CONFIG_FILE))
    raw: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        logger.warning("configuration file %s not found, using defaults", path)

    application = raw.get("application") or {}
    offchain_db = raw.get("offchain_db") or {}
    blockchain = raw.get("blockchain") or {}
    market = raw.get("market") or {}
    communities_raw = raw.get("communities") or {}

    cfg = Config(
        host=application.get("host", "0.0.0.0"),
        port=int(application.get("port", 8081)),
        offchain_db_url=offchain_db.get("base_url", "http://localhost:8080"),
        blockchain_mode=str(blockchain.get("mode", "mock")).lower(),
        rpc_url=blockchain.get("rpc_url", ""),
        contract_address=blockchain.get("contract_address", ""),
        private_key="",
        time_slot_sec=int(market.get("time_slot_sec", 900)),
        communities={uuid: _community_from_yaml(c or {})
                     for uuid, c in communities_raw.items()},
        default_community=_community_from_yaml(raw.get("default_community") or {}),
    )

    # Environment variables always win (guide §4.3).
    env = os.environ
    overrides: dict = {}
    if env.get("OFFCHAIN_DB_URL"):
        overrides["offchain_db_url"] = env["OFFCHAIN_DB_URL"]
    if env.get("RPC_URL"):
        overrides["rpc_url"] = env["RPC_URL"]
    if env.get("CONTRACT_ADDRESS"):
        overrides["contract_address"] = env["CONTRACT_ADDRESS"]
    if env.get("CLEARING_NODE_PRIVATE_KEY"):
        overrides["private_key"] = env["CLEARING_NODE_PRIVATE_KEY"]
    if env.get("TIME_SLOT_SEC"):
        overrides["time_slot_sec"] = int(env["TIME_SLOT_SEC"])
    if env.get("BLOCKCHAIN_MODE"):
        overrides["blockchain_mode"] = env["BLOCKCHAIN_MODE"].lower()
    if env.get("HOST"):
        overrides["host"] = env["HOST"]
    if env.get("PORT"):
        overrides["port"] = int(env["PORT"])
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def resolve_community(cfg: Config, community_uuid: str,
                      overrides: dict | None = None) -> CommunityParams:
    """Sigmoid parameters for a community.

    Falls back to `default_community` for unknown communities. `overrides`
    lets the trigger payload supply parameters directly — a PoC convenience
    for the demo UI. TODO(confirm-with-supervisor): in production, parameter
    governance lives in the contract owner / configuration, not the trigger.
    """
    params = cfg.communities.get(community_uuid, cfg.default_community)
    if overrides:
        known = {k: float(v) for k, v in overrides.items()
                 if k in ("k_upper", "k_lower", "theta", "steepness")
                 and v is not None}
        if known:
            params = replace(params, **known)
    return params
