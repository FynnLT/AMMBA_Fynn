"""Configuration loading: configuration.yaml with environment-variable overrides.

Environment variables always win (same pattern as the GSY-DEX services,
guide §4.3). Recognized overrides:

    OFFCHAIN_DB_URL, RPC_URL, CONTRACT_ADDRESS, CLEARING_NODE_PRIVATE_KEY,
    TIME_SLOT_SEC, BLOCKCHAIN_MODE, HOST, PORT, CONFIG_FILE

    PREFERENCES_ENABLED, PREFERENCE_ORDER, MULTIPLIERS_ENABLED,
    MULTIPLIER_MODE, MULTIPLIER_SIDES, GREEN_MULTIPLIER, GREY_LEVY, LEVY_CAP
"""

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

logger = logging.getLogger("amm-clearing-node.config")

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configuration.yaml"

# Environment variables always win (guide §4.3): env var -> (Config field, cast)
_ENV_OVERRIDES = {
    "OFFCHAIN_DB_URL": ("offchain_db_url", str),
    "RPC_URL": ("rpc_url", str),
    "CONTRACT_ADDRESS": ("contract_address", str),
    "CLEARING_NODE_PRIVATE_KEY": ("private_key", str),
    "MIN_PRIORITY_FEE_WEI": ("min_priority_fee_wei", int),
    "TIME_SLOT_SEC": ("time_slot_sec", int),
    "BLOCKCHAIN_MODE": ("blockchain_mode", str.lower),
    "HOST": ("host", str),
    "PORT": ("port", int),
}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# Preference overrides live in a nested dataclass, so they need their own
# table: env var -> (PreferenceConfig field, cast).
_PREFERENCE_ENV_OVERRIDES = {
    "PREFERENCES_ENABLED": ("enabled", _as_bool),
    "PREFERENCE_ORDER": ("order", str),
    "MULTIPLIERS_ENABLED": ("multipliers_enabled", _as_bool),
    "MULTIPLIER_MODE": ("mode", str),
    "MULTIPLIER_SIDES": ("sides", str),
    "GREEN_MULTIPLIER": ("green_multiplier", float),
    "GREY_LEVY": ("grey_levy", float),
    "LEVY_CAP": ("levy_cap", float),
}

# Allocation order, multiplier formulation and the side(s) the multiplier is
# applied to stay runnable as configuration axes (D-19): `preferences_first`
# is the design and `pro_rata_first` the comparison variant, and
# `sides = "seller"` is the evaluation default (D-30).
PREFERENCE_ORDERS = ("preferences_first", "pro_rata_first")
MULTIPLIER_MODES = ("multiplicative", "additive")
MULTIPLIER_SIDES = ("seller", "both")


class PreferenceConfigError(ValueError):
    """Invalid preference configuration (yaml, env override or trigger)."""


@dataclass(frozen=True)
class CommunityParams:
    k_upper: float = 40.0   # retail buy price, ct/kWh
    k_lower: float = 8.0    # feed-in tariff, ct/kWh
    theta: float = 1.0      # sigmoid midpoint (optimized offline)
    steepness: float = 0.6  # sigmoid steepness B (optimized offline)
    pool_id: str | None = None

    def pool_for(self, community_uuid: str) -> str:
        # NOTE(poc-scope): the PoC names the pool AMM_POOL_{community_uuid}
        # and does not register it as an area in `community_areas`. A
        # production naming scheme is outside the proof of concept.
        return self.pool_id or f"AMM_POOL_{community_uuid}"


@dataclass(frozen=True)
class PreferenceConfig:
    """Phase-2 preference mechanisms (guide §4.4 step 6, §6).

    `order` selects whether mutually preferred pairs are served before the
    pro-rata split ("preferences_first") or whether the pairs are only
    detected and reported while quantities stay purely pro-rata
    ("pro_rata_first", the pre-Phase-2 baseline).

    `mode` selects the multiplier formulation: in "multiplicative" the levy
    is the parameter, the bonus is scaled to what the levy funds, and any
    over-collection stays with the pool; in "additive" the bonus is the
    parameter, the levy follows from funding it, and the round is zero-sum.

    `sides` selects who sees an adjusted rate: only sellers (current UI
    semantics) or buyers as well.
    """

    enabled: bool = True
    order: str = "preferences_first"
    multipliers_enabled: bool = True
    mode: str = "multiplicative"
    sides: str = "seller"
    green_multiplier: float = 0.10
    grey_levy: float = 0.10
    levy_cap: float = 0.20

    def __post_init__(self) -> None:
        # Validated here rather than in load_config so env overrides and
        # per-request overrides are checked with the same rules.
        for name, value, allowed in (
                ("order", self.order, PREFERENCE_ORDERS),
                ("mode", self.mode, MULTIPLIER_MODES),
                ("sides", self.sides, MULTIPLIER_SIDES)):
            if value not in allowed:
                raise PreferenceConfigError(
                    f"preferences.{name} must be one of {list(allowed)}, "
                    f"got {value!r}")
        for name, value in (("green_multiplier", self.green_multiplier),
                            ("grey_levy", self.grey_levy),
                            ("levy_cap", self.levy_cap)):
            if not isinstance(value, (int, float)) or value < 0:
                raise PreferenceConfigError(
                    f"preferences.multipliers.{name} must be >= 0, "
                    f"got {value!r}")

    def as_dict(self) -> dict:
        """Run-level preference settings as written into trade `parameters`
        (so the idempotent re-trigger path can rebuild the summary)."""
        return {
            "enabled": self.enabled,
            "order": self.order,
            "multipliers_enabled": self.multipliers_enabled,
            "mode": self.mode,
            "sides": self.sides,
            "green_multiplier": self.green_multiplier,
            "grey_levy": self.grey_levy,
            "levy_cap": self.levy_cap,
        }


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8081
    offchain_db_url: str = "http://localhost:8080"
    blockchain_mode: str = "mock"          # "mock" | "live"
    rpc_url: str = ""
    contract_address: str = ""
    private_key: str = ""
    # Floor for the EIP-1559 priority fee, in wei. web3 derives the priority
    # fee from recent blocks; on a PoA chain whose blocks are almost always
    # empty that estimate is 0, the transaction is priced at 2 x base fee and
    # validators never include it. Observed on Volta 2026-08-29.
    min_priority_fee_wei: int = 1_000_000_000  # 1 gwei
    time_slot_sec: int = 900
    communities: dict = field(default_factory=dict)  # community_uuid -> CommunityParams
    default_community: CommunityParams = field(default_factory=CommunityParams)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)


def _preferences_from_yaml(raw: dict) -> PreferenceConfig:
    multipliers = raw.get("multipliers") or {}
    defaults = PreferenceConfig()
    return PreferenceConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        order=str(raw.get("order", defaults.order)),
        multipliers_enabled=bool(
            multipliers.get("enabled", defaults.multipliers_enabled)),
        mode=str(multipliers.get("mode", defaults.mode)),
        sides=str(multipliers.get("sides", defaults.sides)),
        green_multiplier=float(multipliers.get("green_multiplier",
                                               defaults.green_multiplier)),
        grey_levy=float(multipliers.get("grey_levy", defaults.grey_levy)),
        levy_cap=float(multipliers.get("levy_cap", defaults.levy_cap)),
    )


def _community_from_yaml(raw: dict) -> CommunityParams:
    return CommunityParams(
        k_upper=float(raw.get("k_upper_ct_per_kwh", 40.0)),
        k_lower=float(raw.get("k_lower_ct_per_kwh", 8.0)),
        theta=float(raw.get("theta", 1.0)),
        steepness=float(raw.get("steepness", 0.6)),
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
        min_priority_fee_wei=int(blockchain.get("min_priority_fee_wei",
                                                1_000_000_000)),
        time_slot_sec=int(market.get("time_slot_sec", 900)),
        communities={uuid: _community_from_yaml(c or {})
                     for uuid, c in communities_raw.items()},
        default_community=_community_from_yaml(raw.get("default_community") or {}),
        preferences=_preferences_from_yaml(raw.get("preferences") or {}),
    )

    env = os.environ
    overrides = {field: cast(env[name])
                 for name, (field, cast) in _ENV_OVERRIDES.items()
                 if env.get(name)}
    if overrides:
        cfg = replace(cfg, **overrides)

    pref_overrides = {field: cast(env[name])
                      for name, (field, cast) in _PREFERENCE_ENV_OVERRIDES.items()
                      if env.get(name)}
    if pref_overrides:
        cfg = replace(cfg, preferences=replace(cfg.preferences, **pref_overrides))
    return cfg


def resolve_community(cfg: Config, community_uuid: str,
                      overrides: dict | None = None) -> CommunityParams:
    """Sigmoid parameters for a community.

    Falls back to `default_community` for unknown communities. `overrides`
    lets the trigger payload supply parameters directly.
    NOTE(poc-scope): a PoC convenience for the demo UI and the evaluation
    harness. In production the parameters belong to the configuration and
    the contract owner; what the artifact enforces today is the start-up
    check against the on-chain parameters (`verify_community_params`).
    """
    params = cfg.communities.get(community_uuid, cfg.default_community)
    if overrides:
        known = {k: float(v) for k, v in overrides.items()
                 if k in ("k_upper", "k_lower", "theta", "steepness")
                 and v is not None}
        if known:
            params = replace(params, **known)
    return params


_PREFERENCE_OVERRIDE_CASTS = {
    "enabled": bool, "order": str, "multipliers_enabled": bool,
    "mode": str, "sides": str, "green_multiplier": float,
    "grey_levy": float, "levy_cap": float,
}


def resolve_preferences(cfg: Config,
                        overrides: dict | None = None) -> PreferenceConfig:
    """Preference settings for one clearing run.

    `overrides` lets the trigger payload supply the parameters directly —
    the same PoC convenience as `sigmoid_params`, so both `order` variants
    and both multiplier `mode`s are runnable from the demo UI without a
    redeploy. NOTE(poc-scope): a PoC convenience for the demo UI and the
    evaluation harness. In production the parameters belong to the
    configuration and the contract owner; what the artifact enforces today
    is the start-up check against the on-chain parameters
    (`verify_community_params`).
    """
    params = cfg.preferences
    if overrides:
        known = {k: _PREFERENCE_OVERRIDE_CASTS[k](v)
                 for k, v in overrides.items()
                 if k in _PREFERENCE_OVERRIDE_CASTS and v is not None}
        if known:
            params = replace(params, **known)
    return params
