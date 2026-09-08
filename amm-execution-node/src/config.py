"""Configuration loading: configuration.yaml with environment-variable overrides.

Recognized overrides (env always wins, guide §4.3 pattern):
    OFFCHAIN_DB_URL, TIME_SLOT_SEC, EXECUTION_OFFSET_MIN,
    POLLING_INTERVAL_SEC, POLLING_ENABLED, PENALTY_GAMMA,
    PENALTY_TOLERANCE_KWH, PENALTY_ETA_RELATIVE, HOST, PORT, CONFIG_FILE

RPC_URL / CONTRACT_ADDRESS are accepted (the compose file passes them for
forward-compatibility) but unused: the Execution Node is fully self-contained
from trade objects (guide §5.4) and never talks to the chain.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configuration.yaml"


@dataclass(frozen=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8082
    offchain_db_url: str = "http://localhost:8080"
    time_slot_sec: int = 900
    execution_offset_min: int = -120   # GSY-DEX default: 2h after delivery start
    polling_interval_sec: int = 300
    polling_enabled: bool = False      # PoC default: triggered via the demo UI
    poll_communities: tuple = ()       # community uuids scanned by the poller
    penalty_gamma: float = 1.1         # γ > 1, K_sho = γ * K_upper
    penalty_eta_kwh: float = 0.0       # η tolerance threshold, absolute kWh
    # η relative to the trade's own quantity (D-26/D-44). None keeps the
    # absolute `penalty_eta_kwh` path, which is the reproduction path for the
    # runs recorded before 09/2026.
    penalty_eta_relative: float | None = None


def _to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


# Environment variables always win (guide §4.3): env var -> (Config field, cast)
_ENV_OVERRIDES = {
    "OFFCHAIN_DB_URL": ("offchain_db_url", str),
    "TIME_SLOT_SEC": ("time_slot_sec", int),
    "EXECUTION_OFFSET_MIN": ("execution_offset_min", int),
    "POLLING_INTERVAL_SEC": ("polling_interval_sec", int),
    "POLLING_ENABLED": ("polling_enabled", _to_bool),
    "PENALTY_GAMMA": ("penalty_gamma", float),
    "PENALTY_TOLERANCE_KWH": ("penalty_eta_kwh", float),
    "PENALTY_ETA_RELATIVE": ("penalty_eta_relative", float),
    "HOST": ("host", str),
    "PORT": ("port", int),
}


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(os.environ.get("CONFIG_FILE", path or DEFAULT_CONFIG_FILE))
    raw: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    application = raw.get("application") or {}
    offchain_db = raw.get("offchain_db") or {}
    execution = raw.get("execution") or {}
    penalty = raw.get("penalty") or {}

    cfg = Config(
        host=application.get("host", "0.0.0.0"),
        port=int(application.get("port", 8082)),
        offchain_db_url=offchain_db.get("base_url", "http://localhost:8080"),
        time_slot_sec=int(execution.get("time_slot_sec", 900)),
        execution_offset_min=int(execution.get("execution_offset_min", -120)),
        polling_interval_sec=int(execution.get("polling_interval_sec", 300)),
        polling_enabled=bool(execution.get("polling_enabled", False)),
        poll_communities=tuple(execution.get("poll_communities") or ()),
        penalty_gamma=float(penalty.get("gamma", 1.1)),
        penalty_eta_kwh=float(penalty.get("eta_tolerance_kwh", 0.0)),
        penalty_eta_relative=(None if penalty.get("eta_relative") is None
                              else float(penalty["eta_relative"])),
    )

    env = os.environ
    overrides = {field: cast(env[name])
                 for name, (field, cast) in _ENV_OVERRIDES.items()
                 if env.get(name)}
    if overrides:
        cfg = replace(cfg, **overrides)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Checked after the env overrides, so both channels obey the same rule.

    `penalty_eta_relative >= 1.0` would mean the deadband swallows the whole
    trade and no deviation could ever be penalised; a negative one is not a
    tolerance at all. The service has no configuration-error type of its own,
    so this raises `ValueError` at load time.
    """
    eta_relative = cfg.penalty_eta_relative
    if eta_relative is None:
        return
    if eta_relative < 0.0 or eta_relative >= 1.0:
        raise ValueError(
            f"penalty.eta_relative must be in [0.0, 1.0), got {eta_relative!r}")
