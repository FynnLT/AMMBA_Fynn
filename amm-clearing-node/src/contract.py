"""AMM Smart Contract clients (guide §4.4 step 4).

Two interchangeable implementations:

* ``MockContractClient`` (default, ``BLOCKCHAIN_MODE=mock``): simulates the
  on-chain anchor locally and returns a deterministic-format simulated tx
  hash. Used because the PoC docker-compose stack runs without a blockchain
  node; the demo UI labels the hash as simulated.
* ``Web3ContractClient`` (``BLOCKCHAIN_MODE=live``): signs and sends a real
  ``clearMarket`` transaction via web3.py and waits for the receipt. Works
  against a local ``npx hardhat node`` (see amm-smart-contract/) or the
  Energy Web Volta testnet.
"""

import asyncio
import logging
import re
from uuid import uuid4

from src.config import Config
from src.sigmoid import to_node_int
from src.trade_builder import blake2b_hash

logger = logging.getLogger("amm-clearing-node.contract")

_HEX32_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")

# Minimal ABI: just what the Clearing Node calls.
AMMBA_ABI = [
    {
        "type": "function",
        "name": "clearMarket",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "market_id", "type": "bytes32"},
            {"name": "community_uuid", "type": "bytes32"},
            {"name": "time_slot", "type": "uint256"},
            {"name": "total_supply", "type": "uint256"},
            {"name": "total_demand", "type": "uint256"},
            {"name": "clearing_price", "type": "uint256"},
        ],
        "outputs": [{"name": "executed", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "getClearingResult",
        "stateMutability": "view",
        "inputs": [{"name": "market_id", "type": "bytes32"}],
        "outputs": [
            {"name": "time_slot", "type": "uint256"},
            {"name": "supply", "type": "uint256"},
            {"name": "demand", "type": "uint256"},
            {"name": "price", "type": "uint256"},
        ],
    },
]


class ContractError(RuntimeError):
    """Raised when the on-chain transaction fails (clearing must abort,
    guide §4.7: do not write trades, allow retry on re-trigger)."""


def hex_str_to_bytes32(value: str) -> bytes:
    """Convert a 0x-prefixed 32-byte hex string (e.g. a market_id hash)."""
    cleaned = value[2:] if value.startswith("0x") else value
    raw = bytes.fromhex(cleaned)
    if len(raw) != 32:
        raise ValueError(f"expected 32 bytes, got {len(raw)}: {value!r}")
    return raw


class BaseContractClient:
    mode = "base"

    async def clear_market(self, *, market_id: str, community_uuid: str,
                           time_slot: int, total_supply_kwh: float,
                           total_demand_kwh: float,
                           clearing_price: float) -> str:
        """Record the clearing on-chain; returns the transaction hash."""
        raise NotImplementedError


class MockContractClient(BaseContractClient):
    """Simulated on-chain anchor — no blockchain required.

    Mirrors the contract's stored state in memory so the rest of the clearing
    flow (tx hash in trade objects, idempotent re-trigger behavior) is
    exercised exactly as in live mode.
    """

    mode = "mock"

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def clear_market(self, *, market_id: str, community_uuid: str,
                           time_slot: int, total_supply_kwh: float,
                           total_demand_kwh: float,
                           clearing_price: float) -> str:
        record = {
            "market_id": market_id,
            "community_uuid": community_uuid,
            "time_slot": time_slot,
            "total_supply": to_node_int(total_supply_kwh),
            "total_demand": to_node_int(total_demand_kwh),
            "clearing_price": to_node_int(clearing_price),
        }
        # Salted so re-clearing attempts are distinguishable in logs/demos.
        tx_hash = blake2b_hash({**record, "salt": uuid4().hex})
        self.records[market_id] = {**record, "tx_hash": tx_hash}
        logger.info(
            "MOCK clearMarket(market_id=%s, community=%s, time_slot=%s, "
            "supply=%s, demand=%s, price=%s) -> simulated tx %s",
            market_id, community_uuid, time_slot, record["total_supply"],
            record["total_demand"], record["clearing_price"], tx_hash)
        return tx_hash


class Web3ContractClient(BaseContractClient):
    mode = "live"

    def __init__(self, rpc_url: str, contract_address: str,
                 private_key: str) -> None:
        # web3 imported lazily so mock mode runs without the dependency.
        from web3 import Web3

        self._Web3 = Web3
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._account = self._w3.eth.account.from_key(private_key)
        self._contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=AMMBA_ABI)
        logger.info("live contract client: %s as %s via %s",
                    contract_address, self._account.address, rpc_url)

    def _string_to_bytes32(self, value: str) -> bytes:
        """community_uuid strings (e.g. "communityid_1") are not 32 bytes.
        Convention: pass 32-byte hex through, otherwise keccak256(utf-8).
        Must match scripts/deploy.js `toBytes32`.
        TODO(confirm-with-supervisor): agree on the canonical conversion."""
        if _HEX32_RE.match(value):
            return hex_str_to_bytes32(value)
        return bytes(self._Web3.keccak(text=value))

    async def clear_market(self, *, market_id: str, community_uuid: str,
                           time_slot: int, total_supply_kwh: float,
                           total_demand_kwh: float,
                           clearing_price: float) -> str:
        return await asyncio.to_thread(
            self._clear_market_sync, market_id, community_uuid, time_slot,
            total_supply_kwh, total_demand_kwh, clearing_price)

    def _clear_market_sync(self, market_id: str, community_uuid: str,
                           time_slot: int, total_supply_kwh: float,
                           total_demand_kwh: float,
                           clearing_price: float) -> str:
        try:
            fn = self._contract.functions.clearMarket(
                hex_str_to_bytes32(market_id),
                self._string_to_bytes32(community_uuid),
                int(time_slot),
                to_node_int(total_supply_kwh),
                to_node_int(total_demand_kwh),
                to_node_int(clearing_price),
            )
            tx = fn.build_transaction({
                "from": self._account.address,
                "nonce": self._w3.eth.get_transaction_count(
                    self._account.address),
            })
            signed = self._account.sign_transaction(tx)
            # web3 v6 names it rawTransaction, v7 raw_transaction
            raw = getattr(signed, "raw_transaction", None)
            if raw is None:
                raw = signed.rawTransaction
            tx_hash = self._w3.eth.send_raw_transaction(raw)
            receipt = self._w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=120)
        except Exception as exc:  # noqa: BLE001 - normalize all web3 failures
            raise ContractError(f"clearMarket transaction failed: {exc}") from exc
        if receipt.status != 1:
            raise ContractError(
                f"clearMarket transaction reverted: {receipt.transactionHash!r}")
        return "0x" + bytes(receipt.transactionHash).hex()


def build_contract_client(cfg: Config) -> BaseContractClient:
    if cfg.blockchain_mode == "live":
        missing = [name for name, value in (
            ("RPC_URL", cfg.rpc_url),
            ("CONTRACT_ADDRESS", cfg.contract_address),
            ("CLEARING_NODE_PRIVATE_KEY", cfg.private_key)) if not value]
        if missing:
            raise ValueError(
                f"BLOCKCHAIN_MODE=live requires {', '.join(missing)}")
        return Web3ContractClient(cfg.rpc_url, cfg.contract_address,
                                  cfg.private_key)
    if cfg.blockchain_mode != "mock":
        raise ValueError(f"unknown BLOCKCHAIN_MODE {cfg.blockchain_mode!r}")
    return MockContractClient()
