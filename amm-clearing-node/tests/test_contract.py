"""Contract client behavior that the clearing tests do not exercise."""

import importlib.util

import pytest

from src.config import CommunityParams
from src.contract import ContractError, MockContractClient, Web3ContractClient
from src.sigmoid import to_node_int

MARKET = "0x" + "bb" * 32

# A fixed test input, not a configuration value -- the same reasoning as
# `GOLDEN_SIGMOID` in test_clearing.py. The x10,000 integers and the
# divergence messages in these tests are written out in full, so the band
# they were written for belongs in the test rather than being read from the
# config default. Taking `CommunityParams()` here instead would tie them to
# whatever the community is configured to, and they would break on every
# parameter change -- K_upper on 17.09.2026 (D-77), theta and B when the
# calibration lands (T-19).
GOLDEN_COMMUNITY = CommunityParams(k_upper=28.5, k_lower=8.0,
                                   theta=1.0, steepness=2.5)

# Constructing the live client needs a well-formed address and key, not a
# reachable node: web3's HTTP provider connects lazily, so nothing here
# touches a chain.
RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x" + "11" * 20
PRIVATE_KEY = "0x" + "22" * 32


def clearing_args(**overrides):
    args = {"market_id": MARKET, "community_uuid": "communityid_1",
            "time_slot": 900, "total_supply_kwh": 12.5,
            "total_demand_kwh": 10.0, "clearing_price": 15.1472}
    args.update(overrides)
    return args


@pytest.mark.anyio
async def test_mock_client_rejects_second_clearing_of_same_market():
    """The mock mirrors the contract's `market already cleared` revert."""
    chain = MockContractClient()
    tx_hash = await chain.clear_market(**clearing_args())
    assert tx_hash

    with pytest.raises(ContractError, match="already cleared"):
        await chain.clear_market(**clearing_args())

    # the first anchor is untouched by the rejected second attempt
    assert chain.records[MARKET]["tx_hash"] == tx_hash


@pytest.mark.anyio
async def test_mock_client_clears_distinct_markets():
    chain = MockContractClient()
    await chain.clear_market(**clearing_args())
    await chain.clear_market(**clearing_args(market_id="0x" + "cc" * 32))
    assert len(chain.records) == 2


# `web3` lives in requirements-live.txt, not requirements.txt. Installing only
# the base requirements — the documented path, and what a CI or an examiner
# cloning the repository does — turned the eight live-client tests below into
# failures instead of skips: nothing is broken there, only the signal (#29).
# The two mock-client tests above need no chain library and keep running.
requires_web3 = pytest.mark.skipif(
    importlib.util.find_spec("web3") is None,
    reason="web3 is in requirements-live.txt, not requirements.txt")


# ------------------------------------- on-chain parameter comparison (#17)

class FakeContractCall:
    def __init__(self, result):
        self._result = result

    def call(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeContractFunctions:
    def __init__(self, result):
        self._result = result
        self.called_with = None

    def getCommunityParams(self, community_uuid):  # noqa: N802 - ABI name
        self.called_with = community_uuid
        return FakeContractCall(self._result)


class FakeContract:
    def __init__(self, result):
        self.functions = FakeContractFunctions(result)


def live_client(on_chain_result) -> Web3ContractClient:
    """A live client whose contract call is canned.

    Everything above the call — the bytes32 conversion, the x10,000 integer
    comparison, the divergence message and the revert translation — is the
    real code, and none of it is reached by the `test_startup_*` tests, which
    stub the whole client out.
    """
    client = Web3ContractClient(RPC_URL, CONTRACT_ADDRESS, PRIVATE_KEY)
    client._contract = FakeContract(on_chain_result)
    return client


@requires_web3
def test_verify_community_params_accepts_matching_values():
    local = GOLDEN_COMMUNITY
    # The contract stores the parameters scaled by 10,000.
    on_chain = [to_node_int(local.k_upper), to_node_int(local.k_lower),
                to_node_int(local.theta), to_node_int(local.steepness)]
    assert on_chain == [285000, 80000, 10000, 25000]

    client = live_client(on_chain)
    client._verify_community_params_sync("communityid_1", local)  # no raise

    # a non-hex community uuid is keccak'd into bytes32
    assert len(client._contract.functions.called_with) == 32


@requires_web3
def test_verify_community_params_names_only_the_diverging_field():
    """#18: the message used to print `!=` for fields that agreed."""
    local = GOLDEN_COMMUNITY
    on_chain = [to_node_int(local.k_upper), to_node_int(local.k_lower),
                to_node_int(1.2), to_node_int(local.steepness)]

    with pytest.raises(ContractError) as excinfo:
        live_client(on_chain)._verify_community_params_sync("communityid_1",
                                                            local)
    message = str(excinfo.value)
    assert "theta: on-chain 1.2 != local 1.0" in message
    for untouched in ("k_upper", "k_lower", "steepness"):
        assert untouched not in message


@requires_web3
def test_verify_community_params_reports_every_diverging_field():
    local = GOLDEN_COMMUNITY
    on_chain = [to_node_int(30.0), to_node_int(local.k_lower),
                to_node_int(local.theta), to_node_int(3.0)]

    with pytest.raises(ContractError) as excinfo:
        live_client(on_chain)._verify_community_params_sync("communityid_1",
                                                            local)
    message = str(excinfo.value)
    assert "k_upper: on-chain 30.0 != local 28.5" in message
    assert "steepness: on-chain 3.0 != local 2.5" in message
    assert "k_lower" not in message


@requires_web3
def test_unregistered_community_gets_an_actionable_error():
    revert = Exception("execution reverted: AMMBA: community params not set")
    with pytest.raises(ContractError, match="setCommunityParams"):
        live_client(revert)._verify_community_params_sync("communityid_1",
                                                          CommunityParams())


@requires_web3
def test_other_call_failures_are_normalized():
    with pytest.raises(ContractError, match="getCommunityParams"):
        live_client(Exception("connection refused"))._verify_community_params_sync(
            "communityid_1", CommunityParams())


# ------------------------------------------- live-mode fee guards (#25)

class FakeEth:
    """The two node calls `_fee_params` makes."""

    def __init__(self, max_priority_fee, base_fee=7):
        self._max_priority_fee = max_priority_fee
        self._base_fee = base_fee

    @property
    def max_priority_fee(self):
        if isinstance(self._max_priority_fee, Exception):
            raise self._max_priority_fee
        return self._max_priority_fee

    def get_block(self, _which):
        return {"baseFeePerGas": self._base_fee}


def fee_client(max_priority_fee, base_fee=7,
               min_priority_fee_wei=1_000_000_000) -> Web3ContractClient:
    client = Web3ContractClient(RPC_URL, CONTRACT_ADDRESS, PRIVATE_KEY,
                                min_priority_fee_wei)
    client._w3 = type("W3", (), {"eth": FakeEth(max_priority_fee, base_fee)})()
    return client


@requires_web3
def test_fee_floor_applies_when_the_estimate_is_below_it():
    """The Volta failure: empty blocks estimate 0, so maxFeePerGas became 14
    wei, the transaction was never included and it blocked its own nonce."""
    fees = fee_client(max_priority_fee=0)._fee_params()
    assert fees["maxPriorityFeePerGas"] == 1_000_000_000
    assert fees["maxFeePerGas"] == 1_000_000_000 + 14


@requires_web3
def test_a_higher_estimate_wins_over_the_floor():
    fees = fee_client(max_priority_fee=3_000_000_000)._fee_params()
    assert fees["maxPriorityFeePerGas"] == 3_000_000_000
    assert fees["maxFeePerGas"] == 3_000_000_000 + 14


@requires_web3
def test_a_node_without_max_priority_fee_falls_back_to_the_floor():
    # Not every node implements eth_maxPriorityFeePerGas.
    fees = fee_client(
        max_priority_fee=ValueError("method not supported"))._fee_params()
    assert fees["maxPriorityFeePerGas"] == 1_000_000_000
    assert fees["maxFeePerGas"] == 1_000_000_000 + 14
