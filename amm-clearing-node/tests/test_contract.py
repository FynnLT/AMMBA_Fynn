"""Contract client behavior that the clearing tests do not exercise."""

import pytest

from src.contract import ContractError, MockContractClient

MARKET = "0x" + "bb" * 32


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
