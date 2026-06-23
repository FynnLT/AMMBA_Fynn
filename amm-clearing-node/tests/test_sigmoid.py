import pytest

from src.sigmoid import (NODE_FLOAT_SCALING_FACTOR, clamp_price,
                         from_node_int, sigmoid_price, to_node_int)

K_UPPER, K_LOWER, THETA, B = 28.5, 8.0, 1.0, 2.5


def test_price_at_midpoint_is_band_center():
    # At ratio == theta the sigmoid term is 1/2.
    assert sigmoid_price(1.0, K_UPPER, K_LOWER, THETA, B) == pytest.approx(
        (K_UPPER + K_LOWER) / 2)


def test_guide_example_supply_12_5_demand_10():
    # ratio 1.25 -> 28.5 - 20.5 / (1 + exp(-2.5 * 0.25)) ~= 15.1472
    assert sigmoid_price(1.25, K_UPPER, K_LOWER, THETA, B) == pytest.approx(
        15.1472, abs=1e-3)


def test_price_decreases_as_supply_ratio_increases():
    ratios = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0]
    prices = [sigmoid_price(r, K_UPPER, K_LOWER, THETA, B) for r in ratios]
    assert prices == sorted(prices, reverse=True)


def test_price_stays_within_band():
    for ratio in [0.0, 0.01, 1.0, 3.99, 100.0]:
        price = sigmoid_price(ratio, K_UPPER, K_LOWER, THETA, B)
        assert K_LOWER < price < K_UPPER or price in (K_LOWER, K_UPPER)


def test_scarce_supply_approaches_k_upper():
    assert sigmoid_price(0.0, K_UPPER, K_LOWER, THETA, 50.0) == pytest.approx(
        K_UPPER, abs=1e-6)


def test_abundant_supply_approaches_k_lower():
    assert sigmoid_price(100.0, K_UPPER, K_LOWER, THETA, B) == pytest.approx(
        K_LOWER, abs=1e-6)


def test_degenerate_band_returns_k_lower():
    # Guide §3.3 edge case: K_upper <= K_lower -> K_lower
    assert sigmoid_price(1.0, 8.0, 8.0, THETA, B) == 8.0
    assert sigmoid_price(1.0, 5.0, 8.0, THETA, B) == 8.0


def test_extreme_steepness_does_not_overflow():
    # exp() argument is guarded; saturates instead of raising OverflowError.
    assert sigmoid_price(0.0, K_UPPER, K_LOWER, 1000.0, 1000.0) == K_UPPER
    assert sigmoid_price(2000.0, K_UPPER, K_LOWER, 1.0, 1000.0) == K_LOWER


def test_clamp_price():
    assert clamp_price(30.0, K_LOWER, K_UPPER) == K_UPPER
    assert clamp_price(5.0, K_LOWER, K_UPPER) == K_LOWER
    assert clamp_price(15.0, K_LOWER, K_UPPER) == 15.0


def test_node_int_scaling_matches_gsy_dex_convention():
    assert NODE_FLOAT_SCALING_FACTOR == 10000
    assert to_node_int(0.26) == 2600
    assert to_node_int(28.5) == 285000
    assert from_node_int(2600) == pytest.approx(0.26)
    # round-trips at the supported resolution
    assert from_node_int(to_node_int(15.1472)) == pytest.approx(15.1472)
