"""Sigmoid clearing-price function and integer scaling utilities (guide §3.3, §4.3)."""

import math

# Matches the GSY-DEX NODE_FLOAT_SCALING_FACTOR: all energy/price values that
# go on-chain are integers scaled by this factor (0.26 kWh -> 2600).
NODE_FLOAT_SCALING_FACTOR = 10000

# math.exp overflows above ~709.78; beyond these bounds the sigmoid has
# saturated to K_upper / K_lower anyway.
_EXP_ARG_LIMIT = 700.0


def to_node_int(float_val: float) -> int:
    return int(round(float_val * NODE_FLOAT_SCALING_FACTOR))


def from_node_int(int_val: int) -> float:
    return int_val / NODE_FLOAT_SCALING_FACTOR


def sigmoid_price(ratio: float, k_upper: float, k_lower: float,
                  theta: float, steepness: float) -> float:
    """Uniform clearing price for a given supply/demand ratio.

        price = K_upper - (K_upper - K_lower) / (1 + exp(-B * (ratio - theta)))

    Scarce supply (ratio << theta) pushes the price towards K_upper (retail
    buy price); abundant supply pushes it towards K_lower (feed-in tariff).
    """
    if k_upper <= k_lower:
        # Degenerate band (guide §3.3 edge case)
        return k_lower
    arg = -steepness * (ratio - theta)
    if arg > _EXP_ARG_LIMIT:
        return k_upper
    if arg < -_EXP_ARG_LIMIT:
        return k_lower
    return k_upper - (k_upper - k_lower) / (1.0 + math.exp(arg))


def clamp_price(price: float, k_lower: float, k_upper: float) -> float:
    """Clamp into [K_lower, K_upper] (guide §4.4 step 3)."""
    return max(k_lower, min(k_upper, price))
