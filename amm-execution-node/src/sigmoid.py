"""Sigmoid clearing-price function (identical to the Clearing Node's copy).

Kept as a local module so each service container stays self-contained, the
same way GSY-DEX services duplicate small shared utilities. Needed here for
counterfactual price calculations (guide §5.3).
"""

import math

NODE_FLOAT_SCALING_FACTOR = 10000

_EXP_ARG_LIMIT = 700.0


def to_node_int(float_val: float) -> int:
    return int(round(float_val * NODE_FLOAT_SCALING_FACTOR))


def from_node_int(int_val: int) -> float:
    return int_val / NODE_FLOAT_SCALING_FACTOR


def sigmoid_price(ratio: float, k_upper: float, k_lower: float,
                  theta: float, steepness: float) -> float:
    if k_upper <= k_lower:
        return k_lower
    arg = -steepness * (ratio - theta)
    if arg > _EXP_ARG_LIMIT:
        return k_upper
    if arg < -_EXP_ARG_LIMIT:
        return k_lower
    return k_upper - (k_upper - k_lower) / (1.0 + math.exp(arg))
