"""Guards the intentionally duplicated cross-service utilities.

The guide (§4.5) keeps each service container self-contained, so small
shared utilities are copied between services instead of imported from a
common package. These tests pin the copies together at the AST level
(comments and docstrings may differ, logic may not): if one copy changes,
its twin must change too.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEARING_SRC = REPO_ROOT / "amm-clearing-node" / "src"
EXECUTION_SRC = REPO_ROOT / "amm-execution-node" / "src"
MOCK_SRC = REPO_ROOT / "mock-offchain-db" / "src"

# Sibling services are absent when this suite runs inside the service's
# own Docker image; the guard only makes sense in the full checkout.
pytestmark = pytest.mark.skipif(
    not (EXECUTION_SRC.exists() and MOCK_SRC.exists()),
    reason="sibling services not present in this checkout")


def _function_logic(path: Path, name: str) -> list[str]:
    """AST dump of a function's body, docstring stripped."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return [ast.dump(stmt) for stmt in body]
    raise AssertionError(f"function {name!r} not found in {path}")


@pytest.mark.parametrize("name", ["sigmoid_price", "to_node_int",
                                  "from_node_int"])
def test_sigmoid_copies_match(name):
    assert (_function_logic(CLEARING_SRC / "sigmoid.py", name)
            == _function_logic(EXECUTION_SRC / "sigmoid.py", name))


def test_blake2b_hash_copies_match():
    assert (_function_logic(CLEARING_SRC / "trade_builder.py", "blake2b_hash")
            == _function_logic(MOCK_SRC / "store.py", "blake2b_hash"))


def test_round_type_semantics_match():
    # Both nodes must classify a market the same way (same epsilon).
    assert (_function_logic(CLEARING_SRC / "clearing.py", "round_type")
            == _function_logic(EXECUTION_SRC / "penalties.py",
                               "determine_round_type"))
