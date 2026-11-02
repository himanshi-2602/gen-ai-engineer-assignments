"""Deterministic tools: a safe arithmetic evaluator for the calculations the LLM writes.

LLMs are unreliable at exact arithmetic, so the executor writes each calculation as an
expression (e.g. "to_minutes('18:05') - to_minutes('14:30')") and the verifier re-computes it
here, in plain Python. Only arithmetic is allowed: no variables, attributes, imports or
arbitrary function calls, so model output can never run unsafe code.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable, Dict, Union

Number = Union[int, float]


class CalculationError(ValueError):
    """The expression is invalid, unsafe or cannot be computed."""


_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def to_minutes(hhmm: str) -> int:
    """'14:30' -> 870 (minutes since midnight). Rejects invalid times like '25:10' or '9:75'."""
    match = _TIME.match(str(hhmm).strip())
    if not match:
        raise CalculationError(f"invalid time {hhmm!r}; expected HH:MM between 00:00 and 23:59")
    return int(match.group(1)) * 60 + int(match.group(2))


FUNCTIONS: Dict[str, Callable] = {
    "to_minutes": to_minutes,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(expression: str) -> Number:
    """Safely evaluate an arithmetic expression. Raises CalculationError on anything else."""
    if len(expression) > 300:
        raise CalculationError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as err:
        raise CalculationError(f"not a valid expression: {expression!r}") from err
    try:
        return _eval(tree.body)
    except ZeroDivisionError as err:
        raise CalculationError("division by zero") from err


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalculationError("exponent too large")
        _require_numbers(left, right)
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        value = _eval(node.operand)
        _require_numbers(value)
        return _UNARY[type(node.op)](value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name not in FUNCTIONS:
            raise CalculationError(f"unsupported function {name!r}; allowed: {', '.join(FUNCTIONS)}")
        return FUNCTIONS[name](*[_eval(arg) for arg in node.args])
    raise CalculationError(f"unsupported syntax: {ast.dump(node)[:60]}")


def _require_numbers(*values) -> None:
    for value in values:
        if not isinstance(value, (int, float)):
            raise CalculationError(f"expected a number, got {value!r}")


def numbers_match(a: Number, b: Number, tolerance: float = 1e-6) -> bool:
    """Compare two numbers allowing tiny floating-point differences."""
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
