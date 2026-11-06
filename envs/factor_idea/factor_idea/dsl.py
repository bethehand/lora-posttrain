"""Safe parser for factor expressions.

Without Minute(...) the grammar is a strict subset of the research platform's Factor
DSL, so a formula accepted here can be pasted into the platform unchanged. Nothing is
ever eval'ed: the text goes through ``ast.parse`` and each node must match the whitelist
below. There is no operator that reads a later row and every period, window and span
must be a positive integer, so a formula cannot ask for the future.
tests/test_no_lookahead.py checks that claim on every operator.

Minute(...) is what the platform writes as a Base: it works inside one coin's day and a
day_ operator turns it into one number per day. A minute value can't meet a daily value
before that, and nothing inside a day can see another day or another coin, so a day's
number depends on that day's own minutes alone. tests/test_minute.py checks that too.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_base_vol", "taker_quote_vol")
MARKET_SERIES = ("market_return", "market_volume_share")
MINUTE_FIELDS = (*FIELDS, "return", *MARKET_SERIES)

MAX_CHARS = 4096
MAX_NODES = 1024
MAX_DEPTH = 200
MAX_WINDOW = 365
MAX_SPAN = 120


class DslError(ValueError):
    """The text is not a valid expression in this sandbox; the message says why."""


@dataclass(frozen=True)
class Field:
    name: str


@dataclass(frozen=True)
class MinuteField:
    name: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Op:
    name: str
    args: tuple["Node", ...]
    params: tuple[tuple[str, Any], ...] = ()


Node = Field | MinuteField | Const | Op

_REQUIRED = object()

# Argument kinds
_POS = "positive integer"
_POS_NONE = "positive integer or None"
_NUM = "number"
_NUM_NONE = "number or None"
_BOOL = "True or False"
_F = "Factor"
_F_NUM = "Factor or number"
_HOUR = "whole hour from 0 to 24"

_ROLLING_SIG = [("window", _POS, _REQUIRED), ("min_periods", _POS_NONE, None)]

# method -> (operator, positional-or-keyword parameters, keyword-only parameters, keyword aliases)
_METHODS: dict[str, tuple[str, list, list, dict]] = {
    "Abs": ("abs", [], [], {}),
    "log": ("log", [], [], {}),
    "ln": ("ln", [], [], {}),
    "sign": ("sign", [], [], {}),
    "pow": ("pow", [("exp", _NUM, _REQUIRED)], [], {}),
    "clip": ("clip", [("lower", _NUM_NONE, None), ("upper", _NUM_NONE, None)], [], {}),
    "max": ("max", [("other", _F_NUM, _REQUIRED)], [], {}),
    "min": ("min", [("other", _F_NUM, _REQUIRED)], [], {}),
    "fillna": ("fillna", [("value", _F_NUM, _REQUIRED)], [], {}),
    "safe_div": ("safe_div", [("other", _F_NUM, _REQUIRED)], [], {}),
    "where": ("where", [("true_value", _F_NUM, _REQUIRED), ("false_value", _F_NUM, _REQUIRED)], [], {}),
    **{name: (name, _ROLLING_SIG, [], {}) for name in (
        "ts_mean", "ts_std", "ts_var", "ts_skew", "ts_kurt", "ts_sum", "ts_max", "ts_min", "ts_median",
        "ts_rank", "ts_argmax", "ts_argmin", "ts_decay_linear", "ts_product", "ts_zscore",
    )},
    "ts_quantile": ("ts_quantile", [("window", _POS, _REQUIRED), ("q", _NUM, _REQUIRED), ("min_periods", _POS_NONE, None)], [], {}),
    "ts_ewm_mean": ("ts_ewm_mean", [("span", _POS, _REQUIRED), ("min_periods", _POS_NONE, None)], [("window", _POS_NONE, None)], {}),
    "EMA": ("ema", [("span", _POS, 10), ("min_periods", _POS_NONE, None)], [("window", _POS_NONE, None)], {"n": "span"}),
    "lag": ("lag", [("periods", _POS, _REQUIRED)], [], {}),
    "delay": ("lag", [("periods", _POS, _REQUIRED)], [], {"n": "periods"}),
    "diff": ("diff", [("periods", _POS, 1)], [], {}),
    "pct_change": ("pct_change", [("periods", _POS, 1)], [], {}),
    "ts_delta": ("diff", [("periods", _POS, _REQUIRED)], [], {}),
    "ts_corr": ("ts_corr", [("other", _F, _REQUIRED), ("window", _POS, _REQUIRED), ("min_periods", _POS_NONE, None)], [], {}),
    "ts_cov": ("ts_cov", [("other", _F, _REQUIRED), ("window", _POS, _REQUIRED), ("min_periods", _POS_NONE, None)], [], {}),
    "sec_rank": ("rank", [("pct", _BOOL, True)], [], {}),
    "rank": ("rank", [("pct", _BOOL, True)], [], {}),
    "percentile": ("rank", [], [], {}),
    "zscore": ("zscore", [], [], {}),
    "demean": ("demean", [], [], {}),
    "sec_demean": ("demean", [], [], {}),
    "sec_pct": ("sec_pct", [], [], {}),
    "winsorize": ("winsorize", [("limit", _NUM, 3.0)], [], {}),
    "sec_winsorize": ("sec_winsorize", [("limit", _NUM, 3.0)], [], {}),
    "scale": ("scale", [("a", _NUM, 1.0)], [], {}),
}
_MINUTE_METHODS: dict[str, tuple[str, list, list, dict]] = {
    "between": ("between", [("start_hour", _HOUR, _REQUIRED), ("end_hour", _HOUR, _REQUIRED)], [], {}),
    "zscore_in_day": ("zscore_in_day", [], [], {}),
    **{name: (name, [], [], {}) for name in (
        "day_sum", "day_mean", "day_std", "day_max", "day_min", "day_first", "day_last", "day_center",
    )},
    "day_corr": ("day_corr", [("other", _F, _REQUIRED)], [], {}),
}
_NARY = {"neutralize", "orthogonalize"}
_COMPARE = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne"}
_BINOP = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}

# Operators that turn a day of minutes into one number per day.
DAY_OPS = frozenset(name for name in _MINUTE_METHODS if name.startswith("day_"))
# Operators that only make sense inside a day.
_INSIDE_DAY = frozenset({"between", "zscore_in_day"})
# Daily operators that also work minute by minute, because each looks at one value at a time.
_ELEMENTWISE = frozenset({
    "add", "sub", "mul", "div", "safe_div", "pow", "neg", "abs", "log", "ln", "sign", "clip", "max", "min", "fillna",
    "where", "gt", "ge", "lt", "le", "eq", "ne",
})
_TO_ONE_NUMBER = "first turn the Minute(...) part into one number per day with .day_sum(), .day_mean() or another day_ operator"


def parse(source: str) -> Node:
    """Parse expression text into a tree, or raise DslError."""
    if not isinstance(source, str) or not source.strip():
        raise DslError("expression is empty")
    if len(source) > MAX_CHARS:
        raise DslError(f"expression is too long (max {MAX_CHARS} characters)")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise DslError(f"invalid expression syntax: {getattr(exc, 'msg', type(exc).__name__)}") from exc
    pending, count = [(tree, 1)], 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_NODES:
            raise DslError(f"expression is too complex (max {MAX_NODES} syntax nodes)")
        if depth > MAX_DEPTH:
            raise DslError(f"expression is too deeply nested (max depth {MAX_DEPTH})")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    result = _compile(tree.body)
    if _level(result) == "minute":
        raise DslError(f"a factor is one number per coin per day: {_TO_ONE_NUMBER}")
    if not isinstance(result, (Field, Op)):
        raise DslError('expression must use at least one Factor("field")')
    return result


def _level(node: Node) -> str | None:
    """"minute" or "day" for what a node gives, None for a plain number; raises if the levels clash."""
    if isinstance(node, Const):
        return None
    if isinstance(node, Field):
        return "day"
    if isinstance(node, MinuteField):
        return "minute"
    levels = {_level(a) for a in node.args} - {None}
    if node.name in DAY_OPS or node.name in _INSIDE_DAY:
        if levels != {"minute"}:
            raise DslError(f"{node.name}() works on Minute(...) data, not on a daily value")
        return "day" if node.name in DAY_OPS else "minute"
    if levels == {"minute", "day"}:
        raise DslError(f"a minute value can't be combined with a daily value: {_TO_ONE_NUMBER}")
    if "minute" in levels:
        if node.name not in _ELEMENTWISE:
            raise DslError(f"{node.name} needs daily values: {_TO_ONE_NUMBER}")
        return "minute"
    return "day"


def _compile(node: ast.AST) -> Node:
    if isinstance(node, ast.Call):
        return _call(node)
    if isinstance(node, ast.BinOp):
        return _binop(node)
    if isinstance(node, ast.Compare):
        return _compare(node)
    if isinstance(node, ast.UnaryOp):
        return _unary(node)
    if isinstance(node, ast.Constant):
        return _constant(node)
    raise DslError(f"unsupported syntax: {type(node).__name__}")


def _constant(node: ast.Constant) -> Const:
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise DslError(f"unsupported literal: {node.value!r}")
    try:
        return Const(float(node.value))
    except OverflowError:
        raise DslError("number is too large") from None


def _unary(node: ast.UnaryOp) -> Node:
    value = _compile(node.operand)
    if isinstance(node.op, ast.UAdd):
        return value
    if isinstance(node.op, ast.USub):
        return Const(-value.value) if isinstance(value, Const) else Op("neg", (value,))
    raise DslError(f"unsupported unary operator: {type(node.op).__name__}")


def _binop(node: ast.BinOp) -> Node:
    name = _BINOP.get(type(node.op))
    if name is None:
        raise DslError(f"unsupported binary operator: {type(node.op).__name__}")
    left, right = _compile(node.left), _compile(node.right)
    if isinstance(left, Const) and isinstance(right, Const):
        return Const(_fold(name, left.value, right.value))
    return Op(name, (left, right))


def _fold(name: str, a: float, b: float) -> float:
    try:
        if name == "add":
            value = a + b
        elif name == "sub":
            value = a - b
        elif name == "mul":
            value = a * b
        elif name == "div":
            value = a / b
        else:
            value = a**b
    except (ArithmeticError, ValueError) as exc:
        raise DslError(f"invalid constant arithmetic: {exc}") from exc
    if isinstance(value, complex) or value != value or value in (float("inf"), float("-inf")):
        raise DslError("invalid constant arithmetic")
    return value


def _compare(node: ast.Compare) -> Node:
    if len(node.ops) != 1:
        raise DslError("chained comparisons are not supported")
    name = _COMPARE.get(type(node.ops[0]))
    if name is None:
        raise DslError(f"unsupported comparison: {type(node.ops[0]).__name__}")
    left, right = _compile(node.left), _compile(node.comparators[0])
    if isinstance(left, Const) and isinstance(right, Const):
        raise DslError("a comparison needs at least one Factor")
    return Op(name, (left, right))


def _call(node: ast.Call) -> Node:
    if isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "Factor":
            return _factor(node)
        if name == "Minute":
            return _minute(node)
        if name == "where" or name == "safe_div" or name in _NARY:
            if not node.args:
                raise DslError(f"{name}() needs a Factor as its first argument")
            first = _compile(node.args[0])
            if isinstance(first, Const):
                if name == "safe_div":
                    return _safe_div_const_left(first, node)
                raise DslError(f"{name}() needs a Factor as its first argument")
            return _method(first, name, node.args[1:], node.keywords)
        if name == "Alpha":
            raise DslError("Alpha() references are not available in this sandbox")
        raise DslError(f"unsupported function: {name}")
    if isinstance(node.func, ast.Attribute):
        receiver = _compile(node.func.value)
        if isinstance(receiver, Const):
            raise DslError("a method must be called on a Factor")
        return _method(receiver, node.func.attr, node.args, node.keywords)
    raise DslError("unsupported function call")


def _safe_div_const_left(left: Const, node: ast.Call) -> Node:
    if len(node.args) != 2 or node.keywords:
        raise DslError("safe_div() takes two arguments")
    right = _compile(node.args[1])
    if isinstance(right, Const):
        raise DslError("safe_div() needs at least one Factor")
    return Op("safe_div", (left, right))


def _factor(node: ast.Call) -> Field:
    if node.keywords or len(node.args) != 1:
        raise DslError("Factor() takes exactly one field name")
    arg = node.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        raise DslError("Factor() takes a field name in quotes")
    if arg.value not in FIELDS:
        raise DslError(f"unknown field {arg.value!r}; available fields: {', '.join(FIELDS)}")
    return Field(arg.value)


def _minute(node: ast.Call) -> MinuteField:
    if node.keywords or len(node.args) != 1:
        raise DslError("Minute() takes exactly one field name")
    arg = node.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        raise DslError("Minute() takes a field name in quotes")
    if arg.value not in MINUTE_FIELDS:
        raise DslError(f"unknown minute field {arg.value!r}; available: {', '.join(MINUTE_FIELDS)}")
    return MinuteField(arg.value)


def _method(receiver: Node, method: str, args: list[ast.AST], keywords: list[ast.keyword]) -> Op:
    if method in _NARY:
        return _nary(receiver, method, args, keywords)
    spec = _METHODS.get(method) or _MINUTE_METHODS.get(method)
    if spec is None:
        raise DslError(f"unsupported method: {method}")
    op, positional, keyword_only, aliases = spec
    if len(args) > len(positional):
        raise DslError(f"{method}() takes at most {len(positional)} positional arguments")
    bound: dict[str, Any] = {}
    for (name, kind, _), value in zip(positional, args):
        bound[name] = _argument(value, kind, method, name)
    kinds = {name: kind for name, kind, _ in positional + keyword_only}
    for kw in keywords:
        if kw.arg in ("pool", "index_pool"):
            raise DslError("pool arguments are not available in this sandbox: cross-sectional operators use every coin that has a value that day")
        name = aliases.get(kw.arg, kw.arg)
        if name not in kinds:
            raise DslError(f"{method}() has no argument {kw.arg!r}")
        if name in bound:
            raise DslError(f"{method}() got {name} twice")
        bound[name] = _argument(kw.value, kinds[name], method, name)
    for name, _, default in positional + keyword_only:
        if name not in bound:
            if default is _REQUIRED:
                raise DslError(f"{method}() requires {name}")
            bound[name] = default
    _check_ranges(method, bound)
    children: list[Node] = [receiver]
    params: list[tuple[str, Any]] = []
    for name, kind, _ in positional + keyword_only:
        if kind in (_F, _F_NUM):
            children.append(bound[name])
        else:
            params.append((name, bound[name]))
    params.sort(key=lambda item: item[0])
    return Op(op, tuple(children), tuple(params))


def _nary(receiver: Node, method: str, args: list[ast.AST], keywords: list[ast.keyword]) -> Op:
    for kw in keywords:
        if kw.arg in ("pool", "index_pool"):
            raise DslError("pool arguments are not available in this sandbox: cross-sectional operators use every coin that has a value that day")
        raise DslError(f"{method}() has no argument {kw.arg!r}")
    if not args:
        raise DslError(f"{method}() needs at least one Factor to regress on")
    others = [_argument(a, _F, method, "factor") for a in args]
    return Op("neutralize", (receiver, *others))


def _argument(node: ast.AST, kind: str, method: str, name: str) -> Any:
    if kind == _HOUR:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) and 0 <= node.value <= 24:
            return node.value
        raise DslError(f"{method}(): {name} must be a {kind}")
    if kind in (_POS, _POS_NONE):
        if kind == _POS_NONE and isinstance(node, ast.Constant) and node.value is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) and node.value > 0:
            return node.value
        raise DslError(f"{method}(): {name} must be a {kind}")
    if kind == _BOOL:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise DslError(f"{method}(): {name} must be {kind}")
    if kind in (_NUM, _NUM_NONE):
        if kind == _NUM_NONE and isinstance(node, ast.Constant) and node.value is None:
            return None
        value = _compile(node)
        if isinstance(value, Const):
            return value.value
        raise DslError(f"{method}(): {name} must be a {kind}")
    value = _compile(node)
    if kind == _F and isinstance(value, Const):
        raise DslError(f"{method}(): {name} must be a Factor")
    return value


def _check_ranges(method: str, bound: dict[str, Any]) -> None:
    window = bound.get("window")
    min_periods = bound.get("min_periods")
    span = bound.get("span")
    if window is not None and window > (MAX_WINDOW if span is None else 10 * MAX_SPAN):
        raise DslError(f"{method}(): window is too long")
    if span is not None and span > MAX_SPAN:
        raise DslError(f"{method}(): span must be at most {MAX_SPAN}")
    if bound.get("periods") is not None and bound["periods"] > MAX_WINDOW:
        raise DslError(f"{method}(): periods must be at most {MAX_WINDOW}")
    reach = window if window is not None else (None if span is None else 10 * span)
    if min_periods is not None and reach is not None and min_periods > reach:
        raise DslError(f"{method}(): min_periods must not exceed window")
    # Longer than the gates' history and a formula could stay blank through them, then switch on when scored.
    if min_periods is not None and min_periods > MAX_WINDOW:
        raise DslError(f"{method}(): min_periods must be at most {MAX_WINDOW}")
    if "q" in bound and not 0.0 <= bound["q"] <= 1.0:
        raise DslError(f"{method}(): q must be between 0 and 1")
    if method == "clip" and bound["lower"] is None and bound["upper"] is None:
        raise DslError("clip() requires lower or upper")
    if method == "between" and bound["start_hour"] >= bound["end_hour"]:
        raise DslError("between(): start_hour must come before end_hour")
    if method in ("winsorize", "sec_winsorize") and bound["limit"] <= 0:
        raise DslError(f"{method}(): limit must be positive")


def fields_of(node: Node) -> frozenset[str]:
    """The data fields an expression reads."""
    if isinstance(node, Field):
        return frozenset((node.name,))
    if isinstance(node, MinuteField):
        names = ("close", "open") if node.name == "return" else (node.name,)
        return frozenset(minute_name(name) for name in names)
    if isinstance(node, Const):
        return frozenset()
    return frozenset().union(*(fields_of(a) for a in node.args))


def day_ops_of(node: Node) -> frozenset[Op]:
    """The day_ operators in an expression. Nothing inside one can hold another."""
    if not isinstance(node, Op):
        return frozenset()
    if node.name in DAY_OPS:
        return frozenset((node,))
    return frozenset().union(*(day_ops_of(a) for a in node.args))


def minute_name(field: str) -> str:
    """How fields_of() names a field read through Minute(...), so it can't be taken for the daily one."""
    return f"minute:{field}"


def reads_minutes(node: Node) -> bool:
    return any(name.startswith("minute:") for name in fields_of(node))


def skeleton(node: Node) -> str:
    """The expression's shape with every number dropped, to count how many distinct structures a model uses."""
    if isinstance(node, Field):
        return node.name
    if isinstance(node, MinuteField):
        return f"minute.{node.name}"
    if isinstance(node, Const):
        return "#"
    return f"{node.name}({','.join(skeleton(a) for a in node.args)})"
