"""Evaluate a parsed expression on a dict of field arrays."""
from __future__ import annotations

import time

import numpy as np

from factor_idea import ops
from factor_idea.dsl import DAY_OPS, Const, Field, Node, day_ops_of


class EvalTimeout(RuntimeError):
    """The expression took longer than its time budget."""


class NoMinuteData(RuntimeError):
    """The expression reads Minute(...) and this sandbox has no minute bars."""


def evaluate(node: Node, fields: dict[str, np.ndarray], *, deadline: float | None = None, memo: dict | None = None, minutes=None):
    """The expression's T x N values. Repeated sub-expressions are computed once.

    ``minutes`` supplies what day_ operators compute: MinuteBars, or PatchedMinutes for a perturbed market.
    """
    memo = {} if memo is None else memo
    if minutes is not None:
        minutes.prepare([op for op in day_ops_of(node) if op not in memo], fields["close"].shape[0], deadline)
    return _eval(node, fields, deadline, memo, minutes)


def _eval(node: Node, fields, deadline, memo, minutes):
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Field):
        return fields[node.name]
    if node in memo:
        return memo[node]
    if deadline is not None and time.monotonic() > deadline:
        raise EvalTimeout("expression took too long to compute")
    if node.name in DAY_OPS:
        if minutes is None:
            raise NoMinuteData("minute data is not available in this sandbox")
        out = minutes.day_values(node, fields["close"].shape[0], deadline)
        memo[node] = out
        return out
    args = [_eval(a, fields, deadline, memo, minutes) for a in node.args]
    # A chain does all its work on the way back up, after the check above has passed at every level.
    if deadline is not None and time.monotonic() > deadline:
        raise EvalTimeout("expression took too long to compute")
    with np.errstate(all="ignore"):
        out = ops.clean(ops.TABLE[node.name](*args, **dict(node.params)))
    out.setflags(write=False)
    memo[node] = out
    return out
