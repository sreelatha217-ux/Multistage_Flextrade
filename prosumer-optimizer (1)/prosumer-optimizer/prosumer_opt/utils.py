"""Small, dependency-light helpers shared by all modules."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import ConfigurationError, DataValidationError


def require(cond: bool, msg: str, exc: type = ConfigurationError) -> None:
    """Raise `exc(msg)` unless `cond` holds."""
    if not cond:
        raise exc(msg)


def as_series(x, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate an array of length n; must be finite."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        raise DataValidationError(f"{name}: cannot convert to float array ({err})") from err
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise DataValidationError(f"{name}: expected length {n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise DataValidationError(f"{name}: contains NaN or inf")
    return arr


def var_value(var) -> float:
    """Value of a Pyomo variable, 0.0 when the solver left it unset."""
    v = var.value
    return 0.0 if v is None else float(v)


def cvar(costs: Sequence[float], probs: Sequence[float], alpha: float = 0.95) -> float:
    """Conditional value at risk of a discrete cost distribution (upper tail)."""
    c = np.asarray(costs, float)
    p = np.asarray(probs, float)
    order = np.argsort(-c)
    c, p = c[order], p[order]
    tail = 1.0 - alpha
    if tail <= 0:
        return float(c[0])
    acc, total = 0.0, 0.0
    for ci, pi in zip(c, p):
        w = min(pi, tail - acc)
        if w <= 0:
            break
        total += w * ci
        acc += w
    return float(total / tail)
