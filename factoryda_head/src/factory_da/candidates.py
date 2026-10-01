"""Time-indexed feasible batch start candidates."""
from dataclasses import dataclass
from typing import List

import numpy as np

from ._internal import EPS
from .data import Instance
from .exceptions import InstanceValidationError


@dataclass(frozen=True)
class Candidate:
    m: int
    p: int
    k: int
    start_h: float
    dur_h: float
    power_mw: float
    occ: np.ndarray          # hourly occupancy in [0,1] (exact overlap with each hour)

def build_candidates(inst: Instance, step: float) -> List[Candidate]:
    T = inst.horizon_h
    n_grid = int(round(T / step))
    hours = np.arange(T, dtype=float)
    out: List[Candidate] = []
    for m in range(len(inst.machines)):
        for p in range(len(inst.tasks)):
            D = float(inst.duration_h[m, p])
            for k in range(n_grid):
                a = k * step
                if a + D > T + EPS:
                    break
                occ = np.clip(np.minimum(a + D, hours + 1) - np.maximum(a, hours), 0.0, 1.0)
                out.append(Candidate(m, p, k, a, D, float(inst.power_mw[m, p]), occ))
    if not out:
        raise InstanceValidationError("no feasible start time exists for any task")
    return out
