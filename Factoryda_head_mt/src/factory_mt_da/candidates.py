"""Time-indexed feasible batch start candidates."""

from dataclasses import dataclass

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
    occ: np.ndarray

    @property
    def machine_index(self) -> int:
        return self.m

    @property
    def task_index(self) -> int:
        return self.p

    @property
    def grid_index(self) -> int:
        return self.k

    @property
    def duration_h(self) -> float:
        return self.dur_h

    @property
    def occupancy(self) -> np.ndarray:
        return self.occ


def build_candidates(inst: Instance, step_h: float) -> list[Candidate]:
    """Create all non-preemptive starts whose full duration fits in the horizon."""
    horizon = inst.horizon_h
    grid_count = round(horizon / step_h)
    hours = np.arange(horizon, dtype=float)
    candidates = []
    for machine_index in range(len(inst.machines)):
        for task_index in range(len(inst.tasks)):
            duration = float(inst.duration_h[machine_index, task_index])
            for grid_index in range(grid_count):
                start_h = grid_index * step_h
                if start_h + duration > horizon + EPS:
                    break
                occupancy = np.clip(
                    np.minimum(start_h + duration, hours + 1) - np.maximum(start_h, hours),
                    0.0,
                    1.0,
                )
                candidates.append(Candidate(
                    machine_index,
                    task_index,
                    grid_index,
                    start_h,
                    duration,
                    float(inst.power_mw[machine_index, task_index]),
                    occupancy,
                ))
    if not candidates:
        raise InstanceValidationError("no feasible start time exists for any task")
    return candidates