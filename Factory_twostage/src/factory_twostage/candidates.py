"""Candidate generation for time-indexed production batches."""

from dataclasses import dataclass

import numpy as np

from .data import Instance
from .exceptions import InstanceValidationError
from .parameters import EPS


@dataclass(frozen=True)
class Candidate:
    machine_index: int
    task_index: int
    grid_index: int
    start_h: float
    duration_h: float
    power_mw: float
    occupancy: np.ndarray


def build_candidates(inst: Instance, step_h: float) -> list[Candidate]:
    hours_count = inst.horizon_h
    grid_count = int(round(hours_count / step_h))
    hours = np.arange(hours_count, dtype=float)
    candidates = []
    for machine_index in range(len(inst.machines)):
        for task_index in range(len(inst.tasks)):
            duration = float(inst.duration_h[machine_index, task_index])
            for grid_index in range(grid_count):
                start = grid_index * step_h
                if start + duration > hours_count + EPS:
                    break
                occupancy = np.clip(
                    np.minimum(start + duration, hours + 1)
                    - np.maximum(start, hours),
                    0.0,
                    1.0,
                )
                candidates.append(Candidate(
                    machine_index,
                    task_index,
                    grid_index,
                    start,
                    duration,
                    float(inst.power_mw[machine_index, task_index]),
                    occupancy,
                ))
    if not candidates:
        raise InstanceValidationError("no feasible start time exists for any task")
    return candidates
