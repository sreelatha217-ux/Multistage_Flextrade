"""Modular factory, microturbine, and BESS day-ahead scheduler."""

from ._internal import __version__
from .benchmark import make_benchmark_instance, tou_price_vectors
from .candidates import Candidate, build_candidates
from .config import RunConfig
from .data import BESS, Instance, Microturbine, SchedulerConfig, SchedulingResult, SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SchedulingError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .model import build_model
from .optimizer import optimize_day_ahead
from .plotting import plot_schedule
from .results import extract_result, hhmm, mt_fuel_cost_by_hour, verify
from .solver import solve_model

__all__ = [
    "BESS",
    "Candidate",
    "InfeasibleScheduleError",
    "Instance",
    "InstanceValidationError",
    "Microturbine",
    "RunConfig",
    "SchedulerConfig",
    "SchedulingError",
    "SchedulingResult",
    "SolveFailedError",
    "SolverSettings",
    "SolverUnavailableError",
    "VerificationError",
    "__version__",
    "build_candidates",
    "build_model",
    "extract_result",
    "hhmm",
    "make_benchmark_instance",
    "mt_fuel_cost_by_hour",
    "optimize_day_ahead",
    "plot_schedule",
    "solve_model",
    "tou_price_vectors",
    "verify",
]
