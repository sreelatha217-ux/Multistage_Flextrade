"""Day-ahead MILP scheduler for industrial batch production."""
from ._internal import __version__
from .benchmark import make_benchmark_instance, tou_price_vector
from .candidates import Candidate, build_candidates
from .data import Instance, SchedulerConfig, SchedulingResult, SolverSettings
from .exceptions import (InfeasibleScheduleError, InstanceValidationError, SchedulingError,
                         SolveFailedError, SolverUnavailableError, VerificationError)
from .model import build_model
from .optimizer import optimize_day_ahead
from .plotting import plot_schedule
from .results import extract_result, verify
from .solver import solve_model

__all__ = [
    "__version__",
    "Candidate",
    "InfeasibleScheduleError",
    "Instance",
    "InstanceValidationError",
    "SchedulerConfig",
    "SchedulingError",
    "SchedulingResult",
    "SolveFailedError",
    "SolverSettings",
    "SolverUnavailableError",
    "VerificationError",
    "build_candidates",
    "build_model",
    "extract_result",
    "make_benchmark_instance",
    "optimize_day_ahead",
    "plot_schedule",
    "solve_model",
    "tou_price_vector",
    "verify",
]