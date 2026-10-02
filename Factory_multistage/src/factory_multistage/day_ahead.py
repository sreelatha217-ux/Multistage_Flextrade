"""Public day-ahead scheduling API."""

from ._internal import DT_H, EPS, __version__, log
from .benchmark import make_benchmark_instance, tou_price_vectors
from .candidates import Candidate, build_candidates
from .cli import apply_instance_overrides, main
from .data import (
    BESS,
    Instance,
    Microturbine,
    SchedulerConfig,
    SchedulingResult,
    SolverSettings,
    _arr,
)
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SchedulingError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .model import add_batch_block, add_mt_commitment, build_model
from .optimizer import optimize_day_ahead
from .plotting import plot_schedule
from .results import (
    _build_hourly,
    extract_result,
    hhmm,
    jobs_from_starts,
    mt_fuel_cost_by_hour,
    verify,
)
from .solver import solve_model

__all__ = [
    "BESS",
    "Candidate",
    "DT_H",
    "EPS",
    "InfeasibleScheduleError",
    "Instance",
    "InstanceValidationError",
    "Microturbine",
    "SchedulerConfig",
    "SchedulingError",
    "SchedulingResult",
    "SolveFailedError",
    "SolverSettings",
    "SolverUnavailableError",
    "VerificationError",
    "_arr",
    "_build_hourly",
    "__version__",
    "add_batch_block",
    "add_mt_commitment",
    "apply_instance_overrides",
    "build_candidates",
    "build_model",
    "extract_result",
    "hhmm",
    "jobs_from_starts",
    "log",
    "main",
    "make_benchmark_instance",
    "mt_fuel_cost_by_hour",
    "optimize_day_ahead",
    "plot_schedule",
    "solve_model",
    "tou_price_vectors",
    "verify",
]