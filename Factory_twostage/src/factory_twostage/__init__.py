"""Modular day-ahead and intraday factory scheduling package."""

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
from .intraday_model import build_intraday_model
from .intraday_optimizer import optimize_intraday, value_of_stochastic_solution
from .intraday_plotting import plot_intraday
from .intraday_results import IntradayResult
from .scenarios import IntradayMarket, ScenarioSet
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
    "IntradayMarket",
    "IntradayResult",
    "Microturbine",
    "RunConfig",
    "SchedulerConfig",
    "ScenarioSet",
    "SchedulingError",
    "SchedulingResult",
    "SolveFailedError",
    "SolverSettings",
    "SolverUnavailableError",
    "VerificationError",
    "__version__",
    "build_candidates",
    "build_model",
    "build_intraday_model",
    "extract_result",
    "hhmm",
    "make_benchmark_instance",
    "mt_fuel_cost_by_hour",
    "optimize_day_ahead",
    "optimize_intraday",
    "plot_schedule",
    "plot_intraday",
    "solve_model",
    "tou_price_vectors",
    "verify",
    "value_of_stochastic_solution",
]
