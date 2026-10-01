"""
Exception hierarchy. Every error raised by this package derives from
:class:`ProsumerOptimizationError`, so callers can catch one base class.
"""
from __future__ import annotations


class ProsumerOptimizationError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(ProsumerOptimizationError):
    """Physical or economic parameters are inconsistent."""


class DataValidationError(ProsumerOptimizationError):
    """Time series or scenario data are malformed."""


class SolverUnavailableError(ProsumerOptimizationError):
    """The requested solver cannot be found or started."""


class ModelInfeasibleError(ProsumerOptimizationError):
    """The MILP is infeasible or unbounded."""


class SolveFailedError(ProsumerOptimizationError):
    """The solver stopped without a usable solution."""
