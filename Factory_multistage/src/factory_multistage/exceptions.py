from __future__ import annotations


class SchedulingError(Exception):
    """Base class for all errors of this module."""


class InstanceValidationError(SchedulingError):
    """Input data is inconsistent or malformed."""


class SolverUnavailableError(SchedulingError):
    """Solver cannot be created or is not installed."""


class InfeasibleScheduleError(SchedulingError):
    """No schedule satisfies all constraints."""


class SolveFailedError(SchedulingError):
    """Solver stopped without a usable solution."""


class VerificationError(SchedulingError):
    """The independent solution check found a violated constraint."""
