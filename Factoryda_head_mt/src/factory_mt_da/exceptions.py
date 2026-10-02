"""Domain exceptions raised by the factory scheduler."""


class SchedulingError(Exception):
    """Base class for scheduler errors."""


class InstanceValidationError(SchedulingError):
    """Input data is inconsistent or malformed."""


class SolverUnavailableError(SchedulingError):
    """The requested optimization solver is unavailable."""


class InfeasibleScheduleError(SchedulingError):
    """No schedule satisfies the model constraints."""


class SolveFailedError(SchedulingError):
    """The solver stopped without a usable solution."""


class VerificationError(SchedulingError):
    """Independent verification found a violated constraint."""