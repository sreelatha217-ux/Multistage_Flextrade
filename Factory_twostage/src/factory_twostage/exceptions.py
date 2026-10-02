"""Domain exceptions raised by the BESS day-ahead scheduler."""


class SchedulingError(Exception):
    """Base class for scheduler failures."""


class InstanceValidationError(SchedulingError):
    """Input data is inconsistent or malformed."""


class SolverUnavailableError(SchedulingError):
    """The requested optimization solver is unavailable."""


class InfeasibleScheduleError(SchedulingError):
    """No schedule satisfies all model constraints."""


class SolveFailedError(SchedulingError):
    """The solver stopped without a usable solution."""


class VerificationError(SchedulingError):
    """Independent verification found a violated constraint."""