"""Error types. Every error carries a stable code that is written to output.json (meta.error.code)."""
from __future__ import annotations


class ServiceError(Exception):
    code = "INTERNAL"

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.field = field

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.field:
            d["field"] = self.field
        return d


class InvalidInput(ServiceError):
    code = "INVALID_INPUT"


class DataUnavailable(ServiceError):
    code = "DATA_UNAVAILABLE"


class InfeasibleSchedule(ServiceError):
    code = "INFEASIBLE"


class SolverFailure(ServiceError):
    code = "SOLVER_ERROR"


class RunTimeout(ServiceError):
    code = "TIMEOUT"
