"""Solver selection, invocation, and termination handling."""

import time

import pyomo.environ as pyo

from ._internal import EPS, log
from .data import SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    SchedulingError,
    SolveFailedError,
    SolverUnavailableError,
)


def solve_model(model: pyo.ConcreteModel, settings: SolverSettings) -> dict[str, object]:
    """Solve a built model and return normalized status and solver statistics."""
    try:
        if settings.name == "appsi_highs":
            from pyomo.contrib.appsi.solvers import Highs

            optimizer = Highs()
            availability = optimizer.available()
            available = availability in (True, 1) or getattr(availability, "name", "") == "FullLicense"
        else:
            optimizer = pyo.SolverFactory(settings.name)
            available = optimizer.available(exception_flag=False)
    except Exception as err:
        raise SolverUnavailableError(f"cannot create solver {settings.name!r}: {err}") from err
    if not available:
        raise SolverUnavailableError(
            f"solver {settings.name!r} unavailable (install highspy or use gurobi/cplex/cbc)"
        )

    stats = {
        "variables": sum(1 for _ in model.component_data_objects(pyo.Var)),
        "binaries": sum(1 for variable in model.component_data_objects(pyo.Var) if variable.is_binary()),
        "constraints": sum(
            1 for _ in model.component_data_objects(pyo.Constraint, active=True)
        ),
    }
    log.info(
        "solving: %(variables)d vars (%(binaries)d binary), %(constraints)d constraints",
        stats,
    )
    started = time.perf_counter()
    gap = None
    try:
        if settings.name.startswith("appsi_"):
            optimizer.config.load_solution = False
            optimizer.config.stream_solver = settings.verbose
            optimizer.config.time_limit = settings.time_limit_s
            optimizer.config.mip_gap = settings.mip_gap
            if settings.name == "appsi_highs" and settings.threads:
                optimizer.highs_options["threads"] = int(settings.threads)
            result = optimizer.solve(model)
            termination = str(result.termination_condition).lower()
            has_solution = result.best_feasible_objective is not None
            if has_solution:
                result.solution_loader.load_vars()
                upper_bound = result.best_feasible_objective
                lower_bound = result.best_objective_bound
                if lower_bound is not None and abs(upper_bound) > EPS:
                    gap = max(0.0, (upper_bound - lower_bound) / abs(upper_bound))
        else:
            option_names = {
                "gurobi": ("MIPGap", "TimeLimit"),
                "cplex": ("mipgap", "timelimit"),
                "cbc": ("ratioGap", "seconds"),
                "glpk": ("mipgap", "tmlim"),
            }
            gap_option, time_option = option_names.get(settings.name, ("mipgap", "timelimit"))
            optimizer.options[gap_option] = settings.mip_gap
            optimizer.options[time_option] = settings.time_limit_s
            result = optimizer.solve(model, tee=settings.verbose, load_solutions=False)
            termination = str(result.solver.termination_condition).lower()
            has_solution = len(result.solution) > 0
            if has_solution:
                model.solutions.load_from(result)
    except SchedulingError:
        raise
    except Exception as err:
        raise SolveFailedError(f"solver crashed: {err}") from err

    elapsed = time.perf_counter() - started
    if "infeasible" in termination or "unbounded" in termination:
        raise InfeasibleScheduleError(
            f"model is {termination}. Check demand, machine availability, grid limit, inventory, and MT/load match."
        )
    if not has_solution:
        raise SolveFailedError(
            f"no feasible solution within the limits (termination: {termination}, {elapsed:.1f}s)"
        )
    status = "optimal" if "optimal" in termination else f"feasible ({termination})"
    if status != "optimal":
        log.warning("stopped early: %s, gap=%s", termination, gap)
    return {"status": status, "gap": gap, "time": elapsed, "stats": stats}