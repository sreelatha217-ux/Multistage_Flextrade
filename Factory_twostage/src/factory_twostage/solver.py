"""Solver selection and Pyomo solution handling."""

import logging
import time

import pyomo.environ as pyo

from .data import SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    SchedulingError,
    SolveFailedError,
    SolverUnavailableError,
)
from .parameters import EPS

log = logging.getLogger("factory_twostage")


def solve_model(model: pyo.ConcreteModel, settings: SolverSettings) -> dict[str, object]:
    try:
        if settings.name == "appsi_highs":
            from pyomo.contrib.appsi.solvers import Highs

            solver = Highs()
            available = solver.available()
            is_available = available in (True, 1) or getattr(available, "name", "") == "FullLicense"
        else:
            solver = pyo.SolverFactory(settings.name)
            is_available = solver.available(exception_flag=False)
    except Exception as err:
        raise SolverUnavailableError(f"cannot create solver '{settings.name}': {err}") from err
    if not is_available:
        raise SolverUnavailableError(
            f"solver '{settings.name}' unavailable (install highspy or use gurobi/cplex/cbc)"
        )

    stats = {
        "variables": sum(1 for _ in model.component_data_objects(pyo.Var)),
        "binaries": sum(1 for variable in model.component_data_objects(pyo.Var) if variable.is_binary()),
        "constraints": sum(
            1 for _ in model.component_data_objects(pyo.Constraint, active=True)
        ),
    }
    log.info("solving: %(variables)d vars (%(binaries)d binary), %(constraints)d constraints", stats)
    start_time = time.perf_counter()
    gap = None
    try:
        if settings.name.startswith("appsi_"):
            solver.config.load_solution = False
            solver.config.stream_solver = settings.verbose
            solver.config.time_limit = settings.time_limit_s
            solver.config.mip_gap = settings.mip_gap
            if settings.name == "appsi_highs" and settings.threads:
                solver.highs_options["threads"] = int(settings.threads)
            result = solver.solve(model)
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
            gap_name, time_name = option_names.get(settings.name, ("mipgap", "timelimit"))
            solver.options[gap_name] = settings.mip_gap
            solver.options[time_name] = settings.time_limit_s
            result = solver.solve(model, tee=settings.verbose, load_solutions=False)
            termination = str(result.solver.termination_condition).lower()
            has_solution = len(result.solution) > 0
            if has_solution:
                model.solutions.load_from(result)
    except SchedulingError:
        raise
    except Exception as err:
        raise SolveFailedError(f"solver crashed: {err}") from err

    elapsed = time.perf_counter() - start_time
    if "infeasible" in termination or "unbounded" in termination:
        raise InfeasibleScheduleError(
            f"model is {termination}; check demand, production capacity, grid limits, and batch limits"
        )
    if not has_solution:
        raise SolveFailedError(
            f"no feasible solution within the limits (termination: {termination}, {elapsed:.1f}s)"
        )
    status = "optimal" if "optimal" in termination else f"feasible ({termination})"
    if status != "optimal":
        log.warning("stopped early: %s, gap=%s", termination, gap)
    return {"status": status, "gap": gap, "time": elapsed, "stats": stats}
