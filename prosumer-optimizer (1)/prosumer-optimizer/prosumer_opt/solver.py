"""
Solver layer: create a solver, run it, and normalise status / gap / statistics.

Two back-ends are supported:

* **APPSI** (default ``appsi_highs``). The APPSI class is instantiated *directly*
  (``pyomo.contrib.appsi.solvers.highs.Highs``). Recent Pyomo releases
  (>= 6.9) make ``SolverFactory('appsi_highs')`` return a legacy wrapper whose results
  object has no ``termination_condition``, so the factory must not be used for APPSI.
* **Legacy LP-file solvers** through ``SolverFactory`` (``gurobi``, ``cplex``,
  ``cbc``, ``glpk``, ...).
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import pyomo.environ as pyo

from .exceptions import (ModelInfeasibleError, ProsumerOptimizationError,
                         SolveFailedError, SolverUnavailableError)
from .parameters import SolverSettings

log = logging.getLogger("prosumer_opt")

class _Appsi(NamedTuple):
    module: str
    cls: str
    options_attr: str       # name of the options dict on the solver object
    threads_key: str
    output_key: str         # solver option that switches console output on / off


_APPSI: Dict[str, _Appsi] = {
    "appsi_highs": _Appsi("pyomo.contrib.appsi.solvers.highs", "Highs", "highs_options", "threads", "output_flag"),
    "appsi_gurobi": _Appsi("pyomo.contrib.appsi.solvers.gurobi", "Gurobi", "gurobi_options", "Threads", "OutputFlag"),
}

# Pyomo always forwards captured solver output to a logger. Give it one that goes nowhere, so solver
# chatter never lands in the user's log handlers; `verbose` streams it to stdout instead.
_SOLVER_OUTPUT_LOGGER = logging.getLogger("prosumer_opt.solver_output")
_SOLVER_OUTPUT_LOGGER.addHandler(logging.NullHandler())
_SOLVER_OUTPUT_LOGGER.propagate = False

_GAP_KEY = {"gurobi": "MIPGap", "cplex": "mipgap", "cbc": "ratioGap", "glpk": "mipgap"}
_TIME_KEY = {"gurobi": "TimeLimit", "cplex": "timelimit", "cbc": "seconds", "glpk": "tmlim"}


@dataclass
class SolveInfo:
    """Normalised outcome of one solver run."""
    status: str
    gap: Optional[float]
    time: float
    stats: Dict[str, int]


def model_statistics(model: pyo.ConcreteModel) -> Dict[str, int]:
    """Number of free variables, free binaries and active constraints."""
    free = [v for v in model.component_data_objects(pyo.Var) if not v.fixed]
    return dict(
        variables=len(free),
        binaries=sum(1 for v in free if v.is_binary()),
        constraints=sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True)),
    )


def _solve_appsi(model, s: SolverSettings) -> Tuple[str, bool, Optional[float]]:
    spec = _APPSI[s.name]
    try:
        opt = getattr(importlib.import_module(spec.module), spec.cls)()
        available = bool(opt.available())
    except Exception as err:   # noqa: BLE001 - import/licence problems come in many types
        raise SolverUnavailableError(f"cannot create solver '{s.name}': {err}") from err
    if not available:
        raise SolverUnavailableError(f"solver '{s.name}' is not available (pip install highspy).")

    opt.config.load_solution = False
    opt.config.stream_solver = s.verbose
    opt.config.solver_output_logger = _SOLVER_OUTPUT_LOGGER
    opt.config.time_limit = s.time_limit_s
    opt.config.mip_gap = s.mip_gap
    options = getattr(opt, spec.options_attr)
    options[spec.output_key] = bool(s.verbose) if s.name == "appsi_highs" else int(s.verbose)
    if s.threads:
        options[spec.threads_key] = int(s.threads)
    for k, v in s.extra_options.items():      # explicit user options always win
        options[k] = v

    res = opt.solve(model)
    term = str(res.termination_condition).lower()
    has_sol = res.best_feasible_objective is not None
    gap = None
    if has_sol:
        res.solution_loader.load_vars()
        ub, lb = res.best_feasible_objective, res.best_objective_bound
        if lb is not None and abs(ub) > 1e-9:
            gap = max(0.0, abs(ub - lb) / max(abs(ub), 1e-9))
    return term, has_sol, gap


def _solve_legacy(model, s: SolverSettings) -> Tuple[str, bool, Optional[float]]:
    try:
        opt = pyo.SolverFactory(s.name)
        available = opt.available(exception_flag=False)
    except Exception as err:   # noqa: BLE001
        raise SolverUnavailableError(f"cannot create solver '{s.name}': {err}") from err
    if not available:
        raise SolverUnavailableError(
            f"solver '{s.name}' is not available. Try `pip install highspy` (appsi_highs), "
            f"or set solver.name to gurobi / cplex / cbc.")

    opt.options.update(s.extra_options)
    opt.options[_GAP_KEY.get(s.name, "mipgap")] = s.mip_gap
    opt.options[_TIME_KEY.get(s.name, "timelimit")] = s.time_limit_s
    if s.threads and s.name in ("gurobi", "cplex"):
        opt.options["threads"] = int(s.threads)
    res = opt.solve(model, tee=s.verbose, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    has_sol = len(res.solution) > 0
    gap = None
    if has_sol:
        model.solutions.load_from(res)
        lb = getattr(res.problem, "lower_bound", None)
        ub = getattr(res.problem, "upper_bound", None)
        if lb is not None and ub is not None and abs(ub) > 1e-9:
            gap = max(0.0, abs(ub - lb) / abs(ub))
    return term, has_sol, gap


def solve_model(model: pyo.ConcreteModel, settings: SolverSettings, label: str) -> SolveInfo:
    """Solve ``model`` and load the solution into it. Raises on infeasibility or failure."""
    stats = model_statistics(model)
    log.info("[%s] solving: %d vars (%d binary), %d constraints",
             label, stats["variables"], stats["binaries"], stats["constraints"])

    t_start = time.perf_counter()
    try:
        backend = _solve_appsi if settings.name in _APPSI else _solve_legacy
        if settings.name.startswith("appsi_") and settings.name not in _APPSI:
            raise SolverUnavailableError(
                f"unsupported APPSI solver '{settings.name}'; supported: {sorted(_APPSI)}")
        term, has_sol, gap = backend(model, settings)
    except ProsumerOptimizationError:
        raise
    except Exception as err:   # noqa: BLE001
        raise SolveFailedError(f"[{label}] solver crashed: {err}") from err
    elapsed = time.perf_counter() - t_start

    if "infeasible" in term or "unbounded" in term:
        raise ModelInfeasibleError(f"[{label}] model is {term}. Check limits, windows, inventory and initial state.")
    if not has_sol:
        raise SolveFailedError(f"[{label}] no feasible solution found (termination: {term}, {elapsed:.1f}s). "
                               f"Increase time_limit_s or loosen mip_gap.")
    status = "optimal" if "optimal" in term else f"feasible ({term})"
    if status != "optimal":
        log.warning("[%s] stopped with %s; gap=%s", label, term, gap)
    return SolveInfo(status=status, gap=gap, time=elapsed, stats=stats)
