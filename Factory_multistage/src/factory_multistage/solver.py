from __future__ import annotations

import time
from typing import Dict

import pyomo.environ as pyo

from ._internal import EPS, log
from .data import SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    SchedulingError,
    SolveFailedError,
    SolverUnavailableError,
)


def solve_model(model: pyo.ConcreteModel, st: SolverSettings) -> Dict[str, object]:
    try:
        if st.name == "appsi_highs":
            from pyomo.contrib.appsi.solvers import (
                Highs,  # native APPSI object (not the legacy wrapper)
            )
            opt = Highs()
            ok = opt.available() in (True, 1) or bool(getattr(opt.available(), "name", "") == "FullLicense")
        else:
            opt = pyo.SolverFactory(st.name)
            ok = opt.available(exception_flag=False)
    except Exception as err:   # noqa: BLE001
        raise SolverUnavailableError(f"cannot create solver '{st.name}': {err}") from err
    if not ok:
        raise SolverUnavailableError(f"solver '{st.name}' unavailable (pip install highspy, or use gurobi/cplex/cbc)")
    stats = dict(
        variables=sum(1 for _ in model.component_data_objects(pyo.Var)),
        binaries=sum(1 for v in model.component_data_objects(pyo.Var) if v.is_binary()),
        constraints=sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True)))
    log.info("solving: %(variables)d vars (%(binaries)d binary), %(constraints)d constraints", stats)
    t0 = time.perf_counter()
    gap = None
    try:
        if st.name.startswith("appsi_"):
            opt.config.load_solution = False
            opt.config.stream_solver = st.verbose
            opt.config.time_limit = st.time_limit_s
            opt.config.mip_gap = st.mip_gap
            if st.name == "appsi_highs" and st.threads:
                opt.highs_options["threads"] = int(st.threads)
            res = opt.solve(model)
            term = str(res.termination_condition).lower()
            has_sol = res.best_feasible_objective is not None
            if has_sol:
                res.solution_loader.load_vars()
                ub, lb = res.best_feasible_objective, res.best_objective_bound
                if lb is not None and abs(ub) > EPS:
                    gap = max(0.0, (ub - lb) / abs(ub))
        else:
            keys = {"gurobi": ("MIPGap", "TimeLimit"), "cplex": ("mipgap", "timelimit"),
                    "cbc": ("ratioGap", "seconds"), "glpk": ("mipgap", "tmlim")}
            gk, tk = keys.get(st.name, ("mipgap", "timelimit"))
            opt.options[gk], opt.options[tk] = st.mip_gap, st.time_limit_s
            res = opt.solve(model, tee=st.verbose, load_solutions=False)
            term = str(res.solver.termination_condition).lower()
            has_sol = len(res.solution) > 0
            if has_sol:
                model.solutions.load_from(res)
    except SchedulingError:
        raise
    except Exception as err:   # noqa: BLE001
        raise SolveFailedError(f"solver crashed: {err}") from err
    elapsed = time.perf_counter() - t0
    if "infeasible" in term or "unbounded" in term:
        raise InfeasibleScheduleError(
            f"model is {term}. Typical causes: demand too high for the available machine time, "
            f"grid import limit too low, N_T >= N_0 not reachable, or max_batches_per_machine too small.")
    if not has_sol:
        raise SolveFailedError(f"no feasible solution within the limits (termination: {term}, {elapsed:.1f}s)")
    status = "optimal" if "optimal" in term else f"feasible ({term})"
    if status != "optimal":
        log.warning("stopped early: %s, gap=%s", term, gap)
    return dict(status=status, gap=gap, time=elapsed, stats=stats)
