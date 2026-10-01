#!/usr/bin/env python3
"""
factory_da_scheduler.py
=======================

Deterministic Day-Ahead (DA) scheduling of industrial batch production.

The program chooses WHICH production tasks run, ON WHICH furnace/line, and AT WHAT
TIME, so that the DA electricity bill of a large industrial consumer is minimised while
product deliveries, warehouse limits and the substation line limit are respected.

    min  sum_t  lambda_t * P_t                                         (EUR)

    P_t = l_base,t + sum_{m,p} d_{m,p} * o_{m,p,t}                     power balance
    0 <= P_t <= Q_md                                                   substation limit
    machine m: batches never overlap and are separated by t_buffer     sequencing
    sum_t o_{m,p,t} = td_{m,p}  if task p runs on m                    task energy
    n_prod,t = sum Y_p * o_{m,p,t} / td_{m,p}                          yield coupling
    N_t = N_{t-1} + n_prod,t - dem_t,  0 <= N_t <= N_max,  N_T >= N_0 inventory

Formulation note (important)
----------------------------
The reference model uses X[m,n,p], continuous start times ts[m,n] and a continuous
occupancy fraction o[m,n,t,p]. Taken literally, nothing ties o to ts, so a task could be
split into arbitrary pieces over cheap hours (pre-emption), which a furnace batch cannot
do. This program therefore uses the exactly equivalent TIME-INDEXED form:

    s[m,p,k] = 1  if task p starts on machine m at grid time a_k = k * start_step_h

The hourly occupancy o[m,p,t] is then the exact overlap of [a_k, a_k + td_{m,p}] with
hour t (a constant), so batches are contiguous and non-interruptible, durations such as
1.3 h are handled exactly, and the model stays a pure MILP with a tight LP relaxation.
The batch position index n and start time ts[m,n] are recovered after the solve (sorted
by start time per machine). N_max batches per machine is kept as a cap.

INPUTS
------
Instance (see dataclass; can be loaded from / saved to JSON)
    horizon_h              scheduling horizon, hours (24)
    machines, tasks        names (5 machines, 30 tasks)
    power_mw[m,p]          power of task p on machine m          (MW)
    duration_h[m,p]        processing time                       (h)
    yield_units[p]         finished goods per completed task     (units)
    buffer_h               setup/cooling gap between batches     (h)
    base_load_mw[t]        non-shiftable load                    (MW)
    price_eur_mwh[t]       DA price                              (EUR/MWh)
    grid_limit_mw          substation import limit Q_md          (MW)
    max_batches_per_machine  N
    inventory_init / inventory_max / demand_units[t]
SchedulerConfig
    start_step_h           start-time grid (smaller = better solution, bigger model)
    unique_tasks           each task p may run at most once per day
    solver settings        name, MIP gap, time limit, threads

OUTPUTS (SchedulingResult)
--------------------------
    status, objective_eur, mip_gap, solve_time_s, model_stats
    jobs      DataFrame: machine, position n, task, start/end (h and HH:MM), power, yield,
              energy (MWh) and energy cost (EUR) of every executed batch
    hourly    DataFrame: price, base load, batch load, P_DA, production, demand, inventory, cost
    kpis      cost, energy, average paid price, peak load, load factor, cost per unit,
              saving vs an as-soon-as-possible baseline
    verification  independent re-computation of every constraint from the job list
    Files written by save(): jobs.csv, hourly.csv, summary.json (+ schedule.png with --plot)

Requires: numpy, pandas, pyomo, highspy (default solver appsi_highs). matplotlib is optional.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

__version__ = "1.0.0"
log = logging.getLogger("factory_da")
EPS = 1e-9


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
def _arr(x, n: Optional[int], name: str, ndim: int = 1) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        raise InstanceValidationError(f"{name}: not numeric ({err})") from err
    if ndim == 1 and a.ndim == 0 and n is not None:
        a = np.full(n, float(a))
    if a.ndim != ndim:
        raise InstanceValidationError(f"{name}: expected {ndim}-D array, got shape {a.shape}")
    if n is not None and ndim == 1 and a.shape[0] != n:
        raise InstanceValidationError(f"{name}: expected length {n}, got {a.shape[0]}")
    if not np.all(np.isfinite(a)):
        raise InstanceValidationError(f"{name}: contains NaN/inf")
    return a


@dataclass
class Instance:
    machines: List[str]
    tasks: List[str]
    power_mw: np.ndarray            # [M, P]
    duration_h: np.ndarray          # [M, P]
    yield_units: np.ndarray         # [P]
    base_load_mw: np.ndarray        # [T]
    price_eur_mwh: np.ndarray       # [T]
    demand_units: np.ndarray        # [T]
    horizon_h: int = 24
    buffer_h: float = 1.0
    grid_limit_mw: float = 400.0
    max_batches_per_machine: int = 10
    inventory_init: float = 50.0
    inventory_max: float = 500.0

    # ------------------------------------------------------------------ checks
    def validate(self) -> "Instance":
        T, M, P = self.horizon_h, len(self.machines), len(self.tasks)
        if T <= 0 or M <= 0 or P <= 0:
            raise InstanceValidationError("horizon, machines and tasks must be non-empty")
        if len(set(self.machines)) != M or len(set(self.tasks)) != P:
            raise InstanceValidationError("machine and task names must be unique")
        self.power_mw = _arr(self.power_mw, None, "power_mw", 2)
        self.duration_h = _arr(self.duration_h, None, "duration_h", 2)
        self.yield_units = _arr(self.yield_units, P, "yield_units")
        self.base_load_mw = _arr(self.base_load_mw, T, "base_load_mw")
        self.price_eur_mwh = _arr(self.price_eur_mwh, T, "price_eur_mwh")
        self.demand_units = _arr(self.demand_units, T, "demand_units")
        for nm, a in (("power_mw", self.power_mw), ("duration_h", self.duration_h)):
            if a.shape != (M, P):
                raise InstanceValidationError(f"{nm}: expected shape {(M, P)}, got {a.shape}")
        if np.any(self.power_mw < 0) or np.any(self.duration_h <= 0):
            raise InstanceValidationError("power must be >= 0 and duration > 0")
        if np.any(self.duration_h > T):
            raise InstanceValidationError("a task duration exceeds the horizon")
        if np.any(self.yield_units < 0) or np.any(self.demand_units < 0) or np.any(self.base_load_mw < 0):
            raise InstanceValidationError("yield, demand and base load must be >= 0")
        if self.buffer_h < 0:
            raise InstanceValidationError("buffer_h must be >= 0")
        if self.max_batches_per_machine < 1:
            raise InstanceValidationError("max_batches_per_machine must be >= 1")
        if not (0 <= self.inventory_init <= self.inventory_max):
            raise InstanceValidationError("inventory_init must lie in [0, inventory_max]")
        if self.grid_limit_mw <= 0:
            raise InstanceValidationError("grid_limit_mw must be positive")
        if np.any(self.base_load_mw > self.grid_limit_mw):
            raise InstanceValidationError("base load alone exceeds the substation limit")
        # Quick necessary condition: even running everything cannot be required to exceed the best yield.
        if self.demand_units.sum() > self.inventory_max * 0 + self.yield_units.sum() * 1.0 + self.inventory_init:
            raise InstanceValidationError(
                f"daily demand {self.demand_units.sum():.0f} exceeds initial stock plus total yield of all tasks "
                f"({self.inventory_init + self.yield_units.sum():.0f}); the instance cannot be feasible")
        return self

    # ---------------------------------------------------------------- JSON I/O
    def to_dict(self) -> dict:
        return dict(machines=self.machines, tasks=self.tasks, power_mw=self.power_mw.tolist(),
                    duration_h=self.duration_h.tolist(), yield_units=self.yield_units.tolist(),
                    base_load_mw=self.base_load_mw.tolist(), price_eur_mwh=self.price_eur_mwh.tolist(),
                    demand_units=self.demand_units.tolist(), horizon_h=self.horizon_h, buffer_h=self.buffer_h,
                    grid_limit_mw=self.grid_limit_mw, max_batches_per_machine=self.max_batches_per_machine,
                    inventory_init=self.inventory_init, inventory_max=self.inventory_max)

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        try:
            return cls(**d).validate()
        except TypeError as err:
            raise InstanceValidationError(f"bad instance schema: {err}") from err

    @classmethod
    def load(cls, path) -> "Instance":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read instance '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class SolverSettings:
    name: str = "appsi_highs"
    mip_gap: float = 1e-4            # 0.01 % as recommended for the reference model
    time_limit_s: float = 180.0
    threads: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        if not (0 <= self.mip_gap < 1) or self.time_limit_s <= 0:
            raise InstanceValidationError("mip_gap must be in [0,1) and time_limit_s > 0")


@dataclass
class SchedulerConfig:
    start_step_h: float = 0.5
    unique_tasks: bool = True
    solver: SolverSettings = field(default_factory=SolverSettings)
    strict_verification: bool = True

    def check(self, inst: Instance) -> None:
        r = inst.horizon_h / self.start_step_h
        if self.start_step_h <= 0 or abs(r - round(r)) > 1e-9:
            raise InstanceValidationError("horizon_h must be an integer multiple of start_step_h")


@dataclass
class SchedulingResult:
    status: str
    objective_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    jobs: pd.DataFrame
    hourly: pd.DataFrame
    kpis: Dict[str, float]
    verification: Dict[str, object]

    def summary(self) -> str:
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        k = self.kpis
        return (f"status={self.status}  cost={self.objective_eur:,.2f} EUR  gap={gap}  time={self.solve_time_s:.1f}s\n"
                f"batches={int(k['n_batches'])}  units={k['units_produced']:.0f}  grid_energy={k['grid_energy_mwh']:.1f} MWh  "
                f"avg_price={k['avg_price_paid']:.2f} EUR/MWh  peak={k['peak_mw']:.1f} MW  "
                f"cost/unit={k['cost_per_unit']:.2f} EUR\n"
                f"ASAP baseline={k['baseline_asap_cost_eur']:,.2f} EUR  saving={k['saving_vs_asap_pct']:.1f}%  "
                f"verification={'PASS' if self.verification['passed'] else 'FAIL'}")

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(out / "jobs.csv", index=False)
        self.hourly.to_csv(out / "hourly.csv", index=False)
        meta = dict(version=__version__, status=self.status, objective_eur=self.objective_eur, mip_gap=self.mip_gap,
                    solve_time_s=self.solve_time_s, model_stats=self.model_stats, kpis=self.kpis,
                    verification=self.verification)
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
        return out


# --------------------------------------------------------------------------- #
# Benchmark instance (parameters from the reference model)
# --------------------------------------------------------------------------- #
_POWER_RANGES = {"m1": (70, 95), "m2": (60, 80), "m3": (70, 90), "m4": (90, 110), "m5": (65, 85)}
_TOU = [  # (EUR/MWh, [(start_h, end_h), ...])
    (66.42, [(2, 4), (12, 14), (18, 20), (22, 24)]),     # off-peak
    (88.56, [(4, 6)]),                                   # flat
    (118.08, [(0, 2), (6, 8), (14, 16), (20, 22)]),      # mid
    (177.12, [(8, 12), (16, 18)]),                       # on-peak
]


def tou_price_vector(horizon_h: int = 24) -> np.ndarray:
    price = np.zeros(horizon_h)
    for p, blocks in _TOU:
        for a, b in blocks:
            for h in range(a, min(b, horizon_h)):
                price[h] = p
    return price


def make_benchmark_instance(seed: int = 2024, n_tasks: int = 30, max_batches: int = 10) -> Instance:
    """5 furnaces, 30 tasks, TOU tariff, 8 units/h demand, N0=50, Nmax=500, Q_md=400 MW."""
    rng = np.random.default_rng(seed)
    machines = list(_POWER_RANGES)
    tasks = [f"p{i + 1}" for i in range(n_tasks)]
    power = np.array([np.round(rng.uniform(*_POWER_RANGES[m], n_tasks), 1) for m in machines])
    dur = np.round(rng.uniform(1.3, 4.9, (len(machines), n_tasks)), 1)
    yld = rng.integers(5, 16, n_tasks).astype(float)
    t = np.arange(24)
    base = np.round(3.5 + 1.0 * np.sin(2 * np.pi * (t - 6) / 24), 2)   # within [2.5, 4.5]
    return Instance(machines, tasks, power, dur, yld, base, tou_price_vector(24), np.full(24, 8.0),
                    max_batches_per_machine=max_batches).validate()


# --------------------------------------------------------------------------- #
# Candidate batches (time-indexed start grid)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    m: int
    p: int
    k: int
    start_h: float
    dur_h: float
    power_mw: float
    occ: np.ndarray          # hourly occupancy in [0,1] (exact overlap with each hour)


def build_candidates(inst: Instance, step: float) -> List[Candidate]:
    T = inst.horizon_h
    n_grid = int(round(T / step))
    hours = np.arange(T, dtype=float)
    out: List[Candidate] = []
    for m in range(len(inst.machines)):
        for p in range(len(inst.tasks)):
            D = float(inst.duration_h[m, p])
            for k in range(n_grid):
                a = k * step
                if a + D > T + EPS:
                    break
                occ = np.clip(np.minimum(a + D, hours + 1) - np.maximum(a, hours), 0.0, 1.0)
                out.append(Candidate(m, p, k, a, D, float(inst.power_mw[m, p]), occ))
    if not out:
        raise InstanceValidationError("no feasible start time exists for any task")
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(inst: Instance, cfg: SchedulerConfig, cands: List[Candidate]) -> pyo.ConcreteModel:
    T, M, P = inst.horizon_h, len(inst.machines), len(inst.tasks)
    step = cfg.start_step_h
    n_grid = int(round(T / step))
    m = pyo.ConcreteModel("FactoryDayAhead")
    m.H = pyo.RangeSet(0, T - 1)
    m.C = pyo.RangeSet(0, len(cands) - 1)
    m.s = pyo.Var(m.C, domain=pyo.Binary)                                   # batch start decision
    m.P = pyo.Var(m.H, bounds=(0, inst.grid_limit_mw))                      # DA purchase (MW)
    m.N = pyo.Var(m.H, bounds=(0, inst.inventory_max))                      # end-of-hour inventory

    by_hour: List[List[Tuple[int, float, float]]] = [[] for _ in range(T)]  # (cand, MW, units)
    by_task: Dict[int, List[int]] = {p: [] for p in range(P)}
    by_machine: Dict[int, List[int]] = {k: [] for k in range(M)}
    slot_users: Dict[Tuple[int, int], List[int]] = {}
    for c, cd in enumerate(cands):
        by_task[cd.p].append(c)
        by_machine[cd.m].append(c)
        for t in np.nonzero(cd.occ > 0)[0]:
            by_hour[t].append((c, cd.power_mw * cd.occ[t], inst.yield_units[cd.p] * cd.occ[t] / cd.dur_h))
        j_end = min(n_grid - 1, math.ceil((cd.start_h + cd.dur_h + inst.buffer_h - EPS) / step) - 1)
        for j in range(cd.k, j_end + 1):
            slot_users.setdefault((cd.m, j), []).append(c)

    m.c_power = pyo.Constraint(m.H, rule=lambda mm, t: mm.P[t] == inst.base_load_mw[t]
                               + pyo.quicksum(mw * mm.s[c] for c, mw, _ in by_hour[t]))
    m.c_inv = pyo.Constraint(m.H, rule=lambda mm, t: mm.N[t] == (mm.N[t - 1] if t > 0 else inst.inventory_init)
                             + pyo.quicksum(u * mm.s[c] for c, _, u in by_hour[t]) - inst.demand_units[t])
    m.c_terminal = pyo.Constraint(expr=m.N[T - 1] >= inst.inventory_init)
    if cfg.unique_tasks:
        m.c_unique = pyo.Constraint(range(P), rule=lambda mm, p: pyo.quicksum(mm.s[c] for c in by_task[p]) <= 1)
    m.c_cap = pyo.Constraint(range(M), rule=lambda mm, k: pyo.quicksum(mm.s[c] for c in by_machine[k])
                             <= inst.max_batches_per_machine)
    m.c_seq = pyo.ConstraintList()
    for (_, _), users in sorted(slot_users.items()):
        if len(users) > 1:
            m.c_seq.add(pyo.quicksum(m.s[c] for c in users) <= 1)
    m.obj = pyo.Objective(expr=pyo.quicksum(inst.price_eur_mwh[t] * m.P[t] for t in range(T)), sense=pyo.minimize)
    return m


# --------------------------------------------------------------------------- #
# Solve
# --------------------------------------------------------------------------- #
def solve_model(model: pyo.ConcreteModel, st: SolverSettings) -> Dict[str, object]:
    try:
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
            f"grid limit too low, N_T >= N_0 not reachable, or max_batches_per_machine too small.")
    if not has_sol:
        raise SolveFailedError(f"no feasible solution within the limits (termination: {term}, {elapsed:.1f}s)")
    status = "optimal" if "optimal" in term else f"feasible ({term})"
    if status != "optimal":
        log.warning("stopped early: %s, gap=%s", term, gap)
    return dict(status=status, gap=gap, time=elapsed, stats=stats)


# --------------------------------------------------------------------------- #
# Post-processing, verification, KPIs
# --------------------------------------------------------------------------- #
def hhmm(h: float) -> str:
    mins = int(round(h * 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _hourly_from_jobs(inst: Instance, jobs: pd.DataFrame) -> pd.DataFrame:
    """Independent (interval based) recomputation of the hourly profile from the job list."""
    T = inst.horizon_h
    hours = np.arange(T, dtype=float)
    batch = np.zeros(T)
    prod = np.zeros(T)
    for r in jobs.itertuples():
        ov = np.clip(np.minimum(r.end_h, hours + 1) - np.maximum(r.start_h, hours), 0, 1)
        batch += r.power_mw * ov
        prod += r.units_out * ov / (r.end_h - r.start_h)
    inv = inst.inventory_init + np.cumsum(prod - inst.demand_units)
    return pd.DataFrame(dict(hour=np.arange(T), price_eur_mwh=inst.price_eur_mwh, base_load_mw=inst.base_load_mw,
                             batch_load_mw=batch, p_da_mw=inst.base_load_mw + batch, production_units=prod,
                             demand_units=inst.demand_units, inventory_units=inv))


def verify(inst: Instance, cfg: SchedulerConfig, jobs: pd.DataFrame, objective: float,
           tol: float = 1e-4) -> Dict[str, object]:
    issues: List[str] = []
    hourly = _hourly_from_jobs(inst, jobs)
    cost = float((hourly.price_eur_mwh * hourly.p_da_mw).sum())
    if abs(cost - objective) > max(tol, 1e-6 * abs(objective)):
        issues.append(f"objective mismatch: recomputed {cost:.4f} vs solver {objective:.4f}")
    if hourly.p_da_mw.max() > inst.grid_limit_mw + tol:
        issues.append(f"substation limit exceeded: {hourly.p_da_mw.max():.2f} MW")
    if hourly.inventory_units.min() < -tol or hourly.inventory_units.max() > inst.inventory_max + tol:
        issues.append("inventory bounds violated")
    if hourly.inventory_units.iloc[-1] < inst.inventory_init - tol:
        issues.append("terminal stock below initial stock")
    for mach, g in jobs.groupby("machine"):
        g = g.sort_values("start_h")
        if len(g) > inst.max_batches_per_machine:
            issues.append(f"{mach}: more than N batches")
        gaps = g.start_h.values[1:] - g.end_h.values[:-1]
        if len(gaps) and gaps.min() < inst.buffer_h - tol:
            issues.append(f"{mach}: buffer violated (min gap {gaps.min():.3f} h)")
    if (jobs.end_h > inst.horizon_h + tol).any() or (jobs.start_h < -tol).any():
        issues.append("a batch lies outside the horizon")
    if cfg.unique_tasks and jobs.task.duplicated().any():
        issues.append("a task was scheduled more than once")
    return dict(passed=not issues, issues=issues, recomputed_cost_eur=cost)


def _asap_baseline(inst: Instance, jobs: pd.DataFrame) -> float:
    """Cost of the same batches, same machine order, started as early as possible (price blind)."""
    rows = []
    for _, g in jobs.sort_values("start_h").groupby("machine"):
        t = 0.0
        for r in g.itertuples():
            d = r.end_h - r.start_h
            rows.append(dict(start_h=t, end_h=t + d, power_mw=r.power_mw, units_out=r.units_out))
            t += d + inst.buffer_h
    base = pd.DataFrame(rows)
    if base.empty or (base.end_h > inst.horizon_h + 1e-9).any():
        return float("nan")
    h = _hourly_from_jobs(inst, base)
    return float((h.price_eur_mwh * h.p_da_mw).sum())


def extract_result(inst: Instance, cfg: SchedulerConfig, model: pyo.ConcreteModel, cands: List[Candidate],
                   info: Dict[str, object]) -> SchedulingResult:
    chosen = [c for c in range(len(cands)) if (model.s[c].value or 0.0) > 0.5]
    rows = []
    for c in chosen:
        cd = cands[c]
        rows.append(dict(machine=inst.machines[cd.m], task=inst.tasks[cd.p], start_h=cd.start_h,
                         end_h=cd.start_h + cd.dur_h, duration_h=cd.dur_h, power_mw=cd.power_mw,
                         units_out=float(inst.yield_units[cd.p])))
    jobs = pd.DataFrame(rows, columns=["machine", "task", "start_h", "end_h", "duration_h", "power_mw", "units_out"])
    jobs = jobs.sort_values(["machine", "start_h"]).reset_index(drop=True)
    jobs.insert(1, "position_n", jobs.groupby("machine").cumcount() + 1)
    jobs["start_time"] = jobs.start_h.map(hhmm)
    jobs["end_time"] = jobs.end_h.map(hhmm)
    jobs["energy_mwh"] = jobs.power_mw * jobs.duration_h
    hourly = _hourly_from_jobs(inst, jobs)
    hourly["p_da_model_mw"] = [model.P[t].value or 0.0 for t in range(inst.horizon_h)]
    hourly["cost_eur"] = hourly.price_eur_mwh * hourly.p_da_mw
    # energy cost per batch (exact, hour by hour)
    hrs = np.arange(inst.horizon_h, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]

    obj = float(pyo.value(model.obj))
    ver = verify(inst, cfg, jobs, obj)
    if not ver["passed"]:
        msg = "; ".join(ver["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)

    units = float(jobs.units_out.sum())
    e = float(hourly.p_da_mw.sum())
    base_cost = _asap_baseline(inst, jobs)
    kpis = dict(
        n_batches=float(len(jobs)), units_produced=units, grid_energy_mwh=e,
        batch_energy_mwh=float(jobs.energy_mwh.sum()), avg_price_paid=obj / e if e > 0 else float("nan"),
        peak_mw=float(hourly.p_da_mw.max()), load_factor=float(hourly.p_da_mw.mean() / hourly.p_da_mw.max()),
        cost_per_unit=obj / units if units > 0 else float("nan"), baseline_asap_cost_eur=base_cost,
        saving_vs_asap_pct=100 * (base_cost - obj) / base_cost if base_cost and not math.isnan(base_cost) else float("nan"),
        final_inventory=float(hourly.inventory_units.iloc[-1]))
    return SchedulingResult(info["status"], obj, info["gap"], info["time"], info["stats"], jobs, hourly, kpis, ver)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def optimize_day_ahead(inst: Instance, cfg: Optional[SchedulerConfig] = None) -> SchedulingResult:
    """Validate -> build candidates -> build MILP -> solve -> verify -> package results."""
    cfg = cfg or SchedulerConfig()
    inst.validate()
    cfg.check(inst)
    cands = build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts (machines=%d, tasks=%d, step=%.2f h)", len(cands),
             len(inst.machines), len(inst.tasks), cfg.start_step_h)
    model = build_model(inst, cfg, cands)
    info = solve_model(model, cfg.solver)
    return extract_result(inst, cfg, model, cands, info)


def plot_schedule(inst: Instance, res: SchedulingResult, path) -> bool:
    """Gantt chart plus load/price profile. Returns False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[3, 2]), sharex=True)
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Day-ahead batch schedule  |  cost {res.objective_eur:,.0f} EUR")
    h = res.hourly
    a2.bar(h.hour + 0.5, h.p_da_mw, width=0.9, color="tab:blue", label="P_DA (MW)")
    a2.axhline(inst.grid_limit_mw, color="r", ls="--", lw=1, label="Q_md")
    a2.set_ylabel("MW")
    a3 = a2.twinx()
    a3.step(np.append(h.hour, inst.horizon_h), np.append(h.price_eur_mwh, h.price_eur_mwh.iloc[-1]),
            where="post", color="tab:orange", label="DA price")
    a3.set_ylabel("EUR/MWh")
    a2.set_xlabel("hour of day")
    a2.set_xlim(0, inst.horizon_h)
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Factory day-ahead batch scheduling (MILP)")
    ap.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    ap.add_argument("--seed", type=int, default=2024, help="benchmark generator seed")
    ap.add_argument("--max-batches", type=int, default=10, help="N, max batches per machine")
    ap.add_argument("--start-step", type=float, default=0.5, help="start-time grid in hours")
    ap.add_argument("--allow-repeat-tasks", action="store_true", help="a task may run more than once")
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="da_results")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--export-instance", help="write the instance used to this JSON path")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        inst = Instance.load(a.instance) if a.instance else make_benchmark_instance(a.seed, max_batches=a.max_batches)
        if a.export_instance:
            inst.save(a.export_instance)
        cfg = SchedulerConfig(a.start_step, not a.allow_repeat_tasks,
                              SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_day_ahead(inst, cfg)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw",
                        "units_out", "energy_mwh", "energy_cost_eur"]].round(2).to_string(index=False))
        print(res.hourly[["hour", "price_eur_mwh", "base_load_mw", "batch_load_mw", "p_da_mw",
                          "inventory_units"]].round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_schedule(inst, res, out / "schedule.png")
        print(f"\nResults written to {out.resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, InfeasibleScheduleError, SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
