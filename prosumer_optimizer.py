#!/usr/bin/env python3
"""
prosumer_optimizer.py
=====================

Multi-stage stochastic MILP for a large industrial prosumer that owns

    * batch production processes (parallel machines, non-interruptible tasks),
    * a Battery Energy Storage System (BESS),
    * an on-site Microturbine (MT, unit commitment + piecewise-linear fuel cost),

and trades on the Day-Ahead (DA), Intraday (ID) and Real-Time balancing markets.
All money is in EUR, power in MW, energy in MWh, time in hours.

Decision architecture
---------------------
    STAGE 1  Day-Ahead ("here-and-now"), solved once before gate closure
             * DA purchase / sale schedule           p_da_buy[t], p_da_sell[t]
             * MT commitment                         u[t], x[t] (start), y[t] (stop)
             * Master batch schedule                 z[job, start_step]
             These variables are NOT scenario indexed, so the non-anticipativity
             condition holds by construction.

    STAGE 2  Intraday recourse (one copy per ID scenario i)
             * ID buy / sell                         id_buy[t,i], id_sell[t,i]
             * MT re-dispatch inside the DA commitment
             * BESS charge / discharge, SoC
             * Batch start shifting inside +/- window  z2[job, start_step, i]

    STAGE 3  Real-time settlement (one copy per ID scenario i and RT branch r)
             * Imbalance volumes  dpos[t,i,r] (surplus), dneg[t,i,r] (shortfall)
             * Settled at DA reference price times ratios r_plus (<=1) / r_minus (>=1)

Two entry points
----------------
    MultiStageProsumerOptimizer.solve_day_ahead(tree)          -> StageResult
    MultiStageProsumerOptimizer.solve_intraday(da_result, inp) -> StageResult

`solve_intraday` re-optimises the remaining hours of the day *after* the DA
commitments are frozen, using refreshed ID prices, refreshed load forecast and
the measured state (SoC, MT output, inventory) at the re-optimisation step.

INPUTS (summary, details in the dataclasses below)
--------------------------------------------------
    TimeGrid                 horizon_h, dt_h
    GridParams               import / export limit (MW)
    MicroturbineParams       limits, ramps, min up/down, start/stop cost, fuel blocks
    BESSParams               power, energy, SoC window, efficiencies, throughput cost
    FactoryParams            base load, BatchJob list, buffer, inventory, demand
    ScenarioTree             ID scenarios (probabilities, DA/ID buy+sell prices)
                             each with RT branches (load deviation, r_plus, r_minus)
    SolverSettings           solver name, MIP gap, time limit, threads
    InitialState             SoC, MT status/output, inventory at start of horizon
    IntradayInputs           t0, measured state, updated scenario, optional load forecast

OUTPUTS (StageResult)
---------------------
    status, objective_eur, mip_gap, solve_time_s, model_stats
    schedule        DataFrame per time step: DA position, MT commitment, expected
                    MT / BESS / ID / load / imbalance values, expected SoC, inventory
    batch_plan      DataFrame per job: machine, DA start/end, power, per-scenario start
    scenario_detail long DataFrame with every scenario level decision
    cost_breakdown  expected EUR by component (DA, ID, BAL, MT, BESS, UNMET)
    scenario_costs  DataFrame with cost by scenario plus mean, std, worst, CVaR
    plan            DAPlan (frozen DA commitments, input for the intraday stage)

Modelling notes and deliberate deviations from the textbook formulation
-----------------------------------------------------------------------
    1. MT cost curve. The reference blocks (5.3, 7.2, 7.2, 5.3 MW) sum to 25 MW
       and describe the curve from 0 MW, while P_MT = P_min*u + sum(blocks) with
       P_min = 10 MW would allow 35 MW. Blocks are therefore clipped to the range
       above P_min (0, 2.5, 7.2, 5.3 MW). The cost of the first P_min MW is added
       as a no-load cost per committed hour, otherwise running at P_min would be free.
    2. Market sides. Buy and sell use separate variables and prices (sell <= buy is
       validated), which is the linear equivalent of lambda*P with signed P when a
       spread exists.
    3. Batch tasks use a time-indexed start formulation, so durations are rounded
       up to whole time steps and start times are linear in the binaries.
    4. Product delivery shortfall is a soft constraint (penalised slack), so the
       model stays feasible when the intraday state makes a delivery impossible.
    5. Non-decreasing bid-curve constraints are not needed: the DA position is a
       single scenario-independent schedule (self-schedule of a price taker).

Usage
-----
    python prosumer_optimizer.py --id-scenarios 5 --rt-branches 3 --intraday-step 10 --out results/

Requires: pyomo, numpy, pandas, and a MILP solver (default `appsi_highs`: pip install highspy).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

__version__ = "1.0.0"
log = logging.getLogger("prosumer_opt")
_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ProsumerOptimizationError(Exception):
    """Base class for every error raised by this module."""


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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require(cond: bool, msg: str, exc: type = ConfigurationError) -> None:
    if not cond:
        raise exc(msg)


def _series(x, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate an array of length n; must be finite."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        raise DataValidationError(f"{name}: cannot convert to float array ({err})") from err
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise DataValidationError(f"{name}: expected length {n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise DataValidationError(f"{name}: contains NaN or inf")
    return arr


def _val(var) -> float:
    v = var.value
    return 0.0 if v is None else float(v)


def cvar(costs: Sequence[float], probs: Sequence[float], alpha: float = 0.95) -> float:
    """Conditional value at risk of a discrete cost distribution (upper tail)."""
    c = np.asarray(costs, float)
    p = np.asarray(probs, float)
    order = np.argsort(-c)
    c, p = c[order], p[order]
    tail = 1.0 - alpha
    if tail <= 0:
        return float(c[0])
    acc, total = 0.0, 0.0
    for ci, pi in zip(c, p):
        w = min(pi, tail - acc)
        if w <= 0:
            break
        total += w * ci
        acc += w
    return float(total / tail)


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeGrid:
    """Optimisation horizon. `horizon_h` must be an integer multiple of `dt_h`."""
    horizon_h: float = 24.0
    dt_h: float = 1.0

    def __post_init__(self):
        _require(self.dt_h > 0 and self.horizon_h > 0, "horizon_h and dt_h must be positive")
        n = self.horizon_h / self.dt_h
        _require(abs(n - round(n)) < 1e-9, "horizon_h must be an integer multiple of dt_h")

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_h / self.dt_h))

    def steps(self, hours: float) -> int:
        """Hours -> number of steps, rounded up (conservative for durations)."""
        return int(math.ceil(hours / self.dt_h - 1e-9))

    def hour(self, t: int) -> float:
        return t * self.dt_h


@dataclass(frozen=True)
class GridParams:
    """Point of common coupling limits (MW)."""
    import_limit_mw: float = 400.0
    export_limit_mw: float = 400.0

    def __post_init__(self):
        _require(self.import_limit_mw > 0 and self.export_limit_mw >= 0, "grid limits must be positive")


@dataclass(frozen=True)
class MicroturbineParams:
    """
    Microturbine data. `cost_blocks` is a tuple of (width_MW, marginal_cost_EUR_per_MWh)
    describing the fuel-cost curve from 0 MW upwards; marginal costs must be non-decreasing
    (convex), which keeps the piecewise-linear cost an LP without extra binaries.
    """
    p_min_mw: float = 10.0
    p_max_mw: float = 25.0
    ramp_up_mw_h: float = 20.0
    ramp_down_mw_h: float = 20.0
    startup_ramp_mw_h: float = 20.0
    shutdown_ramp_mw_h: float = 20.0
    startup_cost_eur: float = 87.40
    shutdown_cost_eur: float = 8.74
    min_up_h: float = 4.0
    min_down_h: float = 2.0
    cost_blocks: Tuple[Tuple[float, float], ...] = (
        (5.30, 48.41), (7.20, 48.78), (7.20, 51.84), (5.30, 55.40))

    def __post_init__(self):
        _require(0 <= self.p_min_mw <= self.p_max_mw, "MT: need 0 <= p_min <= p_max")
        _require(min(self.ramp_up_mw_h, self.ramp_down_mw_h, self.startup_ramp_mw_h,
                     self.shutdown_ramp_mw_h) > 0, "MT: ramp limits must be positive")
        _require(self.startup_cost_eur >= 0 and self.shutdown_cost_eur >= 0, "MT: costs must be >= 0")
        _require(self.min_up_h >= 0 and self.min_down_h >= 0, "MT: min up/down must be >= 0")
        _require(len(self.cost_blocks) > 0, "MT: cost_blocks is empty")
        _require(all(w > 0 for w, _ in self.cost_blocks), "MT: block widths must be positive")
        costs = [c for _, c in self.cost_blocks]
        _require(all(b >= a - _TOL for a, b in zip(costs, costs[1:])),
                 "MT: marginal costs must be non-decreasing (convex curve)")
        _require(sum(w for w, _ in self.cost_blocks) >= self.p_max_mw - 1e-6,
                 "MT: cost blocks do not cover p_max")

    def effective_segments(self) -> List[Tuple[float, float]]:
        """Blocks clipped to the range (p_min, p_max]: list of (width, marginal cost)."""
        segs, lo = [], 0.0
        for w, c in self.cost_blocks:
            hi = lo + w
            a, b = max(lo, self.p_min_mw), min(hi, self.p_max_mw)
            if b - a > 1e-9:
                segs.append((b - a, c))
            lo = hi
        return segs

    def no_load_cost_eur_per_h(self) -> float:
        """Fuel cost of producing p_min (paid for every committed hour)."""
        cost, lo = 0.0, 0.0
        for w, c in self.cost_blocks:
            hi = lo + w
            b = min(hi, self.p_min_mw)
            if b > lo:
                cost += (b - lo) * c
            lo = hi
        return cost


@dataclass(frozen=True)
class BESSParams:
    """Battery data. Energies in MWh, powers in MW (AC side)."""
    p_max_mw: float = 40.0
    e_max_mwh: float = 200.0
    soc_min_mwh: float = 20.0
    soc_max_mwh: float = 200.0
    soc_init_mwh: float = 20.0
    eta_ch: float = 0.80
    eta_dis: float = 0.95
    throughput_cost_eur_mwh: float = 30.0
    terminal_soc_mwh: Optional[float] = None   # default: return to soc_init
    enforce_exclusivity: bool = True           # binaries forbidding simultaneous ch/dis

    def __post_init__(self):
        _require(self.p_max_mw > 0 and self.e_max_mwh > 0, "BESS: capacities must be positive")
        _require(0 <= self.soc_min_mwh <= self.soc_max_mwh <= self.e_max_mwh + _TOL,
                 "BESS: need 0 <= soc_min <= soc_max <= e_max")
        _require(self.soc_min_mwh - _TOL <= self.soc_init_mwh <= self.soc_max_mwh + _TOL,
                 "BESS: soc_init outside [soc_min, soc_max]")
        _require(0 < self.eta_ch <= 1 and 0 < self.eta_dis <= 1, "BESS: efficiencies must be in (0,1]")
        _require(self.throughput_cost_eur_mwh >= 0, "BESS: throughput cost must be >= 0")
        if self.terminal_soc_mwh is not None:
            _require(self.soc_min_mwh - _TOL <= self.terminal_soc_mwh <= self.soc_max_mwh + _TOL,
                     "BESS: terminal SoC outside [soc_min, soc_max]")

    @property
    def terminal_target(self) -> float:
        return self.soc_init_mwh if self.terminal_soc_mwh is None else self.terminal_soc_mwh

    @classmethod
    def medium_scale(cls, **kw) -> "BESSParams":
        base = dict(p_max_mw=2.0, e_max_mwh=4.0, soc_min_mwh=0.4, soc_max_mwh=3.6, soc_init_mwh=0.63)
        base.update(kw)
        return cls(**base)

    @classmethod
    def large_scale(cls, **kw) -> "BESSParams":
        return cls(**kw)


@dataclass(frozen=True)
class BatchJob:
    """
    One non-interruptible batch task. Jobs on the same machine are executed in
    ascending `sequence`, separated by the factory buffer.
    """
    job_id: str
    machine: str
    sequence: int
    duration_h: float
    power_mw: float
    units_out: float = 100.0
    earliest_start_h: float = 0.0
    latest_finish_h: Optional[float] = None   # default: end of horizon

    def __post_init__(self):
        _require(self.duration_h > 0 and self.power_mw >= 0 and self.units_out >= 0,
                 f"job {self.job_id}: duration/power/units must be positive")
        _require(self.earliest_start_h >= 0, f"job {self.job_id}: earliest_start_h < 0")


@dataclass
class FactoryParams:
    """
    Industrial site data.
    base_load_mw:   scalar or length-T array of non-shiftable load.
    demand_units:   scalar or length-T array of product deliveries per step.
    intraday_shift_window_h: max deviation of a batch start from the DA plan in
                    the intraday stage. 0 disables intraday load shifting.
    """
    jobs: List[BatchJob] = field(default_factory=list)
    base_load_mw: object = 3.5
    buffer_h: float = 1.0
    inventory_init: float = 0.0
    inventory_max: float = 1000.0
    demand_units: object = 0.0
    intraday_shift_window_h: float = 2.0
    unmet_penalty_eur_per_unit: float = 5000.0

    def __post_init__(self):
        ids = [j.job_id for j in self.jobs]
        _require(len(ids) == len(set(ids)), "duplicate job_id")
        keys = [(j.machine, j.sequence) for j in self.jobs]
        _require(len(keys) == len(set(keys)), "duplicate (machine, sequence)")
        _require(self.buffer_h >= 0 and self.intraday_shift_window_h >= 0, "buffer/window must be >= 0")
        _require(0 <= self.inventory_init <= self.inventory_max, "inventory_init outside [0, max]")
        _require(self.unmet_penalty_eur_per_unit >= 0, "unmet penalty must be >= 0")

    def base_series(self, n: int) -> np.ndarray:
        s = _series(self.base_load_mw, n, "base_load_mw")
        _require(bool(np.all(s >= 0)), "base_load_mw must be >= 0", DataValidationError)
        return s

    def demand_series(self, n: int) -> np.ndarray:
        s = _series(self.demand_units, n, "demand_units")
        _require(bool(np.all(s >= 0)), "demand_units must be >= 0", DataValidationError)
        return s


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
@dataclass
class RTBranch:
    """
    Real-time realisation, conditional on an ID scenario.
    prob:            conditional probability.
    load_dev:        fractional deviation of base load vs forecast, length T (0.02 = +2 %).
    r_minus, r_plus: balancing price ratios (shortfall >= 1, surplus <= 1), scalar or length T.
    """
    prob: float
    load_dev: object = 0.0
    r_minus: object = 1.2
    r_plus: object = 0.8


@dataclass
class IDScenario:
    """
    One realisation of the market prices (EUR/MWh, length T each).
    da_buy/da_sell: DA clearing price for buying/selling. da_buy is also the
                    reference for imbalance settlement.
    id_buy/id_sell: intraday prices.
    """
    name: str
    prob: float
    da_buy: object
    da_sell: object
    id_buy: object
    id_sell: object
    rt_branches: List[RTBranch] = field(default_factory=list)


@dataclass
class ScenarioTree:
    scenarios: List[IDScenario]

    def validated(self, n: int) -> "ScenarioTree":
        """Return a copy with all series converted to arrays; raise on inconsistencies."""
        _require(len(self.scenarios) > 0, "scenario tree is empty", DataValidationError)
        out = []
        ptot = 0.0
        for k, s in enumerate(self.scenarios):
            tag = f"scenario[{k}] '{s.name}'"
            _require(0 < s.prob <= 1, f"{tag}: prob must be in (0,1]", DataValidationError)
            ptot += s.prob
            arrs = {a: _series(getattr(s, a), n, f"{tag}.{a}")
                    for a in ("da_buy", "da_sell", "id_buy", "id_sell")}
            _require(bool(np.all(arrs["da_sell"] <= arrs["da_buy"] + _TOL)),
                     f"{tag}: da_sell must be <= da_buy (else simultaneous buy/sell is an arbitrage)",
                     DataValidationError)
            _require(bool(np.all(arrs["id_sell"] <= arrs["id_buy"] + _TOL)),
                     f"{tag}: id_sell must be <= id_buy", DataValidationError)
            _require(len(s.rt_branches) > 0, f"{tag}: needs at least one RT branch", DataValidationError)
            rts, pr = [], 0.0
            for r, b in enumerate(s.rt_branches):
                bt = f"{tag}.rt[{r}]"
                _require(0 < b.prob <= 1, f"{bt}: prob must be in (0,1]", DataValidationError)
                pr += b.prob
                dev = _series(b.load_dev, n, f"{bt}.load_dev")
                rm = _series(b.r_minus, n, f"{bt}.r_minus")
                rp = _series(b.r_plus, n, f"{bt}.r_plus")
                _require(bool(np.all(dev > -1.0)), f"{bt}: load_dev must be > -1", DataValidationError)
                _require(bool(np.all(rm >= rp - _TOL)),
                         f"{bt}: r_minus must be >= r_plus (else imbalance arbitrage)", DataValidationError)
                _require(bool(np.all(rp >= 0)), f"{bt}: r_plus must be >= 0", DataValidationError)
                rts.append(RTBranch(b.prob, dev, rm, rp))
            _require(abs(pr - 1.0) < 1e-6, f"{tag}: RT probabilities sum to {pr:.6f}", DataValidationError)
            out.append(IDScenario(s.name, s.prob, rt_branches=rts, **arrs))
        _require(abs(ptot - 1.0) < 1e-6, f"scenario probabilities sum to {ptot:.6f}", DataValidationError)
        return ScenarioTree(out)


# --------------------------------------------------------------------------- #
# State, plan and result containers
# --------------------------------------------------------------------------- #
@dataclass
class InitialState:
    """System state just before the first optimised step."""
    soc_mwh: float
    mt_online: bool = False
    mt_output_mw: float = 0.0
    mt_hours_in_state: float = 1e6     # hours since last status change
    inventory_units: float = 0.0

    @classmethod
    def default(cls, bess: BESSParams, factory: FactoryParams) -> "InitialState":
        return cls(soc_mwh=bess.soc_init_mwh, inventory_units=factory.inventory_init)

    @classmethod
    def from_result(cls, res: "StageResult", t0: int, time_grid: TimeGrid) -> "InitialState":
        """Expected state at the end of step t0-1 taken from a solved stage (for simulation/demo)."""
        _require(1 <= t0 < time_grid.n_steps, "t0 must be in [1, T-1]", DataValidationError)
        row = res.schedule.iloc[t0 - 1]
        u = res.plan.mt_u
        hrs, k = 0.0, t0 - 1
        while k >= 0 and u[k] == u[t0 - 1]:
            hrs += time_grid.dt_h
            k -= 1
        return cls(soc_mwh=float(row["exp_soc_mwh"]), mt_online=bool(u[t0 - 1]),
                   mt_output_mw=float(row["exp_mt_mw"]), mt_hours_in_state=hrs,
                   inventory_units=float(row["exp_inventory"]))


@dataclass
class DAPlan:
    """Frozen first-stage decisions."""
    p_da_buy: np.ndarray
    p_da_sell: np.ndarray
    mt_u: np.ndarray
    mt_x: np.ndarray
    mt_y: np.ndarray
    job_start_step: Dict[str, int]


@dataclass
class IntradayInputs:
    """
    t0:            first step to re-optimise (steps < t0 are history).
    state:         measured state at the end of step t0-1.
    scenario:      updated prices and RT branches (prob must be 1.0); only steps >= t0 are used.
    base_load_mw:  optional refreshed base-load forecast (scalar or length T).
    """
    t0: int
    state: InitialState
    scenario: IDScenario
    base_load_mw: Optional[object] = None


@dataclass
class StageResult:
    stage: str
    t0: int
    status: str
    objective_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    schedule: pd.DataFrame
    batch_plan: pd.DataFrame
    scenario_detail: pd.DataFrame
    cost_breakdown: Dict[str, float]
    scenario_costs: pd.DataFrame
    risk: Dict[str, float]
    plan: DAPlan

    def summary(self) -> str:
        cb = "  ".join(f"{k}={v:,.1f}" for k, v in self.cost_breakdown.items())
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        return (f"[{self.stage}] status={self.status} expected_cost={self.objective_eur:,.2f} EUR "
                f"gap={gap} time={self.solve_time_s:.1f}s\n  {cb}\n  "
                f"risk: " + "  ".join(f"{k}={v:,.1f}" for k, v in self.risk.items()))

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        tag = self.stage.lower()
        self.schedule.to_csv(out / f"{tag}_schedule.csv", index_label="step")
        self.batch_plan.to_csv(out / f"{tag}_batch_plan.csv", index=False)
        self.scenario_detail.to_csv(out / f"{tag}_scenario_detail.csv", index=False)
        self.scenario_costs.to_csv(out / f"{tag}_scenario_costs.csv", index=False)
        meta = dict(stage=self.stage, t0=self.t0, status=self.status, objective_eur=self.objective_eur,
                    mip_gap=self.mip_gap, solve_time_s=self.solve_time_s, model_stats=self.model_stats,
                    cost_breakdown=self.cost_breakdown, risk=self.risk, version=__version__)
        (out / f"{tag}_summary.json").write_text(json.dumps(meta, indent=2))
        return out


@dataclass
class SolverSettings:
    name: str = "appsi_highs"
    mip_gap: float = 1e-4          # 0.01 %
    time_limit_s: float = 300.0
    threads: Optional[int] = None
    verbose: bool = False
    extra_options: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        _require(0 <= self.mip_gap < 1, "mip_gap must be in [0,1)")
        _require(self.time_limit_s > 0, "time_limit_s must be positive")


@dataclass
class _Ctx:
    """Bookkeeping produced while building a model, consumed when extracting results."""
    t0: int
    T_opt: List[int]
    segs: List[Tuple[float, float]]
    job_dur: Dict[str, int]
    uses_z2: Dict[str, bool]
    plan_start: Dict[str, int]
    allowed: Dict[str, List[int]]


# --------------------------------------------------------------------------- #
# The optimizer
# --------------------------------------------------------------------------- #
class MultiStageProsumerOptimizer:
    def __init__(self, grid: GridParams, factory: FactoryParams, mt: MicroturbineParams,
                 bess: BESSParams, time_grid: TimeGrid = TimeGrid(),
                 solver: SolverSettings = SolverSettings()):
        self.grid, self.factory, self.mt, self.bess = grid, factory, mt, bess
        self.time, self.solver = time_grid, solver
        self._check_static_feasibility()

    # ------------------------------------------------------------------ checks
    def _check_static_feasibility(self) -> None:
        tg, T = self.time, self.time.n_steps
        self.factory.base_series(T)
        self.factory.demand_series(T)
        by_machine: Dict[str, float] = {}
        for j in self.factory.jobs:
            d = tg.steps(j.duration_h)
            e = tg.steps(j.earliest_start_h)
            lf = T if j.latest_finish_h is None else min(T, int(math.floor(j.latest_finish_h / tg.dt_h + 1e-9)))
            _require(lf - d >= e, f"job {j.job_id}: cannot fit inside its time window")
            by_machine[j.machine] = by_machine.get(j.machine, 0.0) + d + tg.steps(self.factory.buffer_h)
        for m_id, occ in by_machine.items():
            _require(occ - tg.steps(self.factory.buffer_h) <= T,
                     f"machine {m_id}: sequence needs {occ} steps, horizon has {T}")

    def _validate_state(self, st: InitialState) -> None:
        b = self.bess
        _require(b.soc_min_mwh - 1e-6 <= st.soc_mwh <= b.soc_max_mwh + 1e-6,
                 f"initial SoC {st.soc_mwh:.3f} outside [{b.soc_min_mwh}, {b.soc_max_mwh}]", DataValidationError)
        _require(0 <= st.mt_output_mw <= self.mt.p_max_mw + 1e-6, "initial MT output out of range", DataValidationError)
        _require(bool(st.mt_online) or st.mt_output_mw < 1e-6, "MT offline but output > 0", DataValidationError)
        _require(0 <= st.inventory_units <= self.factory.inventory_max + 1e-6,
                 "initial inventory outside [0, inventory_max]", DataValidationError)

    # ------------------------------------------------------------- public API
    def solve_day_ahead(self, tree: ScenarioTree, state: Optional[InitialState] = None) -> StageResult:
        """Stage 1 + 2 + 3 stochastic problem. DA variables are scenario independent."""
        T = self.time.n_steps
        tree = tree.validated(T)
        state = state or InitialState.default(self.bess, self.factory)
        self._validate_state(state)
        log.info("Building DA model: %d ID scenarios, %d steps", len(tree.scenarios), T)
        model, ctx = self._build_model(tree, 0, state, None, self.factory)
        info = self._solve(model, "DA")
        return self._extract(model, ctx, tree, "DA", info, None, self.factory)

    def solve_intraday(self, da: StageResult, inp: IntradayInputs) -> StageResult:
        """Re-optimise steps >= t0 with the DA commitments frozen."""
        T = self.time.n_steps
        _require(da.stage == "DA", "solve_intraday needs a day-ahead StageResult", DataValidationError)
        _require(0 <= inp.t0 < T, f"t0 must be in [0, {T - 1}]", DataValidationError)
        self._validate_state(inp.state)
        _require(abs(inp.scenario.prob - 1.0) < 1e-9, "intraday scenario prob must be 1.0", DataValidationError)
        tree = ScenarioTree([inp.scenario]).validated(T)
        factory = self.factory
        if inp.base_load_mw is not None:
            factory = replace(factory, base_load_mw=_series(inp.base_load_mw, T, "intraday base_load_mw"))
        log.info("Building ID model from step %d (hour %.2f)", inp.t0, self.time.hour(inp.t0))
        model, ctx = self._build_model(tree, inp.t0, inp.state, da.plan, factory)
        info = self._solve(model, "ID")
        return self._extract(model, ctx, tree, "ID", info, da.plan, factory)

    # ------------------------------------------------------------ model build
    def _build_model(self, tree: ScenarioTree, t0: int, st: InitialState,
                     plan: Optional[DAPlan], fa: FactoryParams):
        tg, mt, bs, gr = self.time, self.mt, self.bess, self.grid
        T, dt = tg.n_steps, tg.dt_h
        nI = len(tree.scenarios)
        T_opt = list(range(t0, T))
        base = fa.base_series(T)
        demand = fa.demand_series(T)
        segs = mt.effective_segments()
        noload = mt.no_load_cost_eur_per_h()
        ru, rd = mt.ramp_up_mw_h * dt, mt.ramp_down_mw_h * dt
        sru, srd = mt.startup_ramp_mw_h * dt, mt.shutdown_ramp_mw_h * dt

        m = pyo.ConcreteModel("ProsumerMultiStage")
        m.TO = pyo.Set(initialize=T_opt, ordered=True)
        m.I = pyo.Set(initialize=list(range(nI)), ordered=True)
        m.TI = pyo.Set(dimen=2, initialize=[(t, i) for t in T_opt for i in range(nI)], ordered=True)
        tir = [(t, i, r) for i in range(nI) for r in range(len(tree.scenarios[i].rt_branches)) for t in T_opt]
        m.TIR = pyo.Set(dimen=3, initialize=tir, ordered=True)
        m.K = pyo.RangeSet(0, len(segs) - 1)

        # ---- Stage 1 variables (scenario independent) -----------------------
        m.p_da_buy = pyo.Var(m.TO, bounds=(0, gr.import_limit_mw))
        m.p_da_sell = pyo.Var(m.TO, bounds=(0, gr.export_limit_mw))
        m.u = pyo.Var(m.TO, domain=pyo.Binary)
        m.x = pyo.Var(m.TO, domain=pyo.Binary)
        m.y = pyo.Var(m.TO, domain=pyo.Binary)

        # ---- Stage 2 variables (per ID scenario) ----------------------------
        m.id_buy = pyo.Var(m.TI, bounds=(0, gr.import_limit_mw))
        m.id_sell = pyo.Var(m.TI, bounds=(0, gr.export_limit_mw))
        m.seg = pyo.Var(m.K, m.TI, bounds=lambda mm, k, t, i: (0, segs[k][0]))
        m.ch = pyo.Var(m.TI, bounds=(0, bs.p_max_mw))
        m.dis = pyo.Var(m.TI, bounds=(0, bs.p_max_mw))
        m.soc = pyo.Var(m.TI, bounds=(bs.soc_min_mwh, bs.soc_max_mwh))
        if bs.enforce_exclusivity:
            m.v_ch = pyo.Var(m.TI, domain=pyo.Binary)
            m.v_dis = pyo.Var(m.TI, domain=pyo.Binary)
        m.inv = pyo.Var(m.TI, bounds=(0, fa.inventory_max))
        m.unmet = pyo.Var(m.TI, domain=pyo.NonNegativeReals)

        # ---- Stage 3 variables (per ID scenario and RT branch) --------------
        m.dpos = pyo.Var(m.TIR, domain=pyo.NonNegativeReals)
        m.dneg = pyo.Var(m.TIR, domain=pyo.NonNegativeReals)

        # ---- Microturbine ---------------------------------------------------
        m.p_mt = pyo.Expression(m.TI, rule=lambda mm, t, i: mt.p_min_mw * mm.u[t] + sum(mm.seg[k, t, i] for k in mm.K))
        m.c_seg = pyo.Constraint(m.K, m.TI, rule=lambda mm, k, t, i: mm.seg[k, t, i] <= segs[k][0] * mm.u[t])

        def p_prev(mm, t, i):
            return mm.p_mt[t - 1, i] if t > t0 else st.mt_output_mw

        def u_prev(mm, t):
            return mm.u[t - 1] if t > t0 else float(st.mt_online)

        m.c_ru = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.p_mt[t, i] - p_prev(mm, t, i) <= ru * u_prev(mm, t) + sru * mm.x[t])
        m.c_rd = pyo.Constraint(m.TI, rule=lambda mm, t, i: p_prev(mm, t, i) - mm.p_mt[t, i] <= rd * mm.u[t] + srd * mm.y[t])

        if plan is None:   # commitment logic is only a decision in the DA stage
            n_up, n_dn = tg.steps(mt.min_up_h), tg.steps(mt.min_down_h)
            m.c_uc = pyo.Constraint(m.TO, rule=lambda mm, t: mm.x[t] - mm.y[t] == mm.u[t] - u_prev(mm, t))
            m.c_xy = pyo.Constraint(m.TO, rule=lambda mm, t: mm.x[t] + mm.y[t] <= 1)
            m.c_mut = pyo.Constraint(m.TO, rule=lambda mm, t: sum(mm.x[k] for k in range(max(t0, t - n_up + 1), t + 1)) <= mm.u[t])
            m.c_mdt = pyo.Constraint(m.TO, rule=lambda mm, t: sum(mm.y[k] for k in range(max(t0, t - n_dn + 1), t + 1)) <= 1 - mm.u[t])
            if st.mt_online and st.mt_hours_in_state < mt.min_up_h - _TOL:
                for t in T_opt[:tg.steps(mt.min_up_h - st.mt_hours_in_state)]:
                    m.u[t].fix(1)
            if (not st.mt_online) and st.mt_hours_in_state < mt.min_down_h - _TOL:
                for t in T_opt[:tg.steps(mt.min_down_h - st.mt_hours_in_state)]:
                    m.u[t].fix(0)

        # ---- BESS -----------------------------------------------------------
        def _soc(mm, t, i):
            prev = mm.soc[t - 1, i] if t > t0 else st.soc_mwh
            return mm.soc[t, i] == prev + bs.eta_ch * mm.ch[t, i] * dt - mm.dis[t, i] * dt / bs.eta_dis
        m.c_soc = pyo.Constraint(m.TI, rule=_soc)
        if bs.enforce_exclusivity:
            m.c_ch = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.ch[t, i] <= bs.p_max_mw * mm.v_ch[t, i])
            m.c_dis = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.dis[t, i] <= bs.p_max_mw * mm.v_dis[t, i])
            m.c_excl = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.v_ch[t, i] + mm.v_dis[t, i] <= 1)
        reach = st.soc_mwh + bs.eta_ch * bs.p_max_mw * dt * len(T_opt)
        term = min(bs.terminal_target, bs.soc_max_mwh, reach)
        m.c_term = pyo.Constraint(m.I, rule=lambda mm, i: mm.soc[T - 1, i] >= term)

        # ---- Batch scheduling (time-indexed start variables) ----------------
        jobs = fa.jobs
        buf = tg.steps(fa.buffer_h)
        W = tg.steps(fa.intraday_shift_window_h)
        dur = {j.job_id: tg.steps(j.duration_h) for j in jobs}
        allowed: Dict[str, List[int]] = {}
        for j in jobs:
            e = tg.steps(j.earliest_start_h)
            lf = T if j.latest_finish_h is None else min(T, int(math.floor(j.latest_finish_h / dt + 1e-9)))
            allowed[j.job_id] = list(range(e, lf - dur[j.job_id] + 1))
        plan_start: Dict[str, int] = {}
        if plan is not None:
            for j in jobs:
                _require(j.job_id in plan.job_start_step, f"DA plan misses job {j.job_id}", DataValidationError)
                plan_start[j.job_id] = int(plan.job_start_step[j.job_id])
        started = {j.job_id: (plan is not None and plan_start[j.job_id] < t0) for j in jobs}
        uses_z2 = {j.job_id: (W > 0 and not started[j.job_id]) for j in jobs}
        z2_dom: Dict[str, List[int]] = {}
        for j in jobs:
            jid = j.job_id
            if not uses_z2[jid]:
                continue
            if plan is None:
                z2_dom[jid] = allowed[jid]
            else:
                z2_dom[jid] = [k for k in allowed[jid] if k >= t0 and abs(k - plan_start[jid]) <= W]

        m.JT = pyo.Set(dimen=2, initialize=[(jid, k) for jid, ks in allowed.items() for k in ks], ordered=True)
        m.z = pyo.Var(m.JT, domain=pyo.Binary)
        m.JTI = pyo.Set(dimen=3, initialize=[(jid, k, i) for jid, ks in z2_dom.items() for k in ks for i in range(nI)], ordered=True)
        m.z2 = pyo.Var(m.JTI, domain=pyo.Binary)
        m.JOBS = pyo.Set(initialize=[j.job_id for j in jobs], ordered=True)
        m.JI = pyo.Set(dimen=2, initialize=[(jid, i) for jid in z2_dom for i in range(nI)], ordered=True)

        def ind(jid, k, i):
            if uses_z2[jid]:
                return m.z2[jid, k, i] if k in z2_dom[jid] else None
            return m.z[jid, k] if k in allowed[jid] else None

        def start_expr(jid, i):
            ks = z2_dom[jid] if uses_z2[jid] else allowed[jid]
            return sum(k * ind(jid, k, i) for k in ks)

        m.c_once = pyo.Constraint(m.JOBS, rule=lambda mm, jid: sum(mm.z[jid, k] for k in allowed[jid]) == 1)
        m.c_once2 = pyo.Constraint(m.JI, rule=lambda mm, jid, i: sum(mm.z2[jid, k, i] for k in z2_dom[jid]) == 1)
        m.c_shift = pyo.Constraint(
            m.JI, rule=lambda mm, jid, i: pyo.inequality(
                -W, sum(k * mm.z2[jid, k, i] for k in z2_dom[jid]) - sum(k * mm.z[jid, k] for k in allowed[jid]), W))

        m.c_prec = pyo.ConstraintList()
        machines: Dict[str, List[BatchJob]] = {}
        for j in jobs:
            machines.setdefault(j.machine, []).append(j)
        for lst in machines.values():
            lst.sort(key=lambda j: j.sequence)
            for a, b in zip(lst, lst[1:]):
                need = dur[a.job_id] + buf
                if plan is None:   # DA master schedule must itself be feasible
                    zs = lambda jid: sum(k * m.z[jid, k] for k in allowed[jid])
                    m.c_prec.add(zs(b.job_id) >= zs(a.job_id) + need)
                if not (uses_z2[a.job_id] or uses_z2[b.job_id]):
                    continue
                for i in range(nI):
                    m.c_prec.add(start_expr(b.job_id, i) >= start_expr(a.job_id, i) + need)

        def _load(mm, t, i):
            terms = []
            for j in jobs:
                for k in range(max(0, t - dur[j.job_id] + 1), t + 1):
                    v = ind(j.job_id, k, i)
                    if v is not None:
                        terms.append(j.power_mw * v)
            return pyo.quicksum(terms)

        def _prod(mm, t, i):
            terms = []
            for j in jobs:
                v = ind(j.job_id, t - dur[j.job_id] + 1, i)
                if v is not None:
                    terms.append(j.units_out * v)
            return pyo.quicksum(terms)
        m.batch_load = pyo.Expression(m.TI, rule=_load)
        m.prod = pyo.Expression(m.TI, rule=_prod)

        def _inv(mm, t, i):
            prev = mm.inv[t - 1, i] if t > t0 else st.inventory_units
            return mm.inv[t, i] == prev + mm.prod[t, i] - demand[t] + mm.unmet[t, i]
        m.c_inv = pyo.Constraint(m.TI, rule=_inv)

        # ---- Power balance, grid limits ------------------------------------
        def supply(mm, t, i):
            return (mm.p_da_buy[t] - mm.p_da_sell[t] + mm.id_buy[t, i] - mm.id_sell[t, i]
                    + mm.p_mt[t, i] + mm.dis[t, i] - mm.ch[t, i])

        def dev(t, i, r):
            return tree.scenarios[i].rt_branches[r].load_dev[t]

        m.c_bal = pyo.Constraint(
            m.TIR, rule=lambda mm, t, i, r: supply(mm, t, i) - base[t] * (1 + dev(t, i, r)) - mm.batch_load[t, i]
            == mm.dpos[t, i, r] - mm.dneg[t, i, r])

        def phys(mm, t, i, r):   # physical import, positive = from grid
            return base[t] * (1 + dev(t, i, r)) + mm.batch_load[t, i] + mm.ch[t, i] - mm.dis[t, i] - mm.p_mt[t, i]
        m.c_imp = pyo.Constraint(m.TIR, rule=lambda mm, t, i, r: phys(mm, t, i, r) <= gr.import_limit_mw)
        m.c_exp = pyo.Constraint(m.TIR, rule=lambda mm, t, i, r: phys(mm, t, i, r) >= -gr.export_limit_mw)
        net = lambda mm, t, i: mm.p_da_buy[t] - mm.p_da_sell[t] + mm.id_buy[t, i] - mm.id_sell[t, i]
        m.c_pos_hi = pyo.Constraint(m.TI, rule=lambda mm, t, i: net(mm, t, i) <= gr.import_limit_mw)
        m.c_pos_lo = pyo.Constraint(m.TI, rule=lambda mm, t, i: net(mm, t, i) >= -gr.export_limit_mw)

        # ---- Cost components per ID scenario --------------------------------
        sc = tree.scenarios
        m.cost_da = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
            sc[i].da_buy[t] * mm.p_da_buy[t] - sc[i].da_sell[t] * mm.p_da_sell[t] for t in T_opt))
        m.cost_id = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
            sc[i].id_buy[t] * mm.id_buy[t, i] - sc[i].id_sell[t] * mm.id_sell[t, i] for t in T_opt))

        def _bal_cost(mm, i):
            tot = 0
            for r, br in enumerate(sc[i].rt_branches):
                tot += br.prob * dt * sum(sc[i].da_buy[t] * (br.r_minus[t] * mm.dneg[t, i, r] - br.r_plus[t] * mm.dpos[t, i, r])
                                          for t in T_opt)
            return tot
        m.cost_bal = pyo.Expression(m.I, rule=_bal_cost)
        m.cost_mt = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
            sum(segs[k][1] * mm.seg[k, t, i] for k in mm.K) + noload * mm.u[t] for t in T_opt)
            + sum(mt.startup_cost_eur * mm.x[t] + mt.shutdown_cost_eur * mm.y[t] for t in T_opt))
        m.cost_bess = pyo.Expression(m.I, rule=lambda mm, i: dt * bs.throughput_cost_eur_mwh * sum(
            mm.ch[t, i] + mm.dis[t, i] for t in T_opt))
        m.cost_unmet = pyo.Expression(m.I, rule=lambda mm, i: fa.unmet_penalty_eur_per_unit * sum(mm.unmet[t, i] for t in T_opt))
        m.obj = pyo.Objective(expr=sum(
            sc[i].prob * (m.cost_da[i] + m.cost_id[i] + m.cost_bal[i] + m.cost_mt[i] + m.cost_bess[i] + m.cost_unmet[i])
            for i in range(nI)), sense=pyo.minimize)

        # ---- Freeze first-stage decisions in the intraday stage -------------
        if plan is not None:
            for t in T_opt:
                m.p_da_buy[t].fix(float(plan.p_da_buy[t]))
                m.p_da_sell[t].fix(float(plan.p_da_sell[t]))
                m.u[t].fix(int(round(plan.mt_u[t])))
                m.x[t].fix(int(round(plan.mt_x[t])))
                m.y[t].fix(int(round(plan.mt_y[t])))
            for jid, ks in allowed.items():
                for k in ks:
                    m.z[jid, k].fix(1 if k == plan_start[jid] else 0)

        return m, _Ctx(t0, T_opt, segs, dur, uses_z2, plan_start, allowed)

    # ------------------------------------------------------------------ solve
    def _solve(self, model: pyo.ConcreteModel, label: str) -> Dict[str, object]:
        s = self.solver
        try:
            opt = pyo.SolverFactory(s.name)
            ok = opt.available(exception_flag=False)
        except Exception as err:   # noqa: BLE001 - pyomo raises many types here
            raise SolverUnavailableError(f"cannot create solver '{s.name}': {err}") from err
        if not ok:
            raise SolverUnavailableError(
                f"solver '{s.name}' is not available. Try `pip install highspy` (appsi_highs), or set --solver gurobi/cplex/cbc.")

        gap_key = {"gurobi": "MIPGap", "cplex": "mipgap", "cbc": "ratioGap", "glpk": "mipgap"}
        t_key = {"gurobi": "TimeLimit", "cplex": "timelimit", "cbc": "seconds", "glpk": "tmlim"}
        n_bin = sum(1 for v in model.component_data_objects(pyo.Var) if v.is_binary() and not v.fixed)
        n_var = sum(1 for v in model.component_data_objects(pyo.Var) if not v.fixed)
        n_con = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))
        stats = dict(variables=n_var, binaries=n_bin, constraints=n_con)
        log.info("[%s] solving: %d vars (%d binary), %d constraints", label, n_var, n_bin, n_con)

        t_start = time.perf_counter()
        gap, term = None, ""
        try:
            if s.name.startswith("appsi_"):
                opt.config.load_solution = False
                opt.config.stream_solver = s.verbose
                opt.config.time_limit = s.time_limit_s
                opt.config.mip_gap = s.mip_gap
                if s.name == "appsi_highs":
                    if s.threads:
                        opt.highs_options["threads"] = int(s.threads)
                    for k, v in s.extra_options.items():
                        opt.highs_options[k] = v
                res = opt.solve(model)
                term = str(res.termination_condition).lower()
                has_sol = res.best_feasible_objective is not None
                if has_sol:
                    res.solution_loader.load_vars()
                    ub, lb = res.best_feasible_objective, res.best_objective_bound
                    if lb is not None and abs(ub) > 1e-9:
                        gap = max(0.0, abs(ub - lb) / max(abs(ub), 1e-9))
            else:
                opt.options.update(s.extra_options)
                opt.options[gap_key.get(s.name, "mipgap")] = s.mip_gap
                opt.options[t_key.get(s.name, "timelimit")] = s.time_limit_s
                if s.threads and s.name in ("gurobi", "cplex"):
                    opt.options["threads"] = int(s.threads)
                res = opt.solve(model, tee=s.verbose, load_solutions=False)
                term = str(res.solver.termination_condition).lower()
                has_sol = len(res.solution) > 0
                if has_sol:
                    model.solutions.load_from(res)
                    lb = getattr(res.problem, "lower_bound", None)
                    ub = getattr(res.problem, "upper_bound", None)
                    if lb is not None and ub is not None and abs(ub) > 1e-9:
                        gap = max(0.0, abs(ub - lb) / abs(ub))
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
        return dict(status=status, gap=gap, time=elapsed, stats=stats)

    # ---------------------------------------------------------------- extract
    def _extract(self, m, ctx: _Ctx, tree: ScenarioTree, stage: str, info, old_plan: Optional[DAPlan],
                 fa: FactoryParams) -> StageResult:
        tg = self.time
        T, dt, t0 = tg.n_steps, tg.dt_h, ctx.t0
        nI = len(tree.scenarios)
        probs = np.array([s.prob for s in tree.scenarios])
        z = lambda: np.zeros(T)

        # first-stage plan
        if old_plan is None:
            p_buy, p_sell, u, x, y = z(), z(), z(), z(), z()
        else:
            p_buy, p_sell = old_plan.p_da_buy.copy(), old_plan.p_da_sell.copy()
            u, x, y = old_plan.mt_u.copy(), old_plan.mt_x.copy(), old_plan.mt_y.copy()
        for t in ctx.T_opt:
            p_buy[t], p_sell[t] = _val(m.p_da_buy[t]), _val(m.p_da_sell[t])
            u[t], x[t], y[t] = round(_val(m.u[t])), round(_val(m.x[t])), round(_val(m.y[t]))

        jobs = self.factory.jobs
        da_start = {}
        for j in jobs:
            jid = j.job_id
            da_start[jid] = ctx.plan_start[jid] if old_plan is not None else \
                int(round(sum(k * _val(m.z[jid, k]) for k in ctx.allowed[jid])))
        scen_start = {}
        for i in range(nI):
            for j in jobs:
                jid = j.job_id
                if ctx.uses_z2[jid]:
                    scen_start[i, jid] = int(round(sum(k * _val(m.z2[jid, k, i]) for k in range(T) if (jid, k, i) in m.z2)))
                else:
                    scen_start[i, jid] = da_start[jid]

        rows = []
        for i, s in enumerate(tree.scenarios):
            for t in ctx.T_opt:
                dp = sum(br.prob * _val(m.dpos[t, i, r]) for r, br in enumerate(s.rt_branches))
                dn = sum(br.prob * _val(m.dneg[t, i, r]) for r, br in enumerate(s.rt_branches))
                rows.append(dict(
                    scenario=s.name, prob=s.prob, step=t, hour=tg.hour(t),
                    da_buy_mw=p_buy[t], da_sell_mw=p_sell[t],
                    id_buy_mw=_val(m.id_buy[t, i]), id_sell_mw=_val(m.id_sell[t, i]),
                    mt_on=u[t], mt_mw=pyo.value(m.p_mt[t, i]),
                    bess_ch_mw=_val(m.ch[t, i]), bess_dis_mw=_val(m.dis[t, i]), soc_mwh=_val(m.soc[t, i]),
                    batch_load_mw=pyo.value(m.batch_load[t, i]), inventory=_val(m.inv[t, i]),
                    unmet_units=_val(m.unmet[t, i]), imb_surplus_mw=dp, imb_shortfall_mw=dn,
                    da_price=s.da_buy[t], id_price=s.id_buy[t]))
        detail = pd.DataFrame(rows)

        exp = detail.assign(w=detail["prob"]).groupby("step").apply(
            lambda g: pd.Series({
                "hour": g["hour"].iloc[0],
                "da_buy_mw": g["da_buy_mw"].iloc[0], "da_sell_mw": g["da_sell_mw"].iloc[0],
                "da_net_mw": g["da_buy_mw"].iloc[0] - g["da_sell_mw"].iloc[0],
                "mt_on": g["mt_on"].iloc[0],
                "mt_startup": x[int(g.name)], "mt_shutdown": y[int(g.name)],
                "exp_mt_mw": np.average(g["mt_mw"], weights=g["w"]),
                "exp_id_net_mw": np.average(g["id_buy_mw"] - g["id_sell_mw"], weights=g["w"]),
                "exp_bess_ch_mw": np.average(g["bess_ch_mw"], weights=g["w"]),
                "exp_bess_dis_mw": np.average(g["bess_dis_mw"], weights=g["w"]),
                "exp_soc_mwh": np.average(g["soc_mwh"], weights=g["w"]),
                "exp_batch_load_mw": np.average(g["batch_load_mw"], weights=g["w"]),
                "exp_inventory": np.average(g["inventory"], weights=g["w"]),
                "exp_imb_surplus_mw": np.average(g["imb_surplus_mw"], weights=g["w"]),
                "exp_imb_shortfall_mw": np.average(g["imb_shortfall_mw"], weights=g["w"]),
                "exp_da_price": np.average(g["da_price"], weights=g["w"]),
            }), include_groups=False)
        exp.index.name = "step"
        if old_plan is not None and t0 > 0:      # keep full-day index for downstream use
            exp = exp.reindex(range(T))

        plan_rows = []
        for j in jobs:
            d = ctx.job_dur[j.job_id]
            row = dict(job_id=j.job_id, machine=j.machine, sequence=j.sequence, power_mw=j.power_mw,
                       units_out=j.units_out, start_step=da_start[j.job_id],
                       start_hour=tg.hour(da_start[j.job_id]), end_hour=tg.hour(da_start[j.job_id] + d),
                       duration_steps=d)
            for i, s in enumerate(tree.scenarios):
                row[f"start_hour_{s.name}"] = tg.hour(scen_start[i, j.job_id])
            plan_rows.append(row)
        batch_plan = pd.DataFrame(plan_rows)

        comps = dict(DA=m.cost_da, ID=m.cost_id, BAL=m.cost_bal, MT=m.cost_mt, BESS=m.cost_bess, UNMET=m.cost_unmet)
        sc_rows = []
        for i, s in enumerate(tree.scenarios):
            r = {"scenario": s.name, "prob": s.prob}
            r.update({k: pyo.value(e[i]) for k, e in comps.items()})
            r["TOTAL"] = sum(r[k] for k in comps)
            sc_rows.append(r)
        sc_costs = pd.DataFrame(sc_rows)
        breakdown = {k: float(np.dot(probs, sc_costs[k])) for k in comps}
        breakdown["TOTAL"] = float(np.dot(probs, sc_costs["TOTAL"]))
        tot = sc_costs["TOTAL"].to_numpy()
        mean = breakdown["TOTAL"]
        risk = dict(mean=mean, std=float(math.sqrt(np.dot(probs, (tot - mean) ** 2))),
                    worst=float(tot.max()), best=float(tot.min()), cvar95=cvar(tot, probs, 0.95))

        plan = DAPlan(p_buy, p_sell, u, x, y, da_start)
        return StageResult(stage=stage, t0=t0, status=info["status"], objective_eur=float(pyo.value(m.obj)),
                           mip_gap=info["gap"], solve_time_s=info["time"], model_stats=info["stats"],
                           schedule=exp, batch_plan=batch_plan, scenario_detail=detail, cost_breakdown=breakdown,
                           scenario_costs=sc_costs, risk=risk, plan=plan)


# --------------------------------------------------------------------------- #
# Reference data: TOU tariff, demo factory and scenario generator
# --------------------------------------------------------------------------- #
_TOU_TIERS = {   # tier: (buy EUR/MWh, sell EUR/MWh, list of hour-of-day blocks [start, end))
    "off": (66.42, 36.53, [(2, 4), (12, 14), (18, 20), (22, 24)]),
    "flat": (88.56, 48.71, [(4, 6)]),
    "mid": (118.08, 64.94, [(0, 2), (6, 8), (14, 16), (20, 22)]),
    "on": (177.12, 97.42, [(8, 12), (16, 18)]),
}


def tou_prices(tg: TimeGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Reference TOU buy and sell prices (EUR/MWh) on the optimisation grid."""
    buy, sell = np.zeros(tg.n_steps), np.zeros(tg.n_steps)
    for t in range(tg.n_steps):
        h = (t * tg.dt_h) % 24.0
        for b, s, blocks in _TOU_TIERS.values():
            if any(a <= h < e for a, e in blocks):
                buy[t], sell[t] = b, s
    return buy, sell


def build_demo_factory(tg: TimeGrid, seed: int = 7, n_machines: int = 5, jobs_per_machine: int = 2,
                       shift_window_h: float = 2.0) -> FactoryParams:
    rng = np.random.default_rng(seed)
    jobs = []
    for mi in range(n_machines):
        for q in range(jobs_per_machine):
            jobs.append(BatchJob(job_id=f"M{mi + 1}-B{q + 1}", machine=f"M{mi + 1}", sequence=q,
                                 duration_h=float(np.round(rng.uniform(1.5, 4.9), 1)),
                                 power_mw=float(np.round(rng.uniform(60, 110), 1)), units_out=100.0))
    demand = np.zeros(tg.n_steps)
    demand[[t for t in range(tg.n_steps) if 8 <= t * tg.dt_h < 24]] = 25.0 * tg.dt_h
    base = np.array([3.0 + 1.5 * (8 <= t * tg.dt_h < 18) for t in range(tg.n_steps)])
    return FactoryParams(jobs=jobs, base_load_mw=base, buffer_h=1.0, inventory_init=0.0, inventory_max=1000.0,
                         demand_units=demand, intraday_shift_window_h=shift_window_h)


def build_demo_tree(tg: TimeGrid, n_id: int = 5, n_rt: int = 3, seed: int = 11,
                    sell_ratio: float = 0.55) -> ScenarioTree:
    """Synthetic scenarios around the TOU tariff: AR(1) DA shocks, ID basis noise, RT load/price noise."""
    _require(n_id >= 1 and n_rt >= 1, "need at least one ID scenario and one RT branch", DataValidationError)
    rng = np.random.default_rng(seed)
    T = tg.n_steps
    base, _ = tou_prices(tg)
    scen = []
    for i in range(n_id):
        e = np.zeros(T)
        for t in range(T):
            e[t] = (0.7 * e[t - 1] if t else 0.0) + rng.normal(0, 0.10)
        da_buy = base * np.exp(e)
        id_buy = da_buy * np.exp(rng.normal(0, 0.06, T))
        rts = []
        for _ in range(n_rt):
            rts.append(RTBranch(prob=1.0 / n_rt, load_dev=rng.normal(0, 0.03, T),
                                r_minus=1.10 + 0.25 * rng.random(T), r_plus=0.90 - 0.25 * rng.random(T)))
        scen.append(IDScenario(name=f"S{i + 1}", prob=1.0 / n_id, da_buy=da_buy, da_sell=sell_ratio * da_buy,
                               id_buy=id_buy, id_sell=sell_ratio * id_buy, rt_branches=rts))
    return ScenarioTree(scen)


def build_demo_intraday_scenario(tg: TimeGrid, da_tree: ScenarioTree, seed: int = 99,
                                 price_spike_from_step: Optional[int] = None, spike: float = 1.4) -> IDScenario:
    """A single refreshed intraday realisation (optionally with a price spike) with fresh RT branches."""
    rng = np.random.default_rng(seed)
    T = tg.n_steps
    p = np.array([s.prob for s in da_tree.scenarios])
    da = sum(pi * np.asarray(s.da_buy) for pi, s in zip(p, da_tree.scenarios))
    idp = da * np.exp(rng.normal(0, 0.05, T))
    if price_spike_from_step is not None:
        idp[price_spike_from_step:] *= spike
    rts = [RTBranch(prob=0.5, load_dev=rng.normal(0.02, 0.02, T), r_minus=1.2, r_plus=0.8),
           RTBranch(prob=0.5, load_dev=rng.normal(-0.02, 0.02, T), r_minus=1.3, r_plus=0.7)]
    return IDScenario(name="ID-refresh", prob=1.0, da_buy=da, da_sell=0.55 * da, id_buy=idp, id_sell=0.55 * idp,
                      rt_branches=rts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-stage DA + intraday optimisation for an industrial prosumer")
    ap.add_argument("--id-scenarios", type=int, default=5)
    ap.add_argument("--rt-branches", type=int, default=3)
    ap.add_argument("--dt", type=float, default=1.0, help="time step in hours")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shift-window", type=float, default=2.0, help="intraday batch shift window (h), 0 disables")
    ap.add_argument("--intraday-step", type=int, default=10, help="re-optimise from this step (0 = skip)")
    ap.add_argument("--spike", type=float, default=1.4, help="ID price multiplier after re-optimisation step")
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--mip-gap", type=float, default=1e-3)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--bess", choices=["large", "medium"], default="large")
    ap.add_argument("--out", default="results")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        tg = TimeGrid(24.0, args.dt)
        bess = BESSParams.large_scale() if args.bess == "large" else BESSParams.medium_scale()
        opt = MultiStageProsumerOptimizer(
            GridParams(), build_demo_factory(tg, args.seed, shift_window_h=args.shift_window),
            MicroturbineParams(), bess, tg,
            SolverSettings(args.solver, args.mip_gap, args.time_limit, verbose=args.verbose))
        tree = build_demo_tree(tg, args.id_scenarios, args.rt_branches, args.seed + 4)

        da = opt.solve_day_ahead(tree)
        print(da.summary())
        print(da.batch_plan[["job_id", "machine", "start_hour", "end_hour", "power_mw"]].to_string(index=False))
        print(da.schedule[["hour", "da_net_mw", "mt_on", "exp_mt_mw", "exp_bess_ch_mw",
                           "exp_bess_dis_mw", "exp_soc_mwh", "exp_batch_load_mw"]].round(2).to_string())
        da.save(args.out)

        if args.intraday_step > 0:
            t0 = args.intraday_step
            scen = build_demo_intraday_scenario(tg, tree, args.seed + 92, t0, args.spike)
            inp = IntradayInputs(t0, InitialState.from_result(da, t0, tg), scen)
            idr = opt.solve_intraday(da, inp)
            print(idr.summary())
            print(idr.schedule.loc[t0:, ["hour", "da_net_mw", "exp_id_net_mw", "exp_mt_mw", "exp_bess_ch_mw",
                                          "exp_bess_dis_mw", "exp_soc_mwh", "exp_batch_load_mw",
                                          "exp_imb_surplus_mw", "exp_imb_shortfall_mw"]].round(2).to_string())
            idr.save(args.out)
        print(f"\nResults written to {Path(args.out).resolve()}")
        return 0
    except (ConfigurationError, DataValidationError) as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, ModelInfeasibleError, SolveFailedError) as err:
        log.error("Optimisation failed: %s", err)
        return 3
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
