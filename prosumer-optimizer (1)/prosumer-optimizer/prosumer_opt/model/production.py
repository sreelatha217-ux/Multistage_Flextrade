"""
Industrial batch production and product inventory  (skill section 2.3).

Time-indexed start formulation: binary z[job, k] = 1 if the job starts at step k.
Durations are rounded up to whole steps, so the load of a job is linear in the binaries.

    first stage   z[job, k]        master production schedule (DA stage, scenario independent)
    second stage  z2[job, k, i]    intraday re-schedule inside +/- shift window (per ID scenario)

    sum_k z = 1                                    every job runs exactly once
    start_b >= start_a + dur_a + buffer            no overlap + setup/cooling buffer per machine
    inv_t = inv_{t-1} + produced_t - demand_t + unmet_t     (unmet is a penalised slack)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import pyomo.environ as pyo

from ..exceptions import DataValidationError
from ..parameters import BatchJob
from ..utils import require
from .context import BuildSpec


@dataclass
class ProductionInfo:
    """Job-level bookkeeping needed to read the schedule back from the solution."""
    job_dur: Dict[str, int]
    allowed: Dict[str, List[int]]
    plan_start: Dict[str, int]
    uses_z2: Dict[str, bool]


def add_production(m: pyo.ConcreteModel, s: BuildSpec) -> ProductionInfo:
    tg, fa, plan, t0, dt, T = s.time, s.factory, s.plan, s.t0, s.dt, s.T
    nI = s.n_scenarios
    jobs = fa.jobs
    buf = tg.steps(fa.buffer_h)
    W = tg.steps(fa.intraday_shift_window_h)

    # ---- feasible start windows ----------------------------------------------------
    dur = {j.job_id: tg.steps(j.duration_h) for j in jobs}
    allowed: Dict[str, List[int]] = {}
    for j in jobs:
        e = tg.steps(j.earliest_start_h)
        lf = T if j.latest_finish_h is None else min(T, int(math.floor(j.latest_finish_h / dt + 1e-9)))
        allowed[j.job_id] = list(range(e, lf - dur[j.job_id] + 1))

    plan_start: Dict[str, int] = {}
    if plan is not None:
        for j in jobs:
            require(j.job_id in plan.job_start_step, f"DA plan misses job {j.job_id}", DataValidationError)
            plan_start[j.job_id] = int(plan.job_start_step[j.job_id])

    # a job already started before t0 cannot be moved; others may shift within +/- W
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

    # ---- variables -----------------------------------------------------------------
    m.JT = pyo.Set(dimen=2, initialize=[(jid, k) for jid, ks in allowed.items() for k in ks], ordered=True)
    m.z = pyo.Var(m.JT, domain=pyo.Binary)
    m.JTI = pyo.Set(dimen=3, initialize=[(jid, k, i) for jid, ks in z2_dom.items() for k in ks
                                         for i in range(nI)], ordered=True)
    m.z2 = pyo.Var(m.JTI, domain=pyo.Binary)
    m.JOBS = pyo.Set(initialize=[j.job_id for j in jobs], ordered=True)
    m.JI = pyo.Set(dimen=2, initialize=[(jid, i) for jid in z2_dom for i in range(nI)], ordered=True)

    def ind(jid, k, i):
        """Start indicator that is effective in scenario i (z2 if the job may shift, else z)."""
        if uses_z2[jid]:
            return m.z2[jid, k, i] if k in z2_dom[jid] else None
        return m.z[jid, k] if k in allowed[jid] else None

    def start_expr(jid, i):
        ks = z2_dom[jid] if uses_z2[jid] else allowed[jid]
        return sum(k * ind(jid, k, i) for k in ks)

    # ---- assignment and shifting ---------------------------------------------------------
    m.c_once = pyo.Constraint(m.JOBS, rule=lambda mm, jid: sum(mm.z[jid, k] for k in allowed[jid]) == 1)
    m.c_once2 = pyo.Constraint(m.JI, rule=lambda mm, jid, i: sum(mm.z2[jid, k, i] for k in z2_dom[jid]) == 1)
    m.c_shift = pyo.Constraint(
        m.JI, rule=lambda mm, jid, i: pyo.inequality(
            -W, sum(k * mm.z2[jid, k, i] for k in z2_dom[jid]) - sum(k * mm.z[jid, k] for k in allowed[jid]), W))

    # ---- precedence with buffer, per machine -------------------------------------------------
    m.c_prec = pyo.ConstraintList()
    machines: Dict[str, List[BatchJob]] = {}
    for j in jobs:
        machines.setdefault(j.machine, []).append(j)
    for lst in machines.values():
        lst.sort(key=lambda j: j.sequence)
        for a, b in zip(lst, lst[1:]):
            need = dur[a.job_id] + buf
            if plan is None:   # the DA master schedule must itself be feasible
                zs = lambda jid: sum(k * m.z[jid, k] for k in allowed[jid])
                m.c_prec.add(zs(b.job_id) >= zs(a.job_id) + need)
            if not (uses_z2[a.job_id] or uses_z2[b.job_id]):
                continue
            for i in range(nI):
                m.c_prec.add(start_expr(b.job_id, i) >= start_expr(a.job_id, i) + need)

    # ---- load and output expressions ---------------------------------------------------------
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

    # ---- inventory with soft delivery shortfall ------------------------------------------------
    m.inv = pyo.Var(m.TI, bounds=(0, fa.inventory_max))
    m.unmet = pyo.Var(m.TI, domain=pyo.NonNegativeReals)

    def _inv(mm, t, i):
        prev = mm.inv[t - 1, i] if t > t0 else s.state.inventory_units
        return mm.inv[t, i] == prev + mm.prod[t, i] - s.demand[t] + mm.unmet[t, i]
    m.c_inv = pyo.Constraint(m.TI, rule=_inv)

    m.cost_unmet = pyo.Expression(
        m.I, rule=lambda mm, i: fa.unmet_penalty_eur_per_unit * sum(mm.unmet[t, i] for t in s.t_opt))

    return ProductionInfo(dur, allowed, plan_start, uses_z2)
