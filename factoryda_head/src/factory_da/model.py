"""Pyomo MILP construction for day-ahead factory scheduling."""
import math
from typing import Dict, List, Tuple

import numpy as np
import pyomo.environ as pyo

from ._internal import EPS
from .candidates import Candidate
from .data import Instance, SchedulerConfig
from .exceptions import InstanceValidationError


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
