from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pyomo.environ as pyo

from ._internal import DT_H, EPS
from .candidates import Candidate
from .data import Instance, SchedulerConfig


def add_batch_block(m: pyo.ConcreteModel, inst: Instance, cfg: SchedulerConfig,
                    cands: List[Candidate]) -> List[List[Tuple[int, float, float]]]:
    """Factory block shared by the day-ahead and intraday models (requires m.H).

    Creates the batch start binaries m.s[c], the inventory m.N[t] and the constraints c_inv, c_terminal,
    c_unique, c_cap and c_seq.  Returns by_hour[t] = [(candidate, MW, units)] so the caller can build the
    power balance from the same occupancy data."""
    T, M, P = inst.horizon_h, len(inst.machines), len(inst.tasks)
    step = cfg.start_step_h
    n_grid = int(round(T / step))
    m.C = pyo.RangeSet(0, len(cands) - 1)
    m.s = pyo.Var(m.C, domain=pyo.Binary)                                   # batch start decision
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
    return by_hour


def add_mt_commitment(m: pyo.ConcreteModel, inst: Instance) -> None:
    """MT unit-commitment block (binaries u, x, y; logic; MUT / MDT).  Requires m.H.  The pre-horizon state
    comes from Microturbine.initial_on; MUT / MDT windows are truncated at t=0."""
    mt = inst.mt
    m.u = pyo.Var(m.H, domain=pyo.Binary)                                   # online
    m.x = pyo.Var(m.H, domain=pyo.Binary)                                   # start-up
    m.y = pyo.Var(m.H, domain=pyo.Binary)                                   # shut-down
    u0 = 1.0 if mt.initial_on else 0.0
    up = lambda mm, t: mm.u[t - 1] if t > 0 else u0
    m.c_uc = pyo.Constraint(m.H, rule=lambda mm, t: mm.x[t] - mm.y[t] == mm.u[t] - up(mm, t))
    m.c_xy = pyo.Constraint(m.H, rule=lambda mm, t: mm.x[t] + mm.y[t] <= 1)
    m.c_mut = pyo.Constraint(m.H, rule=lambda mm, t: pyo.quicksum(
        mm.x[tau] for tau in range(max(0, t - mt.min_up_h + 1), t + 1)) <= mm.u[t])
    m.c_mdt = pyo.Constraint(m.H, rule=lambda mm, t: pyo.quicksum(
        mm.y[tau] for tau in range(max(0, t - mt.min_down_h + 1), t + 1)) <= 1 - mm.u[t])


def build_model(inst: Instance, cfg: SchedulerConfig, cands: List[Candidate]) -> pyo.ConcreteModel:
    T = inst.horizon_h
    mt = inst.mt
    m = pyo.ConcreteModel("FactoryMicroturbineDayAhead")
    m.H = pyo.RangeSet(0, T - 1)
    m.Pbuy = pyo.Var(m.H, bounds=(0, inst.grid_limit_mw))                   # DA purchase (MW)
    m.Psell = pyo.Var(m.H, bounds=(0, inst.grid_sell_limit_mw))             # DA sale (MW)
    by_hour = add_batch_block(m, inst, cfg, cands)                          # s, N, inventory, sequencing

    # ---------------- microturbine variables ----------------
    if mt is not None:
        nb = len(mt.block_width_mw)
        m.B = pyo.RangeSet(0, nb - 1)
        m.Pmt = pyo.Var(m.H, bounds=(0, mt.p_max_mw))                       # total MT output (MW)
        m.Pb = pyo.Var(m.B, m.H, bounds=lambda mm, b, t: (0, mt.block_width_mw[b]))
        add_mt_commitment(m, inst)                                          # u, x, y + logic + MUT/MDT
        u0 = 1.0 if mt.initial_on else 0.0
        p0 = float(mt.initial_power_mw) if mt.initial_on else 0.0
        up = lambda mm, t: mm.u[t - 1] if t > 0 else u0
        pp = lambda mm, t: mm.Pmt[t - 1] if t > 0 else p0

        # piecewise generation:  P = Pmin*u + sum_b P_b ;  P_b <= W_b*u   (sum_b W_b = Pmax - Pmin => P <= Pmax*u)
        m.c_mt_sum = pyo.Constraint(m.H, rule=lambda mm, t: mm.Pmt[t] == mt.p_min_mw * mm.u[t]
                                    + pyo.quicksum(mm.Pb[b, t] for b in mm.B))
        m.c_mt_blk = pyo.Constraint(m.B, m.H, rule=lambda mm, b, t: mm.Pb[b, t] <= mt.block_width_mw[b] * mm.u[t])
        # ramping
        m.c_ru = pyo.Constraint(m.H, rule=lambda mm, t: mm.Pmt[t] - pp(mm, t)
                                <= mt.ramp_up_mw_h * up(mm, t) + mt.startup_ramp_mw_h * mm.x[t])
        m.c_rd = pyo.Constraint(m.H, rule=lambda mm, t: pp(mm, t) - mm.Pmt[t]
                                <= mt.ramp_down_mw_h * mm.u[t] + mt.shutdown_ramp_mw_h * mm.y[t])

    # ---------------- BESS ----------------
    bess = inst.bess
    if bess is not None:
        m.Pch = pyo.Var(m.H, bounds=(0, bess.p_max_mw))                     # charging power (MW, AC side)
        m.Pdis = pyo.Var(m.H, bounds=(0, bess.p_max_mw))                    # discharging power (MW, AC side)
        m.SoC = pyo.Var(m.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))   # end-of-hour stored energy (MWh)
        soc_prev = lambda mm, t: mm.SoC[t - 1] if t > 0 else bess.soc_init_mwh
        m.c_soc = pyo.Constraint(m.H, rule=lambda mm, t: mm.SoC[t] == soc_prev(mm, t)
                                 + bess.eta_ch * mm.Pch[t] * DT_H - mm.Pdis[t] * DT_H / bess.eta_dis)
        m.c_soc_term = pyo.Constraint(expr=m.SoC[T - 1] >= bess.soc_init_mwh)
        if bess.enforce_exclusive:
            m.vch = pyo.Var(m.H, domain=pyo.Binary)
            m.vdis = pyo.Var(m.H, domain=pyo.Binary)
            m.c_ch = pyo.Constraint(m.H, rule=lambda mm, t: mm.Pch[t] <= bess.p_max_mw * mm.vch[t])
            m.c_dis = pyo.Constraint(m.H, rule=lambda mm, t: mm.Pdis[t] <= bess.p_max_mw * mm.vdis[t])
            m.c_excl = pyo.Constraint(m.H, rule=lambda mm, t: mm.vch[t] + mm.vdis[t] <= 1)
    bess_out = (lambda mm, t: mm.Pdis[t] - mm.Pch[t]) if bess is not None else (lambda mm, t: 0.0)

    # ---------------- factory constraints ----------------
    mt_out = (lambda mm, t: mm.Pmt[t]) if mt is not None else (lambda mm, t: 0.0)
    m.c_power = pyo.Constraint(m.H, rule=lambda mm, t: mm.Pbuy[t] - mm.Psell[t] + mt_out(mm, t) + bess_out(mm, t) == inst.base_load_mw[t]
                               + pyo.quicksum(mw * mm.s[c] for c, mw, _ in by_hour[t]))

    # ---------------- objective ----------------
    deg = 0.0
    if bess is not None:
        w_ch = 1.0 if bess.degradation_basis == "throughput" else 0.0      # weight of charge energy in C_TP
        deg = pyo.quicksum(bess.degradation_eur_mwh * (w_ch * m.Pch[t] + m.Pdis[t]) for t in range(T))
    grid = pyo.quicksum(inst.price_buy_eur_mwh[t] * m.Pbuy[t] - inst.price_sell_eur_mwh[t] * m.Psell[t] for t in range(T))
    if mt is not None:
        fuel = pyo.quicksum(mt.block_cost_eur_mwh[b] * m.Pb[b, t] for b in range(len(mt.block_width_mw)) for t in range(T))
        fuel += pyo.quicksum(mt.base_cost_eur_mwh * mt.p_min_mw * m.u[t] for t in range(T))   # C0*Pmin*u (always charged)
        sust = pyo.quicksum(mt.startup_cost_eur * m.x[t] + mt.shutdown_cost_eur * m.y[t] for t in range(T))
        m.obj = pyo.Objective(expr=grid + fuel + sust + deg, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=grid + deg, sense=pyo.minimize)
    return m
