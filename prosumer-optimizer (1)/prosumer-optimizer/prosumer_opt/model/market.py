"""
Electricity markets, power balance and grid limits  (skill sections 2.1, 2.2, 2.6).

    Stage 1  p_da_buy / p_da_sell        scenario independent  -> non-anticipativity by construction
    Stage 2  id_buy / id_sell            per ID scenario
    Stage 3  dpos / dneg                 per ID scenario and RT branch (imbalance volumes)

Balance for every (t, i, r):
    P_DA + P_ID + P_MT + dis - ch  -  factory(t, i, r)  =  dpos - dneg
Imbalance settled at the DA reference price times ratios r_plus (<= 1) / r_minus (>= 1).
"""
from __future__ import annotations

import pyomo.environ as pyo

from .context import BuildSpec


def add_market(m: pyo.ConcreteModel, s: BuildSpec) -> None:
    """Add market variables, balance/limit constraints and the DA / ID / BAL cost expressions.
    Must run after the MT, BESS and production modules (uses ``p_mt``, ``ch``, ``dis``, ``batch_load``)."""
    gr, dt, sc = s.grid, s.dt, s.tree.scenarios

    # ---- variables ------------------------------------------------------------------
    m.p_da_buy = pyo.Var(m.TO, bounds=(0, gr.import_limit_mw))
    m.p_da_sell = pyo.Var(m.TO, bounds=(0, gr.export_limit_mw))
    m.id_buy = pyo.Var(m.TI, bounds=(0, gr.import_limit_mw))
    m.id_sell = pyo.Var(m.TI, bounds=(0, gr.export_limit_mw))
    m.dpos = pyo.Var(m.TIR, domain=pyo.NonNegativeReals)
    m.dneg = pyo.Var(m.TIR, domain=pyo.NonNegativeReals)

    # ---- power balance -----------------------------------------------------------------
    def supply(mm, t, i):
        return (mm.p_da_buy[t] - mm.p_da_sell[t] + mm.id_buy[t, i] - mm.id_sell[t, i]
                + mm.p_mt[t, i] + mm.dis[t, i] - mm.ch[t, i])

    def dev(t, i, r):
        return sc[i].rt_branches[r].load_dev[t]

    m.c_bal = pyo.Constraint(
        m.TIR, rule=lambda mm, t, i, r: supply(mm, t, i) - s.base[t] * (1 + dev(t, i, r)) - mm.batch_load[t, i]
        == mm.dpos[t, i, r] - mm.dneg[t, i, r])

    # ---- grid connection limits ----------------------------------------------------------
    def phys(mm, t, i, r):   # physical import at the meter, positive = from grid
        return (s.base[t] * (1 + dev(t, i, r)) + mm.batch_load[t, i]
                + mm.ch[t, i] - mm.dis[t, i] - mm.p_mt[t, i])
    m.c_imp = pyo.Constraint(m.TIR, rule=lambda mm, t, i, r: phys(mm, t, i, r) <= gr.import_limit_mw)
    m.c_exp = pyo.Constraint(m.TIR, rule=lambda mm, t, i, r: phys(mm, t, i, r) >= -gr.export_limit_mw)

    net = lambda mm, t, i: mm.p_da_buy[t] - mm.p_da_sell[t] + mm.id_buy[t, i] - mm.id_sell[t, i]
    m.c_pos_hi = pyo.Constraint(m.TI, rule=lambda mm, t, i: net(mm, t, i) <= gr.import_limit_mw)
    m.c_pos_lo = pyo.Constraint(m.TI, rule=lambda mm, t, i: net(mm, t, i) >= -gr.export_limit_mw)

    # ---- costs per ID scenario (EUR) -----------------------------------------------------------
    m.cost_da = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
        sc[i].da_buy[t] * mm.p_da_buy[t] - sc[i].da_sell[t] * mm.p_da_sell[t] for t in s.t_opt))
    m.cost_id = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
        sc[i].id_buy[t] * mm.id_buy[t, i] - sc[i].id_sell[t] * mm.id_sell[t, i] for t in s.t_opt))

    def _bal_cost(mm, i):
        tot = 0
        for r, br in enumerate(sc[i].rt_branches):
            tot += br.prob * dt * sum(
                sc[i].da_buy[t] * (br.r_minus[t] * mm.dneg[t, i, r] - br.r_plus[t] * mm.dpos[t, i, r])
                for t in s.t_opt)
        return tot
    m.cost_bal = pyo.Expression(m.I, rule=_bal_cost)


def freeze_first_stage(m: pyo.ConcreteModel, s: BuildSpec, allowed) -> None:
    """Intraday stage: fix every first-stage decision to the committed day-ahead plan."""
    plan = s.plan
    for t in s.t_opt:
        m.p_da_buy[t].fix(float(plan.p_da_buy[t]))
        m.p_da_sell[t].fix(float(plan.p_da_sell[t]))
        m.u[t].fix(int(round(plan.mt_u[t])))
        m.x[t].fix(int(round(plan.mt_x[t])))
        m.y[t].fix(int(round(plan.mt_y[t])))
    for jid, ks in allowed.items():
        start = int(plan.job_start_step[jid])
        for k in ks:
            m.z[jid, k].fix(1 if k == start else 0)
