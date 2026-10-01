"""
Microturbine (MT): unit commitment + piecewise-linear fuel cost  (skill section 2.4).

First-stage (scenario independent):  u (online), x (start-up), y (shut-down)
Second-stage (per ID scenario):      seg[k] power in cost block k above P_min

    P_MT = P_min * u + sum_k seg_k            0 <= seg_k <= width_k * u
    ramp up / down with start-up / shut-down ramp allowances
    min up / min down time, commitment logic  x - y = u - u_prev,  x + y <= 1
"""
from __future__ import annotations

import pyomo.environ as pyo

from ..constants import TOL
from .context import BuildSpec


def add_microturbine(m: pyo.ConcreteModel, s: BuildSpec) -> None:
    mt, st, t0, dt = s.mt, s.state, s.t0, s.dt
    segs = s.segments
    ru, rd = mt.ramp_up_mw_h * dt, mt.ramp_down_mw_h * dt
    sru, srd = mt.startup_ramp_mw_h * dt, mt.shutdown_ramp_mw_h * dt

    # ---- variables ------------------------------------------------------------
    m.u = pyo.Var(m.TO, domain=pyo.Binary)
    m.x = pyo.Var(m.TO, domain=pyo.Binary)
    m.y = pyo.Var(m.TO, domain=pyo.Binary)
    m.seg = pyo.Var(m.K, m.TI, bounds=lambda mm, k, t, i: (0, segs[k][0]))

    # ---- output and block limits ------------------------------------------------
    m.p_mt = pyo.Expression(
        m.TI, rule=lambda mm, t, i: mt.p_min_mw * mm.u[t] + sum(mm.seg[k, t, i] for k in mm.K))
    m.c_seg = pyo.Constraint(m.K, m.TI, rule=lambda mm, k, t, i: mm.seg[k, t, i] <= segs[k][0] * mm.u[t])

    def p_prev(mm, t, i):
        return mm.p_mt[t - 1, i] if t > t0 else st.mt_output_mw

    def u_prev(mm, t):
        return mm.u[t - 1] if t > t0 else float(st.mt_online)

    # ---- ramping ----------------------------------------------------------------
    m.c_ru = pyo.Constraint(
        m.TI, rule=lambda mm, t, i: mm.p_mt[t, i] - p_prev(mm, t, i) <= ru * u_prev(mm, t) + sru * mm.x[t])
    m.c_rd = pyo.Constraint(
        m.TI, rule=lambda mm, t, i: p_prev(mm, t, i) - mm.p_mt[t, i] <= rd * mm.u[t] + srd * mm.y[t])

    # ---- commitment logic: only a decision in the day-ahead stage ----------------
    if not s.is_intraday:
        tg = s.time
        n_up, n_dn = tg.steps(mt.min_up_h), tg.steps(mt.min_down_h)
        m.c_uc = pyo.Constraint(m.TO, rule=lambda mm, t: mm.x[t] - mm.y[t] == mm.u[t] - u_prev(mm, t))
        m.c_xy = pyo.Constraint(m.TO, rule=lambda mm, t: mm.x[t] + mm.y[t] <= 1)
        m.c_mut = pyo.Constraint(
            m.TO, rule=lambda mm, t: sum(mm.x[k] for k in range(max(t0, t - n_up + 1), t + 1)) <= mm.u[t])
        m.c_mdt = pyo.Constraint(
            m.TO, rule=lambda mm, t: sum(mm.y[k] for k in range(max(t0, t - n_dn + 1), t + 1)) <= 1 - mm.u[t])
        # carry-over of min up / down time from before the horizon
        if st.mt_online and st.mt_hours_in_state < mt.min_up_h - TOL:
            for t in s.t_opt[:tg.steps(mt.min_up_h - st.mt_hours_in_state)]:
                m.u[t].fix(1)
        if (not st.mt_online) and st.mt_hours_in_state < mt.min_down_h - TOL:
            for t in s.t_opt[:tg.steps(mt.min_down_h - st.mt_hours_in_state)]:
                m.u[t].fix(0)

    # ---- cost (EUR per ID scenario): fuel blocks + no-load + start/stop ---------------
    m.cost_mt = pyo.Expression(m.I, rule=lambda mm, i: dt * sum(
        sum(segs[k][1] * mm.seg[k, t, i] for k in mm.K) + s.no_load_cost * mm.u[t] for t in s.t_opt)
        + sum(mt.startup_cost_eur * mm.x[t] + mt.shutdown_cost_eur * mm.y[t] for t in s.t_opt))
