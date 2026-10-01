"""
Battery Energy Storage System (BESS)  (skill section 2.5).

Per ID scenario:  ch, dis (MW), soc (MWh), optional binaries v_ch / v_dis that forbid
simultaneous charging and discharging.

    soc_t = soc_{t-1} + eta_ch * ch * dt - dis * dt / eta_dis
    soc_min <= soc <= soc_max,   ch <= P_max * v_ch,   dis <= P_max * v_dis,   v_ch + v_dis <= 1
    soc_T >= terminal target (capped at what is physically reachable)
Cost: throughput degradation  C_TP * (ch + dis) * dt
"""
from __future__ import annotations

import pyomo.environ as pyo

from .context import BuildSpec


def add_bess(m: pyo.ConcreteModel, s: BuildSpec) -> None:
    bs, st, t0, dt, T = s.bess, s.state, s.t0, s.dt, s.T

    m.ch = pyo.Var(m.TI, bounds=(0, bs.p_max_mw))
    m.dis = pyo.Var(m.TI, bounds=(0, bs.p_max_mw))
    m.soc = pyo.Var(m.TI, bounds=(bs.soc_min_mwh, bs.soc_max_mwh))
    if bs.enforce_exclusivity:
        m.v_ch = pyo.Var(m.TI, domain=pyo.Binary)
        m.v_dis = pyo.Var(m.TI, domain=pyo.Binary)

    def _soc(mm, t, i):
        prev = mm.soc[t - 1, i] if t > t0 else st.soc_mwh
        return mm.soc[t, i] == prev + bs.eta_ch * mm.ch[t, i] * dt - mm.dis[t, i] * dt / bs.eta_dis
    m.c_soc = pyo.Constraint(m.TI, rule=_soc)

    if bs.enforce_exclusivity:
        m.c_ch = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.ch[t, i] <= bs.p_max_mw * mm.v_ch[t, i])
        m.c_dis = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.dis[t, i] <= bs.p_max_mw * mm.v_dis[t, i])
        m.c_excl = pyo.Constraint(m.TI, rule=lambda mm, t, i: mm.v_ch[t, i] + mm.v_dis[t, i] <= 1)

    # Terminal SoC: never demand more than can be charged in the remaining steps.
    reach = st.soc_mwh + bs.eta_ch * bs.p_max_mw * dt * len(s.t_opt)
    term = min(bs.terminal_target, bs.soc_max_mwh, reach)
    m.c_term = pyo.Constraint(m.I, rule=lambda mm, i: mm.soc[T - 1, i] >= term)

    m.cost_bess = pyo.Expression(m.I, rule=lambda mm, i: dt * bs.throughput_cost_eur_mwh * sum(
        mm.ch[t, i] + mm.dis[t, i] for t in s.t_opt))
