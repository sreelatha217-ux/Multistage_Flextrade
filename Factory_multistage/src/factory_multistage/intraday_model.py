from __future__ import annotations

from typing import List

import pyomo.environ as pyo

from . import day_ahead as da
from ._internal import DT_H
from .day_ahead import Instance, SchedulerConfig
from .scenarios import IntradayMarket, ScenarioSet


def build_intraday_model(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, cfg: SchedulerConfig,
                         cands: List[da.Candidate]) -> pyo.ConcreteModel:
    """Deterministic equivalent of the two-stage problem (extensive form)."""
    T, S = inst.horizon_h, scen.n
    mt, bess = inst.mt, inst.bess
    m = pyo.ConcreteModel("FactoryMicroturbineIntraday")
    m.H = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, S - 1)

    # ---------------- Stage 1 ----------------
    m.Pbuy = pyo.Var(m.H, bounds=(0, inst.grid_limit_mw))                   # DA purchase (MW)
    m.Psell = pyo.Var(m.H, bounds=(0, inst.grid_sell_limit_mw))             # DA sale (MW)
    by_hour = da.add_batch_block(m, inst, cfg, cands)                       # s, N, inventory, sequencing
    m.Lbatch = pyo.Expression(m.H, rule=lambda mm, t: pyo.quicksum(mw * mm.s[c] for c, mw, _ in by_hour[t]))
    if mt is not None:
        da.add_mt_commitment(m, inst)                                       # u, x, y, MUT / MDT

    # ---------------- Stage 2 ----------------
    m.Ibuy = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_buy_mw))                  # ID purchase (MW)
    m.Isell = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_sell_mw))                # ID sale (MW)
    if mt is not None:
        m.B = pyo.RangeSet(0, len(mt.block_width_mw) - 1)
        m.Pmt = pyo.Var(m.S, m.H, bounds=(0, mt.p_max_mw))
        m.Pb = pyo.Var(m.B, m.S, m.H, bounds=lambda mm, b, k, t: (0, mt.block_width_mw[b]))
        u0 = 1.0 if mt.initial_on else 0.0
        p0 = float(mt.initial_power_mw) if mt.initial_on else 0.0
        up = lambda mm, t: mm.u[t - 1] if t > 0 else u0
        pp = lambda mm, k, t: mm.Pmt[k, t - 1] if t > 0 else p0
        m.c_mt_sum = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] == mt.p_min_mw * mm.u[t]
                                    + pyo.quicksum(mm.Pb[b, k, t] for b in mm.B))
        m.c_mt_blk = pyo.Constraint(m.B, m.S, m.H, rule=lambda mm, b, k, t: mm.Pb[b, k, t] <= mt.block_width_mw[b] * mm.u[t])
        m.c_ru = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] - pp(mm, k, t)
                                <= mt.ramp_up_mw_h * up(mm, t) + mt.startup_ramp_mw_h * mm.x[t])
        m.c_rd = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: pp(mm, k, t) - mm.Pmt[k, t]
                                <= mt.ramp_down_mw_h * mm.u[t] + mt.shutdown_ramp_mw_h * mm.y[t])
    if bess is not None:
        m.Pch = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.Pdis = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.SoC = pyo.Var(m.S, m.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))
        soc_prev = lambda mm, k, t: mm.SoC[k, t - 1] if t > 0 else bess.soc_init_mwh
        m.c_soc = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.SoC[k, t] == soc_prev(mm, k, t)
                                 + bess.eta_ch * mm.Pch[k, t] * DT_H - mm.Pdis[k, t] * DT_H / bess.eta_dis)
        m.c_soc_term = pyo.Constraint(m.S, rule=lambda mm, k: mm.SoC[k, T - 1] >= bess.soc_init_mwh)
        if bess.enforce_exclusive:
            m.vch = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.vdis = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.c_ch = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pch[k, t] <= bess.p_max_mw * mm.vch[k, t])
            m.c_dis = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pdis[k, t] <= bess.p_max_mw * mm.vdis[k, t])
            m.c_excl = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.vch[k, t] + mm.vdis[k, t] <= 1)

    # ---------------- balance and substation limits ----------------
    mt_out = (lambda mm, k, t: mm.Pmt[k, t]) if mt is not None else (lambda mm, k, t: 0.0)
    bess_out = (lambda mm, k, t: mm.Pdis[k, t] - mm.Pch[k, t]) if bess is not None else (lambda mm, k, t: 0.0)
    net = lambda mm, k, t: mm.Pbuy[t] - mm.Psell[t] + mm.Ibuy[k, t] - mm.Isell[k, t]
    m.c_power = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: net(mm, k, t) + mt_out(mm, k, t) + bess_out(mm, k, t)
                               == inst.base_load_mw[t] + scen.load_dev_mw[k, t] + mm.Lbatch[t])
    m.c_net_imp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: net(mm, k, t) <= inst.grid_limit_mw)
    m.c_net_exp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: -net(mm, k, t) <= inst.grid_sell_limit_mw)

    # ---------------- objective ----------------
    pi = scen.prob
    cost = pyo.quicksum(inst.price_buy_eur_mwh[t] * m.Pbuy[t] - inst.price_sell_eur_mwh[t] * m.Psell[t] for t in range(T))
    cost += pyo.quicksum(pi[k] * (scen.id_buy_eur_mwh[k, t] * m.Ibuy[k, t] - scen.id_sell_eur_mwh[k, t] * m.Isell[k, t])
                         for k in range(S) for t in range(T))
    if mt is not None:
        cost += pyo.quicksum(mt.base_cost_eur_mwh * mt.p_min_mw * m.u[t] + mt.startup_cost_eur * m.x[t]
                             + mt.shutdown_cost_eur * m.y[t] for t in range(T))          # Stage-1 commitment cost
        cost += pyo.quicksum(pi[k] * mt.block_cost_eur_mwh[b] * m.Pb[b, k, t]
                             for b in range(len(mt.block_width_mw)) for k in range(S) for t in range(T))
    if bess is not None:
        w_ch = 1.0 if bess.degradation_basis == "throughput" else 0.0
        cost += pyo.quicksum(pi[k] * bess.degradation_eur_mwh * (w_ch * m.Pch[k, t] + m.Pdis[k, t])
                             for k in range(S) for t in range(T))
    m.obj = pyo.Objective(expr=cost, sense=pyo.minimize)
    return m


def _val(var, default: float = 0.0) -> float:
    v = var.value
    return default if v is None else float(v)
