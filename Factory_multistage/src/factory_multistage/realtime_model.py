from __future__ import annotations

from typing import List

import numpy as np
import pyomo.environ as pyo

from . import day_ahead as da
from . import intraday as idm
from .day_ahead import Instance, SchedulerConfig
from .realtime_scenarios import BalancingMarket, RealTimeSet
from .scenarios import IntradayMarket, ScenarioSet


def build_realtime_model(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, rt: RealTimeSet,
                         bal: BalancingMarket, cfg: SchedulerConfig, cands: List[da.Candidate]) -> pyo.ConcreteModel:
    """Extensive form of the three-stage problem.  The Stage 1/2 model of factory_mt_id_scheduler is reused; its
    scenario balance (which has no Stage-3 slack) and its objective are replaced."""
    T, S, W = inst.horizon_h, scen.n, rt.n_w
    mt, bess = inst.mt, inst.bess
    m = idm.build_intraday_model(inst, mkt, scen, cfg, cands)
    stage12_cost = m.obj.expr                      # DA + ID + MT + BESS cost, unchanged
    m.del_component(m.c_power)
    m.del_component(m.obj)
    m.W = pyo.RangeSet(0, W - 1)

    # ---------------- Stage 3 variables ----------------
    d_buy_max, d_sell_max = bal.bounds(inst)
    m.Dp = pyo.Var(m.S, m.W, m.H, bounds=(0, d_buy_max))                    # deficit bought (v2 Delta+)
    m.Dm = pyo.Var(m.S, m.W, m.H, bounds=(0, d_sell_max))                   # surplus sold   (v2 Delta-)
    if bal.enforce_exclusive:
        m.zb = pyo.Var(m.S, m.W, m.H, domain=pyo.Binary)
        m.c_dp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dp[k, w, t] <= d_buy_max * mm.zb[k, w, t])
        m.c_dm = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dm[k, w, t] <= d_sell_max * (1 - mm.zb[k, w, t]))

    # ---------------- balance, physical limits ----------------
    mt_out = (lambda mm, k, t: mm.Pmt[k, t]) if mt is not None else (lambda mm, k, t: 0.0)
    bess_out = (lambda mm, k, t: mm.Pdis[k, t] - mm.Pch[k, t]) if bess is not None else (lambda mm, k, t: 0.0)
    market = lambda mm, k, t: mm.Pbuy[t] - mm.Psell[t] + mm.Ibuy[k, t] - mm.Isell[k, t]
    supply = lambda mm, k, t: market(mm, k, t) + mt_out(mm, k, t) + bess_out(mm, k, t)
    load2 = lambda mm, k, t: inst.base_load_mw[t] + scen.load_dev_mw[k, t] + mm.Lbatch[t]     # Stage-2 factory load
    m.c_power_rt = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: supply(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]
                                  == (1.0 + rt.load_rel_dev[k, w, t]) * load2(mm, k, t))
    phys = lambda mm, k, w, t: market(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]             # physical grid exchange
    m.c_phys_imp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: phys(mm, k, w, t) <= inst.grid_limit_mw)
    m.c_phys_exp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: -phys(mm, k, w, t) <= inst.grid_sell_limit_mw)
    if bal.mode == "passive":                      # plan must balance the expected RT load
        e_eta = np.einsum("sw,swt->st", rt.prob, rt.load_rel_dev)
        m.c_plan = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: supply(mm, k, t) == (1.0 + e_eta[k, t]) * load2(mm, k, t))

    # ---------------- objective ----------------
    lam_p, lam_m = rt.lam_plus(inst), rt.lam_minus(inst)
    bal_cost = pyo.quicksum(scen.prob[k] * rt.prob[k, w] * (lam_p[k, w, t] * m.Dp[k, w, t] - lam_m[k, w, t] * m.Dm[k, w, t])
                            for k in range(S) for w in range(W) for t in range(T))
    m.obj = pyo.Objective(expr=stage12_cost + bal_cost, sense=pyo.minimize)
    return m
