from __future__ import annotations

from typing import Dict, List, Optional

import pyomo.environ as pyo

from . import day_ahead as da
from ._internal import log
from .day_ahead import Instance, SchedulerConfig
from .intraday_model import _val, build_intraday_model
from .intraday_results import IntradayResult, extract_intraday
from .scenarios import IntradayMarket, ScenarioSet


def value_of_stochastic_solution(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, cfg: SchedulerConfig,
                                 cands: List[da.Candidate], rp_eur: float) -> Dict[str, float]:
    """VSS = EEV - RP.  EEV: solve the expected-value problem, fix its Stage-1 decisions, re-optimise the
    recourse in every scenario.  RP is the stochastic solution value already computed."""
    ev_model = build_intraday_model(inst, mkt, scen.expected_value(), cfg, cands)
    da.solve_model(ev_model, cfg.solver)
    full = build_intraday_model(inst, mkt, scen, cfg, cands)
    for c in full.C:
        full.s[c].fix(round(_val(ev_model.s[c])))
    for t in range(inst.horizon_h):
        full.Pbuy[t].fix(_val(ev_model.Pbuy[t]))
        full.Psell[t].fix(_val(ev_model.Psell[t]))
        if inst.mt is not None:
            for nm in ("u", "x", "y"):
                getattr(full, nm)[t].fix(round(_val(getattr(ev_model, nm)[t])))
    try:
        da.solve_model(full, cfg.solver)
        eev = float(pyo.value(full.obj))
    except da.InfeasibleScheduleError:
        log.warning("the expected-value plan is infeasible in at least one scenario: EEV = inf")
        eev = float("inf")
    return dict(ev_objective_eur=float(pyo.value(ev_model.obj)), eev_eur=eev, rp_eur=rp_eur, vss_eur=eev - rp_eur,
                vss_pct=100 * (eev - rp_eur) / abs(rp_eur) if rp_eur else float("nan"))


def optimize_intraday(inst: Instance, scen: ScenarioSet, mkt: Optional[IntradayMarket] = None,
                      cfg: Optional[SchedulerConfig] = None, compute_vss: bool = False) -> IntradayResult:
    """Validate -> candidates -> extensive-form MILP -> solve -> verify -> package results."""
    cfg, mkt = cfg or SchedulerConfig(), (mkt or IntradayMarket()).validate()
    inst.validate()
    cfg.check(inst)
    scen.validate(inst)
    cands = da.build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts, %d scenarios, MT=%s, BESS=%s", len(cands), scen.n,
             "yes" if inst.mt else "no", "yes" if inst.bess else "no")
    model = build_intraday_model(inst, mkt, scen, cfg, cands)
    info = da.solve_model(model, cfg.solver)
    res = extract_intraday(inst, mkt, scen, cfg, model, cands, info)
    if compute_vss:
        res.vss = value_of_stochastic_solution(inst, mkt, scen, cfg, cands, res.objective_eur)
    return res
