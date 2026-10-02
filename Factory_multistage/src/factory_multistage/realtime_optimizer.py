from __future__ import annotations

from typing import Dict, List, Optional

import pyomo.environ as pyo

from . import day_ahead as da
from . import intraday as idm
from ._internal import log
from .day_ahead import Instance, SchedulerConfig
from .intraday_model import _val
from .realtime_model import build_realtime_model
from .realtime_results import RealTimeResult, extract_realtime
from .realtime_scenarios import BalancingMarket, RealTimeSet
from .scenarios import IntradayMarket, ScenarioSet


def value_of_rt_modelling(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, rt: RealTimeSet,
                          bal: BalancingMarket, cfg: SchedulerConfig, cands: List[da.Candidate],
                          rp_eur: float) -> Dict[str, float]:
    """EEV_RT - RP.  Solve the Stage 1/2 model WITHOUT real-time risk, fix its Stage-1 decisions (batch starts, DA
    position, MT commitment) in the three-stage model and re-optimise Stage 2 / 3.  The gap to the full solution
    (RP) is the value of anticipating real-time imbalance when committing in Stage 1."""
    id_model = idm.build_intraday_model(inst, mkt, scen, cfg, cands)
    da.solve_model(id_model, cfg.solver)
    full = build_realtime_model(inst, mkt, scen, rt, bal, cfg, cands)
    for c in full.C:
        full.s[c].fix(round(_val(id_model.s[c])))
    for t in range(inst.horizon_h):
        full.Pbuy[t].fix(_val(id_model.Pbuy[t]))
        full.Psell[t].fix(_val(id_model.Psell[t]))
        if inst.mt is not None:
            for nm in ("u", "x", "y"):
                getattr(full, nm)[t].fix(round(_val(getattr(id_model, nm)[t])))
    try:
        da.solve_model(full, cfg.solver)
        eev = float(pyo.value(full.obj))
    except da.InfeasibleScheduleError:
        log.warning("the RT-blind Stage-1 plan is infeasible in at least one RT scenario: EEV_RT = inf")
        eev = float("inf")
    return dict(id_model_objective_eur=float(pyo.value(id_model.obj)), eev_rt_eur=eev, rp_eur=rp_eur,
                vss_rt_eur=eev - rp_eur, vss_rt_pct=100 * (eev - rp_eur) / abs(rp_eur) if rp_eur else float("nan"))


def optimize_realtime(inst: Instance, scen: ScenarioSet, rt: RealTimeSet, mkt: Optional[IntradayMarket] = None,
                      bal: Optional[BalancingMarket] = None, cfg: Optional[SchedulerConfig] = None,
                      compute_vss_rt: bool = False) -> RealTimeResult:
    """Validate -> candidates -> extensive-form MILP -> solve -> verify -> package results."""
    cfg, mkt, bal = cfg or SchedulerConfig(), (mkt or IntradayMarket()).validate(), (bal or BalancingMarket()).validate()
    inst.validate()
    cfg.check(inst)
    scen.validate(inst)
    rt.validate(inst, scen)
    cands = da.build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts, %d ID x %d RT scenarios, mode=%s, MT=%s, BESS=%s", len(cands), scen.n, rt.n_w,
             bal.mode, "yes" if inst.mt else "no", "yes" if inst.bess else "no")
    model = build_realtime_model(inst, mkt, scen, rt, bal, cfg, cands)
    info = da.solve_model(model, cfg.solver)
    res = extract_realtime(inst, mkt, scen, rt, bal, cfg, model, cands, info)
    if compute_vss_rt:
        res.vss_rt = value_of_rt_modelling(inst, mkt, scen, rt, bal, cfg, cands, res.objective_eur)
    return res
