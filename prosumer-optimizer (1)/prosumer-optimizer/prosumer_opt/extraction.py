"""
Turn a solved Pyomo model into a :class:`~prosumer_opt.results.StageResult`.

Split into small functions, one per output table:

    first_stage_plan   -> DAPlan arrays
    job_start_steps    -> DA start and per-scenario start of every job
    scenario_detail    -> long table, one row per (scenario, step)
    expected_schedule  -> probability-weighted per-step table
    batch_plan         -> one row per job
    cost_tables        -> per-scenario costs, expected breakdown, risk metrics
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .model.context import COST_COMPONENTS, ModelContext
from .parameters import BatchJob, TimeGrid
from .results import StageResult
from .scenarios import ScenarioTree
from .solver import SolveInfo
from .state import DAPlan
from .utils import cvar, var_value

_EXPECTED_COLUMNS = {   # schedule column -> column of the scenario-detail table it averages
    "exp_mt_mw": "mt_mw", "exp_bess_ch_mw": "bess_ch_mw", "exp_bess_dis_mw": "bess_dis_mw",
    "exp_soc_mwh": "soc_mwh", "exp_batch_load_mw": "batch_load_mw", "exp_inventory": "inventory",
    "exp_imb_surplus_mw": "imb_surplus_mw", "exp_imb_shortfall_mw": "imb_shortfall_mw",
    "exp_da_price": "da_price",
}


def first_stage_plan(m, ctx: ModelContext, old_plan: Optional[DAPlan], T: int,
                     da_start: Dict[str, int]) -> DAPlan:
    """Read first-stage decisions; steps before t0 are carried over from the earlier plan."""
    if old_plan is None:
        p_buy, p_sell, u, x, y = (np.zeros(T) for _ in range(5))
    else:
        p_buy, p_sell = old_plan.p_da_buy.copy(), old_plan.p_da_sell.copy()
        u, x, y = old_plan.mt_u.copy(), old_plan.mt_x.copy(), old_plan.mt_y.copy()
    for t in ctx.T_opt:
        p_buy[t], p_sell[t] = var_value(m.p_da_buy[t]), var_value(m.p_da_sell[t])
        u[t], x[t], y[t] = round(var_value(m.u[t])), round(var_value(m.x[t])), round(var_value(m.y[t]))
    return DAPlan(p_buy, p_sell, u, x, y, da_start)


def job_start_steps(m, ctx: ModelContext, jobs: List[BatchJob], tree: ScenarioTree, T: int,
                    old_plan: Optional[DAPlan]) -> Tuple[Dict[str, int], Dict[Tuple[int, str], int]]:
    da_start: Dict[str, int] = {}
    for j in jobs:
        jid = j.job_id
        da_start[jid] = ctx.plan_start[jid] if old_plan is not None else \
            int(round(sum(k * var_value(m.z[jid, k]) for k in ctx.allowed[jid])))
    scen_start: Dict[Tuple[int, str], int] = {}
    for i in range(len(tree.scenarios)):
        for j in jobs:
            jid = j.job_id
            if ctx.uses_z2[jid]:
                scen_start[i, jid] = int(round(sum(
                    k * var_value(m.z2[jid, k, i]) for k in range(T) if (jid, k, i) in m.z2)))
            else:
                scen_start[i, jid] = da_start[jid]
    return da_start, scen_start


def scenario_detail(m, ctx: ModelContext, tree: ScenarioTree, plan: DAPlan, tg: TimeGrid) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(tree.scenarios):
        for t in ctx.T_opt:
            dp = sum(br.prob * var_value(m.dpos[t, i, r]) for r, br in enumerate(s.rt_branches))
            dn = sum(br.prob * var_value(m.dneg[t, i, r]) for r, br in enumerate(s.rt_branches))
            rows.append(dict(
                scenario=s.name, prob=s.prob, step=t, hour=tg.hour(t),
                da_buy_mw=plan.p_da_buy[t], da_sell_mw=plan.p_da_sell[t],
                id_buy_mw=var_value(m.id_buy[t, i]), id_sell_mw=var_value(m.id_sell[t, i]),
                mt_on=plan.mt_u[t], mt_mw=pyo.value(m.p_mt[t, i]),
                bess_ch_mw=var_value(m.ch[t, i]), bess_dis_mw=var_value(m.dis[t, i]),
                soc_mwh=var_value(m.soc[t, i]),
                batch_load_mw=pyo.value(m.batch_load[t, i]), inventory=var_value(m.inv[t, i]),
                unmet_units=var_value(m.unmet[t, i]), imb_surplus_mw=dp, imb_shortfall_mw=dn,
                da_price=s.da_buy[t], id_price=s.id_buy[t]))
    return pd.DataFrame(rows)


def expected_schedule(detail: pd.DataFrame, plan: DAPlan, T: int, t0: int, partial: bool) -> pd.DataFrame:
    """Probability-weighted per-step schedule. ``partial`` re-indexes an intraday result to the full day."""
    d = detail.assign(id_net_mw=detail["id_buy_mw"] - detail["id_sell_mw"])
    w = d["prob"]
    by_step = d["step"]
    wsum = w.groupby(by_step).sum()

    def wavg(col: str) -> pd.Series:
        return (d[col] * w).groupby(by_step).sum() / wsum

    first = d.groupby("step")[["hour", "da_buy_mw", "da_sell_mw", "mt_on"]].first()
    steps = first.index.to_numpy()
    out = pd.DataFrame(index=first.index)
    out["hour"] = first["hour"]
    out["da_buy_mw"], out["da_sell_mw"] = first["da_buy_mw"], first["da_sell_mw"]
    out["da_net_mw"] = first["da_buy_mw"] - first["da_sell_mw"]
    out["mt_on"] = first["mt_on"]
    out["mt_startup"], out["mt_shutdown"] = plan.mt_x[steps], plan.mt_y[steps]
    out["exp_mt_mw"] = wavg("mt_mw")
    out["exp_id_net_mw"] = wavg("id_net_mw")
    for col in ("exp_bess_ch_mw", "exp_bess_dis_mw", "exp_soc_mwh", "exp_batch_load_mw",
                "exp_inventory", "exp_imb_surplus_mw", "exp_imb_shortfall_mw", "exp_da_price"):
        out[col] = wavg(_EXPECTED_COLUMNS[col])
    out.index.name = "step"
    if partial and t0 > 0:      # keep the full-day index for downstream use
        out = out.reindex(range(T))
    return out


def batch_plan(jobs: List[BatchJob], ctx: ModelContext, tree: ScenarioTree, tg: TimeGrid,
               da_start: Dict[str, int], scen_start) -> pd.DataFrame:
    rows = []
    for j in jobs:
        d = ctx.job_dur[j.job_id]
        row = dict(job_id=j.job_id, machine=j.machine, sequence=j.sequence, power_mw=j.power_mw,
                   units_out=j.units_out, start_step=da_start[j.job_id],
                   start_hour=tg.hour(da_start[j.job_id]), end_hour=tg.hour(da_start[j.job_id] + d),
                   duration_steps=d)
        for i, s in enumerate(tree.scenarios):
            row[f"start_hour_{s.name}"] = tg.hour(scen_start[i, j.job_id])
        rows.append(row)
    return pd.DataFrame(rows)


def cost_tables(m, tree: ScenarioTree) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Per-scenario costs, probability-weighted breakdown and risk metrics (mean, std, worst, best, CVaR95)."""
    probs = np.array([s.prob for s in tree.scenarios])
    exprs = {label: getattr(m, name) for label, name in COST_COMPONENTS.items()}
    rows = []
    for i, s in enumerate(tree.scenarios):
        r = {"scenario": s.name, "prob": s.prob}
        r.update({k: pyo.value(e[i]) for k, e in exprs.items()})
        r["TOTAL"] = sum(r[k] for k in exprs)
        rows.append(r)
    sc_costs = pd.DataFrame(rows)
    breakdown = {k: float(np.dot(probs, sc_costs[k])) for k in exprs}
    breakdown["TOTAL"] = float(np.dot(probs, sc_costs["TOTAL"]))
    tot = sc_costs["TOTAL"].to_numpy()
    mean = breakdown["TOTAL"]
    risk = dict(mean=mean, std=float(math.sqrt(np.dot(probs, (tot - mean) ** 2))),
                worst=float(tot.max()), best=float(tot.min()), cvar95=cvar(tot, probs, 0.95))
    return sc_costs, breakdown, risk


def extract_stage_result(m, ctx: ModelContext, tree: ScenarioTree, stage: str, info: SolveInfo,
                         old_plan: Optional[DAPlan], jobs: List[BatchJob], tg: TimeGrid) -> StageResult:
    T = tg.n_steps
    da_start, scen_start = job_start_steps(m, ctx, jobs, tree, T, old_plan)
    plan = first_stage_plan(m, ctx, old_plan, T, da_start)
    detail = scenario_detail(m, ctx, tree, plan, tg)
    schedule = expected_schedule(detail, plan, T, ctx.t0, partial=old_plan is not None)
    sc_costs, breakdown, risk = cost_tables(m, tree)
    return StageResult(
        stage=stage, t0=ctx.t0, status=info.status, objective_eur=float(pyo.value(m.obj)),
        mip_gap=info.gap, solve_time_s=info.time, model_stats=info.stats,
        schedule=schedule, batch_plan=batch_plan(jobs, ctx, tree, tg, da_start, scen_start),
        scenario_detail=detail, cost_breakdown=breakdown, scenario_costs=sc_costs,
        risk=risk, plan=plan)
