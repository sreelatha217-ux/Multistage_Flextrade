from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ._internal import DT_H, log
from .candidates import Candidate
from .data import Instance, Microturbine, SchedulerConfig, SchedulingResult
from .exceptions import VerificationError


def hhmm(h: float) -> str:
    mins = int(round(h * 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _load_from_jobs(inst: Instance, jobs: pd.DataFrame) -> pd.DataFrame:
    """Independent (interval based) recomputation of load, production and inventory from the job list."""
    T = inst.horizon_h
    hours = np.arange(T, dtype=float)
    batch = np.zeros(T)
    prod = np.zeros(T)
    for r in jobs.itertuples():
        ov = np.clip(np.minimum(r.end_h, hours + 1) - np.maximum(r.start_h, hours), 0, 1)
        batch += r.power_mw * ov
        prod += r.units_out * ov / (r.end_h - r.start_h)
    inv = inst.inventory_init + np.cumsum(prod - inst.demand_units)
    return pd.DataFrame(dict(hour=np.arange(T), price_buy_eur_mwh=inst.price_buy_eur_mwh,
                             price_sell_eur_mwh=inst.price_sell_eur_mwh, base_load_mw=inst.base_load_mw,
                             batch_load_mw=batch, total_load_mw=inst.base_load_mw + batch, production_units=prod,
                             demand_units=inst.demand_units, inventory_units=inv))


def mt_fuel_cost_by_hour(mt: Microturbine, p_mt: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fuel cost = C0*Pmin*u (always charged while online) + merit-order block cost (valid because block
    costs are non-decreasing).  Returns (cost per hour, MW that did not fit into any block - must be ~0)."""
    rem = np.maximum(p_mt - mt.p_min_mw * u, 0.0)
    cost = mt.base_cost_eur_mwh * mt.p_min_mw * u
    for w, c in zip(mt.block_width_mw, mt.block_cost_eur_mwh):
        seg = np.minimum(rem, w)
        cost = cost + c * seg
        rem = rem - seg
    return cost, rem


def _build_hourly(inst: Instance, cfg: SchedulerConfig, jobs: pd.DataFrame,
                  p_mt: np.ndarray, u: np.ndarray, p_ch: np.ndarray, p_dis: np.ndarray) -> pd.DataFrame:
    """Hourly table from the job list, the MT commitment/dispatch and the BESS dispatch."""
    h = _load_from_jobs(inst, jobs)
    T = inst.horizon_h
    h["p_mt_mw"] = p_mt
    h["mt_on"] = u.astype(int)
    u_prev = np.concatenate(([1.0 if (inst.mt and inst.mt.initial_on) else 0.0], u[:-1]))
    h["mt_startup"] = ((u > 0.5) & (u_prev < 0.5)).astype(int)
    h["mt_shutdown"] = ((u < 0.5) & (u_prev > 0.5)).astype(int)
    h["p_bess_ch_mw"] = p_ch
    h["p_bess_dis_mw"] = p_dis
    b = inst.bess
    soc, prev = np.zeros(T), (b.soc_init_mwh if b else 0.0)
    for t in range(T):                                  # independent SoC recomputation from the dispatch
        if b:
            prev = prev + b.eta_ch * p_ch[t] * DT_H - p_dis[t] * DT_H / b.eta_dis
        soc[t] = prev
    h["soc_mwh"] = soc
    net = h.total_load_mw - h.p_mt_mw - h.p_bess_dis_mw + h.p_bess_ch_mw   # net grid exchange: > 0 import, < 0 export
    h["p_net_grid_mw"] = net
    h["p_buy_mw"] = np.maximum(net, 0.0)
    h["p_sell_mw"] = np.maximum(-net, 0.0)
    if inst.mt is not None:
        fuel, _ = mt_fuel_cost_by_hour(inst.mt, p_mt, u)
        ss = inst.mt.startup_cost_eur * h.mt_startup + inst.mt.shutdown_cost_eur * h.mt_shutdown
    else:
        fuel, ss = np.zeros(T), np.zeros(T)
    h["grid_buy_cost_eur"] = h.price_buy_eur_mwh * h.p_buy_mw
    h["grid_sell_revenue_eur"] = h.price_sell_eur_mwh * h.p_sell_mw
    h["grid_cost_eur"] = h.grid_buy_cost_eur - h.grid_sell_revenue_eur       # net grid cost
    h["mt_fuel_cost_eur"] = fuel
    h["mt_startstop_cost_eur"] = ss
    w_ch = 1.0 if (b and b.degradation_basis == "throughput") else 0.0
    h["bess_degradation_cost_eur"] = (b.degradation_eur_mwh * (w_ch * h.p_bess_ch_mw + h.p_bess_dis_mw)) if b else 0.0
    h["cost_eur"] = h.grid_cost_eur + h.mt_fuel_cost_eur + h.mt_startstop_cost_eur + h.bess_degradation_cost_eur
    return h


def verify(inst: Instance, cfg: SchedulerConfig, jobs: pd.DataFrame, hourly: pd.DataFrame,
           objective: Optional[float],
           tol: float = 1e-4) -> Dict[str, object]:
    """Re-check every constraint of the model from the extracted solution only."""
    issues: List[str] = []
    cost = float(hourly.cost_eur.sum())
    if objective is not None and abs(cost - objective) > max(tol, 1e-6 * abs(objective)):
        issues.append(f"objective mismatch: recomputed {cost:.4f} vs solver {objective:.4f}")
    # substation balance & limits (import AND export)
    if hourly.p_buy_mw.max() > inst.grid_limit_mw + tol:
        issues.append(f"substation import limit exceeded: {hourly.p_buy_mw.max():.2f} MW")
    if hourly.p_sell_mw.max() > inst.grid_sell_limit_mw + tol:
        issues.append(f"substation export limit exceeded: {hourly.p_sell_mw.max():.2f} MW")
    if np.abs(hourly.p_buy_mw - hourly.p_sell_mw + hourly.p_mt_mw + hourly.p_bess_dis_mw
              - hourly.p_bess_ch_mw - hourly.total_load_mw).max() > tol:
        issues.append("power balance violated")
    # BESS
    b = inst.bess
    if b is not None:
        ch, dis, soc = hourly.p_bess_ch_mw.values, hourly.p_bess_dis_mw.values, hourly.soc_mwh.values
        if ch.max() > b.p_max_mw + tol or dis.max() > b.p_max_mw + tol or ch.min() < -tol or dis.min() < -tol:
            issues.append("BESS power limit violated")
        if soc.min() < b.soc_min_mwh - tol or soc.max() > b.soc_max_mwh + tol:
            issues.append("BESS SoC bounds violated")
        if soc[-1] < b.soc_init_mwh - tol:
            issues.append("BESS terminal SoC below initial SoC")
        if b.enforce_exclusive and np.any((ch > tol) & (dis > tol)):
            issues.append("BESS charges and discharges in the same hour")
    elif (hourly.p_bess_ch_mw.abs().max() > tol) or (hourly.p_bess_dis_mw.abs().max() > tol):
        issues.append("BESS power without a BESS")
    # warehouse
    if hourly.inventory_units.min() < -tol or hourly.inventory_units.max() > inst.inventory_max + tol:
        issues.append("inventory bounds violated")
    if hourly.inventory_units.iloc[-1] < inst.inventory_init - tol:
        issues.append("terminal stock below initial stock")
    # batches
    for mach, g in jobs.groupby("machine"):
        g = g.sort_values("start_h")
        if len(g) > inst.max_batches_per_machine:
            issues.append(f"{mach}: more than N batches")
        gaps = g.start_h.values[1:] - g.end_h.values[:-1]
        if len(gaps) and gaps.min() < inst.buffer_h - tol:
            issues.append(f"{mach}: buffer violated (min gap {gaps.min():.3f} h)")
    if (jobs.end_h > inst.horizon_h + tol).any() or (jobs.start_h < -tol).any():
        issues.append("a batch lies outside the horizon")
    if cfg.unique_tasks and jobs.task.duplicated().any():
        issues.append("a task was scheduled more than once")
    # microturbine
    mt = inst.mt
    if mt is not None:
        u = hourly.mt_on.values.astype(float)
        p = hourly.p_mt_mw.values
        on = u > 0.5
        if np.any(p[~on] > tol):
            issues.append("MT produces power while offline")
        if np.any(p[on] < mt.p_min_mw - tol) or np.any(p[on] > mt.p_max_mw + tol):
            issues.append("MT output outside [Pmin, Pmax] while online")
        _, leftover = mt_fuel_cost_by_hour(mt, p, u)
        if leftover.max() > tol:
            issues.append(f"MT output above the capacity of the fuel blocks ({leftover.max():.3f} MW)")
        u_prev = np.concatenate(([1.0 if mt.initial_on else 0.0], u[:-1]))
        p_prev = np.concatenate(([mt.initial_power_mw if mt.initial_on else 0.0], p[:-1]))
        x = ((u > 0.5) & (u_prev < 0.5)).astype(float)
        y = ((u < 0.5) & (u_prev > 0.5)).astype(float)
        if np.any(p - p_prev > mt.ramp_up_mw_h * u_prev + mt.startup_ramp_mw_h * x + tol):
            issues.append("MT ramp-up / start-up limit violated")
        if np.any(p_prev - p > mt.ramp_down_mw_h * u + mt.shutdown_ramp_mw_h * y + tol):
            issues.append("MT ramp-down / shut-down limit violated")
        for t in np.nonzero(x > 0.5)[0]:        # a start at t forces ON through t+MUT-1 (inside the horizon)
            if not u[t:t + mt.min_up_h].all():
                issues.append(f"MT minimum up time violated (start at hour {t})")
        for t in np.nonzero(y > 0.5)[0]:        # a stop at t forces OFF through t+MDT-1 (inside the horizon)
            if u[t:t + mt.min_down_h].any():
                issues.append(f"MT minimum down time violated (stop at hour {t})")
    return dict(passed=not issues, issues=issues, recomputed_cost_eur=cost)


def _asap_baseline(inst: Instance, jobs: pd.DataFrame) -> float:
    """Grid-only cost of the same batches/machine order, started as early as possible (price blind)."""
    rows = []
    for _, g in jobs.sort_values("start_h").groupby("machine"):
        t = 0.0
        for r in g.itertuples():
            d = r.end_h - r.start_h
            rows.append(dict(start_h=t, end_h=t + d, power_mw=r.power_mw, units_out=r.units_out))
            t += d + inst.buffer_h
    base = pd.DataFrame(rows)
    if base.empty or (base.end_h > inst.horizon_h + 1e-9).any():
        return float("nan")
    h = _load_from_jobs(inst, base)
    return float((h.price_buy_eur_mwh * h.total_load_mw).sum())


def jobs_from_starts(inst: Instance, cands: List[Candidate], model: pyo.ConcreteModel) -> pd.DataFrame:
    """Job table (machine, position n, task, start/end, power, yield, energy) from the chosen batch starts."""
    chosen = [c for c in range(len(cands)) if (model.s[c].value or 0.0) > 0.5]
    rows = [dict(machine=inst.machines[cands[c].m], task=inst.tasks[cands[c].p], start_h=cands[c].start_h,
                 end_h=cands[c].start_h + cands[c].dur_h, duration_h=cands[c].dur_h, power_mw=cands[c].power_mw,
                 units_out=float(inst.yield_units[cands[c].p])) for c in chosen]
    jobs = pd.DataFrame(rows, columns=["machine", "task", "start_h", "end_h", "duration_h", "power_mw", "units_out"])
    jobs = jobs.sort_values(["machine", "start_h"]).reset_index(drop=True)
    jobs.insert(1, "position_n", jobs.groupby("machine").cumcount() + 1)
    jobs["start_time"] = jobs.start_h.map(hhmm)
    jobs["end_time"] = jobs.end_h.map(hhmm)
    jobs["energy_mwh"] = jobs.power_mw * jobs.duration_h
    return jobs


def extract_result(inst: Instance, cfg: SchedulerConfig, model: pyo.ConcreteModel, cands: List[Candidate],
                   info: Dict[str, object]) -> SchedulingResult:
    T = inst.horizon_h
    jobs = jobs_from_starts(inst, cands, model)

    if inst.mt is not None:
        u = np.array([round(model.u[t].value or 0.0) for t in range(T)], dtype=float)
        p_mt = np.array([max(0.0, model.Pmt[t].value or 0.0) for t in range(T)]) * u
    else:
        u, p_mt = np.zeros(T), np.zeros(T)
    if inst.bess is not None:
        p_ch = np.array([max(0.0, model.Pch[t].value or 0.0) for t in range(T)])
        p_dis = np.array([max(0.0, model.Pdis[t].value or 0.0) for t in range(T)])
        p_ch[p_ch < 1e-9], p_dis[p_dis < 1e-9] = 0.0, 0.0
    else:
        p_ch, p_dis = np.zeros(T), np.zeros(T)
    hourly = _build_hourly(inst, cfg, jobs, p_mt, u, p_ch, p_dis)
    hourly["soc_model_mwh"] = [model.SoC[t].value or 0.0 for t in range(T)] if inst.bess is not None else np.zeros(T)
    hourly["p_buy_model_mw"] = [model.Pbuy[t].value or 0.0 for t in range(T)]
    hourly["p_sell_model_mw"] = [model.Psell[t].value or 0.0 for t in range(T)]

    hrs = np.arange(T, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_buy_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]

    obj = float(pyo.value(model.obj))
    ver = verify(inst, cfg, jobs, hourly, obj)
    model_net = hourly.p_buy_model_mw - hourly.p_sell_model_mw
    if abs(model_net - hourly.p_net_grid_mw).max() > 1e-3:
        ver["issues"].append("model net grid exchange differs from the load-balance recomputation")
        ver["passed"] = False
    if inst.bess is not None and abs(hourly.soc_model_mwh - hourly.soc_mwh).max() > 1e-3:
        ver["issues"].append("model SoC differs from the independent SoC recomputation")
        ver["passed"] = False
    if not ver["passed"]:
        msg = "; ".join(ver["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)

    units = float(jobs.units_out.sum())
    buy_e, sell_e, mt_e = (float(hourly[c].sum()) for c in ("p_buy_mw", "p_sell_mw", "p_mt_mw"))
    load_e = float(hourly.total_load_mw.sum())
    grid_cost, fuel, ss = (float(hourly[c].sum()) for c in ("grid_cost_eur", "mt_fuel_cost_eur", "mt_startstop_cost_eur"))
    buy_cost, sell_rev = float(hourly.grid_buy_cost_eur.sum()), float(hourly.grid_sell_revenue_eur.sum())
    no_mt = float((hourly.price_buy_eur_mwh * hourly.total_load_mw).sum())   # same batches, all from the grid
    base_cost = _asap_baseline(inst, jobs)
    peak_buy = float(hourly.p_buy_mw.max())
    kpis = dict(
        n_batches=float(len(jobs)), units_produced=units, total_load_mwh=load_e,
        grid_buy_energy_mwh=buy_e, grid_sell_energy_mwh=sell_e, grid_energy_mwh=buy_e,
        mt_energy_mwh=mt_e, mt_share_pct=100 * mt_e / load_e if load_e > 0 else 0.0,
        batch_energy_mwh=float(jobs.energy_mwh.sum()),
        grid_cost_eur=grid_cost, grid_buy_cost_eur=buy_cost, grid_sell_revenue_eur=sell_rev,
        mt_fuel_cost_eur=fuel, mt_startstop_cost_eur=ss,
        mt_starts=float(hourly.mt_startup.sum()), mt_stops=float(hourly.mt_shutdown.sum()),
        mt_on_hours=float(hourly.mt_on.sum()),
        mt_avg_fuel_cost_eur_mwh=fuel / mt_e if mt_e > 0 else float("nan"),
        avg_cost_per_mwh=obj / load_e if load_e > 0 else float("nan"),
        peak_grid_mw=peak_buy, peak_export_mw=float(hourly.p_sell_mw.max()), peak_load_mw=float(hourly.total_load_mw.max()),
        load_factor=float(hourly.p_buy_mw.mean() / peak_buy) if peak_buy > 0 else 0.0,
        cost_per_unit=obj / units if units > 0 else float("nan"),
        bess_charge_mwh=float(hourly.p_bess_ch_mw.sum()), bess_discharge_mwh=float(hourly.p_bess_dis_mw.sum()),
        bess_degradation_cost_eur=float(hourly.bess_degradation_cost_eur.sum()),
        bess_final_soc_mwh=float(hourly.soc_mwh.iloc[-1]),
        bess_equiv_cycles=(float(hourly.p_bess_dis_mw.sum()) / (inst.bess.soc_max_mwh - inst.bess.soc_min_mwh)
                           if inst.bess else 0.0),
        grid_only_same_schedule_eur=no_mt, der_saving_eur=no_mt - obj,
        baseline_asap_cost_eur=base_cost,
        saving_vs_asap_pct=100 * (base_cost - obj) / base_cost if base_cost and not math.isnan(base_cost) else float("nan"),
        final_inventory=float(hourly.inventory_units.iloc[-1]))
    return SchedulingResult(info["status"], obj, info["gap"], info["time"], info["stats"], jobs, hourly, kpis, ver)
