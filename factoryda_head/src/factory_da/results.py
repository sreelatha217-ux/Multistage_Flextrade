"""Solution extraction, independent verification, and KPI calculation."""
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ._internal import __version__, log
from .candidates import Candidate
from .data import Instance, SchedulerConfig, SchedulingResult
from .exceptions import VerificationError


def hhmm(h: float) -> str:
    mins = int(round(h * 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"

def _hourly_from_jobs(inst: Instance, jobs: pd.DataFrame) -> pd.DataFrame:
    """Independent (interval based) recomputation of the hourly profile from the job list."""
    T = inst.horizon_h
    hours = np.arange(T, dtype=float)
    batch = np.zeros(T)
    prod = np.zeros(T)
    for r in jobs.itertuples():
        ov = np.clip(np.minimum(r.end_h, hours + 1) - np.maximum(r.start_h, hours), 0, 1)
        batch += r.power_mw * ov
        prod += r.units_out * ov / (r.end_h - r.start_h)
    inv = inst.inventory_init + np.cumsum(prod - inst.demand_units)
    return pd.DataFrame(dict(hour=np.arange(T), price_eur_mwh=inst.price_eur_mwh, base_load_mw=inst.base_load_mw,
                             batch_load_mw=batch, p_da_mw=inst.base_load_mw + batch, production_units=prod,
                             demand_units=inst.demand_units, inventory_units=inv))

def verify(inst: Instance, cfg: SchedulerConfig, jobs: pd.DataFrame, objective: float,
           tol: float = 1e-4) -> Dict[str, object]:
    issues: List[str] = []
    hourly = _hourly_from_jobs(inst, jobs)
    cost = float((hourly.price_eur_mwh * hourly.p_da_mw).sum())
    if abs(cost - objective) > max(tol, 1e-6 * abs(objective)):
        issues.append(f"objective mismatch: recomputed {cost:.4f} vs solver {objective:.4f}")
    if hourly.p_da_mw.max() > inst.grid_limit_mw + tol:
        issues.append(f"substation limit exceeded: {hourly.p_da_mw.max():.2f} MW")
    if hourly.inventory_units.min() < -tol or hourly.inventory_units.max() > inst.inventory_max + tol:
        issues.append("inventory bounds violated")
    if hourly.inventory_units.iloc[-1] < inst.inventory_init - tol:
        issues.append("terminal stock below initial stock")
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
    return dict(passed=not issues, issues=issues, recomputed_cost_eur=cost)

def _asap_baseline(inst: Instance, jobs: pd.DataFrame) -> float:
    """Cost of the same batches, same machine order, started as early as possible (price blind)."""
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
    h = _hourly_from_jobs(inst, base)
    return float((h.price_eur_mwh * h.p_da_mw).sum())

def extract_result(inst: Instance, cfg: SchedulerConfig, model: pyo.ConcreteModel, cands: List[Candidate],
                   info: Dict[str, object]) -> SchedulingResult:
    chosen = [c for c in range(len(cands)) if (model.s[c].value or 0.0) > 0.5]
    rows = []
    for c in chosen:
        cd = cands[c]
        rows.append(dict(machine=inst.machines[cd.m], task=inst.tasks[cd.p], start_h=cd.start_h,
                         end_h=cd.start_h + cd.dur_h, duration_h=cd.dur_h, power_mw=cd.power_mw,
                         units_out=float(inst.yield_units[cd.p])))
    jobs = pd.DataFrame(rows, columns=["machine", "task", "start_h", "end_h", "duration_h", "power_mw", "units_out"])
    jobs = jobs.sort_values(["machine", "start_h"]).reset_index(drop=True)
    jobs.insert(1, "position_n", jobs.groupby("machine").cumcount() + 1)
    jobs["start_time"] = jobs.start_h.map(hhmm)
    jobs["end_time"] = jobs.end_h.map(hhmm)
    jobs["energy_mwh"] = jobs.power_mw * jobs.duration_h
    hourly = _hourly_from_jobs(inst, jobs)
    hourly["p_da_model_mw"] = [model.P[t].value or 0.0 for t in range(inst.horizon_h)]
    hourly["cost_eur"] = hourly.price_eur_mwh * hourly.p_da_mw
    # energy cost per batch (exact, hour by hour)
    hrs = np.arange(inst.horizon_h, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]

    obj = float(pyo.value(model.obj))
    ver = verify(inst, cfg, jobs, obj)
    if not ver["passed"]:
        msg = "; ".join(ver["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)

    units = float(jobs.units_out.sum())
    e = float(hourly.p_da_mw.sum())
    base_cost = _asap_baseline(inst, jobs)
    kpis = dict(
        n_batches=float(len(jobs)), units_produced=units, grid_energy_mwh=e,
        batch_energy_mwh=float(jobs.energy_mwh.sum()), avg_price_paid=obj / e if e > 0 else float("nan"),
        peak_mw=float(hourly.p_da_mw.max()), load_factor=float(hourly.p_da_mw.mean() / hourly.p_da_mw.max()),
        cost_per_unit=obj / units if units > 0 else float("nan"), baseline_asap_cost_eur=base_cost,
        saving_vs_asap_pct=100 * (base_cost - obj) / base_cost if base_cost and not math.isnan(base_cost) else float("nan"),
        final_inventory=float(hourly.inventory_units.iloc[-1]))
    return SchedulingResult(info["status"], obj, info["gap"], info["time"], info["stats"], jobs, hourly, kpis, ver)
