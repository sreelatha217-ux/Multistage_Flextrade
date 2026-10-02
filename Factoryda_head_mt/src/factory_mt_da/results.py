"""Result extraction, independent verification, and KPI calculation."""

import math

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ._internal import EPS, log
from .candidates import Candidate
from .data import Instance, Microturbine, SchedulerConfig, SchedulingResult
from .exceptions import VerificationError


def hhmm(hour: float) -> str:
    """Format a fractional hour as a 24-hour clock time."""
    minutes = round(hour * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _load_from_jobs(inst: Instance, jobs: pd.DataFrame) -> pd.DataFrame:
    """Recompute hourly load, production, and inventory from job intervals."""
    hours = np.arange(inst.horizon_h, dtype=float)
    batch_load = np.zeros(inst.horizon_h)
    production = np.zeros(inst.horizon_h)
    for job in jobs.itertuples():
        overlap = np.clip(
            np.minimum(job.end_h, hours + 1) - np.maximum(job.start_h, hours),
            0,
            1,
        )
        batch_load += job.power_mw * overlap
        production += job.units_out * overlap / (job.end_h - job.start_h)
    inventory = inst.inventory_init + np.cumsum(production - inst.demand_units)
    return pd.DataFrame({
        "hour": np.arange(inst.horizon_h),
        "price_eur_mwh": inst.price_eur_mwh,
        "base_load_mw": inst.base_load_mw,
        "batch_load_mw": batch_load,
        "total_load_mw": inst.base_load_mw + batch_load,
        "production_units": production,
        "demand_units": inst.demand_units,
        "inventory_units": inventory,
    })


def mt_fuel_cost_by_hour(
    mt: Microturbine,
    p_mt: np.ndarray,
    online: np.ndarray,
    charge_min_power: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate merit-order MT fuel cost and output not covered by fuel blocks."""
    remaining = np.maximum(p_mt - mt.p_min_mw * online, 0.0)
    cost = np.zeros_like(p_mt)
    for width, marginal_cost in zip(mt.block_width_mw, mt.block_cost_eur_mwh):
        segment = np.minimum(remaining, width)
        cost += marginal_cost * segment
        remaining -= segment
    if charge_min_power:
        cost += mt.block_cost_eur_mwh[0] * mt.p_min_mw * online
    return cost, remaining


def _build_hourly(
    inst: Instance,
    cfg: SchedulerConfig,
    jobs: pd.DataFrame,
    p_mt: np.ndarray,
    online: np.ndarray,
) -> pd.DataFrame:
    hourly = _load_from_jobs(inst, jobs)
    hourly["p_mt_mw"] = p_mt
    hourly["mt_on"] = online.astype(int)
    previous_online = np.concatenate((
        [1.0 if inst.mt is not None and inst.mt.initial_on else 0.0],
        online[:-1],
    ))
    hourly["mt_startup"] = ((online > 0.5) & (previous_online < 0.5)).astype(int)
    hourly["mt_shutdown"] = ((online < 0.5) & (previous_online > 0.5)).astype(int)
    hourly["p_da_mw"] = hourly.total_load_mw - hourly.p_mt_mw
    if inst.mt is None:
        fuel_cost = np.zeros(inst.horizon_h)
        start_stop_cost = np.zeros(inst.horizon_h)
    else:
        fuel_cost, _ = mt_fuel_cost_by_hour(inst.mt, p_mt, online, cfg.charge_min_power_fuel)
        start_stop_cost = (
            inst.mt.startup_cost_eur * hourly.mt_startup
            + inst.mt.shutdown_cost_eur * hourly.mt_shutdown
        )
    hourly["grid_cost_eur"] = hourly.price_eur_mwh * hourly.p_da_mw
    hourly["mt_fuel_cost_eur"] = fuel_cost
    hourly["mt_startstop_cost_eur"] = start_stop_cost
    hourly["cost_eur"] = hourly.grid_cost_eur + fuel_cost + start_stop_cost
    return hourly


def verify(
    inst: Instance,
    cfg: SchedulerConfig,
    jobs: pd.DataFrame,
    hourly: pd.DataFrame,
    objective: float,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    """Check scheduling constraints independently of the Pyomo model."""
    issues = []
    recomputed_cost = float(hourly.cost_eur.sum())
    if abs(recomputed_cost - objective) > max(tolerance, 1e-6 * abs(objective)):
        issues.append(f"objective mismatch: recomputed {recomputed_cost:.4f} vs solver {objective:.4f}")
    if hourly.p_da_mw.min() < -tolerance:
        issues.append("negative grid import (MT output exceeds load)")
    if hourly.p_da_mw.max() > inst.grid_limit_mw + tolerance:
        issues.append(f"substation limit exceeded: {hourly.p_da_mw.max():.2f} MW")
    if np.abs(hourly.p_da_mw + hourly.p_mt_mw - hourly.total_load_mw).max() > tolerance:
        issues.append("power balance violated")
    if hourly.inventory_units.min() < -tolerance or hourly.inventory_units.max() > inst.inventory_max + tolerance:
        issues.append("inventory bounds violated")
    if hourly.inventory_units.iloc[-1] < inst.inventory_init - tolerance:
        issues.append("terminal stock below initial stock")

    for machine, group in jobs.groupby("machine"):
        group = group.sort_values("start_h")
        if len(group) > inst.max_batches_per_machine:
            issues.append(f"{machine}: more than N batches")
        gaps = group.start_h.values[1:] - group.end_h.values[:-1]
        if len(gaps) and gaps.min() < inst.buffer_h - tolerance:
            issues.append(f"{machine}: buffer violated (min gap {gaps.min():.3f} h)")
    if (jobs.end_h > inst.horizon_h + tolerance).any() or (jobs.start_h < -tolerance).any():
        issues.append("a batch lies outside the horizon")
    if cfg.unique_tasks and jobs.task.duplicated().any():
        issues.append("a task was scheduled more than once")

    mt = inst.mt
    if mt is not None:
        online = hourly.mt_on.to_numpy(dtype=float)
        output = hourly.p_mt_mw.to_numpy(dtype=float)
        is_online = online > 0.5
        if np.any(output[~is_online] > tolerance):
            issues.append("MT produces power while offline")
        if np.any(output[is_online] < mt.p_min_mw - tolerance) or np.any(
            output[is_online] > mt.p_max_mw + tolerance
        ):
            issues.append("MT output outside [Pmin, Pmax] while online")
        _, unused_block_output = mt_fuel_cost_by_hour(
            mt, output, online, cfg.charge_min_power_fuel
        )
        if unused_block_output.max() > tolerance:
            issues.append("MT output exceeds fuel-block capacity")
        previous_online = np.concatenate(([1.0 if mt.initial_on else 0.0], online[:-1]))
        previous_output = np.concatenate(([mt.initial_power_mw if mt.initial_on else 0.0], output[:-1]))
        startup = (online > 0.5) & (previous_online < 0.5)
        shutdown = (online < 0.5) & (previous_online > 0.5)
        ramp_up_limit = mt.ramp_up_mw_h * previous_online + mt.startup_ramp_mw_h * startup
        ramp_down_limit = mt.ramp_down_mw_h * online + mt.shutdown_ramp_mw_h * shutdown
        if np.any(output - previous_output > ramp_up_limit + tolerance):
            issues.append("MT ramp-up / start-up limit violated")
        if np.any(previous_output - output > ramp_down_limit + tolerance):
            issues.append("MT ramp-down / shut-down limit violated")
        for hour in np.flatnonzero(startup):
            if not online[hour:hour + mt.min_up_h].all():
                issues.append(f"MT minimum up time violated (start at hour {hour})")
        for hour in np.flatnonzero(shutdown):
            if online[hour:hour + mt.min_down_h].any():
                issues.append(f"MT minimum down time violated (stop at hour {hour})")
    return {"passed": not issues, "issues": issues, "recomputed_cost_eur": recomputed_cost}


def _asap_baseline(inst: Instance, jobs: pd.DataFrame) -> float:
    """Cost the selected batches in their machine order at earliest starts."""
    rows = []
    for _, group in jobs.sort_values("start_h").groupby("machine"):
        next_start = 0.0
        for job in group.itertuples():
            duration = job.end_h - job.start_h
            rows.append({
                "start_h": next_start,
                "end_h": next_start + duration,
                "power_mw": job.power_mw,
                "units_out": job.units_out,
            })
            next_start += duration + inst.buffer_h
    baseline_jobs = pd.DataFrame(rows)
    if baseline_jobs.empty or (baseline_jobs.end_h > inst.horizon_h + EPS).any():
        return float("nan")
    baseline_hourly = _load_from_jobs(inst, baseline_jobs)
    return float((baseline_hourly.price_eur_mwh * baseline_hourly.total_load_mw).sum())


def extract_result(
    inst: Instance,
    cfg: SchedulerConfig,
    model: pyo.ConcreteModel,
    candidates: list[Candidate],
    solve_info: dict[str, object],
) -> SchedulingResult:
    """Convert a solved model into job/hour tables, KPIs, and verification results."""
    selected = [index for index in range(len(candidates)) if (model.s[index].value or 0.0) > 0.5]
    job_rows = []
    for index in selected:
        candidate = candidates[index]
        job_rows.append({
            "machine": inst.machines[candidate.machine_index],
            "task": inst.tasks[candidate.task_index],
            "start_h": candidate.start_h,
            "end_h": candidate.start_h + candidate.duration_h,
            "duration_h": candidate.duration_h,
            "power_mw": candidate.power_mw,
            "units_out": float(inst.yield_units[candidate.task_index]),
        })
    jobs = pd.DataFrame(job_rows, columns=[
        "machine", "task", "start_h", "end_h", "duration_h", "power_mw", "units_out",
    ])
    jobs = jobs.sort_values(["machine", "start_h"]).reset_index(drop=True)
    jobs.insert(1, "position_n", jobs.groupby("machine").cumcount() + 1)
    jobs["start_time"] = jobs.start_h.map(hhmm)
    jobs["end_time"] = jobs.end_h.map(hhmm)
    jobs["energy_mwh"] = jobs.power_mw * jobs.duration_h

    if inst.mt is not None:
        online = np.array([round(model.u[hour].value or 0.0) for hour in range(inst.horizon_h)])
        p_mt = np.array([
            max(0.0, model.Pmt[hour].value or 0.0) for hour in range(inst.horizon_h)
        ]) * online
    else:
        online = np.zeros(inst.horizon_h)
        p_mt = np.zeros(inst.horizon_h)
    hourly = _build_hourly(inst, cfg, jobs, p_mt, online)
    hourly["p_da_model_mw"] = [model.P[hour].value or 0.0 for hour in range(inst.horizon_h)]

    hours = np.arange(inst.horizon_h, dtype=float)
    jobs["energy_cost_eur"] = [
        float(
            (np.clip(np.minimum(job.end_h, hours + 1) - np.maximum(job.start_h, hours), 0, 1)
             * inst.price_eur_mwh).sum()
            * job.power_mw
        )
        for job in jobs.itertuples()
    ]

    objective = float(pyo.value(model.obj))
    verification = verify(inst, cfg, jobs, hourly, objective)
    if np.abs(hourly.p_da_model_mw - hourly.p_da_mw).max() > 1e-3:
        verification["issues"].append("model P_DA differs from the load-balance recomputation")
        verification["passed"] = False
    if not verification["passed"]:
        message = "; ".join(verification["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {message}")
        log.error("verification failed: %s", message)

    units = float(jobs.units_out.sum())
    grid_energy = float(hourly.p_da_mw.sum())
    mt_energy = float(hourly.p_mt_mw.sum())
    total_load = float(hourly.total_load_mw.sum())
    grid_cost = float(hourly.grid_cost_eur.sum())
    fuel_cost = float(hourly.mt_fuel_cost_eur.sum())
    start_stop_cost = float(hourly.mt_startstop_cost_eur.sum())
    grid_only_cost = float((hourly.price_eur_mwh * hourly.total_load_mw).sum())
    baseline_cost = _asap_baseline(inst, jobs)
    peak_grid = float(hourly.p_da_mw.max())
    kpis = {
        "n_batches": float(len(jobs)),
        "units_produced": units,
        "total_load_mwh": total_load,
        "grid_energy_mwh": grid_energy,
        "mt_energy_mwh": mt_energy,
        "mt_share_pct": 100 * mt_energy / total_load if total_load > 0 else 0.0,
        "batch_energy_mwh": float(jobs.energy_mwh.sum()),
        "grid_cost_eur": grid_cost,
        "mt_fuel_cost_eur": fuel_cost,
        "mt_startstop_cost_eur": start_stop_cost,
        "mt_starts": float(hourly.mt_startup.sum()),
        "mt_stops": float(hourly.mt_shutdown.sum()),
        "mt_on_hours": float(hourly.mt_on.sum()),
        "mt_avg_fuel_cost_eur_mwh": fuel_cost / mt_energy if mt_energy > 0 else float("nan"),
        "avg_cost_per_mwh": objective / total_load if total_load > 0 else float("nan"),
        "peak_grid_mw": peak_grid,
        "peak_load_mw": float(hourly.total_load_mw.max()),
        "load_factor": float(hourly.p_da_mw.mean() / peak_grid) if peak_grid > 0 else 0.0,
        "cost_per_unit": objective / units if units > 0 else float("nan"),
        "grid_only_same_schedule_eur": grid_only_cost,
        "mt_saving_eur": grid_only_cost - objective,
        "baseline_asap_cost_eur": baseline_cost,
        "saving_vs_asap_pct": (
            100 * (baseline_cost - objective) / baseline_cost
            if baseline_cost and not math.isnan(baseline_cost) else float("nan")
        ),
        "final_inventory": float(hourly.inventory_units.iloc[-1]),
    }
    return SchedulingResult(
        solve_info["status"],
        objective,
        solve_info["gap"],
        solve_info["time"],
        solve_info["stats"],
        jobs,
        hourly,
        kpis,
        verification,
    )