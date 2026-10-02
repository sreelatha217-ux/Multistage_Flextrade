"""Solution extraction, independent verification, and KPI calculations."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .candidates import Candidate
from .data import Instance, Microturbine, SchedulerConfig, SchedulingResult
from .exceptions import VerificationError
from .parameters import DT_H


def hhmm(hour: float) -> str:
    minutes = int(round(hour * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def load_from_jobs(inst: Instance, jobs: pd.DataFrame) -> pd.DataFrame:
    """Independently rebuild hourly load, production, and inventory from jobs."""
    hours = np.arange(inst.horizon_h, dtype=float)
    batch_load = np.zeros(inst.horizon_h)
    production = np.zeros(inst.horizon_h)
    for row in jobs.itertuples():
        overlap = np.clip(
            np.minimum(row.end_h, hours + 1) - np.maximum(row.start_h, hours), 0, 1
        )
        batch_load += row.power_mw * overlap
        production += row.units_out * overlap / (row.end_h - row.start_h)
    inventory = inst.inventory_init + np.cumsum(production - inst.demand_units)
    return pd.DataFrame({
        "hour": np.arange(inst.horizon_h),
        "price_buy_eur_mwh": inst.price_buy_eur_mwh,
        "price_sell_eur_mwh": inst.price_sell_eur_mwh,
        "base_load_mw": inst.base_load_mw,
        "batch_load_mw": batch_load,
        "total_load_mw": inst.base_load_mw + batch_load,
        "production_units": production,
        "demand_units": inst.demand_units,
        "inventory_units": inventory,
    })


def mt_fuel_cost_by_hour(
    mt: Microturbine, power_mt: np.ndarray, online: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return MT fuel cost and unused block capacity for each hour."""
    remaining = np.maximum(power_mt - mt.p_min_mw * online, 0.0)
    cost = mt.base_cost_eur_mwh * mt.p_min_mw * online
    for width, rate in zip(mt.block_width_mw, mt.block_cost_eur_mwh):
        segment = np.minimum(remaining, width)
        cost = cost + rate * segment
        remaining = remaining - segment
    return cost, remaining


def build_hourly(
    inst: Instance,
    jobs: pd.DataFrame,
    power_mt: np.ndarray,
    online: np.ndarray,
    power_charge: np.ndarray,
    power_discharge: np.ndarray,
) -> pd.DataFrame:
    """Build hourly output and independently recompute costs and battery SoC."""
    hourly = load_from_jobs(inst, jobs)
    hours = inst.horizon_h
    hourly["p_mt_mw"] = power_mt
    hourly["mt_on"] = online.astype(int)
    previous_online = np.concatenate(([1.0 if inst.mt and inst.mt.initial_on else 0.0], online[:-1]))
    hourly["mt_startup"] = ((online > 0.5) & (previous_online < 0.5)).astype(int)
    hourly["mt_shutdown"] = ((online < 0.5) & (previous_online > 0.5)).astype(int)
    hourly["p_bess_ch_mw"] = power_charge
    hourly["p_bess_dis_mw"] = power_discharge

    bess = inst.bess
    soc = np.zeros(hours)
    current_soc = bess.soc_init_mwh if bess else 0.0
    for hour in range(hours):
        if bess:
            current_soc += (
                bess.eta_ch * power_charge[hour] * DT_H
                - power_discharge[hour] * DT_H / bess.eta_dis
            )
        soc[hour] = current_soc
    hourly["soc_mwh"] = soc

    net_grid = (
        hourly.total_load_mw - hourly.p_mt_mw
        - hourly.p_bess_dis_mw + hourly.p_bess_ch_mw
    )
    hourly["p_net_grid_mw"] = net_grid
    hourly["p_buy_mw"] = np.maximum(net_grid, 0.0)
    hourly["p_sell_mw"] = np.maximum(-net_grid, 0.0)
    if inst.mt is not None:
        fuel_cost, _ = mt_fuel_cost_by_hour(inst.mt, power_mt, online)
        start_stop_cost = (
            inst.mt.startup_cost_eur * hourly.mt_startup
            + inst.mt.shutdown_cost_eur * hourly.mt_shutdown
        )
    else:
        fuel_cost = np.zeros(hours)
        start_stop_cost = np.zeros(hours)
    hourly["grid_buy_cost_eur"] = hourly.price_buy_eur_mwh * hourly.p_buy_mw
    hourly["grid_sell_revenue_eur"] = hourly.price_sell_eur_mwh * hourly.p_sell_mw
    hourly["grid_cost_eur"] = hourly.grid_buy_cost_eur - hourly.grid_sell_revenue_eur
    hourly["mt_fuel_cost_eur"] = fuel_cost
    hourly["mt_startstop_cost_eur"] = start_stop_cost
    charge_weight = 1.0 if bess and bess.degradation_basis == "throughput" else 0.0
    hourly["bess_degradation_cost_eur"] = (
        bess.degradation_eur_mwh
        * (charge_weight * hourly.p_bess_ch_mw + hourly.p_bess_dis_mw)
        if bess else 0.0
    )
    hourly["cost_eur"] = (
        hourly.grid_cost_eur + hourly.mt_fuel_cost_eur
        + hourly.mt_startstop_cost_eur + hourly.bess_degradation_cost_eur
    )
    return hourly


def verify(
    inst: Instance,
    cfg: SchedulerConfig,
    jobs: pd.DataFrame,
    hourly: pd.DataFrame,
    objective: float | None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Re-check modeled constraints from the extracted schedule."""
    issues = []
    cost = float(hourly.cost_eur.sum())
    if objective is not None and abs(cost - objective) > max(tolerance, 1e-6 * abs(objective)):
        issues.append(f"objective mismatch: recomputed {cost:.4f} vs solver {objective:.4f}")
    if hourly.p_buy_mw.max() > inst.grid_limit_mw + tolerance:
        issues.append("substation import limit exceeded")
    if hourly.p_sell_mw.max() > inst.grid_sell_limit_mw + tolerance:
        issues.append("substation export limit exceeded")
    balance = (
        hourly.p_buy_mw - hourly.p_sell_mw + hourly.p_mt_mw
        + hourly.p_bess_dis_mw - hourly.p_bess_ch_mw - hourly.total_load_mw
    )
    if balance.abs().max() > tolerance:
        issues.append("power balance violated")

    bess = inst.bess
    if bess is not None:
        charge = hourly.p_bess_ch_mw.to_numpy()
        discharge = hourly.p_bess_dis_mw.to_numpy()
        soc = hourly.soc_mwh.to_numpy()
        if charge.max() > bess.p_max_mw + tolerance or discharge.max() > bess.p_max_mw + tolerance:
            issues.append("BESS power limit violated")
        if charge.min() < -tolerance or discharge.min() < -tolerance:
            issues.append("BESS power is negative")
        if soc.min() < bess.soc_min_mwh - tolerance or soc.max() > bess.soc_max_mwh + tolerance:
            issues.append("BESS SoC bounds violated")
        if soc[-1] < bess.soc_init_mwh - tolerance:
            issues.append("BESS terminal SoC below initial SoC")
        if bess.enforce_exclusive and np.any((charge > tolerance) & (discharge > tolerance)):
            issues.append("BESS charges and discharges in the same hour")
    elif hourly.p_bess_ch_mw.abs().max() > tolerance or hourly.p_bess_dis_mw.abs().max() > tolerance:
        issues.append("BESS power used without a BESS")

    if hourly.inventory_units.min() < -tolerance or hourly.inventory_units.max() > inst.inventory_max + tolerance:
        issues.append("inventory bounds violated")
    if hourly.inventory_units.iloc[-1] < inst.inventory_init - tolerance:
        issues.append("terminal stock below initial stock")
    for machine, machine_jobs in jobs.groupby("machine"):
        machine_jobs = machine_jobs.sort_values("start_h")
        if len(machine_jobs) > inst.max_batches_per_machine:
            issues.append(f"{machine}: more than the allowed number of batches")
        gaps = machine_jobs.start_h.values[1:] - machine_jobs.end_h.values[:-1]
        if len(gaps) and gaps.min() < inst.buffer_h - tolerance:
            issues.append(f"{machine}: batch buffer violated")
    if (jobs.end_h > inst.horizon_h + tolerance).any() or (jobs.start_h < -tolerance).any():
        issues.append("a batch lies outside the horizon")
    if cfg.unique_tasks and jobs.task.duplicated().any():
        issues.append("a task was scheduled more than once")

    mt = inst.mt
    if mt is not None:
        online = hourly.mt_on.to_numpy(dtype=float)
        power = hourly.p_mt_mw.to_numpy()
        is_online = online > 0.5
        if np.any(power[~is_online] > tolerance):
            issues.append("MT produces power while offline")
        if np.any(power[is_online] < mt.p_min_mw - tolerance) or np.any(power[is_online] > mt.p_max_mw + tolerance):
            issues.append("MT output outside [Pmin, Pmax] while online")
        _, remaining = mt_fuel_cost_by_hour(mt, power, online)
        if remaining.max() > tolerance:
            issues.append("MT output above fuel-block capacity")
        previous_online = np.concatenate(([1.0 if mt.initial_on else 0.0], online[:-1]))
        previous_power = np.concatenate(([mt.initial_power_mw if mt.initial_on else 0.0], power[:-1]))
        startup = ((online > 0.5) & (previous_online < 0.5)).astype(float)
        shutdown = ((online < 0.5) & (previous_online > 0.5)).astype(float)
        if np.any(power - previous_power > mt.ramp_up_mw_h * previous_online + mt.startup_ramp_mw_h * startup + tolerance):
            issues.append("MT ramp-up or start-up limit violated")
        if np.any(previous_power - power > mt.ramp_down_mw_h * online + mt.shutdown_ramp_mw_h * shutdown + tolerance):
            issues.append("MT ramp-down or shut-down limit violated")
        for hour in np.flatnonzero(startup):
            if not online[hour:hour + mt.min_up_h].all():
                issues.append(f"MT minimum up time violated at hour {hour}")
        for hour in np.flatnonzero(shutdown):
            if online[hour:hour + mt.min_down_h].any():
                issues.append(f"MT minimum down time violated at hour {hour}")

    return {"passed": not issues, "issues": issues, "recomputed_cost_eur": cost}


def _asap_baseline(inst: Instance, jobs: pd.DataFrame) -> float:
    rows = []
    for _, machine_jobs in jobs.sort_values("start_h").groupby("machine"):
        start = 0.0
        for row in machine_jobs.itertuples():
            duration = row.end_h - row.start_h
            rows.append({
                "start_h": start,
                "end_h": start + duration,
                "power_mw": row.power_mw,
                "units_out": row.units_out,
            })
            start += duration + inst.buffer_h
    baseline_jobs = pd.DataFrame(rows)
    if baseline_jobs.empty or (baseline_jobs.end_h > inst.horizon_h + 1e-9).any():
        return float("nan")
    hourly = load_from_jobs(inst, baseline_jobs)
    return float((hourly.price_buy_eur_mwh * hourly.total_load_mw).sum())


def jobs_from_starts(
    inst: Instance,
    candidates: list[Candidate],
    model: pyo.ConcreteModel,
) -> pd.DataFrame:
    """Create the common production job table from selected candidate starts."""
    chosen = [index for index in range(len(candidates)) if (model.s[index].value or 0.0) > 0.5]
    rows = []
    for index in chosen:
        candidate = candidates[index]
        rows.append({
            "machine": inst.machines[candidate.machine_index],
            "task": inst.tasks[candidate.task_index],
            "start_h": candidate.start_h,
            "end_h": candidate.start_h + candidate.duration_h,
            "duration_h": candidate.duration_h,
            "power_mw": candidate.power_mw,
            "units_out": float(inst.yield_units[candidate.task_index]),
        })
    jobs = pd.DataFrame(rows, columns=[
        "machine", "task", "start_h", "end_h", "duration_h", "power_mw", "units_out"
    ])
    jobs = jobs.sort_values(["machine", "start_h"]).reset_index(drop=True)
    jobs.insert(1, "position_n", jobs.groupby("machine").cumcount() + 1)
    jobs["start_time"] = jobs.start_h.map(hhmm)
    jobs["end_time"] = jobs.end_h.map(hhmm)
    jobs["energy_mwh"] = jobs.power_mw * jobs.duration_h
    return jobs


def extract_result(
    inst: Instance,
    cfg: SchedulerConfig,
    model: pyo.ConcreteModel,
    candidates: list[Candidate],
    solve_info: dict[str, object],
) -> SchedulingResult:
    hours = inst.horizon_h
    jobs = jobs_from_starts(inst, candidates, model)

    if inst.mt is not None:
        online = np.array([round(model.u[hour].value or 0.0) for hour in range(hours)], dtype=float)
        power_mt = np.array([max(0.0, model.Pmt[hour].value or 0.0) for hour in range(hours)]) * online
    else:
        online, power_mt = np.zeros(hours), np.zeros(hours)
    if inst.bess is not None:
        power_charge = np.array([max(0.0, model.Pch[hour].value or 0.0) for hour in range(hours)])
        power_discharge = np.array([max(0.0, model.Pdis[hour].value or 0.0) for hour in range(hours)])
        power_charge[power_charge < 1e-9] = 0.0
        power_discharge[power_discharge < 1e-9] = 0.0
    else:
        power_charge, power_discharge = np.zeros(hours), np.zeros(hours)
    hourly = build_hourly(inst, jobs, power_mt, online, power_charge, power_discharge)
    hourly["soc_model_mwh"] = (
        [model.SoC[hour].value or 0.0 for hour in range(hours)] if inst.bess else np.zeros(hours)
    )
    hourly["p_buy_model_mw"] = [model.Pbuy[hour].value or 0.0 for hour in range(hours)]
    hourly["p_sell_model_mw"] = [model.Psell[hour].value or 0.0 for hour in range(hours)]

    hour_indices = np.arange(hours, dtype=float)
    jobs["energy_cost_eur"] = [
        float((np.clip(np.minimum(row.end_h, hour_indices + 1) - np.maximum(row.start_h, hour_indices), 0, 1)
               * inst.price_buy_eur_mwh).sum() * row.power_mw)
        for row in jobs.itertuples()
    ]
    objective = float(pyo.value(model.obj))
    verification = verify(inst, cfg, jobs, hourly, objective)
    model_net = hourly.p_buy_model_mw - hourly.p_sell_model_mw
    if abs(model_net - hourly.p_net_grid_mw).max() > 1e-3:
        verification["issues"].append("model grid exchange differs from independent recomputation")
        verification["passed"] = False
    if inst.bess is not None and abs(hourly.soc_model_mwh - hourly.soc_mwh).max() > 1e-3:
        verification["issues"].append("model SoC differs from independent recomputation")
        verification["passed"] = False
    if not verification["passed"]:
        message = "; ".join(verification["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {message}")

    units = float(jobs.units_out.sum())
    buy_energy = float(hourly.p_buy_mw.sum())
    sell_energy = float(hourly.p_sell_mw.sum())
    mt_energy = float(hourly.p_mt_mw.sum())
    total_load = float(hourly.total_load_mw.sum())
    grid_cost = float(hourly.grid_cost_eur.sum())
    buy_cost = float(hourly.grid_buy_cost_eur.sum())
    sell_revenue = float(hourly.grid_sell_revenue_eur.sum())
    fuel_cost = float(hourly.mt_fuel_cost_eur.sum())
    start_stop_cost = float(hourly.mt_startstop_cost_eur.sum())
    grid_only_cost = float((hourly.price_buy_eur_mwh * hourly.total_load_mw).sum())
    baseline_cost = _asap_baseline(inst, jobs)
    peak_buy = float(hourly.p_buy_mw.max())
    bess_capacity = inst.bess.soc_max_mwh - inst.bess.soc_min_mwh if inst.bess else 0.0
    kpis = {
        "n_batches": float(len(jobs)),
        "units_produced": units,
        "total_load_mwh": total_load,
        "grid_buy_energy_mwh": buy_energy,
        "grid_sell_energy_mwh": sell_energy,
        "grid_energy_mwh": buy_energy,
        "mt_energy_mwh": mt_energy,
        "mt_share_pct": 100 * mt_energy / total_load if total_load > 0 else 0.0,
        "batch_energy_mwh": float(jobs.energy_mwh.sum()),
        "grid_cost_eur": grid_cost,
        "grid_buy_cost_eur": buy_cost,
        "grid_sell_revenue_eur": sell_revenue,
        "mt_fuel_cost_eur": fuel_cost,
        "mt_startstop_cost_eur": start_stop_cost,
        "mt_starts": float(hourly.mt_startup.sum()),
        "mt_stops": float(hourly.mt_shutdown.sum()),
        "mt_on_hours": float(hourly.mt_on.sum()),
        "mt_avg_fuel_cost_eur_mwh": fuel_cost / mt_energy if mt_energy > 0 else float("nan"),
        "avg_cost_per_mwh": objective / total_load if total_load > 0 else float("nan"),
        "peak_grid_mw": peak_buy,
        "peak_export_mw": float(hourly.p_sell_mw.max()),
        "peak_load_mw": float(hourly.total_load_mw.max()),
        "load_factor": float(hourly.p_buy_mw.mean() / peak_buy) if peak_buy > 0 else 0.0,
        "cost_per_unit": objective / units if units > 0 else float("nan"),
        "bess_charge_mwh": float(hourly.p_bess_ch_mw.sum()),
        "bess_discharge_mwh": float(hourly.p_bess_dis_mw.sum()),
        "bess_degradation_cost_eur": float(hourly.bess_degradation_cost_eur.sum()),
        "bess_final_soc_mwh": float(hourly.soc_mwh.iloc[-1]),
        "bess_equiv_cycles": float(hourly.p_bess_dis_mw.sum() / bess_capacity) if bess_capacity > 0 else 0.0,
        "grid_only_same_schedule_eur": grid_only_cost,
        "der_saving_eur": grid_only_cost - objective,
        "baseline_asap_cost_eur": baseline_cost,
        "saving_vs_asap_pct": (
            100 * (baseline_cost - objective) / baseline_cost
            if baseline_cost and not math.isnan(baseline_cost) else float("nan")
        ),
        "final_inventory": float(hourly.inventory_units.iloc[-1]),
    }
    return SchedulingResult(
        solve_info["status"], objective, solve_info["gap"], solve_info["time"],
        solve_info["stats"], jobs, hourly, kpis, verification,
    )
