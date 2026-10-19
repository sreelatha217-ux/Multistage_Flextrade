"""Stage-2 production rescheduling against intraday price scenarios."""
from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd

import factory_mt_da_scheduler as da


def reschedule_production_by_id_scenario(inst, baseline_jobs: pd.DataFrame, mt_commitment,
                                         scenario_set, cfg: da.SchedulerConfig,
                                         reschedule_hour: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-optimize the production plan per ID scenario, freezing batches already started.

    Each scenario is solved with its ID prices as the scheduling tariff. The Stage-1 MT
    commitment is retained, while dispatch, BESS, grid positions, and future batches may
    recourse. The result is a schedule comparison, not a second offering-profit solve.
    """
    horizon = inst.horizon_h
    if not math.isfinite(reschedule_hour) or not 0 <= reschedule_hour <= horizon:
        raise da.InstanceValidationError(f"reschedule_hour must be in [0, {horizon}]")

    candidates = da.build_candidates(inst, cfg.start_step_h)
    all_jobs = []
    summaries = []
    task_index = {task: index for index, task in enumerate(inst.tasks)}
    machine_index = {machine: index for index, machine in enumerate(inst.machines)}
    frozen = baseline_jobs[baseline_jobs.start_h < reschedule_hour - da.EPS]

    def fix_commitment(model, scenario_inst):
        if scenario_inst.mt is not None:
            if len(mt_commitment) != horizon:
                raise da.InstanceValidationError(f"MT commitment must have {horizon} hourly values")
            for hour, committed in enumerate(mt_commitment):
                model.u[hour].fix(int(round(float(committed))))

    def candidate_for_job(job):
        try:
            machine = machine_index[job.machine]
            task = task_index[job.task]
        except KeyError as err:
            raise da.InstanceValidationError(f"job references unknown machine/task: {err}") from err
        match = [index for index, candidate in enumerate(candidates)
                 if candidate.m == machine and candidate.p == task
                 and abs(candidate.start_h - float(job.start_h)) <= da.EPS]
        if len(match) != 1:
            raise da.InstanceValidationError(
                f"cannot map job {job.task} on {job.machine} at {job.start_h:g} h to a start slot"
            )
        return match[0], task

    for scenario in range(scenario_set.n):
        scenario_inst = replace(
            inst,
            base_load_mw=inst.base_load_mw + scenario_set.load_dev_mw[scenario],
            price_buy_eur_mwh=scenario_set.id_buy[scenario].copy(),
            price_sell_eur_mwh=scenario_set.id_sell[scenario].copy(),
        ).validate()
        model = da.build_model(scenario_inst, cfg, candidates)
        fix_commitment(model, scenario_inst)

        frozen_candidate_ids = set()
        for job in frozen.itertuples():
            candidate_id, task = candidate_for_job(job)
            frozen_candidate_ids.add(candidate_id)
            if cfg.unique_tasks:
                for index, candidate in enumerate(candidates):
                    if candidate.p == task:
                        model.s[index].fix(1 if index == candidate_id else 0)
            else:
                model.s[candidate_id].fix(1)

        info = da.solve_model(model, cfg.solver)
        result = da.extract_result(scenario_inst, cfg, model, candidates, info)

        fixed_model = da.build_model(scenario_inst, cfg, candidates)
        fix_commitment(fixed_model, scenario_inst)
        fixed_candidate_ids = {candidate_for_job(job)[0] for job in baseline_jobs.itertuples()}
        for index in range(len(candidates)):
            fixed_model.s[index].fix(1 if index in fixed_candidate_ids else 0)
        fixed_info = da.solve_model(fixed_model, cfg.solver)
        fixed_result = da.extract_result(scenario_inst, cfg, fixed_model, candidates, fixed_info)
        used_fixed_plan = result.objective_eur > fixed_result.objective_eur
        if used_fixed_plan:
            result = fixed_result

        jobs = result.jobs.copy()
        jobs.insert(0, "id_scenario", scenario)
        jobs.insert(1, "scenario_probability", float(scenario_set.prob[scenario]))
        all_jobs.append(jobs)

        before = {row.task: (row.machine, float(row.start_h)) for row in baseline_jobs.itertuples()}
        after = {row.task: (row.machine, float(row.start_h)) for row in result.jobs.itertuples()}
        changed = sum(
            task not in before or task not in after
            or before[task][0] != after[task][0]
            or abs(before[task][1] - after[task][1]) > da.EPS
            for task in before.keys() | after.keys()
        )
        summaries.append(dict(
            id_scenario=scenario,
            probability=float(scenario_set.prob[scenario]),
            operating_cost_eur=result.objective_eur,
            fixed_plan_cost_eur=fixed_result.objective_eur,
            production_reschedule_savings_eur=fixed_result.objective_eur - result.objective_eur,
            used_fixed_plan=used_fixed_plan,
            mip_gap=result.mip_gap,
            solve_time_s=result.solve_time_s,
            batches=int(result.kpis["n_batches"]),
            units_produced=float(result.kpis["units_produced"]),
            jobs_changed=int(changed),
            frozen_jobs=int(len(frozen_candidate_ids)),
            verification_passed=bool(result.verification["passed"]),
        ))

    jobs_frame = pd.concat(all_jobs, ignore_index=True) if all_jobs else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    return jobs_frame, summary_frame