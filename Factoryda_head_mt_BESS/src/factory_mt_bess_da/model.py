"""Pyomo formulation for factory production, microturbine, and BESS dispatch."""

import math
from typing import Any

import pyomo.environ as pyo

from .candidates import Candidate
from .data import Instance, SchedulerConfig
from .parameters import DT_H, EPS


def build_model(inst: Instance, cfg: SchedulerConfig, candidates: list[Candidate]) -> pyo.ConcreteModel:
    hours, machine_count, task_count = inst.horizon_h, len(inst.machines), len(inst.tasks)
    grid_count = int(round(hours / cfg.start_step_h))
    mt = inst.mt
    bess = inst.bess
    model = pyo.ConcreteModel("FactoryMicroturbineBESSDayAhead")
    model.H = pyo.RangeSet(0, hours - 1)
    model.C = pyo.RangeSet(0, len(candidates) - 1)
    model.s = pyo.Var(model.C, domain=pyo.Binary)
    model.Pbuy = pyo.Var(model.H, bounds=(0, inst.grid_limit_mw))
    model.Psell = pyo.Var(model.H, bounds=(0, inst.grid_sell_limit_mw))
    model.N = pyo.Var(model.H, bounds=(0, inst.inventory_max))

    by_hour: list[list[tuple[int, float, float]]] = [[] for _ in range(hours)]
    by_task: dict[int, list[int]] = {task: [] for task in range(task_count)}
    by_machine: dict[int, list[int]] = {machine: [] for machine in range(machine_count)}
    slot_users: dict[tuple[int, int], list[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        by_task[candidate.task_index].append(candidate_index)
        by_machine[candidate.machine_index].append(candidate_index)
        for hour in candidate.occupancy.nonzero()[0]:
            overlap = candidate.occupancy[hour]
            by_hour[hour].append((
                candidate_index,
                candidate.power_mw * overlap,
                inst.yield_units[candidate.task_index] * overlap / candidate.duration_h,
            ))
        last_blocked_slot = min(
            grid_count - 1,
            math.ceil((candidate.start_h + candidate.duration_h + inst.buffer_h - EPS)
                      / cfg.start_step_h) - 1,
        )
        for slot in range(candidate.grid_index, last_blocked_slot + 1):
            slot_users.setdefault((candidate.machine_index, slot), []).append(candidate_index)

    if mt is not None:
        block_count = len(mt.block_width_mw)
        model.B = pyo.RangeSet(0, block_count - 1)
        model.Pmt = pyo.Var(model.H, bounds=(0, mt.p_max_mw))
        model.Pb = pyo.Var(model.B, model.H, bounds=lambda _, b, __: (0, mt.block_width_mw[b]))
        model.u = pyo.Var(model.H, domain=pyo.Binary)
        model.x = pyo.Var(model.H, domain=pyo.Binary)
        model.y = pyo.Var(model.H, domain=pyo.Binary)
        initial_on = 1.0 if mt.initial_on else 0.0
        initial_power = float(mt.initial_power_mw) if mt.initial_on else 0.0
        previous_on = lambda _, hour: model.u[hour - 1] if hour > 0 else initial_on
        previous_power = lambda _, hour: model.Pmt[hour - 1] if hour > 0 else initial_power

        model.c_mt_sum = pyo.Constraint(
            model.H,
            rule=lambda m, hour: m.Pmt[hour] == mt.p_min_mw * m.u[hour]
            + pyo.quicksum(m.Pb[block, hour] for block in m.B),
        )
        model.c_mt_block = pyo.Constraint(
            model.B,
            model.H,
            rule=lambda m, block, hour: m.Pb[block, hour] <= mt.block_width_mw[block] * m.u[hour],
        )
        model.c_ramp_up = pyo.Constraint(
            model.H,
            rule=lambda m, hour: m.Pmt[hour] - previous_power(m, hour)
            <= mt.ramp_up_mw_h * previous_on(m, hour) + mt.startup_ramp_mw_h * m.x[hour],
        )
        model.c_ramp_down = pyo.Constraint(
            model.H,
            rule=lambda m, hour: previous_power(m, hour) - m.Pmt[hour]
            <= mt.ramp_down_mw_h * m.u[hour] + mt.shutdown_ramp_mw_h * m.y[hour],
        )
        model.c_commitment = pyo.Constraint(
            model.H,
            rule=lambda m, hour: m.x[hour] - m.y[hour] == m.u[hour] - previous_on(m, hour),
        )
        model.c_start_stop = pyo.Constraint(
            model.H, rule=lambda m, hour: m.x[hour] + m.y[hour] <= 1
        )
        model.c_min_up = pyo.Constraint(
            model.H,
            rule=lambda m, hour: pyo.quicksum(
                m.x[prior] for prior in range(max(0, hour - mt.min_up_h + 1), hour + 1)
            ) <= m.u[hour],
        )
        model.c_min_down = pyo.Constraint(
            model.H,
            rule=lambda m, hour: pyo.quicksum(
                m.y[prior] for prior in range(max(0, hour - mt.min_down_h + 1), hour + 1)
            ) <= 1 - m.u[hour],
        )

    if bess is not None:
        model.Pch = pyo.Var(model.H, bounds=(0, bess.p_max_mw))
        model.Pdis = pyo.Var(model.H, bounds=(0, bess.p_max_mw))
        model.SoC = pyo.Var(model.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))
        previous_soc = lambda _, hour: model.SoC[hour - 1] if hour > 0 else bess.soc_init_mwh
        model.c_soc = pyo.Constraint(
            model.H,
            rule=lambda m, hour: m.SoC[hour] == previous_soc(m, hour)
            + bess.eta_ch * m.Pch[hour] * DT_H - m.Pdis[hour] * DT_H / bess.eta_dis,
        )
        model.c_soc_terminal = pyo.Constraint(expr=model.SoC[hours - 1] >= bess.soc_init_mwh)
        if bess.enforce_exclusive:
            model.vch = pyo.Var(model.H, domain=pyo.Binary)
            model.vdis = pyo.Var(model.H, domain=pyo.Binary)
            model.c_charge_power = pyo.Constraint(
                model.H, rule=lambda m, hour: m.Pch[hour] <= bess.p_max_mw * m.vch[hour]
            )
            model.c_discharge_power = pyo.Constraint(
                model.H, rule=lambda m, hour: m.Pdis[hour] <= bess.p_max_mw * m.vdis[hour]
            )
            model.c_bess_exclusive = pyo.Constraint(
                model.H, rule=lambda m, hour: m.vch[hour] + m.vdis[hour] <= 1
            )

    bess_output = lambda m, hour: m.Pdis[hour] - m.Pch[hour] if bess is not None else 0.0
    mt_output = lambda m, hour: m.Pmt[hour] if mt is not None else 0.0
    model.c_power_balance = pyo.Constraint(
        model.H,
        rule=lambda m, hour: m.Pbuy[hour] - m.Psell[hour] + mt_output(m, hour)
        + bess_output(m, hour) == inst.base_load_mw[hour]
        + pyo.quicksum(power * m.s[index] for index, power, _ in by_hour[hour]),
    )
    model.c_inventory = pyo.Constraint(
        model.H,
        rule=lambda m, hour: m.N[hour] == (m.N[hour - 1] if hour > 0 else inst.inventory_init)
        + pyo.quicksum(production * m.s[index] for index, _, production in by_hour[hour])
        - inst.demand_units[hour],
    )
    model.c_inventory_terminal = pyo.Constraint(expr=model.N[hours - 1] >= inst.inventory_init)
    if cfg.unique_tasks:
        model.c_unique_tasks = pyo.Constraint(
            range(task_count),
            rule=lambda m, task: pyo.quicksum(m.s[index] for index in by_task[task]) <= 1,
        )
    model.c_batch_cap = pyo.Constraint(
        range(machine_count),
        rule=lambda m, machine: pyo.quicksum(m.s[index] for index in by_machine[machine])
        <= inst.max_batches_per_machine,
    )
    model.c_sequence = pyo.ConstraintList()
    for users in slot_users.values():
        if len(users) > 1:
            model.c_sequence.add(pyo.quicksum(model.s[index] for index in users) <= 1)

    degradation_cost: Any = 0.0
    if bess is not None:
        charge_weight = 1.0 if bess.degradation_basis == "throughput" else 0.0
        degradation_cost = pyo.quicksum(
            bess.degradation_eur_mwh
            * (charge_weight * model.Pch[hour] + model.Pdis[hour])
            for hour in range(hours)
        )
    grid_cost = pyo.quicksum(
        inst.price_buy_eur_mwh[hour] * model.Pbuy[hour]
        - inst.price_sell_eur_mwh[hour] * model.Psell[hour]
        for hour in range(hours)
    )
    if mt is not None:
        fuel_cost = pyo.quicksum(
            mt.block_cost_eur_mwh[block] * model.Pb[block, hour]
            for block in range(len(mt.block_width_mw))
            for hour in range(hours)
        )
        fuel_cost += pyo.quicksum(
            mt.base_cost_eur_mwh * mt.p_min_mw * model.u[hour] for hour in range(hours)
        )
        start_stop_cost = pyo.quicksum(
            mt.startup_cost_eur * model.x[hour] + mt.shutdown_cost_eur * model.y[hour]
            for hour in range(hours)
        )
        objective = grid_cost + fuel_cost + start_stop_cost + degradation_cost
    else:
        objective = grid_cost + degradation_cost
    model.obj = pyo.Objective(expr=objective, sense=pyo.minimize)
    return model
