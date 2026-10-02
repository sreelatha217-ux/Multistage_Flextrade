"""Pyomo MILP construction for factory production and microturbine dispatch."""

import math

import numpy as np
import pyomo.environ as pyo

from ._internal import EPS
from .candidates import Candidate
from .data import Instance, SchedulerConfig


def _add_microturbine_constraints(model: pyo.ConcreteModel, inst: Instance) -> None:
    mt = inst.mt
    if mt is None:
        return
    block_count = len(mt.block_width_mw)
    model.B = pyo.RangeSet(0, block_count - 1)
    model.Pmt = pyo.Var(model.H, bounds=(0, mt.p_max_mw))
    model.Pb = pyo.Var(model.B, model.H, bounds=lambda _m, b, _t: (0, mt.block_width_mw[b]))
    model.u = pyo.Var(model.H, domain=pyo.Binary)
    model.x = pyo.Var(model.H, domain=pyo.Binary)
    model.y = pyo.Var(model.H, domain=pyo.Binary)

    initial_on = 1.0 if mt.initial_on else 0.0
    initial_power = float(mt.initial_power_mw) if mt.initial_on else 0.0

    def previous_on(_model, hour):
        return _model.u[hour - 1] if hour > 0 else initial_on

    def previous_power(_model, hour):
        return _model.Pmt[hour - 1] if hour > 0 else initial_power

    model.c_mt_sum = pyo.Constraint(
        model.H,
        rule=lambda m, t: m.Pmt[t] == mt.p_min_mw * m.u[t]
        + pyo.quicksum(m.Pb[b, t] for b in m.B),
    )
    model.c_mt_blk = pyo.Constraint(
        model.B,
        model.H,
        rule=lambda m, b, t: m.Pb[b, t] <= mt.block_width_mw[b] * m.u[t],
    )
    model.c_mt_cap = pyo.Constraint(model.H, rule=lambda m, t: m.Pmt[t] <= mt.p_max_mw * m.u[t])
    model.c_ru = pyo.Constraint(
        model.H,
        rule=lambda m, t: m.Pmt[t] - previous_power(m, t)
        <= mt.ramp_up_mw_h * previous_on(m, t) + mt.startup_ramp_mw_h * m.x[t],
    )
    model.c_rd = pyo.Constraint(
        model.H,
        rule=lambda m, t: previous_power(m, t) - m.Pmt[t]
        <= mt.ramp_down_mw_h * m.u[t] + mt.shutdown_ramp_mw_h * m.y[t],
    )
    model.c_uc = pyo.Constraint(
        model.H,
        rule=lambda m, t: m.x[t] - m.y[t] == m.u[t] - previous_on(m, t),
    )
    model.c_xy = pyo.Constraint(model.H, rule=lambda m, t: m.x[t] + m.y[t] <= 1)
    model.c_mut = pyo.Constraint(
        model.H,
        rule=lambda m, t: pyo.quicksum(
            m.x[tau] for tau in range(max(0, t - mt.min_up_h + 1), t + 1)
        ) <= m.u[t],
    )
    model.c_mdt = pyo.Constraint(
        model.H,
        rule=lambda m, t: pyo.quicksum(
            m.y[tau] for tau in range(max(0, t - mt.min_down_h + 1), t + 1)
        ) <= 1 - m.u[t],
    )


def build_model(
    inst: Instance,
    cfg: SchedulerConfig,
    candidates: list[Candidate],
) -> pyo.ConcreteModel:
    """Build the time-indexed MILP without invoking a solver."""
    horizon, machine_count, task_count = inst.horizon_h, len(inst.machines), len(inst.tasks)
    start_step = cfg.start_step_h
    grid_count = round(horizon / start_step)
    model = pyo.ConcreteModel("FactoryMicroturbineDayAhead")
    model.H = pyo.RangeSet(0, horizon - 1)
    model.C = pyo.RangeSet(0, len(candidates) - 1)
    model.s = pyo.Var(model.C, domain=pyo.Binary)
    model.P = pyo.Var(model.H, bounds=(0, inst.grid_limit_mw))
    model.N = pyo.Var(model.H, bounds=(0, inst.inventory_max))

    by_hour: list[list[tuple[int, float, float]]] = [[] for _ in range(horizon)]
    by_task: dict[int, list[int]] = {task: [] for task in range(task_count)}
    by_machine: dict[int, list[int]] = {machine: [] for machine in range(machine_count)}
    slot_users: dict[tuple[int, int], list[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        by_task[candidate.task_index].append(candidate_index)
        by_machine[candidate.machine_index].append(candidate_index)
        for hour in np.nonzero(candidate.occupancy > 0)[0]:
            production = (
                inst.yield_units[candidate.task_index]
                * candidate.occupancy[hour]
                / candidate.duration_h
            )
            by_hour[hour].append((
                candidate_index,
                candidate.power_mw * candidate.occupancy[hour],
                production,
            ))
        last_slot = min(
            grid_count - 1,
            math.ceil((candidate.start_h + candidate.duration_h + inst.buffer_h - EPS) / start_step) - 1,
        )
        for slot in range(candidate.grid_index, last_slot + 1):
            slot_users.setdefault((candidate.machine_index, slot), []).append(candidate_index)

    _add_microturbine_constraints(model, inst)
    mt_output = (lambda m, t: m.Pmt[t]) if inst.mt is not None else (lambda _m, _t: 0.0)
    model.c_power = pyo.Constraint(
        model.H,
        rule=lambda m, t: m.P[t] + mt_output(m, t) == inst.base_load_mw[t]
        + pyo.quicksum(power * m.s[c] for c, power, _ in by_hour[t]),
    )
    model.c_inv = pyo.Constraint(
        model.H,
        rule=lambda m, t: m.N[t]
        == (m.N[t - 1] if t > 0 else inst.inventory_init)
        + pyo.quicksum(production * m.s[c] for c, _, production in by_hour[t])
        - inst.demand_units[t],
    )
    model.c_terminal = pyo.Constraint(expr=model.N[horizon - 1] >= inst.inventory_init)
    if cfg.unique_tasks:
        model.c_unique = pyo.Constraint(
            range(task_count),
            rule=lambda m, task: pyo.quicksum(m.s[c] for c in by_task[task]) <= 1,
        )
    model.c_cap = pyo.Constraint(
        range(machine_count),
        rule=lambda m, machine: pyo.quicksum(m.s[c] for c in by_machine[machine])
        <= inst.max_batches_per_machine,
    )
    model.c_seq = pyo.ConstraintList()
    for users in slot_users.values():
        if len(users) > 1:
            model.c_seq.add(pyo.quicksum(model.s[c] for c in users) <= 1)

    objective = pyo.quicksum(inst.price_eur_mwh[t] * model.P[t] for t in range(horizon))
    if inst.mt is not None:
        mt = inst.mt
        fuel = pyo.quicksum(
            mt.block_cost_eur_mwh[block] * model.Pb[block, hour]
            for block in range(len(mt.block_width_mw))
            for hour in range(horizon)
        )
        if cfg.charge_min_power_fuel:
            fuel += pyo.quicksum(
                mt.block_cost_eur_mwh[0] * mt.p_min_mw * model.u[hour]
                for hour in range(horizon)
            )
        start_stop = pyo.quicksum(
            mt.startup_cost_eur * model.x[hour] + mt.shutdown_cost_eur * model.y[hour]
            for hour in range(horizon)
        )
        objective += fuel + start_stop
    model.obj = pyo.Objective(expr=objective, sense=pyo.minimize)
    return model