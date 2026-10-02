"""Reusable factory and microturbine commitment blocks for Pyomo models."""

import math

import pyomo.environ as pyo

from .candidates import Candidate
from .data import Instance, SchedulerConfig
from .parameters import EPS


def add_batch_block(
    model: pyo.ConcreteModel,
    inst: Instance,
    cfg: SchedulerConfig,
    candidates: list[Candidate],
) -> list[list[tuple[int, float, float]]]:
    """Add batch starts, inventory, sequencing, and task capacity constraints."""
    hours = inst.horizon_h
    machine_count = len(inst.machines)
    task_count = len(inst.tasks)
    grid_count = int(round(hours / cfg.start_step_h))
    model.C = pyo.RangeSet(0, len(candidates) - 1)
    model.s = pyo.Var(model.C, domain=pyo.Binary)
    model.N = pyo.Var(model.H, bounds=(0, inst.inventory_max))

    by_hour: list[list[tuple[int, float, float]]] = [[] for _ in range(hours)]
    by_task: dict[int, list[int]] = {task: [] for task in range(task_count)}
    by_machine: dict[int, list[int]] = {machine: [] for machine in range(machine_count)}
    slot_users: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_task[candidate.task_index].append(index)
        by_machine[candidate.machine_index].append(index)
        for hour in candidate.occupancy.nonzero()[0]:
            overlap = candidate.occupancy[hour]
            by_hour[hour].append((
                index,
                candidate.power_mw * overlap,
                inst.yield_units[candidate.task_index] * overlap / candidate.duration_h,
            ))
        last_slot = min(
            grid_count - 1,
            math.ceil(
                (candidate.start_h + candidate.duration_h + inst.buffer_h - EPS)
                / cfg.start_step_h
            ) - 1,
        )
        for slot in range(candidate.grid_index, last_slot + 1):
            slot_users.setdefault((candidate.machine_index, slot), []).append(index)

    model.c_inventory = pyo.Constraint(
        model.H,
        rule=lambda m, hour: m.N[hour]
        == (m.N[hour - 1] if hour > 0 else inst.inventory_init)
        + pyo.quicksum(production * m.s[index] for index, _, production in by_hour[hour])
        - inst.demand_units[hour],
    )
    model.c_inventory_terminal = pyo.Constraint(
        expr=model.N[hours - 1] >= inst.inventory_init
    )
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
    return by_hour


def add_mt_commitment(model: pyo.ConcreteModel, inst: Instance) -> None:
    """Add shared MT on/start/stop binaries and minimum up/down constraints."""
    mt = inst.mt
    if mt is None:
        return
    model.u = pyo.Var(model.H, domain=pyo.Binary)
    model.x = pyo.Var(model.H, domain=pyo.Binary)
    model.y = pyo.Var(model.H, domain=pyo.Binary)
    initial_on = 1.0 if mt.initial_on else 0.0
    previous_on = lambda m, hour: m.u[hour - 1] if hour > 0 else initial_on
    model.c_commitment = pyo.Constraint(
        model.H,
        rule=lambda m, hour: m.x[hour] - m.y[hour]
        == m.u[hour] - previous_on(m, hour),
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