"""Deterministic benchmark instance and time-of-use tariff."""

import numpy as np

from .data import Instance, Microturbine
from .parameters import (
    BENCHMARK_BASE_LOAD_AMPLITUDE_MW,
    BENCHMARK_BASE_LOAD_CENTER_MW,
    BENCHMARK_DEMAND_UNITS_PER_H,
    BENCHMARK_MACHINE_POWER_RANGES_MW,
    BENCHMARK_SEED,
    BENCHMARK_TARIFF_BLOCKS,
    BENCHMARK_TASK_COUNT,
    BENCHMARK_TASK_DURATION_RANGE_H,
    BENCHMARK_TASK_YIELD_RANGE_UNITS,
    DEFAULT_HORIZON_H,
    DEFAULT_MAX_BATCHES_PER_MACHINE,
)


def tou_price_vector(horizon_h: int = DEFAULT_HORIZON_H) -> np.ndarray:
    """Return the hourly time-of-use tariff for the requested horizon."""
    prices = np.zeros(horizon_h)
    for price, blocks in BENCHMARK_TARIFF_BLOCKS:
        for start, end in blocks:
            prices[start:min(end, horizon_h)] = price
    return prices


def make_benchmark_instance(
    seed: int = BENCHMARK_SEED,
    n_tasks: int = BENCHMARK_TASK_COUNT,
    max_batches: int = DEFAULT_MAX_BATCHES_PER_MACHINE,
    with_mt: bool = True,
) -> Instance:
    """Build the five-furnace, 24-hour reference instance."""
    rng = np.random.default_rng(seed)
    machines = list(BENCHMARK_MACHINE_POWER_RANGES_MW)
    tasks = [f"p{index + 1}" for index in range(n_tasks)]
    power = np.array([
        np.round(rng.uniform(*BENCHMARK_MACHINE_POWER_RANGES_MW[machine], n_tasks), 1)
        for machine in machines
    ])
    duration = np.round(
        rng.uniform(*BENCHMARK_TASK_DURATION_RANGE_H, (len(machines), n_tasks)), 1
    )
    yields = rng.integers(*BENCHMARK_TASK_YIELD_RANGE_UNITS, n_tasks).astype(float)
    hours = np.arange(DEFAULT_HORIZON_H)
    base_load = np.round(
        BENCHMARK_BASE_LOAD_CENTER_MW
        + BENCHMARK_BASE_LOAD_AMPLITUDE_MW
        * np.sin(2 * np.pi * (hours - 6) / DEFAULT_HORIZON_H),
        2,
    )
    instance = Instance(
        machines=machines,
        tasks=tasks,
        power_mw=power,
        duration_h=duration,
        yield_units=yields,
        base_load_mw=base_load,
        price_eur_mwh=tou_price_vector(),
        demand_units=np.full(DEFAULT_HORIZON_H, BENCHMARK_DEMAND_UNITS_PER_H),
        max_batches_per_machine=max_batches,
        mt=Microturbine() if with_mt else None,
    )
    return instance.validate()