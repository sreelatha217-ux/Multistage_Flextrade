"""Synthetic benchmark instance and time-of-use tariff generation."""

import numpy as np

from .data import BESS, Instance, Microturbine
from .parameters import (
    BENCHMARK_BASE_LOAD_AMPLITUDE_MW,
    BENCHMARK_BASE_LOAD_CENTER_MW,
    BENCHMARK_DEMAND_UNITS_PER_H,
    BENCHMARK_MACHINE_POWER_RANGES_MW,
    BENCHMARK_SEED,
    BENCHMARK_TASK_COUNT,
    BENCHMARK_TASK_DURATION_RANGE_H,
    BENCHMARK_TASK_YIELD_RANGE_UNITS,
    BENCHMARK_TARIFF_BLOCKS,
    DEFAULT_BESS,
    DEFAULT_MAX_BATCHES_PER_MACHINE,
    DEFAULT_MT,
)


def tou_price_vectors(horizon_h: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Return buy and sell tariffs per hour in EUR/MWh."""
    buy = np.zeros(horizon_h)
    sell = np.zeros(horizon_h)
    for buy_rate, sell_rate, periods in BENCHMARK_TARIFF_BLOCKS:
        for start, end in periods:
            for hour in range(start, min(end, horizon_h)):
                buy[hour], sell[hour] = buy_rate, sell_rate
    return buy, sell


def make_benchmark_instance(
    seed: int = BENCHMARK_SEED,
    n_tasks: int = BENCHMARK_TASK_COUNT,
    max_batches: int = DEFAULT_MAX_BATCHES_PER_MACHINE,
    with_mt: bool = True,
    with_bess: bool = True,
    mt_parameters: dict | None = None,
    bess_parameters: dict | None = None,
) -> Instance:
    """Create the documented 24-hour, five-machine factory benchmark."""
    rng = np.random.default_rng(seed)
    machines = list(BENCHMARK_MACHINE_POWER_RANGES_MW)
    tasks = [f"p{index + 1}" for index in range(n_tasks)]
    power = np.array([
        np.round(rng.uniform(*BENCHMARK_MACHINE_POWER_RANGES_MW[machine], n_tasks), 1)
        for machine in machines
    ])
    duration = np.round(rng.uniform(*BENCHMARK_TASK_DURATION_RANGE_H, (len(machines), n_tasks)), 1)
    yields = rng.integers(*BENCHMARK_TASK_YIELD_RANGE_UNITS, n_tasks).astype(float)
    hours = np.arange(24)
    base_load = np.round(
        BENCHMARK_BASE_LOAD_CENTER_MW
        + BENCHMARK_BASE_LOAD_AMPLITUDE_MW * np.sin(2 * np.pi * (hours - 6) / 24),
        2,
    )
    price_buy, price_sell = tou_price_vectors(24)
    mt_values = dict(DEFAULT_MT)
    mt_values.update(mt_parameters or {})
    bess_values = dict(DEFAULT_BESS)
    bess_values.update(bess_parameters or {})
    return Instance(
        machines=machines,
        tasks=tasks,
        power_mw=power,
        duration_h=duration,
        yield_units=yields,
        base_load_mw=base_load,
        price_buy_eur_mwh=price_buy,
        price_sell_eur_mwh=price_sell,
        demand_units=np.full(24, BENCHMARK_DEMAND_UNITS_PER_H),
        max_batches_per_machine=max_batches,
        mt=Microturbine(**mt_values) if with_mt else None,
        bess=BESS(**bess_values) if with_bess else None,
    ).validate()
