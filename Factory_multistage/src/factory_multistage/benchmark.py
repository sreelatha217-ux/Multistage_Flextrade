from __future__ import annotations

from typing import Tuple

import numpy as np

from .data import BESS, Instance, Microturbine

_POWER_RANGES = {"m1": (70, 95), "m2": (60, 80), "m3": (70, 90), "m4": (90, 110), "m5": (65, 85)}


_TOU = [  # (buy EUR/MWh, sell EUR/MWh, [(start_h, end_h), ...])
    (66.42, 36.53, [(2, 4), (12, 14), (18, 20), (22, 24)]),     # off-peak
    (88.56, 48.71, [(4, 6)]),                                   # flat
    (118.08, 64.94, [(0, 2), (6, 8), (14, 16), (20, 22)]),      # mid
    (177.12, 97.42, [(8, 12), (16, 18)]),                       # on-peak
]


def tou_price_vectors(horizon_h: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """(buy tariff, sell tariff) per hour in EUR/MWh."""
    buy, sell = np.zeros(horizon_h), np.zeros(horizon_h)
    for pb, ps, blocks in _TOU:
        for a, b in blocks:
            for h in range(a, min(b, horizon_h)):
                buy[h], sell[h] = pb, ps
    return buy, sell


def make_benchmark_instance(seed: int = 2024, n_tasks: int = 30, max_batches: int = 10,
                            with_mt: bool = True, with_bess: bool = True) -> Instance:
    """5 furnaces, 30 tasks, TOU buy/sell tariff, 8 units/h demand, N0=50, Nmax=500, Q_md=400 MW (both ways), MT 10-25 MW, BESS 40 MW / 200 MWh."""
    rng = np.random.default_rng(seed)
    machines = list(_POWER_RANGES)
    tasks = [f"p{i + 1}" for i in range(n_tasks)]
    power = np.array([np.round(rng.uniform(*_POWER_RANGES[m], n_tasks), 1) for m in machines])
    dur = np.round(rng.uniform(1.3, 4.9, (len(machines), n_tasks)), 1)
    yld = rng.integers(5, 16, n_tasks).astype(float)
    t = np.arange(24)
    base = np.round(3.5 + 1.0 * np.sin(2 * np.pi * (t - 6) / 24), 2)   # within [2.5, 4.5]
    buy, sell = tou_price_vectors(24)
    return Instance(machines, tasks, power, dur, yld, base, buy, sell, np.full(24, 8.0),
                    max_batches_per_machine=max_batches, mt=Microturbine() if with_mt else None,
                    bess=BESS() if with_bess else None).validate()
