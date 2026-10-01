"""
Synthetic input data: a demo factory, a scenario tree around the TOU tariff, and a refreshed
intraday scenario. Replace these with real forecasts and plant data in production; the
optimizer only needs the same container types (``FactoryParams``, ``ScenarioTree``, ``IDScenario``).

Random-number call order is part of the contract: identical seeds give identical data.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import DemoFactorySettings, IntradaySettings, ScenarioSettings
from ..parameters import BatchJob, FactoryParams, TimeGrid
from ..scenarios import IDScenario, RTBranch, ScenarioTree
from .tariffs import tou_prices


def build_demo_factory(tg: TimeGrid, cfg: DemoFactorySettings = DemoFactorySettings()) -> FactoryParams:
    rng = np.random.default_rng(cfg.seed)
    lo_d, hi_d = cfg.duration_range_h
    lo_p, hi_p = cfg.power_range_mw
    jobs = []
    for mi in range(cfg.n_machines):
        for q in range(cfg.jobs_per_machine):
            jobs.append(BatchJob(job_id=f"M{mi + 1}-B{q + 1}", machine=f"M{mi + 1}", sequence=q,
                                 duration_h=float(np.round(rng.uniform(lo_d, hi_d), 1)),
                                 power_mw=float(np.round(rng.uniform(lo_p, hi_p), 1)),
                                 units_out=cfg.units_out))
    hours = np.arange(tg.n_steps) * tg.dt_h
    d0, d1 = cfg.demand_window_h
    demand = np.where((hours >= d0) & (hours < d1), cfg.demand_units_per_h * tg.dt_h, 0.0)
    s0, s1 = cfg.on_shift_window_h
    base = np.where((hours >= s0) & (hours < s1), cfg.base_load_on_shift_mw, cfg.base_load_off_shift_mw)
    return FactoryParams(jobs=jobs, base_load_mw=base, buffer_h=1.0, inventory_init=0.0, inventory_max=1000.0,
                         demand_units=demand, intraday_shift_window_h=cfg.shift_window_h)


def build_demo_tree(tg: TimeGrid, cfg: ScenarioSettings = ScenarioSettings()) -> ScenarioTree:
    """AR(1) day-ahead price shocks, intraday basis noise, real-time load and balancing-price noise."""
    rng = np.random.default_rng(cfg.seed)
    T = tg.n_steps
    base, _ = tou_prices(tg)
    scen = []
    for i in range(cfg.n_id_scenarios):
        e = np.zeros(T)
        for t in range(T):
            e[t] = (cfg.da_ar_coeff * e[t - 1] if t else 0.0) + rng.normal(0, cfg.da_sigma)
        da_buy = base * np.exp(e)
        id_buy = da_buy * np.exp(rng.normal(0, cfg.id_sigma, T))
        rts = []
        for _ in range(cfg.n_rt_branches):
            rts.append(RTBranch(
                prob=1.0 / cfg.n_rt_branches,
                load_dev=rng.normal(0, cfg.load_sigma, T),
                r_minus=cfg.r_minus_base + cfg.r_minus_spread * rng.random(T),
                r_plus=cfg.r_plus_base - cfg.r_plus_spread * rng.random(T)))
        scen.append(IDScenario(name=f"S{i + 1}", prob=1.0 / cfg.n_id_scenarios, da_buy=da_buy,
                               da_sell=cfg.sell_ratio * da_buy, id_buy=id_buy,
                               id_sell=cfg.sell_ratio * id_buy, rt_branches=rts))
    return ScenarioTree(scen)


def build_demo_intraday_scenario(tg: TimeGrid, da_tree: ScenarioTree,
                                 cfg: IntradaySettings = IntradaySettings(),
                                 spike_from_step: Optional[int] = None) -> IDScenario:
    """One refreshed intraday realisation (optionally with a price spike) and fresh RT branches."""
    rng = np.random.default_rng(cfg.seed)
    T = tg.n_steps
    p = np.array([s.prob for s in da_tree.scenarios])
    da = sum(pi * np.asarray(s.da_buy) for pi, s in zip(p, da_tree.scenarios))
    idp = da * np.exp(rng.normal(0, cfg.price_noise_sigma, T))
    if spike_from_step is not None:
        idp[spike_from_step:] *= cfg.spike
    rts = [RTBranch(prob=prob, load_dev=rng.normal(mean, cfg.load_dev_sigma, T), r_minus=rm, r_plus=rp)
           for prob, mean, rm, rp in cfg.rt_branches]
    return IDScenario(name="ID-refresh", prob=1.0, da_buy=da, da_sell=cfg.sell_ratio * da,
                      id_buy=idp, id_sell=cfg.sell_ratio * idp, rt_branches=rts)
