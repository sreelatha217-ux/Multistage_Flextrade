"""
End-to-end workflow driven by an :class:`~prosumer_opt.config.AppConfig`:

    config -> factory + scenario tree -> optimizer -> day-ahead solve -> (intraday re-solve) -> save

This is the programmatic counterpart of the command line interface::

    from prosumer_opt import AppConfig, run_pipeline
    result = run_pipeline(AppConfig.from_file("configs/default.yaml"))
    print(result.day_ahead.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .data import build_demo_factory, build_demo_intraday_scenario, build_demo_tree
from .optimizer import MultiStageProsumerOptimizer
from .parameters import FactoryParams
from .results import StageResult
from .scenarios import ScenarioTree
from .state import InitialState, IntradayInputs

log = logging.getLogger("prosumer_opt")


@dataclass
class PipelineResult:
    config: AppConfig
    tree: ScenarioTree
    day_ahead: StageResult
    intraday: Optional[StageResult] = None
    output_dir: Optional[Path] = None


def resolve_factory(cfg: AppConfig) -> FactoryParams:
    """Explicit factory from the config if given, otherwise the seeded demo factory."""
    if cfg.factory is not None:
        return cfg.factory
    return build_demo_factory(cfg.time, cfg.demo_factory)


def build_optimizer(cfg: AppConfig) -> MultiStageProsumerOptimizer:
    return MultiStageProsumerOptimizer(cfg.grid, resolve_factory(cfg), cfg.microturbine, cfg.bess,
                                       cfg.time, cfg.solver)


def run_pipeline(cfg: AppConfig) -> PipelineResult:
    optimizer = build_optimizer(cfg)
    tree = build_demo_tree(cfg.time, cfg.scenarios)

    da = optimizer.solve_day_ahead(tree)
    idr: Optional[StageResult] = None
    if cfg.intraday.enabled:
        t0 = cfg.intraday.step
        scenario = build_demo_intraday_scenario(cfg.time, tree, cfg.intraday, spike_from_step=t0)
        inp = IntradayInputs(t0, InitialState.from_result(da, t0, cfg.time), scenario)
        idr = optimizer.solve_intraday(da, inp)

    out: Optional[Path] = None
    if cfg.output.save:
        out = Path(cfg.output.directory)
        da.save(out)
        if idr is not None:
            idr.save(out)
        if cfg.output.save_config:
            (out / "config_used.json").write_text(cfg.to_json())
        log.info("Results written to %s", out.resolve())
    return PipelineResult(cfg, tree, da, idr, out)
