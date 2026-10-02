"""Validated package configuration for all three market stages."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .data import Instance, SchedulerConfig, SolverSettings
from .exceptions import InstanceValidationError

if TYPE_CHECKING:
    from .realtime_scenarios import BalancingMarket, RealTimeSet
    from .scenarios import IntradayMarket, ScenarioSet


@dataclass
class RunConfig:
    seed: int = 2024
    max_batches: int = 10
    with_mt: bool = True
    with_bess: bool = True
    start_step_h: float = 0.5
    unique_tasks: bool = True
    solver: SolverSettings = field(default_factory=SolverSettings)
    id_cap_buy_mw: float = 100.0
    id_cap_sell_mw: float = 100.0
    scenario_count: int = 10
    scenario_seed: int = 7
    price_sigma: float = 0.15
    level_sigma: float = 0.05
    load_sigma_mw: float = 0.0
    rt_scenario_count: int = 5
    rt_seed: int = 11
    rt_load_sigma: float = 0.03
    rt_load_rho: float = 0.6
    r_premium: float = 0.25
    r_discount: float = 0.25
    r_sigma: float = 0.3
    imbalance_mode: str = "strategic"
    bal_exclusive: bool = False
    max_imbalance_mw: float | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunConfig":
        allowed = {"benchmark", "scheduler", "solver", "intraday", "realtime"}
        unknown = set(values) - allowed
        if unknown:
            raise InstanceValidationError(f"unknown config sections: {', '.join(sorted(unknown))}")
        sections = {}
        for name in allowed:
            section = values.get(name, {})
            if not isinstance(section, dict):
                raise InstanceValidationError(f"config section '{name}' must be an object")
            sections[name] = section

        benchmark, scheduler = sections["benchmark"], sections["scheduler"]
        intraday, realtime = sections["intraday"], sections["realtime"]
        try:
            solver = SolverSettings(**sections["solver"])
            config = cls(
                seed=benchmark.get("seed", 2024),
                max_batches=benchmark.get("max_batches", 10),
                with_mt=benchmark.get("with_mt", True),
                with_bess=benchmark.get("with_bess", True),
                start_step_h=scheduler.get("start_step_h", 0.5),
                unique_tasks=scheduler.get("unique_tasks", True),
                solver=solver,
                id_cap_buy_mw=intraday.get("cap_buy_mw", 100.0),
                id_cap_sell_mw=intraday.get("cap_sell_mw", 100.0),
                scenario_count=intraday.get("scenarios", 10),
                scenario_seed=intraday.get("scenario_seed", 7),
                price_sigma=intraday.get("price_sigma", 0.15),
                level_sigma=intraday.get("level_sigma", 0.05),
                load_sigma_mw=intraday.get("load_sigma_mw", 0.0),
                rt_scenario_count=realtime.get("scenarios", 5),
                rt_seed=realtime.get("seed", 11),
                rt_load_sigma=realtime.get("load_sigma", 0.03),
                rt_load_rho=realtime.get("load_rho", 0.6),
                r_premium=realtime.get("premium", 0.25),
                r_discount=realtime.get("discount", 0.25),
                r_sigma=realtime.get("ratio_sigma", 0.3),
                imbalance_mode=realtime.get("imbalance_mode", "strategic"),
                bal_exclusive=realtime.get("exclusive", False),
                max_imbalance_mw=realtime.get("max_imbalance_mw"),
            )
        except (TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad run config: {err}") from err

        if config.max_batches < 1 or config.scenario_count < 1 or config.rt_scenario_count < 1:
            raise InstanceValidationError("batch and scenario counts must be >= 1")
        if config.start_step_h <= 0 or not math.isfinite(config.start_step_h):
            raise InstanceValidationError("start_step_h must be finite and > 0")
        numeric = (
            config.id_cap_buy_mw,
            config.id_cap_sell_mw,
            config.price_sigma,
            config.level_sigma,
            config.load_sigma_mw,
            config.rt_load_sigma,
            config.rt_load_rho,
            config.r_premium,
            config.r_discount,
            config.r_sigma,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise InstanceValidationError("market caps and scenario parameters must be finite and >= 0")
        if config.rt_load_rho >= 1:
            raise InstanceValidationError("realtime.load_rho must be < 1")
        if config.imbalance_mode not in ("strategic", "passive"):
            raise InstanceValidationError("realtime.imbalance_mode must be 'strategic' or 'passive'")
        if config.max_imbalance_mw is not None and (
            not math.isfinite(config.max_imbalance_mw) or config.max_imbalance_mw < 0
        ):
            raise InstanceValidationError("max_imbalance_mw must be finite and >= 0")
        return config

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read config '{path}': {err}") from err
        if not isinstance(values, dict):
            raise InstanceValidationError("run config root must be an object")
        return cls.from_dict(values)

    @classmethod
    def bundled(cls) -> "RunConfig":
        path = files("factory_multistage").joinpath("configs/default.json")
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(values)

    def cli_defaults(self, stage: str) -> dict[str, Any]:
        defaults = {
            "seed": self.seed,
            "max_batches": self.max_batches,
            "start_step": self.start_step_h,
            "allow_repeat_tasks": not self.unique_tasks,
            "no_mt": not self.with_mt,
            "no_bess": not self.with_bess,
            "solver": self.solver.name,
            "mip_gap": self.solver.mip_gap,
            "time_limit": self.solver.time_limit_s,
            "threads": self.solver.threads,
            "verbose": self.solver.verbose,
        }
        if stage in ("id", "rt"):
            defaults.update(
                id_cap_buy=self.id_cap_buy_mw,
                id_cap_sell=self.id_cap_sell_mw,
                scenarios=self.scenario_count,
                scenario_seed=self.scenario_seed,
                price_sigma=self.price_sigma,
                level_sigma=self.level_sigma,
                load_sigma=self.load_sigma_mw,
            )
        if stage == "rt":
            defaults.update(
                rt_scenarios=self.rt_scenario_count,
                rt_seed=self.rt_seed,
                rt_load_sigma=self.rt_load_sigma,
                rt_load_rho=self.rt_load_rho,
                r_premium=self.r_premium,
                r_discount=self.r_discount,
                r_sigma=self.r_sigma,
                imbalance_mode=self.imbalance_mode,
                bal_exclusive=self.bal_exclusive,
                max_imbalance=self.max_imbalance_mw,
            )
        if stage not in ("da", "id", "rt"):
            raise ValueError(f"unknown scheduler stage: {stage}")
        return defaults

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": {
                "seed": self.seed,
                "max_batches": self.max_batches,
                "with_mt": self.with_mt,
                "with_bess": self.with_bess,
            },
            "scheduler": {"start_step_h": self.start_step_h, "unique_tasks": self.unique_tasks},
            "solver": asdict(self.solver),
            "intraday": {
                "cap_buy_mw": self.id_cap_buy_mw,
                "cap_sell_mw": self.id_cap_sell_mw,
                "scenarios": self.scenario_count,
                "scenario_seed": self.scenario_seed,
                "price_sigma": self.price_sigma,
                "level_sigma": self.level_sigma,
                "load_sigma_mw": self.load_sigma_mw,
            },
            "realtime": {
                "scenarios": self.rt_scenario_count,
                "seed": self.rt_seed,
                "load_sigma": self.rt_load_sigma,
                "load_rho": self.rt_load_rho,
                "premium": self.r_premium,
                "discount": self.r_discount,
                "ratio_sigma": self.r_sigma,
                "imbalance_mode": self.imbalance_mode,
                "exclusive": self.bal_exclusive,
                "max_imbalance_mw": self.max_imbalance_mw,
            },
        }

    def make_instance(self) -> Instance:
        from .benchmark import make_benchmark_instance

        return make_benchmark_instance(
            seed=self.seed,
            max_batches=self.max_batches,
            with_mt=self.with_mt,
            with_bess=self.with_bess,
        )

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            start_step_h=self.start_step_h,
            unique_tasks=self.unique_tasks,
            solver=self.solver,
        )

    def intraday_market(self) -> IntradayMarket:
        from .scenarios import IntradayMarket

        return IntradayMarket(self.id_cap_buy_mw, self.id_cap_sell_mw).validate()

    def generate_scenarios(self, instance: Instance) -> ScenarioSet:
        from .scenarios import ScenarioSet

        return ScenarioSet.generate(
            instance,
            self.scenario_count,
            self.scenario_seed,
            self.price_sigma,
            self.level_sigma,
            self.load_sigma_mw,
        )

    def generate_realtime_set(self, instance: Instance, scenarios: ScenarioSet) -> RealTimeSet:
        from .realtime_scenarios import RealTimeSet

        return RealTimeSet.generate(
            instance,
            scenarios.n,
            self.rt_scenario_count,
            self.rt_seed,
            self.rt_load_sigma,
            self.rt_load_rho,
            self.r_premium,
            self.r_discount,
            self.r_sigma,
        )

    def balancing_market(self) -> BalancingMarket:
        from .realtime_scenarios import BalancingMarket

        return BalancingMarket(
            self.imbalance_mode,
            self.bal_exclusive,
            self.max_imbalance_mw,
            self.max_imbalance_mw,
        ).validate()


def load_config_arg(argv: list[str]) -> RunConfig | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    args, _ = parser.parse_known_args(argv)
    return RunConfig.load(args.config) if args.config else None