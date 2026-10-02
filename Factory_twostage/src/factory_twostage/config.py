"""JSON-configurable benchmark and solver run settings."""

import json
import math
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from .data import BESS, Instance, Microturbine, SchedulerConfig, SolverSettings
from .exceptions import InstanceValidationError
from .parameters import (
    BENCHMARK_SEED,
    BENCHMARK_TASK_COUNT,
    DEFAULT_BESS,
    DEFAULT_MAX_BATCHES_PER_MACHINE,
    DEFAULT_MT,
    DEFAULT_START_STEP_H,
    DEFAULT_STRICT_VERIFICATION,
    DEFAULT_UNIQUE_TASKS,
)


@dataclass
class RunConfig:
    seed: int = BENCHMARK_SEED
    n_tasks: int = BENCHMARK_TASK_COUNT
    max_batches: int = DEFAULT_MAX_BATCHES_PER_MACHINE
    with_mt: bool = True
    with_bess: bool = True
    start_step_h: float = DEFAULT_START_STEP_H
    unique_tasks: bool = DEFAULT_UNIQUE_TASKS
    strict_verification: bool = DEFAULT_STRICT_VERIFICATION
    id_cap_buy_mw: float = 100.0
    id_cap_sell_mw: float = 100.0
    scenario_count: int = 10
    scenario_seed: int = 7
    price_sigma: float = 0.15
    level_sigma: float = 0.05
    load_sigma_mw: float = 0.0
    solver: SolverSettings = field(default_factory=SolverSettings)
    microturbine: dict[str, Any] = field(
        default_factory=lambda: {
            **DEFAULT_MT,
            "block_width_mw": list(DEFAULT_MT["block_width_mw"]),
            "block_cost_eur_mwh": list(DEFAULT_MT["block_cost_eur_mwh"]),
        }
    )
    bess: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BESS))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunConfig":
        try:
            benchmark = values.get("benchmark", {})
            scheduler = values.get("scheduler", {})
            intraday = values.get("intraday", {})
            allowed = {"benchmark", "scheduler", "solver", "microturbine", "bess", "intraday"}
            unknown = set(values) - allowed
            if unknown:
                raise InstanceValidationError(f"unknown config sections: {', '.join(sorted(unknown))}")
            for name, section in (
                ("benchmark", benchmark),
                ("scheduler", scheduler),
                ("solver", values.get("solver", {})),
                ("microturbine", values.get("microturbine", {})),
                ("bess", values.get("bess", {})),
                ("intraday", intraday),
            ):
                if not isinstance(section, dict):
                    raise InstanceValidationError(f"config section '{name}' must be an object")
            solver = SolverSettings(**values.get("solver", {}))
            mt = dict(DEFAULT_MT)
            mt.update(values.get("microturbine", {}))
            mt["block_width_mw"] = list(mt["block_width_mw"])
            mt["block_cost_eur_mwh"] = list(mt["block_cost_eur_mwh"])
            bess = dict(DEFAULT_BESS)
            bess.update(values.get("bess", {}))
            intraday_defaults = {
                "cap_buy_mw": 100.0,
                "cap_sell_mw": 100.0,
                "scenarios": 10,
                "scenario_seed": 7,
                "price_sigma": 0.15,
                "level_sigma": 0.05,
                "load_sigma_mw": 0.0,
            }
            intraday_defaults.update(intraday)
            market_caps = (intraday_defaults["cap_buy_mw"], intraday_defaults["cap_sell_mw"])
            if any(not math.isfinite(value) or value < 0 for value in market_caps):
                raise InstanceValidationError("intraday market caps must be finite and >= 0")
            if (
                isinstance(intraday_defaults["scenarios"], bool)
                or not isinstance(intraday_defaults["scenarios"], int)
                or intraday_defaults["scenarios"] < 1
            ):
                raise InstanceValidationError("intraday scenario count must be >= 1")
            volatility_names = ("price_sigma", "level_sigma", "load_sigma_mw")
            if any(
                not math.isfinite(intraday_defaults[name]) or intraday_defaults[name] < 0
                for name in volatility_names
            ):
                raise InstanceValidationError("intraday volatilities must be finite and >= 0")
            Microturbine(**mt).validate()
            BESS(**bess).validate()
            return cls(
                seed=benchmark.get("seed", BENCHMARK_SEED),
                n_tasks=benchmark.get("n_tasks", BENCHMARK_TASK_COUNT),
                max_batches=benchmark.get("max_batches", DEFAULT_MAX_BATCHES_PER_MACHINE),
                with_mt=benchmark.get("with_mt", True),
                with_bess=benchmark.get("with_bess", True),
                start_step_h=scheduler.get("start_step_h", DEFAULT_START_STEP_H),
                unique_tasks=scheduler.get("unique_tasks", DEFAULT_UNIQUE_TASKS),
                strict_verification=scheduler.get("strict_verification", DEFAULT_STRICT_VERIFICATION),
                id_cap_buy_mw=intraday_defaults["cap_buy_mw"],
                id_cap_sell_mw=intraday_defaults["cap_sell_mw"],
                scenario_count=intraday_defaults["scenarios"],
                scenario_seed=intraday_defaults["scenario_seed"],
                price_sigma=intraday_defaults["price_sigma"],
                level_sigma=intraday_defaults["level_sigma"],
                load_sigma_mw=intraday_defaults["load_sigma_mw"],
                solver=solver,
                microturbine=mt,
                bess=bess,
            )
        except (TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad run config: {err}") from err

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read config '{path}': {err}") from err
        if not isinstance(values, dict):
            raise InstanceValidationError("run config root must be a JSON object")
        return cls.from_dict(values)

    @classmethod
    def bundled(cls) -> "RunConfig":
        """Load the default JSON configuration shipped with the package."""
        path = files("factory_twostage").joinpath("configs/default.json")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": {
                "seed": self.seed,
                "n_tasks": self.n_tasks,
                "max_batches": self.max_batches,
                "with_mt": self.with_mt,
                "with_bess": self.with_bess,
            },
            "scheduler": {
                "start_step_h": self.start_step_h,
                "unique_tasks": self.unique_tasks,
                "strict_verification": self.strict_verification,
            },
            "solver": asdict(self.solver),
            "microturbine": self.microturbine,
            "bess": self.bess,
            "intraday": {
                "cap_buy_mw": self.id_cap_buy_mw,
                "cap_sell_mw": self.id_cap_sell_mw,
                "scenarios": self.scenario_count,
                "scenario_seed": self.scenario_seed,
                "price_sigma": self.price_sigma,
                "level_sigma": self.level_sigma,
                "load_sigma_mw": self.load_sigma_mw,
            },
        }

    def make_instance(self) -> Instance:
        from .benchmark import make_benchmark_instance

        mt = Microturbine(**self.microturbine).validate() if self.with_mt else None
        bess = BESS(**self.bess).validate() if self.with_bess else None
        return make_benchmark_instance(
            seed=self.seed,
            n_tasks=self.n_tasks,
            max_batches=self.max_batches,
            with_mt=self.with_mt,
            with_bess=self.with_bess,
            mt_parameters=asdict(mt) if mt else None,
            bess_parameters=asdict(bess) if bess else None,
        )

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            start_step_h=self.start_step_h,
            unique_tasks=self.unique_tasks,
            solver=self.solver,
            strict_verification=self.strict_verification,
        )
