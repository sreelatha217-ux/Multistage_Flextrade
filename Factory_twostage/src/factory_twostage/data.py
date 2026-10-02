"""Validated scheduler inputs, settings, and result types."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._internal import EPS, __version__
from .exceptions import InstanceValidationError
from .parameters import (
    DEFAULT_BESS,
    DEFAULT_BUFFER_H,
    DEFAULT_GRID_LIMIT_MW,
    DEFAULT_GRID_SELL_LIMIT_MW,
    DEFAULT_HORIZON_H,
    DEFAULT_INVENTORY_INIT,
    DEFAULT_INVENTORY_MAX,
    DEFAULT_MAX_BATCHES_PER_MACHINE,
    DEFAULT_MIP_GAP,
    DEFAULT_MT,
    DEFAULT_SOLVER,
    DEFAULT_SOLVER_THREADS,
    DEFAULT_START_STEP_H,
    DEFAULT_STRICT_VERIFICATION,
    DEFAULT_TIME_LIMIT_S,
    DEFAULT_UNIQUE_TASKS,
)

log = logging.getLogger("factory_twostage")


def _arr(value: Any, length: int | None, name: str, ndim: int = 1) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as err:
        raise InstanceValidationError(f"{name}: not numeric ({err})") from err
    if ndim == 1 and array.ndim == 0 and length is not None:
        array = np.full(length, float(array))
    if array.ndim != ndim:
        raise InstanceValidationError(f"{name}: expected {ndim}-D array, got shape {array.shape}")
    if length is not None and ndim == 1 and array.shape[0] != length:
        raise InstanceValidationError(f"{name}: expected length {length}, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise InstanceValidationError(f"{name}: contains NaN/inf")
    return array


@dataclass
class Microturbine:
    """Microturbine operating limits, fuel curve, and commitment costs."""

    p_min_mw: float = DEFAULT_MT["p_min_mw"]
    p_max_mw: float = DEFAULT_MT["p_max_mw"]
    ramp_up_mw_h: float = DEFAULT_MT["ramp_up_mw_h"]
    ramp_down_mw_h: float = DEFAULT_MT["ramp_down_mw_h"]
    startup_ramp_mw_h: float = DEFAULT_MT["startup_ramp_mw_h"]
    shutdown_ramp_mw_h: float = DEFAULT_MT["shutdown_ramp_mw_h"]
    startup_cost_eur: float = DEFAULT_MT["startup_cost_eur"]
    shutdown_cost_eur: float = DEFAULT_MT["shutdown_cost_eur"]
    min_up_h: int = DEFAULT_MT["min_up_h"]
    min_down_h: int = DEFAULT_MT["min_down_h"]
    block_width_mw: list[float] = field(default_factory=lambda: list(DEFAULT_MT["block_width_mw"]))
    block_cost_eur_mwh: list[float] = field(default_factory=lambda: list(DEFAULT_MT["block_cost_eur_mwh"]))
    base_cost_eur_mwh: float = DEFAULT_MT["base_cost_eur_mwh"]
    initial_on: bool = DEFAULT_MT["initial_on"]
    initial_power_mw: float = DEFAULT_MT["initial_power_mw"]

    def validate(self) -> "Microturbine":
        widths = _arr(self.block_width_mw, None, "mt.block_width_mw")
        costs = _arr(self.block_cost_eur_mwh, None, "mt.block_cost_eur_mwh")
        if widths.size == 0 or costs.shape != widths.shape:
            raise InstanceValidationError("mt blocks: width and cost must be equal-length, non-empty 1-D lists")
        if np.any(widths <= 0) or np.any(costs < 0):
            raise InstanceValidationError("mt blocks: widths must be > 0 and costs >= 0")
        if np.any(np.diff(costs) < -EPS):
            raise InstanceValidationError("mt block marginal costs must be non-decreasing")
        if not (0 <= self.p_min_mw <= self.p_max_mw) or self.p_max_mw <= 0:
            raise InstanceValidationError("mt: need 0 <= p_min <= p_max and p_max > 0")
        for name in (
            "ramp_up_mw_h", "ramp_down_mw_h", "startup_ramp_mw_h", "shutdown_ramp_mw_h",
            "startup_cost_eur", "shutdown_cost_eur", "base_cost_eur_mwh",
        ):
            if getattr(self, name) < 0:
                raise InstanceValidationError(f"mt.{name} must be >= 0")
        if self.min_up_h < 1 or self.min_down_h < 1:
            raise InstanceValidationError("mt min_up_h / min_down_h must be >= 1")
        if self.initial_on and not (self.p_min_mw - EPS <= self.initial_power_mw <= self.p_max_mw + EPS):
            raise InstanceValidationError("mt.initial_power_mw must lie in [p_min, p_max] when initial_on")
        if abs(float(widths.sum()) - (self.p_max_mw - self.p_min_mw)) > 1e-6:
            raise InstanceValidationError("mt block widths must sum to p_max_mw - p_min_mw")
        return self


@dataclass
class BESS:
    """Battery energy-storage parameters and operating constraints."""

    p_max_mw: float = DEFAULT_BESS["p_max_mw"]
    e_max_mwh: float = DEFAULT_BESS["e_max_mwh"]
    soc_min_frac: float = DEFAULT_BESS["soc_min_frac"]
    soc_max_frac: float = DEFAULT_BESS["soc_max_frac"]
    soc_init_mwh: float = DEFAULT_BESS["soc_init_mwh"]
    eta_ch: float = DEFAULT_BESS["eta_ch"]
    eta_dis: float = DEFAULT_BESS["eta_dis"]
    degradation_eur_mwh: float = DEFAULT_BESS["degradation_eur_mwh"]
    degradation_basis: str = DEFAULT_BESS["degradation_basis"]
    enforce_exclusive: bool = DEFAULT_BESS["enforce_exclusive"]

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.e_max_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.e_max_mwh

    @property
    def round_trip(self) -> float:
        return self.eta_ch * self.eta_dis

    def validate(self) -> "BESS":
        if self.p_max_mw <= 0 or self.e_max_mwh <= 0:
            raise InstanceValidationError("bess: p_max_mw and e_max_mwh must be > 0")
        if not (0 <= self.soc_min_frac < self.soc_max_frac <= 1):
            raise InstanceValidationError("bess: need 0 <= soc_min_frac < soc_max_frac <= 1")
        if not (self.soc_min_mwh - EPS <= self.soc_init_mwh <= self.soc_max_mwh + EPS):
            raise InstanceValidationError(
                f"bess: soc_init_mwh must lie in [{self.soc_min_mwh:.1f}, {self.soc_max_mwh:.1f}] MWh"
            )
        if not (0 < self.eta_ch <= 1 and 0 < self.eta_dis <= 1):
            raise InstanceValidationError("bess: efficiencies must lie in (0, 1]")
        if self.degradation_eur_mwh < 0:
            raise InstanceValidationError("bess: degradation cost must be >= 0")
        if self.degradation_basis not in ("throughput", "discharge"):
            raise InstanceValidationError("bess: degradation_basis must be 'throughput' or 'discharge'")
        return self


@dataclass
class Instance:
    """One factory day-ahead instance, including optional MT and BESS assets."""

    machines: list[str]
    tasks: list[str]
    power_mw: np.ndarray
    duration_h: np.ndarray
    yield_units: np.ndarray
    base_load_mw: np.ndarray
    price_buy_eur_mwh: np.ndarray
    price_sell_eur_mwh: np.ndarray
    demand_units: np.ndarray
    horizon_h: int = DEFAULT_HORIZON_H
    buffer_h: float = DEFAULT_BUFFER_H
    grid_limit_mw: float = DEFAULT_GRID_LIMIT_MW
    grid_sell_limit_mw: float = DEFAULT_GRID_SELL_LIMIT_MW
    max_batches_per_machine: int = DEFAULT_MAX_BATCHES_PER_MACHINE
    inventory_init: float = DEFAULT_INVENTORY_INIT
    inventory_max: float = DEFAULT_INVENTORY_MAX
    mt: Microturbine | None = field(default_factory=Microturbine)
    bess: BESS | None = field(default_factory=BESS)

    def validate(self) -> "Instance":
        hours, machine_count, task_count = self.horizon_h, len(self.machines), len(self.tasks)
        if hours <= 0 or machine_count <= 0 or task_count <= 0:
            raise InstanceValidationError("horizon, machines, and tasks must be non-empty")
        if len(set(self.machines)) != machine_count or len(set(self.tasks)) != task_count:
            raise InstanceValidationError("machine and task names must be unique")
        self.power_mw = _arr(self.power_mw, None, "power_mw", ndim=2)
        self.duration_h = _arr(self.duration_h, None, "duration_h", ndim=2)
        self.yield_units = _arr(self.yield_units, task_count, "yield_units")
        self.base_load_mw = _arr(self.base_load_mw, hours, "base_load_mw")
        self.price_buy_eur_mwh = _arr(self.price_buy_eur_mwh, hours, "price_buy_eur_mwh")
        self.price_sell_eur_mwh = _arr(self.price_sell_eur_mwh, hours, "price_sell_eur_mwh")
        self.demand_units = _arr(self.demand_units, hours, "demand_units")
        for name, array in (("power_mw", self.power_mw), ("duration_h", self.duration_h)):
            if array.shape != (machine_count, task_count):
                raise InstanceValidationError(
                    f"{name}: expected shape {(machine_count, task_count)}, got {array.shape}"
                )
        if np.any(self.power_mw < 0) or np.any(self.duration_h <= 0) or np.any(self.duration_h > hours):
            raise InstanceValidationError("power must be >= 0 and duration must lie in (0, horizon]")
        if np.any(self.yield_units < 0) or np.any(self.demand_units < 0) or np.any(self.base_load_mw < 0):
            raise InstanceValidationError("yield, demand, and base load must be >= 0")
        if np.any(self.price_buy_eur_mwh < 0) or np.any(self.price_sell_eur_mwh < 0):
            raise InstanceValidationError("negative day-ahead prices are not supported")
        if np.any(self.price_sell_eur_mwh > self.price_buy_eur_mwh + EPS):
            raise InstanceValidationError("sell tariff must not exceed buy tariff")
        if self.buffer_h < 0 or self.max_batches_per_machine < 1:
            raise InstanceValidationError("buffer must be >= 0 and max_batches_per_machine must be >= 1")
        if self.grid_limit_mw <= 0 or self.grid_sell_limit_mw <= 0:
            raise InstanceValidationError("grid import and export limits must be positive")
        if not (0 <= self.inventory_init <= self.inventory_max):
            raise InstanceValidationError("inventory_init must lie in [0, inventory_max]")
        mt_capacity = self.mt.p_max_mw if self.mt else 0.0
        if np.any(self.base_load_mw > self.grid_limit_mw + mt_capacity):
            raise InstanceValidationError("base load exceeds import limit plus MT capacity")
        if self.demand_units.sum() > self.yield_units.sum() + self.inventory_init:
            raise InstanceValidationError("daily demand exceeds initial stock plus total task yield")
        if self.mt is not None:
            self.mt.validate()
        if self.bess is not None:
            self.bess.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "machines": self.machines,
            "tasks": self.tasks,
            "power_mw": self.power_mw.tolist(),
            "duration_h": self.duration_h.tolist(),
            "yield_units": self.yield_units.tolist(),
            "base_load_mw": self.base_load_mw.tolist(),
            "price_buy_eur_mwh": self.price_buy_eur_mwh.tolist(),
            "price_sell_eur_mwh": self.price_sell_eur_mwh.tolist(),
            "demand_units": self.demand_units.tolist(),
            "horizon_h": self.horizon_h,
            "buffer_h": self.buffer_h,
            "grid_limit_mw": self.grid_limit_mw,
            "grid_sell_limit_mw": self.grid_sell_limit_mw,
            "max_batches_per_machine": self.max_batches_per_machine,
            "inventory_init": self.inventory_init,
            "inventory_max": self.inventory_max,
            "mt": asdict(self.mt) if self.mt else None,
            "bess": asdict(self.bess) if self.bess else None,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Instance":
        try:
            data = dict(values)
            if "price_eur_mwh" in data:
                legacy = np.asarray(data.pop("price_eur_mwh"), dtype=float)
                data.setdefault("price_buy_eur_mwh", legacy.tolist())
                data.setdefault("price_sell_eur_mwh", (0.55 * legacy).tolist())
                log.warning("legacy single tariff found; using 55%% as sell tariff")
            if "mt" in data:
                data["mt"] = Microturbine(**data["mt"]) if data["mt"] is not None else None
            if "bess" in data:
                data["bess"] = BESS(**data["bess"]) if data["bess"] is not None else None
            return cls(**data).validate()
        except (TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad instance schema: {err}") from err

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read instance '{path}': {err}") from err

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class SolverSettings:
    name: str = DEFAULT_SOLVER
    mip_gap: float = DEFAULT_MIP_GAP
    time_limit_s: float = DEFAULT_TIME_LIMIT_S
    threads: int | None = DEFAULT_SOLVER_THREADS
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.mip_gap < 1 or self.time_limit_s <= 0:
            raise InstanceValidationError("mip_gap must be in [0, 1) and time_limit_s > 0")
        if self.threads is not None and self.threads < 1:
            raise InstanceValidationError("threads must be >= 1")


@dataclass
class SchedulerConfig:
    start_step_h: float = DEFAULT_START_STEP_H
    unique_tasks: bool = DEFAULT_UNIQUE_TASKS
    solver: SolverSettings = field(default_factory=SolverSettings)
    strict_verification: bool = DEFAULT_STRICT_VERIFICATION

    def check(self, inst: Instance) -> None:
        if self.start_step_h <= 0:
            raise InstanceValidationError("start_step_h must be positive")
        ratio = inst.horizon_h / self.start_step_h
        if abs(ratio - round(ratio)) > EPS:
            raise InstanceValidationError("horizon_h must be an integer multiple of start_step_h")


@dataclass
class SchedulingResult:
    status: str
    objective_eur: float
    mip_gap: float | None
    solve_time_s: float
    model_stats: dict[str, int]
    jobs: pd.DataFrame
    hourly: pd.DataFrame
    kpis: dict[str, float]
    verification: dict[str, object]

    def summary(self) -> str:
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        k = self.kpis
        return (
            f"status={self.status}  cost={self.objective_eur:,.2f} EUR  gap={gap}  "
            f"time={self.solve_time_s:.1f}s\n"
            f"cost split: grid={k['grid_cost_eur']:,.2f}  MT fuel={k['mt_fuel_cost_eur']:,.2f}  "
            f"MT start/stop={k['mt_startstop_cost_eur']:,.2f} EUR\n"
            f"batches={int(k['n_batches'])}  units={k['units_produced']:.0f}  "
            f"total_load={k['total_load_mwh']:.1f} MWh  grid buy={k['grid_buy_energy_mwh']:.1f} MWh  "
            f"grid sell={k['grid_sell_energy_mwh']:.1f} MWh\n"
            f"MT={k['mt_energy_mwh']:.1f} MWh; BESS charge={k['bess_charge_mwh']:.1f} MWh, "
            f"discharge={k['bess_discharge_mwh']:.1f} MWh, final SoC={k['bess_final_soc_mwh']:.1f} MWh\n"
            f"average cost={k['avg_cost_per_mwh']:.2f} EUR/MWh  cost/unit={k['cost_per_unit']:.2f} EUR\n"
            f"verification={'PASS' if self.verification['passed'] else 'FAIL'}"
        )

    def save(self, outdir: str | Path) -> Path:
        output = Path(outdir)
        output.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(output / "jobs.csv", index=False)
        self.hourly.to_csv(output / "hourly.csv", index=False)
        metadata = {
            "version": __version__,
            "status": self.status,
            "objective_eur": self.objective_eur,
            "mip_gap": self.mip_gap,
            "solve_time_s": self.solve_time_s,
            "model_stats": self.model_stats,
            "kpis": self.kpis,
            "verification": self.verification,
        }
        (output / "summary.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
        return output
