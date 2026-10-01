"""Validated scheduler inputs, settings, and result types."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._internal import __version__
from .exceptions import InstanceValidationError


def _arr(x, n: Optional[int], name: str, ndim: int = 1) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        raise InstanceValidationError(f"{name}: not numeric ({err})") from err
    if ndim == 1 and a.ndim == 0 and n is not None:
        a = np.full(n, float(a))
    if a.ndim != ndim:
        raise InstanceValidationError(f"{name}: expected {ndim}-D array, got shape {a.shape}")
    if n is not None and ndim == 1 and a.shape[0] != n:
        raise InstanceValidationError(f"{name}: expected length {n}, got {a.shape[0]}")
    if not np.all(np.isfinite(a)):
        raise InstanceValidationError(f"{name}: contains NaN/inf")
    return a

@dataclass
class Instance:
    machines: List[str]
    tasks: List[str]
    power_mw: np.ndarray            # [M, P]
    duration_h: np.ndarray          # [M, P]
    yield_units: np.ndarray         # [P]
    base_load_mw: np.ndarray        # [T]
    price_eur_mwh: np.ndarray       # [T]
    demand_units: np.ndarray        # [T]
    horizon_h: int = 24
    buffer_h: float = 1.0
    grid_limit_mw: float = 400.0
    max_batches_per_machine: int = 10
    inventory_init: float = 50.0
    inventory_max: float = 500.0

    # ------------------------------------------------------------------ checks
    def validate(self) -> "Instance":
        T, M, P = self.horizon_h, len(self.machines), len(self.tasks)
        if T <= 0 or M <= 0 or P <= 0:
            raise InstanceValidationError("horizon, machines and tasks must be non-empty")
        if len(set(self.machines)) != M or len(set(self.tasks)) != P:
            raise InstanceValidationError("machine and task names must be unique")
        self.power_mw = _arr(self.power_mw, None, "power_mw", 2)
        self.duration_h = _arr(self.duration_h, None, "duration_h", 2)
        self.yield_units = _arr(self.yield_units, P, "yield_units")
        self.base_load_mw = _arr(self.base_load_mw, T, "base_load_mw")
        self.price_eur_mwh = _arr(self.price_eur_mwh, T, "price_eur_mwh")
        self.demand_units = _arr(self.demand_units, T, "demand_units")
        for nm, a in (("power_mw", self.power_mw), ("duration_h", self.duration_h)):
            if a.shape != (M, P):
                raise InstanceValidationError(f"{nm}: expected shape {(M, P)}, got {a.shape}")
        if np.any(self.power_mw < 0) or np.any(self.duration_h <= 0):
            raise InstanceValidationError("power must be >= 0 and duration > 0")
        if np.any(self.duration_h > T):
            raise InstanceValidationError("a task duration exceeds the horizon")
        if np.any(self.yield_units < 0) or np.any(self.demand_units < 0) or np.any(self.base_load_mw < 0):
            raise InstanceValidationError("yield, demand and base load must be >= 0")
        if self.buffer_h < 0:
            raise InstanceValidationError("buffer_h must be >= 0")
        if self.max_batches_per_machine < 1:
            raise InstanceValidationError("max_batches_per_machine must be >= 1")
        if not (0 <= self.inventory_init <= self.inventory_max):
            raise InstanceValidationError("inventory_init must lie in [0, inventory_max]")
        if self.grid_limit_mw <= 0:
            raise InstanceValidationError("grid_limit_mw must be positive")
        if np.any(self.base_load_mw > self.grid_limit_mw):
            raise InstanceValidationError("base load alone exceeds the substation limit")
        # Quick necessary condition: even running everything cannot be required to exceed the best yield.
        if self.demand_units.sum() > self.inventory_max * 0 + self.yield_units.sum() * 1.0 + self.inventory_init:
            raise InstanceValidationError(
                f"daily demand {self.demand_units.sum():.0f} exceeds initial stock plus total yield of all tasks "
                f"({self.inventory_init + self.yield_units.sum():.0f}); the instance cannot be feasible")
        return self

    # ---------------------------------------------------------------- JSON I/O
    def to_dict(self) -> dict:
        return dict(machines=self.machines, tasks=self.tasks, power_mw=self.power_mw.tolist(),
                    duration_h=self.duration_h.tolist(), yield_units=self.yield_units.tolist(),
                    base_load_mw=self.base_load_mw.tolist(), price_eur_mwh=self.price_eur_mwh.tolist(),
                    demand_units=self.demand_units.tolist(), horizon_h=self.horizon_h, buffer_h=self.buffer_h,
                    grid_limit_mw=self.grid_limit_mw, max_batches_per_machine=self.max_batches_per_machine,
                    inventory_init=self.inventory_init, inventory_max=self.inventory_max)

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        try:
            return cls(**d).validate()
        except TypeError as err:
            raise InstanceValidationError(f"bad instance schema: {err}") from err

    @classmethod
    def load(cls, path) -> "Instance":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read instance '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

@dataclass
class SolverSettings:
    name: str = "appsi_highs"
    mip_gap: float = 1e-4            # 0.01 % as recommended for the reference model
    time_limit_s: float = 180.0
    threads: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        if not (0 <= self.mip_gap < 1) or self.time_limit_s <= 0:
            raise InstanceValidationError("mip_gap must be in [0,1) and time_limit_s > 0")

@dataclass
class SchedulerConfig:
    start_step_h: float = 0.5
    unique_tasks: bool = True
    solver: SolverSettings = field(default_factory=SolverSettings)
    strict_verification: bool = True

    def check(self, inst: Instance) -> None:
        r = inst.horizon_h / self.start_step_h
        if self.start_step_h <= 0 or abs(r - round(r)) > 1e-9:
            raise InstanceValidationError("horizon_h must be an integer multiple of start_step_h")

@dataclass
class SchedulingResult:
    status: str
    objective_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    jobs: pd.DataFrame
    hourly: pd.DataFrame
    kpis: Dict[str, float]
    verification: Dict[str, object]

    def summary(self) -> str:
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        k = self.kpis
        return (f"status={self.status}  cost={self.objective_eur:,.2f} EUR  gap={gap}  time={self.solve_time_s:.1f}s\n"
                f"batches={int(k['n_batches'])}  units={k['units_produced']:.0f}  grid_energy={k['grid_energy_mwh']:.1f} MWh  "
                f"avg_price={k['avg_price_paid']:.2f} EUR/MWh  peak={k['peak_mw']:.1f} MW  "
                f"cost/unit={k['cost_per_unit']:.2f} EUR\n"
                f"ASAP baseline={k['baseline_asap_cost_eur']:,.2f} EUR  saving={k['saving_vs_asap_pct']:.1f}%  "
                f"verification={'PASS' if self.verification['passed'] else 'FAIL'}")

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(out / "jobs.csv", index=False)
        self.hourly.to_csv(out / "hourly.csv", index=False)
        meta = dict(version=__version__, status=self.status, objective_eur=self.objective_eur, mip_gap=self.mip_gap,
                    solve_time_s=self.solve_time_s, model_stats=self.model_stats, kpis=self.kpis,
                    verification=self.verification)
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
        return out
