from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._internal import EPS, __version__, log
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
class Microturbine:
    """Microturbine data (defaults = skill benchmark)."""
    p_min_mw: float = 10.0
    p_max_mw: float = 25.0
    ramp_up_mw_h: float = 20.0
    ramp_down_mw_h: float = 20.0
    startup_ramp_mw_h: float = 20.0
    shutdown_ramp_mw_h: float = 20.0
    startup_cost_eur: float = 87.40
    shutdown_cost_eur: float = 8.74
    min_up_h: int = 4
    min_down_h: int = 2
    block_width_mw: List[float] = field(default_factory=lambda: [5.3, 4.7, 5.0])       # sums to Pmax - Pmin
    block_cost_eur_mwh: List[float] = field(default_factory=lambda: [48.41, 48.78, 51.84])
    base_cost_eur_mwh: float = 48.41  # C0: fuel rate on the Pmin*u part (484.10 EUR/h at 10 MW)
    initial_on: bool = False          # commitment state before t = 0
    initial_power_mw: float = 0.0     # output before t = 0 (only used if initial_on)

    def validate(self) -> "Microturbine":
        w = _arr(self.block_width_mw, None, "mt.block_width_mw")
        c = _arr(self.block_cost_eur_mwh, None, "mt.block_cost_eur_mwh")
        if w.ndim != 1 or c.shape != w.shape or w.size == 0:
            raise InstanceValidationError("mt blocks: width and cost must be equal-length, non-empty 1-D lists")
        if np.any(w <= 0) or np.any(c < 0):
            raise InstanceValidationError("mt blocks: widths must be > 0 and costs >= 0")
        if np.any(np.diff(c) < -EPS):
            raise InstanceValidationError(
                "mt block marginal costs must be non-decreasing (convex curve); otherwise the LP "
                "piecewise formulation would not fill blocks in order")
        if not (0 <= self.p_min_mw <= self.p_max_mw) or self.p_max_mw <= 0:
            raise InstanceValidationError("mt: need 0 <= p_min <= p_max and p_max > 0")
        for nm in ("ramp_up_mw_h", "ramp_down_mw_h", "startup_ramp_mw_h", "shutdown_ramp_mw_h",
                   "startup_cost_eur", "shutdown_cost_eur", "base_cost_eur_mwh"):
            if getattr(self, nm) < 0:
                raise InstanceValidationError(f"mt.{nm} must be >= 0")
        if self.min_up_h < 1 or self.min_down_h < 1:
            raise InstanceValidationError("mt min_up_h / min_down_h must be >= 1")
        if self.initial_on and not (self.p_min_mw - 1e-9 <= self.initial_power_mw <= self.p_max_mw + 1e-9):
            raise InstanceValidationError("mt.initial_power_mw must lie in [p_min, p_max] when initial_on")
        if abs(float(w.sum()) - (self.p_max_mw - self.p_min_mw)) > 1e-6:
            raise InstanceValidationError(
                f"mt blocks must sum to p_max - p_min = {self.p_max_mw - self.p_min_mw:.2f} MW "
                f"(got {float(w.sum()):.2f} MW); otherwise Pmin*u + sum(blocks) would break the capacity limit")
        return self


@dataclass
class BESS:
    """Battery data (defaults = skill v5 benchmark, 40 MW / 200 MWh)."""
    p_max_mw: float = 40.0              # max charge and max discharge power
    e_max_mwh: float = 200.0            # nominal energy capacity
    soc_min_frac: float = 0.10          # SoC_min = 10 % of e_max
    soc_max_frac: float = 0.90          # SoC_max = 90 % of e_max
    soc_init_mwh: float = 20.0          # SoC_0 (also the required terminal SoC)
    eta_ch: float = 0.80
    eta_dis: float = 0.95
    degradation_eur_mwh: float = 35.0   # C_TP, skill range 20-50 EUR/MWh of throughput
    # which energy C_TP is charged on: "throughput" = charge + discharge (AC side, skill literal),
    # "discharge" = discharged energy only (= half the cost for the same cycling)
    degradation_basis: str = "throughput"
    enforce_exclusive: bool = True      # binaries v_ch + v_dis <= 1 (skill eq. 3.3.4); redundant while export is not binding

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
        if not (self.soc_min_mwh - 1e-9 <= self.soc_init_mwh <= self.soc_max_mwh + 1e-9):
            raise InstanceValidationError(
                f"bess: soc_init_mwh={self.soc_init_mwh} must lie in [{self.soc_min_mwh:.1f}, {self.soc_max_mwh:.1f}] MWh")
        if not (0 < self.eta_ch <= 1 and 0 < self.eta_dis <= 1):
            raise InstanceValidationError("bess: efficiencies must lie in (0, 1]")
        if self.degradation_eur_mwh < 0:
            raise InstanceValidationError("bess: degradation cost must be >= 0")
        if self.degradation_basis not in ("throughput", "discharge"):
            raise InstanceValidationError("bess: degradation_basis must be 'throughput' or 'discharge'")
        return self


@dataclass
class Instance:
    machines: List[str]
    tasks: List[str]
    power_mw: np.ndarray            # [M, P]
    duration_h: np.ndarray          # [M, P]
    yield_units: np.ndarray         # [P]
    base_load_mw: np.ndarray        # [T]
    price_buy_eur_mwh: np.ndarray   # [T]  DA import tariff
    price_sell_eur_mwh: np.ndarray  # [T]  DA export tariff
    demand_units: np.ndarray        # [T]
    horizon_h: int = 24
    buffer_h: float = 1.0
    grid_limit_mw: float = 400.0        # Q_md^buy  substation import limit
    grid_sell_limit_mw: float = 400.0   # Q_md^sell substation export limit
    max_batches_per_machine: int = 10
    inventory_init: float = 50.0
    inventory_max: float = 500.0
    mt: Optional[Microturbine] = field(default_factory=Microturbine)   # None = no microturbine
    bess: Optional[BESS] = None                                        # None = no battery

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
        self.price_buy_eur_mwh = _arr(self.price_buy_eur_mwh, T, "price_buy_eur_mwh")
        self.price_sell_eur_mwh = _arr(self.price_sell_eur_mwh, T, "price_sell_eur_mwh")
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
        if np.any(self.price_buy_eur_mwh < 0) or np.any(self.price_sell_eur_mwh < 0):
            raise InstanceValidationError("negative DA prices are not supported")
        if np.any(self.price_sell_eur_mwh > self.price_buy_eur_mwh + EPS):
            raise InstanceValidationError(
                "sell tariff must not exceed the buy tariff in any hour (simultaneous buy+sell would be an arbitrage loop)")
        if self.buffer_h < 0:
            raise InstanceValidationError("buffer_h must be >= 0")
        if self.max_batches_per_machine < 1:
            raise InstanceValidationError("max_batches_per_machine must be >= 1")
        if not (0 <= self.inventory_init <= self.inventory_max):
            raise InstanceValidationError("inventory_init must lie in [0, inventory_max]")
        if self.grid_limit_mw <= 0 or self.grid_sell_limit_mw <= 0:
            raise InstanceValidationError("grid_limit_mw and grid_sell_limit_mw must be positive")
        mt_cap = self.mt.p_max_mw if self.mt else 0.0
        if np.any(self.base_load_mw > self.grid_limit_mw + mt_cap):
            raise InstanceValidationError("base load alone exceeds substation limit plus MT capacity")
        # Necessary condition: demand cannot exceed initial stock + yield of every task.
        if self.demand_units.sum() > self.yield_units.sum() + self.inventory_init:
            raise InstanceValidationError(
                f"daily demand {self.demand_units.sum():.0f} exceeds initial stock plus total yield of all tasks "
                f"({self.inventory_init + self.yield_units.sum():.0f}); the instance cannot be feasible")
        if self.mt is not None:
            self.mt.validate()
        if self.bess is not None:
            self.bess.validate()
        return self

    # ---------------------------------------------------------------- JSON I/O
    def to_dict(self) -> dict:
        return dict(machines=self.machines, tasks=self.tasks, power_mw=self.power_mw.tolist(),
                    duration_h=self.duration_h.tolist(), yield_units=self.yield_units.tolist(),
                    base_load_mw=self.base_load_mw.tolist(), price_buy_eur_mwh=self.price_buy_eur_mwh.tolist(),
                    price_sell_eur_mwh=self.price_sell_eur_mwh.tolist(),
                    demand_units=self.demand_units.tolist(), horizon_h=self.horizon_h, buffer_h=self.buffer_h,
                    grid_limit_mw=self.grid_limit_mw, grid_sell_limit_mw=self.grid_sell_limit_mw,
                    max_batches_per_machine=self.max_batches_per_machine,
                    inventory_init=self.inventory_init, inventory_max=self.inventory_max,
                    mt=asdict(self.mt) if self.mt else None, bess=asdict(self.bess) if self.bess else None)

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        try:
            d = dict(d)
            if "price_eur_mwh" in d:            # legacy single-tariff file: treat as buy tariff, sell = 55 % of buy
                legacy = np.asarray(d.pop("price_eur_mwh"), dtype=float)
                d.setdefault("price_buy_eur_mwh", legacy.tolist())
                d.setdefault("price_sell_eur_mwh", (0.55 * legacy).tolist())
                log.warning("legacy 'price_eur_mwh' found: using it as buy tariff and 0.55x as sell tariff")
            if "mt" in d:                       # missing key -> default MT, null -> no MT
                d["mt"] = Microturbine(**d["mt"]) if d["mt"] is not None else None
            if "bess" in d:                     # missing key / null -> no battery
                d["bess"] = BESS(**d["bess"]) if d["bess"] is not None else None
            return cls(**d).validate()
        except (TypeError, ValueError) as err:
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
    mip_gap: float = 1e-4            # 0.01 % as recommended by the skill
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
                f"cost split: grid net={k['grid_cost_eur']:,.2f} (buy {k['grid_buy_cost_eur']:,.2f} - "
                f"sell {k['grid_sell_revenue_eur']:,.2f})  MT fuel={k['mt_fuel_cost_eur']:,.2f}  "
                f"MT start/stop={k['mt_startstop_cost_eur']:,.2f} EUR\n"
                f"batches={int(k['n_batches'])}  units={k['units_produced']:.0f}  total_load={k['total_load_mwh']:.1f} MWh  "
                f"grid buy={k['grid_buy_energy_mwh']:.1f} MWh  grid sell={k['grid_sell_energy_mwh']:.1f} MWh  "
                f"MT={k['mt_energy_mwh']:.1f} MWh ({int(k['mt_starts'])} start(s), {int(k['mt_on_hours'])} h on)\n"
                f"BESS: charge={k['bess_charge_mwh']:.1f} MWh  discharge={k['bess_discharge_mwh']:.1f} MWh  "
                f"degradation={k['bess_degradation_cost_eur']:,.2f} EUR  final SoC={k['bess_final_soc_mwh']:.1f} MWh  "
                f"cycles={k['bess_equiv_cycles']:.2f}\n"
                f"avg_cost={k['avg_cost_per_mwh']:.2f} EUR/MWh  peak_import={k['peak_grid_mw']:.1f} MW  "
                f"cost/unit={k['cost_per_unit']:.2f} EUR\n"
                f"ASAP grid-only baseline={k['baseline_asap_cost_eur']:,.2f} EUR (saving {k['saving_vs_asap_pct']:.1f}%)  "
                f"same schedule grid-only={k['grid_only_same_schedule_eur']:,.2f} EUR "
                f"(MT+BESS save {k['der_saving_eur']:,.2f} EUR)\n"
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
