"""
Physical and economic parameter dataclasses.

Every class validates itself in ``__post_init__`` and raises
:class:`~prosumer_opt.exceptions.ConfigurationError` on inconsistent data, so an
invalid configuration fails fast, before any model is built.
Default values come from :mod:`prosumer_opt.constants`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import constants as C
from .exceptions import DataValidationError
from .utils import as_series, require

TOL = C.TOL


@dataclass(frozen=True)
class TimeGrid:
    """Optimisation horizon. ``horizon_h`` must be an integer multiple of ``dt_h``."""
    horizon_h: float = C.DEFAULT_HORIZON_H
    dt_h: float = C.DEFAULT_DT_H

    def __post_init__(self):
        require(self.dt_h > 0 and self.horizon_h > 0, "horizon_h and dt_h must be positive")
        n = self.horizon_h / self.dt_h
        require(abs(n - round(n)) < 1e-9, "horizon_h must be an integer multiple of dt_h")

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_h / self.dt_h))

    def steps(self, hours: float) -> int:
        """Hours -> number of steps, rounded up (conservative for durations)."""
        return int(math.ceil(hours / self.dt_h - 1e-9))

    def hour(self, t: int) -> float:
        return t * self.dt_h


@dataclass(frozen=True)
class GridParams:
    """Point-of-common-coupling limits (MW)."""
    import_limit_mw: float = C.GRID_IMPORT_LIMIT_MW
    export_limit_mw: float = C.GRID_EXPORT_LIMIT_MW

    def __post_init__(self):
        require(self.import_limit_mw > 0 and self.export_limit_mw >= 0,
                "grid limits must be positive")


@dataclass(frozen=True)
class MicroturbineParams:
    """
    Microturbine data. ``cost_blocks`` is a tuple of (width_MW, marginal_cost_EUR_per_MWh)
    describing the fuel-cost curve from 0 MW upwards. Marginal costs must be
    non-decreasing (convex), which keeps the piecewise-linear cost an LP without
    extra binaries.
    """
    p_min_mw: float = C.MT_P_MIN_MW
    p_max_mw: float = C.MT_P_MAX_MW
    ramp_up_mw_h: float = C.MT_RAMP_UP_MW_H
    ramp_down_mw_h: float = C.MT_RAMP_DOWN_MW_H
    startup_ramp_mw_h: float = C.MT_STARTUP_RAMP_MW_H
    shutdown_ramp_mw_h: float = C.MT_SHUTDOWN_RAMP_MW_H
    startup_cost_eur: float = C.MT_STARTUP_COST_EUR
    shutdown_cost_eur: float = C.MT_SHUTDOWN_COST_EUR
    min_up_h: float = C.MT_MIN_UP_H
    min_down_h: float = C.MT_MIN_DOWN_H
    cost_blocks: Tuple[Tuple[float, float], ...] = C.MT_COST_BLOCKS

    def __post_init__(self):
        require(0 <= self.p_min_mw <= self.p_max_mw, "MT: need 0 <= p_min <= p_max")
        require(min(self.ramp_up_mw_h, self.ramp_down_mw_h, self.startup_ramp_mw_h,
                    self.shutdown_ramp_mw_h) > 0, "MT: ramp limits must be positive")
        require(self.startup_cost_eur >= 0 and self.shutdown_cost_eur >= 0, "MT: costs must be >= 0")
        require(self.min_up_h >= 0 and self.min_down_h >= 0, "MT: min up/down must be >= 0")
        require(len(self.cost_blocks) > 0, "MT: cost_blocks is empty")
        require(all(w > 0 for w, _ in self.cost_blocks), "MT: block widths must be positive")
        costs = [c for _, c in self.cost_blocks]
        require(all(b >= a - TOL for a, b in zip(costs, costs[1:])),
                "MT: marginal costs must be non-decreasing (convex curve)")
        require(sum(w for w, _ in self.cost_blocks) >= self.p_max_mw - 1e-6,
                "MT: cost blocks do not cover p_max")

    def effective_segments(self) -> List[Tuple[float, float]]:
        """Blocks clipped to the range (p_min, p_max]: list of (width, marginal cost)."""
        segs, lo = [], 0.0
        for w, c in self.cost_blocks:
            hi = lo + w
            a, b = max(lo, self.p_min_mw), min(hi, self.p_max_mw)
            if b - a > 1e-9:
                segs.append((b - a, c))
            lo = hi
        return segs

    def no_load_cost_eur_per_h(self) -> float:
        """Fuel cost of producing ``p_min`` (paid for every committed hour)."""
        cost, lo = 0.0, 0.0
        for w, c in self.cost_blocks:
            hi = lo + w
            b = min(hi, self.p_min_mw)
            if b > lo:
                cost += (b - lo) * c
            lo = hi
        return cost


@dataclass(frozen=True)
class BESSParams:
    """Battery data. Energies in MWh, powers in MW (AC side)."""
    p_max_mw: float = 40.0
    e_max_mwh: float = 200.0
    soc_min_mwh: float = 20.0
    soc_max_mwh: float = 200.0
    soc_init_mwh: float = 20.0
    eta_ch: float = C.BESS_ETA_CH
    eta_dis: float = C.BESS_ETA_DIS
    throughput_cost_eur_mwh: float = C.BESS_THROUGHPUT_COST_EUR_MWH
    terminal_soc_mwh: Optional[float] = None   # default: return to soc_init
    enforce_exclusivity: bool = True           # binaries forbidding simultaneous ch/dis

    def __post_init__(self):
        require(self.p_max_mw > 0 and self.e_max_mwh > 0, "BESS: capacities must be positive")
        require(0 <= self.soc_min_mwh <= self.soc_max_mwh <= self.e_max_mwh + TOL,
                "BESS: need 0 <= soc_min <= soc_max <= e_max")
        require(self.soc_min_mwh - TOL <= self.soc_init_mwh <= self.soc_max_mwh + TOL,
                "BESS: soc_init outside [soc_min, soc_max]")
        require(0 < self.eta_ch <= 1 and 0 < self.eta_dis <= 1, "BESS: efficiencies must be in (0,1]")
        require(self.throughput_cost_eur_mwh >= 0, "BESS: throughput cost must be >= 0")
        if self.terminal_soc_mwh is not None:
            require(self.soc_min_mwh - TOL <= self.terminal_soc_mwh <= self.soc_max_mwh + TOL,
                    "BESS: terminal SoC outside [soc_min, soc_max]")

    @property
    def terminal_target(self) -> float:
        return self.soc_init_mwh if self.terminal_soc_mwh is None else self.terminal_soc_mwh

    @classmethod
    def from_profile(cls, profile: str = "large", **overrides) -> "BESSParams":
        """Build from a named size profile (see ``constants.BESS_PROFILES``)."""
        require(profile in C.BESS_PROFILES,
                f"BESS: unknown profile '{profile}', choose from {sorted(C.BESS_PROFILES)}")
        base = dict(C.BESS_PROFILES[profile])
        base.update(overrides)
        return cls(**base)

    @classmethod
    def medium_scale(cls, **kw) -> "BESSParams":
        return cls.from_profile("medium", **kw)

    @classmethod
    def large_scale(cls, **kw) -> "BESSParams":
        return cls.from_profile("large", **kw)


@dataclass(frozen=True)
class BatchJob:
    """
    One non-interruptible batch task. Jobs on the same machine run in ascending
    ``sequence``, separated by the factory buffer.
    """
    job_id: str
    machine: str
    sequence: int
    duration_h: float
    power_mw: float
    units_out: float = C.DEMO_UNITS_OUT_PER_BATCH
    earliest_start_h: float = 0.0
    latest_finish_h: Optional[float] = None   # default: end of horizon

    def __post_init__(self):
        require(self.duration_h > 0 and self.power_mw >= 0 and self.units_out >= 0,
                f"job {self.job_id}: duration/power/units must be positive")
        require(self.earliest_start_h >= 0, f"job {self.job_id}: earliest_start_h < 0")


@dataclass
class FactoryParams:
    """
    Industrial site data.

    base_load_mw:    scalar or length-T array of non-shiftable load.
    demand_units:    scalar or length-T array of product deliveries per step.
    intraday_shift_window_h: max deviation of a batch start from the DA plan in the
                     intraday stage. ``0`` disables intraday load shifting.
    """
    jobs: List[BatchJob] = field(default_factory=list)
    base_load_mw: object = C.FACTORY_BASE_LOAD_MW
    buffer_h: float = C.FACTORY_BUFFER_H
    inventory_init: float = 0.0
    inventory_max: float = C.FACTORY_INVENTORY_MAX
    demand_units: object = 0.0
    intraday_shift_window_h: float = C.FACTORY_SHIFT_WINDOW_H
    unmet_penalty_eur_per_unit: float = C.FACTORY_UNMET_PENALTY_EUR_PER_UNIT

    def __post_init__(self):
        ids = [j.job_id for j in self.jobs]
        require(len(ids) == len(set(ids)), "duplicate job_id")
        keys = [(j.machine, j.sequence) for j in self.jobs]
        require(len(keys) == len(set(keys)), "duplicate (machine, sequence)")
        require(self.buffer_h >= 0 and self.intraday_shift_window_h >= 0, "buffer/window must be >= 0")
        require(0 <= self.inventory_init <= self.inventory_max, "inventory_init outside [0, max]")
        require(self.unmet_penalty_eur_per_unit >= 0, "unmet penalty must be >= 0")

    def base_series(self, n: int) -> np.ndarray:
        s = as_series(self.base_load_mw, n, "base_load_mw")
        require(bool(np.all(s >= 0)), "base_load_mw must be >= 0", DataValidationError)
        return s

    def demand_series(self, n: int) -> np.ndarray:
        s = as_series(self.demand_units, n, "demand_units")
        require(bool(np.all(s >= 0)), "demand_units must be >= 0", DataValidationError)
        return s


@dataclass
class SolverSettings:
    """MILP solver selection and tolerances."""
    name: str = C.SOLVER_NAME
    mip_gap: float = C.SOLVER_MIP_GAP
    time_limit_s: float = C.SOLVER_TIME_LIMIT_S
    threads: Optional[int] = None
    verbose: bool = False
    extra_options: dict = field(default_factory=dict)

    def __post_init__(self):
        require(0 <= self.mip_gap < 1, "mip_gap must be in [0,1)")
        require(self.time_limit_s > 0, "time_limit_s must be positive")
