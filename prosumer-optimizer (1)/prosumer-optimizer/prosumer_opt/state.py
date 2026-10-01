"""System state and the frozen first-stage plan passed between stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from .exceptions import DataValidationError
from .parameters import BESSParams, FactoryParams, TimeGrid
from .scenarios import IDScenario
from .utils import require

if TYPE_CHECKING:   # avoid an import cycle with results.py
    from .results import StageResult


@dataclass
class InitialState:
    """System state just before the first optimised step."""
    soc_mwh: float
    mt_online: bool = False
    mt_output_mw: float = 0.0
    mt_hours_in_state: float = 1e6     # hours since last status change
    inventory_units: float = 0.0

    @classmethod
    def default(cls, bess: BESSParams, factory: FactoryParams) -> "InitialState":
        return cls(soc_mwh=bess.soc_init_mwh, inventory_units=factory.inventory_init)

    @classmethod
    def from_result(cls, res: "StageResult", t0: int, time_grid: TimeGrid) -> "InitialState":
        """Expected state at the end of step ``t0-1`` of a solved stage (for simulation/demo)."""
        require(1 <= t0 < time_grid.n_steps, "t0 must be in [1, T-1]", DataValidationError)
        row = res.schedule.iloc[t0 - 1]
        u = res.plan.mt_u
        hrs, k = 0.0, t0 - 1
        while k >= 0 and u[k] == u[t0 - 1]:
            hrs += time_grid.dt_h
            k -= 1
        return cls(soc_mwh=float(row["exp_soc_mwh"]), mt_online=bool(u[t0 - 1]),
                   mt_output_mw=float(row["exp_mt_mw"]), mt_hours_in_state=hrs,
                   inventory_units=float(row["exp_inventory"]))


@dataclass
class DAPlan:
    """Frozen first-stage decisions (non-anticipative by construction)."""
    p_da_buy: np.ndarray
    p_da_sell: np.ndarray
    mt_u: np.ndarray
    mt_x: np.ndarray
    mt_y: np.ndarray
    job_start_step: Dict[str, int]


@dataclass
class IntradayInputs:
    """
    Inputs for an intraday re-optimisation.

    t0:            first step to re-optimise (steps < t0 are history).
    state:         measured state at the end of step ``t0-1``.
    scenario:      updated prices and RT branches (``prob`` must be 1.0); only steps >= t0 are used.
    base_load_mw:  optional refreshed base-load forecast (scalar or length T).
    """
    t0: int
    state: InitialState
    scenario: IDScenario
    base_load_mw: Optional[object] = None
