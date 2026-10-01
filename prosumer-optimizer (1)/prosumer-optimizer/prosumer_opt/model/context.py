"""
Data shared by the model-building modules.

``BuildSpec``      read-only inputs for one model build (parameters, scenarios, state, DA plan)
``ModelContext``   bookkeeping produced while building, consumed by result extraction
``COST_COMPONENTS`` names of the per-scenario cost expressions each module contributes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..parameters import BESSParams, FactoryParams, GridParams, MicroturbineParams, TimeGrid
from ..scenarios import ScenarioTree
from ..state import DAPlan, InitialState

#: label -> name of the Pyomo Expression (indexed by scenario) that holds the cost.
COST_COMPONENTS: Dict[str, str] = {
    "DA": "cost_da",        # market.py
    "ID": "cost_id",        # market.py
    "BAL": "cost_bal",      # market.py
    "MT": "cost_mt",        # microturbine.py
    "BESS": "cost_bess",    # bess.py
    "UNMET": "cost_unmet",  # production.py
}


@dataclass
class BuildSpec:
    """Everything a model-building module needs to know. Never mutated after creation."""
    time: TimeGrid
    grid: GridParams
    mt: MicroturbineParams
    bess: BESSParams
    factory: FactoryParams
    tree: ScenarioTree
    t0: int
    state: InitialState
    plan: Optional[DAPlan] = None      # None -> day-ahead stage, else intraday with frozen plan

    # derived, filled in __post_init__
    base: np.ndarray = field(init=False)
    demand: np.ndarray = field(init=False)
    segments: List[Tuple[float, float]] = field(init=False)
    no_load_cost: float = field(init=False)

    def __post_init__(self):
        self.base = self.factory.base_series(self.T)
        self.demand = self.factory.demand_series(self.T)
        self.segments = self.mt.effective_segments()
        self.no_load_cost = self.mt.no_load_cost_eur_per_h()

    @property
    def T(self) -> int:
        return self.time.n_steps

    @property
    def dt(self) -> float:
        return self.time.dt_h

    @property
    def n_scenarios(self) -> int:
        return len(self.tree.scenarios)

    @property
    def t_opt(self) -> List[int]:
        return list(range(self.t0, self.T))

    @property
    def is_intraday(self) -> bool:
        return self.plan is not None


@dataclass
class ModelContext:
    """Bookkeeping returned by :func:`build_model` and used to read the solution back."""
    t0: int
    T_opt: List[int]
    segs: List[Tuple[float, float]]
    job_dur: Dict[str, int]
    uses_z2: Dict[str, bool]
    plan_start: Dict[str, int]
    allowed: Dict[str, List[int]]
