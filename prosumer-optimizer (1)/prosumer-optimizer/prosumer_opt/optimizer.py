"""
Public facade of the package.

    optimizer = MultiStageProsumerOptimizer(grid, factory, mt, bess, time_grid, solver)
    da  = optimizer.solve_day_ahead(tree)                  # Stage 1 (+2, 3 as recourse)
    idr = optimizer.solve_intraday(da, IntradayInputs(...))  # re-optimise the remaining hours

Decision architecture
---------------------
    STAGE 1  Day-Ahead ("here-and-now"), solved once before gate closure
             DA purchase / sale schedule, MT commitment, master batch schedule.
             Not scenario indexed -> non-anticipativity holds by construction.
    STAGE 2  Intraday recourse (one copy per ID scenario)
             ID buy / sell, MT re-dispatch, BESS dispatch, batch shifting within +/- window.
    STAGE 3  Real-time settlement (one copy per ID scenario and RT branch)
             Imbalance volumes settled at DA reference price x ratios r_plus / r_minus.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

from .exceptions import DataValidationError
from .extraction import extract_stage_result
from .model import BuildSpec, build_model
from .parameters import (BESSParams, FactoryParams, GridParams, MicroturbineParams,
                         SolverSettings, TimeGrid)
from .results import StageResult
from .scenarios import ScenarioTree
from .solver import solve_model
from .state import InitialState, IntradayInputs
from .utils import as_series, require
from .validation import check_state, check_static_feasibility

log = logging.getLogger("prosumer_opt")


class MultiStageProsumerOptimizer:
    def __init__(self, grid: GridParams, factory: FactoryParams, mt: MicroturbineParams,
                 bess: BESSParams, time_grid: Optional[TimeGrid] = None,
                 solver: Optional[SolverSettings] = None):
        self.grid, self.factory, self.mt, self.bess = grid, factory, mt, bess
        self.time = time_grid or TimeGrid()
        self.solver = solver or SolverSettings()
        check_static_feasibility(self.time, self.factory)

    # ------------------------------------------------------------------ Stage 1
    def solve_day_ahead(self, tree: ScenarioTree, state: Optional[InitialState] = None) -> StageResult:
        """Stage 1 + 2 + 3 stochastic problem. DA variables are scenario independent."""
        T = self.time.n_steps
        tree = tree.validated(T)
        state = state or InitialState.default(self.bess, self.factory)
        check_state(state, self.bess, self.mt, self.factory)
        log.info("Building DA model: %d ID scenarios, %d steps", len(tree.scenarios), T)
        return self._solve("DA", self._spec(tree, 0, state, None, self.factory), tree, None)

    # ------------------------------------------------------------------ Stage 2
    def solve_intraday(self, da: StageResult, inp: IntradayInputs) -> StageResult:
        """Re-optimise steps >= ``t0`` with the day-ahead commitments frozen."""
        T = self.time.n_steps
        require(da.stage == "DA", "solve_intraday needs a day-ahead StageResult", DataValidationError)
        require(0 <= inp.t0 < T, f"t0 must be in [0, {T - 1}]", DataValidationError)
        check_state(inp.state, self.bess, self.mt, self.factory)
        require(abs(inp.scenario.prob - 1.0) < 1e-9, "intraday scenario prob must be 1.0", DataValidationError)
        tree = ScenarioTree([inp.scenario]).validated(T)
        factory = self.factory
        if inp.base_load_mw is not None:
            factory = replace(factory, base_load_mw=as_series(inp.base_load_mw, T, "intraday base_load_mw"))
        log.info("Building ID model from step %d (hour %.2f)", inp.t0, self.time.hour(inp.t0))
        return self._solve("ID", self._spec(tree, inp.t0, inp.state, da.plan, factory), tree, da.plan)

    # ------------------------------------------------------------------ internals
    def _spec(self, tree, t0, state, plan, factory) -> BuildSpec:
        return BuildSpec(time=self.time, grid=self.grid, mt=self.mt, bess=self.bess, factory=factory,
                         tree=tree, t0=t0, state=state, plan=plan)

    def _solve(self, stage: str, spec: BuildSpec, tree: ScenarioTree, old_plan) -> StageResult:
        model, ctx = build_model(spec)
        info = solve_model(model, self.solver, stage)
        return extract_stage_result(model, ctx, tree, stage, info, old_plan, spec.factory.jobs, self.time)
