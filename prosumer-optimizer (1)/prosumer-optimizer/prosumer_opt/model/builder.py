"""
Assembles the full multi-stage stochastic MILP from the individual building blocks.

    build_model(spec) -> (pyo.ConcreteModel, ModelContext)

Build order matters only in that ``market`` needs expressions defined by the other blocks.
"""
from __future__ import annotations

from typing import Tuple

import pyomo.environ as pyo

from .bess import add_bess
from .context import COST_COMPONENTS, BuildSpec, ModelContext
from .market import add_market, freeze_first_stage
from .microturbine import add_microturbine
from .production import add_production


def _add_index_sets(m: pyo.ConcreteModel, s: BuildSpec) -> None:
    """T_opt steps (TO), ID scenarios (I), (t,i) pairs (TI), (t,i,r) triples (TIR), MT cost blocks (K)."""
    nI, T_opt = s.n_scenarios, s.t_opt
    m.TO = pyo.Set(initialize=T_opt, ordered=True)
    m.I = pyo.Set(initialize=list(range(nI)), ordered=True)
    m.TI = pyo.Set(dimen=2, initialize=[(t, i) for t in T_opt for i in range(nI)], ordered=True)
    tir = [(t, i, r) for i in range(nI) for r in range(len(s.tree.scenarios[i].rt_branches)) for t in T_opt]
    m.TIR = pyo.Set(dimen=3, initialize=tir, ordered=True)
    m.K = pyo.RangeSet(0, len(s.segments) - 1)


def _add_objective(m: pyo.ConcreteModel, s: BuildSpec) -> None:
    """Minimise expected total cost:  sum_i p_i * (DA + ID + BAL + MT + BESS + UNMET)."""
    costs = [getattr(m, name) for name in COST_COMPONENTS.values()]
    m.obj = pyo.Objective(
        expr=sum(sc.prob * sum(c[i] for c in costs) for i, sc in enumerate(s.tree.scenarios)),
        sense=pyo.minimize)


def build_model(spec: BuildSpec) -> Tuple[pyo.ConcreteModel, ModelContext]:
    m = pyo.ConcreteModel("ProsumerMultiStage")
    _add_index_sets(m, spec)

    add_microturbine(m, spec)
    add_bess(m, spec)
    prod = add_production(m, spec)
    add_market(m, spec)
    _add_objective(m, spec)

    if spec.is_intraday:
        freeze_first_stage(m, spec, prod.allowed)

    ctx = ModelContext(spec.t0, spec.t_opt, spec.segments, prod.job_dur, prod.uses_z2,
                       prod.plan_start, prod.allowed)
    return m, ctx
