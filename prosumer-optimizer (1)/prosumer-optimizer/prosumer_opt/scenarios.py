"""
Uncertainty representation: a two-level scenario tree.

    ScenarioTree
      └── IDScenario  (market prices; probability p_i)
            └── RTBranch  (real-time load deviation + balancing ratios; conditional q_r)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .exceptions import DataValidationError
from .utils import as_series, require

TOL = 1e-9


@dataclass
class RTBranch:
    """
    Real-time realisation, conditional on an intraday scenario.

    prob:            conditional probability.
    load_dev:        fractional deviation of base load vs forecast, length T (0.02 = +2 %).
    r_minus, r_plus: balancing price ratios (shortfall >= 1, surplus <= 1), scalar or length T.
    """
    prob: float
    load_dev: object = 0.0
    r_minus: object = 1.2
    r_plus: object = 0.8


@dataclass
class IDScenario:
    """
    One realisation of the market prices (EUR/MWh, length T each).

    da_buy / da_sell: day-ahead clearing price for buying / selling. ``da_buy`` is also
                      the reference for imbalance settlement.
    id_buy / id_sell: intraday prices.
    """
    name: str
    prob: float
    da_buy: object
    da_sell: object
    id_buy: object
    id_sell: object
    rt_branches: List[RTBranch] = field(default_factory=list)


@dataclass
class ScenarioTree:
    scenarios: List[IDScenario]

    def validated(self, n: int) -> "ScenarioTree":
        """Return a copy with all series converted to arrays; raise on inconsistencies."""
        require(len(self.scenarios) > 0, "scenario tree is empty", DataValidationError)
        out, ptot = [], 0.0
        for k, s in enumerate(self.scenarios):
            tag = f"scenario[{k}] '{s.name}'"
            require(0 < s.prob <= 1, f"{tag}: prob must be in (0,1]", DataValidationError)
            ptot += s.prob
            arrs = {a: as_series(getattr(s, a), n, f"{tag}.{a}")
                    for a in ("da_buy", "da_sell", "id_buy", "id_sell")}
            require(bool(np.all(arrs["da_sell"] <= arrs["da_buy"] + TOL)),
                    f"{tag}: da_sell must be <= da_buy (else simultaneous buy/sell is an arbitrage)",
                    DataValidationError)
            require(bool(np.all(arrs["id_sell"] <= arrs["id_buy"] + TOL)),
                    f"{tag}: id_sell must be <= id_buy", DataValidationError)
            require(len(s.rt_branches) > 0, f"{tag}: needs at least one RT branch", DataValidationError)
            rts, pr = [], 0.0
            for r, b in enumerate(s.rt_branches):
                bt = f"{tag}.rt[{r}]"
                require(0 < b.prob <= 1, f"{bt}: prob must be in (0,1]", DataValidationError)
                pr += b.prob
                dev = as_series(b.load_dev, n, f"{bt}.load_dev")
                rm = as_series(b.r_minus, n, f"{bt}.r_minus")
                rp = as_series(b.r_plus, n, f"{bt}.r_plus")
                require(bool(np.all(dev > -1.0)), f"{bt}: load_dev must be > -1", DataValidationError)
                require(bool(np.all(rm >= rp - TOL)),
                        f"{bt}: r_minus must be >= r_plus (else imbalance arbitrage)", DataValidationError)
                require(bool(np.all(rp >= 0)), f"{bt}: r_plus must be >= 0", DataValidationError)
                rts.append(RTBranch(b.prob, dev, rm, rp))
            require(abs(pr - 1.0) < 1e-6, f"{tag}: RT probabilities sum to {pr:.6f}", DataValidationError)
            out.append(IDScenario(s.name, s.prob, rt_branches=rts, **arrs))
        require(abs(ptot - 1.0) < 1e-6, f"scenario probabilities sum to {ptot:.6f}", DataValidationError)
        return ScenarioTree(out)
