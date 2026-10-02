from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import day_ahead as da
from ._internal import EPS, log
from .day_ahead import Instance, InstanceValidationError
from .scenarios import ScenarioSet

MODES = ("strategic", "passive")


@dataclass
class BalancingMarket:
    """Settings of the imbalance settlement (v2 section 5)."""
    mode: str = "strategic"                    # "strategic" | "passive" (see module docstring)
    enforce_exclusive: bool = False            # z_BAL binaries (redundant while lam+ >= lam-)
    max_deficit_mw: Optional[float] = None     # Delta_bar_buy  (default Q_buy + Q_sell)
    max_surplus_mw: Optional[float] = None     # Delta_bar_sell (default Q_buy + Q_sell)

    def validate(self) -> "BalancingMarket":
        if self.mode not in MODES:
            raise InstanceValidationError(f"imbalance mode must be one of {MODES}")
        for nm in ("max_deficit_mw", "max_surplus_mw"):
            v = getattr(self, nm)
            if v is not None and v < 0:
                raise InstanceValidationError(f"{nm} must be >= 0")
        return self

    def bounds(self, inst: Instance) -> tuple:
        full = inst.grid_limit_mw + inst.grid_sell_limit_mw
        return (full if self.max_deficit_mw is None else self.max_deficit_mw,
                full if self.max_surplus_mw is None else self.max_surplus_mw)


@dataclass
class RealTimeSet:
    """Real-time scenarios for every ID scenario s: conditional probabilities, relative factory-load
    deviation eta and imbalance price ratios, all [S, W, T] (prob is [S, W])."""
    prob: np.ndarray
    load_rel_dev: np.ndarray        # eta: actual factory load = (1 + eta) * planned load
    r_plus: np.ndarray              # deficit (buy) penalty ratio, >= 1
    r_minus: np.ndarray             # surplus (sell) discount ratio, in [0, 1]

    @property
    def n_s(self) -> int:
        return int(self.prob.shape[0])

    @property
    def n_w(self) -> int:
        return int(self.prob.shape[1])

    # ------------------------------------------------------------------ prices
    def lam_plus(self, inst: Instance) -> np.ndarray:
        """Deficit price lam+ = r+ * lam_DA,buy  [S, W, T]."""
        return self.r_plus * inst.price_buy_eur_mwh[None, None, :]

    def lam_minus(self, inst: Instance) -> np.ndarray:
        """Surplus price lam- = r- * lam_DA,sell  [S, W, T]."""
        return self.r_minus * inst.price_sell_eur_mwh[None, None, :]

    # ------------------------------------------------------------------ checks
    def validate(self, inst: Instance, scen: ScenarioSet) -> "RealTimeSet":
        T = inst.horizon_h
        self.prob = da._arr(self.prob, None, "rt prob", 2)
        S, W = self.prob.shape
        if S != scen.n:
            raise InstanceValidationError(f"rt set has {S} ID scenarios, the ID scenario set has {scen.n}")
        if W == 0:
            raise InstanceValidationError("at least one RT scenario per ID scenario is required")
        for nm in ("load_rel_dev", "r_plus", "r_minus"):
            a = da._arr(getattr(self, nm), None, nm, 3)
            if a.shape != (S, W, T):
                raise InstanceValidationError(f"{nm}: expected shape {(S, W, T)}, got {a.shape}")
            setattr(self, nm, a)
        if np.any(self.prob < 0) or np.abs(self.prob.sum(axis=1) - 1.0).max() > 1e-9:
            raise InstanceValidationError("rt probabilities must be >= 0 and sum to 1 for every ID scenario")
        if np.any(self.load_rel_dev <= -1.0):
            raise InstanceValidationError("relative load deviation must stay > -1 (load would become negative)")
        if np.any(self.r_plus < 1.0 - 1e-9):
            raise InstanceValidationError("r_plus must be >= 1 (deficit may not be cheaper than the DA buy price)")
        if np.any(self.r_minus < -1e-9) or np.any(self.r_minus > 1.0 + 1e-9):
            raise InstanceValidationError("r_minus must lie in [0, 1] (surplus may not earn more than the DA sell price)")
        if np.any(self.lam_plus(inst) < self.lam_minus(inst) - EPS):      # cannot happen for valid ratios; kept as guard
            raise InstanceValidationError("deficit price below surplus price: simultaneous buy/sell would be an arbitrage loop")
        # price arbitrage ID <-> imbalance (issue 5): imbalance cheaper than the ID market in expectation
        e_lp = np.einsum("sw,swt->st", self.prob, self.lam_plus(inst))
        e_lm = np.einsum("sw,swt->st", self.prob, self.lam_minus(inst))
        n_bad = int(np.sum((scen.id_buy_eur_mwh > e_lp + 1e-6) | (scen.id_sell_eur_mwh < e_lm - 1e-6)))
        if n_bad:
            log.warning("in %d of %d (scenario, hour) cells the expected imbalance price beats the ID price: the model "
                        "may buy/sell imbalance deliberately instead of trading intraday (use --imbalance-mode passive "
                        "or larger ratio spreads to suppress)", n_bad, scen.n * T)
        return self

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return dict(prob=self.prob.tolist(), load_rel_dev=self.load_rel_dev.tolist(),
                    r_plus=self.r_plus.tolist(), r_minus=self.r_minus.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "RealTimeSet":
        try:
            return cls(np.asarray(d["prob"], float), np.asarray(d["load_rel_dev"], float),
                       np.asarray(d["r_plus"], float), np.asarray(d["r_minus"], float))
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad RT file schema: {err}") from err

    @classmethod
    def load(cls, path) -> "RealTimeSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read RT file '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---------------------------------------------------------------- builders
    @classmethod
    def constant(cls, n_s: int, n_w: int, horizon_h: int, r_plus: float = 1.25, r_minus: float = 0.75,
                 eta: float = 0.0) -> "RealTimeSet":
        """Equiprobable RT scenarios with constant ratios and a constant deviation (deterministic RT stage)."""
        shape = (n_s, n_w, horizon_h)
        return cls(np.full((n_s, n_w), 1.0 / n_w), np.full(shape, float(eta)), np.full(shape, float(r_plus)),
                   np.full(shape, float(r_minus)))

    @classmethod
    def generate(cls, inst: Instance, n_s: int, n_w: int = 5, seed: int = 11, rel_sigma: float = 0.03,
                 rho: float = 0.6, premium: float = 0.25, discount: float = 0.25,
                 ratio_sigma: float = 0.3) -> "RealTimeSet":
        """eta: stationary AR(1) with std rel_sigma, antithetic pairs (+eta, -eta) plus one zero path when n_w is odd,
        so E[eta] = 0 exactly.  r+ = 1 + premium*g, r- = clip(1 - discount*g, 0, 1), g lognormal with mean one."""
        if n_s < 1 or n_w < 1:
            raise InstanceValidationError("need n_s >= 1 and n_w >= 1")
        if min(rel_sigma, premium, discount, ratio_sigma) < 0 or not (0 <= rho < 1):
            raise InstanceValidationError("need non-negative volatilities and 0 <= rho < 1")
        T = inst.horizon_h
        rng = np.random.default_rng(seed)
        pairs = n_w // 2
        z = rng.standard_normal((n_s, pairs, T))
        e = np.empty_like(z)
        if pairs:
            e[..., 0] = z[..., 0]
            a = math.sqrt(1.0 - rho ** 2)
            for t in range(1, T):
                e[..., t] = rho * e[..., t - 1] + a * z[..., t]
        e = np.clip(rel_sigma * e, -0.5, 0.5)                       # symmetric clip keeps the zero mean
        parts = [e, -e] + ([np.zeros((n_s, 1, T))] if n_w % 2 else [])
        eta = np.concatenate(parts, axis=1)
        g = np.exp(ratio_sigma * rng.standard_normal((n_s, n_w, T)) - ratio_sigma ** 2 / 2.0)
        return cls(np.full((n_s, n_w), 1.0 / n_w), eta, 1.0 + premium * g, np.clip(1.0 - discount * g, 0.0, 1.0))
