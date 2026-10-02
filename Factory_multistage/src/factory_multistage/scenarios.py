from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import day_ahead as da
from ._internal import EPS, log
from .day_ahead import Instance, InstanceValidationError


@dataclass
class IntradayMarket:
    """Depth of the intraday market (MW per hour, each direction)."""
    cap_buy_mw: float = 100.0
    cap_sell_mw: float = 100.0

    def validate(self) -> "IntradayMarket":
        if self.cap_buy_mw < 0 or self.cap_sell_mw < 0:
            raise InstanceValidationError("intraday caps must be >= 0")
        return self


@dataclass
class ScenarioSet:
    """Intraday scenarios: probabilities, ID buy/sell prices [EUR/MWh] and base-load deviation [MW], all [S, T]."""
    prob: np.ndarray
    id_buy_eur_mwh: np.ndarray
    id_sell_eur_mwh: np.ndarray
    load_dev_mw: np.ndarray

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])

    # ------------------------------------------------------------------ checks
    def validate(self, inst: Instance) -> "ScenarioSet":
        T = inst.horizon_h
        self.prob = da._arr(self.prob, None, "scenario prob", 1)
        S = self.prob.shape[0]
        if S == 0:
            raise InstanceValidationError("at least one scenario is required")
        for nm in ("id_buy_eur_mwh", "id_sell_eur_mwh", "load_dev_mw"):
            a = da._arr(getattr(self, nm), None, nm, 2)
            if a.shape != (S, T):
                raise InstanceValidationError(f"{nm}: expected shape {(S, T)}, got {a.shape}")
            setattr(self, nm, a)
        if np.any(self.prob < 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise InstanceValidationError("scenario probabilities must be >= 0 and sum to 1")
        if np.any(self.id_buy_eur_mwh < 0) or np.any(self.id_sell_eur_mwh < 0):
            raise InstanceValidationError("negative intraday prices are not supported")
        if np.any(self.id_sell_eur_mwh > self.id_buy_eur_mwh + EPS):
            raise InstanceValidationError("intraday sell price must not exceed the buy price in any scenario/hour")
        if np.any(inst.base_load_mw[None, :] + self.load_dev_mw < -EPS):
            raise InstanceValidationError("base load plus deviation must stay >= 0")
        # virtual DA/ID arbitrage (formulation issue 9)
        e_buy, e_sell = self.prob @ self.id_buy_eur_mwh, self.prob @ self.id_sell_eur_mwh
        bad = np.nonzero((e_sell > inst.price_buy_eur_mwh + 1e-6) | (e_buy < inst.price_sell_eur_mwh - 1e-6))[0]
        if bad.size:
            log.warning("expected ID prices leave the DA buy/sell band in hours %s: DA/ID virtual arbitrage is "
                        "profitable there and is limited only by Q and the ID caps", bad.tolist())
        return self

    def expected_value(self) -> "ScenarioSet":
        """Single scenario with the probability-weighted prices and deviation (for the EV problem)."""
        return ScenarioSet(np.ones(1), (self.prob @ self.id_buy_eur_mwh)[None, :],
                           (self.prob @ self.id_sell_eur_mwh)[None, :], (self.prob @ self.load_dev_mw)[None, :])

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return dict(prob=self.prob.tolist(), id_buy_eur_mwh=self.id_buy_eur_mwh.tolist(),
                    id_sell_eur_mwh=self.id_sell_eur_mwh.tolist(), load_dev_mw=self.load_dev_mw.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSet":
        try:
            return cls(np.asarray(d["prob"], float), np.asarray(d["id_buy_eur_mwh"], float),
                       np.asarray(d["id_sell_eur_mwh"], float), np.asarray(d["load_dev_mw"], float))
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad scenario file schema: {err}") from err

    @classmethod
    def load(cls, path) -> "ScenarioSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read scenario file '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---------------------------------------------------------------- generator
    @classmethod
    def generate(cls, inst: Instance, n: int = 10, seed: int = 7, price_sigma: float = 0.15,
                 level_sigma: float = 0.05, load_sigma_mw: float = 0.0) -> "ScenarioSet":
        """ID price = DA tariff * exp(eps - var/2); eps = hourly noise + common daily level shock.
        Antithetic sampling (eps and -eps) keeps the sample mean close to the DA tariff."""
        if n < 1 or min(price_sigma, level_sigma, load_sigma_mw) < 0:
            raise InstanceValidationError("need n >= 1 and non-negative volatilities")
        T = inst.horizon_h
        rng = np.random.default_rng(seed)
        half = n // 2                                   # antithetic pairs; an odd n adds one zero (central) path
        eps = price_sigma * rng.standard_normal((half, T)) + level_sigma * rng.standard_normal((half, 1))
        dev = load_sigma_mw * rng.standard_normal((half, T))
        dev = np.clip(dev, -inst.base_load_mw[None, :], inst.base_load_mw[None, :])   # symmetric clip keeps the +/- pairs
        zero = np.zeros((n - 2 * half, T))
        eps, dev = np.vstack([eps, -eps, zero]), np.vstack([dev, -dev, zero])
        factor = np.exp(eps - (price_sigma ** 2 + level_sigma ** 2) / 2.0)
        return cls(np.full(n, 1.0 / n), inst.price_buy_eur_mwh[None, :] * factor,
                   inst.price_sell_eur_mwh[None, :] * factor, dev).validate(inst)
