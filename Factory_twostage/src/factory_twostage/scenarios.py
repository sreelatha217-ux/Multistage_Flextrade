"""Intraday market settings and validated price/load uncertainty scenarios."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Instance, _arr
from .exceptions import InstanceValidationError
from .parameters import EPS

log = logging.getLogger("factory_twostage")


@dataclass
class IntradayMarket:
    """Intraday market depth in MW per hour and direction."""

    cap_buy_mw: float = 100.0
    cap_sell_mw: float = 100.0

    def validate(self) -> "IntradayMarket":
        caps = np.asarray([self.cap_buy_mw, self.cap_sell_mw], dtype=float)
        if not np.isfinite(caps).all() or np.any(caps < 0):
            raise InstanceValidationError("intraday caps must be finite and >= 0")
        return self


@dataclass
class ScenarioSet:
    """Scenario probabilities, buy/sell prices, and base-load deviations."""

    prob: np.ndarray
    id_buy_eur_mwh: np.ndarray
    id_sell_eur_mwh: np.ndarray
    load_dev_mw: np.ndarray

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])

    def validate(self, inst: Instance) -> "ScenarioSet":
        hours = inst.horizon_h
        self.prob = _arr(self.prob, None, "scenario prob")
        count = self.prob.shape[0]
        if count == 0:
            raise InstanceValidationError("at least one scenario is required")
        for name in ("id_buy_eur_mwh", "id_sell_eur_mwh", "load_dev_mw"):
            array = _arr(getattr(self, name), None, name, ndim=2)
            if array.shape != (count, hours):
                raise InstanceValidationError(
                    f"{name}: expected shape {(count, hours)}, got {array.shape}"
                )
            setattr(self, name, array)
        if np.any(self.prob < 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise InstanceValidationError("scenario probabilities must be >= 0 and sum to 1")
        if np.any(self.id_buy_eur_mwh < 0) or np.any(self.id_sell_eur_mwh < 0):
            raise InstanceValidationError("negative intraday prices are not supported")
        if np.any(self.id_sell_eur_mwh > self.id_buy_eur_mwh + EPS):
            raise InstanceValidationError(
                "intraday sell price must not exceed buy price in any scenario/hour"
            )
        if np.any(inst.base_load_mw[None, :] + self.load_dev_mw < -EPS):
            raise InstanceValidationError("base load plus deviation must stay >= 0")

        expected_buy = self.prob @ self.id_buy_eur_mwh
        expected_sell = self.prob @ self.id_sell_eur_mwh
        arbitrage_hours = np.flatnonzero(
            (expected_sell > inst.price_buy_eur_mwh + 1e-6)
            | (expected_buy < inst.price_sell_eur_mwh - 1e-6)
        )
        if arbitrage_hours.size:
            log.warning(
                "expected ID prices leave the DA buy/sell band in hours %s; virtual arbitrage "
                "is limited only by market caps",
                arbitrage_hours.tolist(),
            )
        return self

    def expected_value(self) -> "ScenarioSet":
        """Return a one-scenario set with probability-weighted values."""
        return ScenarioSet(
            np.ones(1),
            (self.prob @ self.id_buy_eur_mwh)[None, :],
            (self.prob @ self.id_sell_eur_mwh)[None, :],
            (self.prob @ self.load_dev_mw)[None, :],
        )

    def to_dict(self) -> dict[str, list]:
        return {
            "prob": self.prob.tolist(),
            "id_buy_eur_mwh": self.id_buy_eur_mwh.tolist(),
            "id_sell_eur_mwh": self.id_sell_eur_mwh.tolist(),
            "load_dev_mw": self.load_dev_mw.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "ScenarioSet":
        try:
            return cls(
                np.asarray(values["prob"], dtype=float),
                np.asarray(values["id_buy_eur_mwh"], dtype=float),
                np.asarray(values["id_sell_eur_mwh"], dtype=float),
                np.asarray(values["load_dev_mw"], dtype=float),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad scenario file schema: {err}") from err

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read scenario file '{path}': {err}") from err

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def generate(
        cls,
        inst: Instance,
        n: int = 10,
        seed: int = 7,
        price_sigma: float = 0.15,
        level_sigma: float = 0.05,
        load_sigma_mw: float = 0.0,
    ) -> "ScenarioSet":
        """Generate antithetic lognormal ID-price scenarios around DA tariffs."""
        if n < 1 or min(price_sigma, level_sigma, load_sigma_mw) < 0:
            raise InstanceValidationError("need n >= 1 and non-negative volatilities")
        hours = inst.horizon_h
        rng = np.random.default_rng(seed)
        half = (n + 1) // 2
        shocks = price_sigma * rng.standard_normal((half, hours))
        shocks += level_sigma * rng.standard_normal((half, 1))
        shocks = np.vstack((shocks, -shocks))[:n]
        factor = np.exp(shocks - (price_sigma**2 + level_sigma**2) / 2.0)
        load_deviation = load_sigma_mw * rng.standard_normal((half, hours))
        load_deviation = np.vstack((load_deviation, -load_deviation))[:n]
        load_deviation = np.maximum(load_deviation, -inst.base_load_mw[None, :])
        return cls(
            np.full(n, 1.0 / n),
            inst.price_buy_eur_mwh[None, :] * factor,
            inst.price_sell_eur_mwh[None, :] * factor,
            load_deviation,
        ).validate(inst)