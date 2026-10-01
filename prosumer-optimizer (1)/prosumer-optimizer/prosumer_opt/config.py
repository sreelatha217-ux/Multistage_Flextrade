"""
Central configuration.

``AppConfig`` groups every setting of a run into typed sections:

    time / grid / microturbine / bess / factory   physical system (see parameters.py)
    demo_factory / scenarios / intraday           synthetic-data generators (used when no real data is given)
    solver                                        MILP solver and tolerances
    output                                        where and what to save

Defaults come from :mod:`prosumer_opt.constants`. Load overrides from a YAML or JSON file::

    cfg = AppConfig.from_file("configs/default.yaml")

Unknown keys raise :class:`ConfigurationError`, so typos never pass silently.
Precedence used by the CLI:  built-in defaults  <  config file  <  command-line flags.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from . import constants as C
from .exceptions import ConfigurationError
from .parameters import (BatchJob, BESSParams, FactoryParams, GridParams, MicroturbineParams,
                         SolverSettings, TimeGrid)
from .utils import require

__all__ = ["AppConfig", "DemoFactorySettings", "ScenarioSettings", "IntradaySettings",
           "OutputSettings", "SolverSettings"]

T_ = TypeVar("T_")


# --------------------------------------------------------------------------- #
# Settings of the synthetic-data generators
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoFactorySettings:
    """Random demo factory: ``n_machines`` machines with ``jobs_per_machine`` batches each."""
    seed: int = 7
    n_machines: int = 5
    jobs_per_machine: int = 2
    duration_range_h: Tuple[float, float] = C.BATCH_DURATION_RANGE_H
    power_range_mw: Tuple[float, float] = C.BATCH_POWER_RANGE_MW
    units_out: float = C.DEMO_UNITS_OUT_PER_BATCH
    shift_window_h: float = C.FACTORY_SHIFT_WINDOW_H     # intraday batch shift window, 0 disables
    demand_units_per_h: float = C.DEMO_DEMAND_UNITS_PER_H
    demand_window_h: Tuple[float, float] = C.DEMO_DEMAND_WINDOW_H
    base_load_off_shift_mw: float = C.DEMO_BASE_LOAD_OFF_SHIFT_MW
    base_load_on_shift_mw: float = C.DEMO_BASE_LOAD_ON_SHIFT_MW
    on_shift_window_h: Tuple[float, float] = C.DEMO_ON_SHIFT_WINDOW_H

    def __post_init__(self):
        require(self.n_machines >= 1 and self.jobs_per_machine >= 1, "demo_factory: need >= 1 machine and job")
        require(0 < self.duration_range_h[0] <= self.duration_range_h[1], "demo_factory: bad duration_range_h")
        require(0 <= self.power_range_mw[0] <= self.power_range_mw[1], "demo_factory: bad power_range_mw")
        require(self.shift_window_h >= 0, "demo_factory: shift_window_h must be >= 0")


@dataclass(frozen=True)
class ScenarioSettings:
    """Synthetic day-ahead / intraday / real-time scenario tree around the TOU tariff."""
    n_id_scenarios: int = 5
    n_rt_branches: int = 3
    seed: int = 11
    sell_ratio: float = C.SCENARIO_SELL_RATIO
    da_ar_coeff: float = C.SCENARIO_DA_AR_COEFF
    da_sigma: float = C.SCENARIO_DA_SIGMA
    id_sigma: float = C.SCENARIO_ID_SIGMA
    load_sigma: float = C.SCENARIO_LOAD_SIGMA
    r_minus_base: float = C.SCENARIO_R_MINUS_BASE
    r_minus_spread: float = C.SCENARIO_R_MINUS_SPREAD
    r_plus_base: float = C.SCENARIO_R_PLUS_BASE
    r_plus_spread: float = C.SCENARIO_R_PLUS_SPREAD

    def __post_init__(self):
        require(self.n_id_scenarios >= 1 and self.n_rt_branches >= 1,
                "scenarios: need at least one ID scenario and one RT branch")
        require(0 <= self.sell_ratio <= 1, "scenarios: sell_ratio must be in [0, 1]")


@dataclass(frozen=True)
class IntradaySettings:
    """Intraday re-optimisation demo. ``step = 0`` skips the intraday stage."""
    step: int = 10
    seed: int = 99
    spike: float = 1.4
    price_noise_sigma: float = C.INTRADAY_PRICE_NOISE_SIGMA
    load_dev_sigma: float = C.INTRADAY_LOAD_DEV_SIGMA
    sell_ratio: float = C.SCENARIO_SELL_RATIO
    #: (probability, mean load deviation, r_minus, r_plus) for each real-time branch
    rt_branches: Tuple[Tuple[float, float, float, float], ...] = C.INTRADAY_RT_BRANCHES

    def __post_init__(self):
        require(self.step >= 0, "intraday: step must be >= 0")
        require(self.spike > 0, "intraday: spike must be > 0")
        require(abs(sum(b[0] for b in self.rt_branches) - 1.0) < 1e-6, "intraday: rt_branches probs must sum to 1")

    @property
    def enabled(self) -> bool:
        return self.step > 0


@dataclass(frozen=True)
class OutputSettings:
    directory: str = "results"
    save: bool = True            # write CSV/JSON tables
    save_config: bool = True     # also write the resolved configuration next to the results


# --------------------------------------------------------------------------- #
# Top-level configuration
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    time: TimeGrid = field(default_factory=TimeGrid)
    grid: GridParams = field(default_factory=GridParams)
    microturbine: MicroturbineParams = field(default_factory=MicroturbineParams)
    bess: BESSParams = field(default_factory=BESSParams)
    #: explicit factory (jobs, loads, demand). ``None`` -> generate one from ``demo_factory``.
    factory: Optional[FactoryParams] = None
    demo_factory: DemoFactorySettings = field(default_factory=DemoFactorySettings)
    scenarios: ScenarioSettings = field(default_factory=ScenarioSettings)
    intraday: IntradaySettings = field(default_factory=IntradaySettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    log_level: str = "INFO"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        data = dict(data or {})
        kw: Dict[str, Any] = {}
        kw["time"] = _make(TimeGrid, data.pop("time", None), "time")
        kw["grid"] = _make(GridParams, data.pop("grid", None), "grid")
        kw["microturbine"] = _make(MicroturbineParams, data.pop("microturbine", None), "microturbine")
        kw["bess"] = _make_bess(data.pop("bess", None))
        kw["factory"] = _make_factory(data.pop("factory", None))
        kw["demo_factory"] = _make(DemoFactorySettings, data.pop("demo_factory", None), "demo_factory")
        kw["scenarios"] = _make(ScenarioSettings, data.pop("scenarios", None), "scenarios")
        kw["intraday"] = _make(IntradaySettings, data.pop("intraday", None), "intraday")
        kw["solver"] = _make(SolverSettings, data.pop("solver", None), "solver")
        kw["output"] = _make(OutputSettings, data.pop("output", None), "output")
        if "log_level" in data:
            kw["log_level"] = str(data.pop("log_level"))
        if data:
            raise ConfigurationError(f"config: unknown top-level key(s) {sorted(data)}")
        return cls(**kw)

    @classmethod
    def from_file(cls, path) -> "AppConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()
        try:
            if suffix == ".json":
                raw = json.loads(text)
            elif suffix in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError as err:   # pragma: no cover
                    raise ConfigurationError("reading YAML needs PyYAML: pip install pyyaml") from err
                raw = yaml.safe_load(text)
            else:
                raise ConfigurationError(f"unsupported config format '{suffix}' (use .yaml, .yml or .json)")
        except (json.JSONDecodeError, ValueError) as err:
            raise ConfigurationError(f"{p}: cannot parse ({err})") from err
        except Exception as err:   # yaml.YAMLError without importing yaml at module level
            if isinstance(err, ConfigurationError):
                raise
            raise ConfigurationError(f"{p}: cannot parse ({err})") from err
        if raw is not None and not isinstance(raw, dict):
            raise ConfigurationError(f"{p}: top level must be a mapping")
        return cls.from_dict(raw or {})

    # ------------------------------------------------------------------ export
    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python (JSON-safe) representation; ``from_dict(to_dict())`` reproduces the config."""
        return _jsonable(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
# Loader helpers
# --------------------------------------------------------------------------- #
def _tuplify(v):
    return tuple(_tuplify(x) for x in v) if isinstance(v, (list, tuple)) else v


def _make(cls: Type[T_], mapping: Optional[Mapping[str, Any]], section: str) -> T_:
    """Instantiate dataclass ``cls`` from a mapping, rejecting unknown keys.
    Lists become tuples for fields whose default is a tuple (YAML/JSON have no tuples)."""
    if mapping is None:
        return cls()
    if not isinstance(mapping, Mapping):
        raise ConfigurationError(f"config.{section}: expected a mapping, got {type(mapping).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        raise ConfigurationError(f"config.{section}: unknown key(s) {unknown}; valid: {sorted(known)}")
    kwargs = {}
    for k, v in mapping.items():
        if isinstance(known[k].default, tuple):
            v = _tuplify(v)
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as err:
        raise ConfigurationError(f"config.{section}: {err}") from err


def _make_bess(mapping: Optional[Mapping[str, Any]]) -> BESSParams:
    """``bess: {profile: large|medium, <any BESSParams field>: override}``."""
    if mapping is None:
        return BESSParams.from_profile("large")
    if not isinstance(mapping, Mapping):
        raise ConfigurationError("config.bess: expected a mapping")
    m = dict(mapping)
    profile = m.pop("profile", "large")
    unknown = sorted(set(m) - {f.name for f in fields(BESSParams)})
    if unknown:
        raise ConfigurationError(f"config.bess: unknown key(s) {unknown}")
    return BESSParams.from_profile(profile, **m)


def _make_factory(mapping: Optional[Mapping[str, Any]]) -> Optional[FactoryParams]:
    """``factory:`` absent / null -> None (use the demo generator). Otherwise explicit jobs and profiles."""
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise ConfigurationError("config.factory: expected a mapping or null")
    m = dict(mapping)
    jobs_raw = m.pop("jobs", [])
    if not isinstance(jobs_raw, (list, tuple)):
        raise ConfigurationError("config.factory.jobs: expected a list")
    jobs = [_make(BatchJob, j, f"factory.jobs[{k}]") for k, j in enumerate(jobs_raw)]
    unknown = sorted(set(m) - {f.name for f in fields(FactoryParams)})
    if unknown:
        raise ConfigurationError(f"config.factory: unknown key(s) {unknown}")
    return FactoryParams(jobs=jobs, **m)


def _jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x
