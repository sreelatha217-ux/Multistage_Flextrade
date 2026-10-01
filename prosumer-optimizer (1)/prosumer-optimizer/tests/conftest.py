"""Shared fixtures. The solved fixtures are module-scoped so the MILPs are solved once per test file."""
from dataclasses import replace

import pytest

pytest.importorskip("pyomo")
pytest.importorskip("highspy", reason="HiGHS (pip install highspy) is needed for solver tests")

from prosumer_opt import AppConfig, ScenarioSettings, IntradaySettings  # noqa: E402
from prosumer_opt.data import build_demo_intraday_scenario, build_demo_tree  # noqa: E402
from prosumer_opt.pipeline import build_optimizer  # noqa: E402
from prosumer_opt.state import InitialState, IntradayInputs  # noqa: E402


def make_config(n_id=3, n_rt=2, gap=1e-6) -> AppConfig:
    """The configuration the original prosumer_optimizer.py regression numbers were recorded with."""
    cfg = AppConfig()
    cfg.scenarios = ScenarioSettings(n_id_scenarios=n_id, n_rt_branches=n_rt)
    cfg.intraday = IntradaySettings(step=10)
    cfg.solver.mip_gap = gap
    cfg.solver.time_limit_s = 120.0
    cfg.output = replace(cfg.output, save=False)
    return cfg


@pytest.fixture(scope="module")
def cfg():
    return make_config()


@pytest.fixture(scope="module")
def optimizer(cfg):
    return build_optimizer(cfg)


@pytest.fixture(scope="module")
def tree(cfg):
    return build_demo_tree(cfg.time, cfg.scenarios)


@pytest.fixture(scope="module")
def da(optimizer, tree):
    return optimizer.solve_day_ahead(tree)


@pytest.fixture(scope="module")
def intraday(optimizer, da, tree, cfg):
    t0 = cfg.intraday.step
    scen = build_demo_intraday_scenario(cfg.time, tree, cfg.intraday, spike_from_step=t0)
    return optimizer.solve_intraday(da, IntradayInputs(t0, InitialState.from_result(da, t0, cfg.time), scen))
