import json

import numpy as np
import pytest

from factory_twostage import (
    BESS,
    Instance,
    IntradayMarket,
    Microturbine,
    RunConfig,
    ScenarioSet,
    SchedulerConfig,
    SolverSettings,
    build_candidates,
    build_model,
    build_intraday_model,
    optimize_intraday,
)
from factory_twostage.exceptions import InstanceValidationError
from factory_twostage.intraday_plotting import plot_bess_scenarios


def small_instance():
    return Instance(
        machines=["furnace-1"],
        tasks=["batch-1"],
        power_mw=np.array([[2.0]]),
        duration_h=np.array([[1.0]]),
        yield_units=np.array([10.0]),
        base_load_mw=np.array([1.0, 1.0]),
        price_buy_eur_mwh=np.array([100.0, 120.0]),
        price_sell_eur_mwh=np.array([50.0, 60.0]),
        demand_units=np.array([0.0, 0.0]),
        horizon_h=2,
        buffer_h=0.0,
        grid_limit_mw=50.0,
        grid_sell_limit_mw=50.0,
        max_batches_per_machine=1,
        mt=Microturbine(),
        bess=BESS(),
    ).validate()


def test_bundled_run_config_and_round_trip():
    config = RunConfig.bundled()

    assert config.scenario_count == 10
    assert config.id_cap_buy_mw == 100.0
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_scenario_generation_is_reproducible_and_valid():
    instance = small_instance()
    first = ScenarioSet.generate(instance, n=3, seed=19, load_sigma_mw=0.1)
    second = ScenarioSet.generate(instance, n=3, seed=19, load_sigma_mw=0.1)

    np.testing.assert_allclose(first.id_buy_eur_mwh, second.id_buy_eur_mwh)
    np.testing.assert_allclose(first.load_dev_mw, second.load_dev_mw)
    np.testing.assert_allclose(first.prob, np.full(3, 1 / 3))


def test_invalid_scenario_probabilities_are_rejected():
    instance = small_instance()
    scenarios = ScenarioSet(
        prob=np.array([0.4, 0.4]),
        id_buy_eur_mwh=np.ones((2, 2)),
        id_sell_eur_mwh=np.zeros((2, 2)),
        load_dev_mw=np.zeros((2, 2)),
    )

    with pytest.raises(InstanceValidationError, match="sum to 1"):
        scenarios.validate(instance)


def test_intraday_model_shares_stage_one_and_indexes_recourse_by_scenario():
    instance = small_instance()
    scenarios = ScenarioSet.generate(instance, n=2, seed=2)
    candidates = build_candidates(instance, step_h=1.0)
    model = build_intraday_model(
        instance, IntradayMarket(), scenarios, SchedulerConfig(start_step_h=1.0), candidates
    )

    assert len(model.s) == 2
    assert len(model.Pbuy) == instance.horizon_h
    assert len(model.u) == instance.horizon_h
    assert len(model.Ibuy) == scenarios.n * instance.horizon_h
    assert len(model.Pmt) == scenarios.n * instance.horizon_h
    assert len(model.SoC) == scenarios.n * instance.horizon_h
    assert len(model.c_power_balance) == scenarios.n * instance.horizon_h


def test_day_ahead_model_keeps_shared_factory_and_commitment_constraints():
    instance = small_instance()
    candidates = build_candidates(instance, step_h=1.0)
    model = build_model(instance, SchedulerConfig(start_step_h=1.0), candidates)

    assert len(model.s) == 2
    assert len(model.c_inventory) == instance.horizon_h
    assert len(model.c_commitment) == instance.horizon_h


def test_run_config_rejects_zero_scenarios():
    with pytest.raises(InstanceValidationError, match="scenario count"):
        RunConfig.from_dict({"intraday": {"scenarios": 0}})


def test_two_stage_solve_verifies_and_writes_results(tmp_path):
    instance = small_instance()
    scenarios = ScenarioSet.generate(instance, n=2, seed=4)
    config = SchedulerConfig(start_step_h=1.0, solver=SolverSettings(time_limit_s=30.0))

    result = optimize_intraday(instance, scenarios, IntradayMarket(), config)

    assert result.verification["passed"]
    assert result.objective_eur == pytest.approx(result.verification["recomputed_cost_eur"])
    output = result.save(tmp_path / "results")
    assert {
        "jobs.csv",
        "da_position.csv",
        "scenarios.csv",
        "hourly_scenarios.csv",
        "scenario_set.json",
        "summary.json",
    } <= {path.name for path in output.iterdir()}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["verification"]["passed"]
    bess_plot = tmp_path / "bess_scenarios.png"
    assert plot_bess_scenarios(instance, result, bess_plot)
    assert bess_plot.stat().st_size > 0