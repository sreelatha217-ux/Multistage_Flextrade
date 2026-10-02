import numpy as np
import pytest

from factory_multistage import (
    BESS,
    BalancingMarket,
    Instance,
    InstanceValidationError,
    IntradayMarket,
    Microturbine,
    RealTimeSet,
    RunConfig,
    ScenarioSet,
    SchedulerConfig,
    SolverSettings,
    build_candidates,
    build_intraday_model,
    build_model,
    build_realtime_model,
    optimize_realtime,
)
from factory_multistage.cli import _parse as parse_da_args
from factory_multistage.intraday_cli import _parse as parse_id_args
from factory_multistage.realtime_cli import _parse as parse_rt_args
from factory_multistage.realtime_plotting import plot_realtime


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


def test_bundled_config_round_trip_and_stage_overrides():
    config = RunConfig.bundled()

    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    assert config.generate_scenarios(small_instance()).n == config.scenario_count

    config_path = "src/factory_multistage/configs/default.json"
    assert parse_da_args(["--config", config_path, "--seed", "23"]).seed == 23
    assert parse_id_args(["--config", config_path, "--scenarios", "2"]).scenarios == 2
    rt_args = parse_rt_args(["--config", config_path, "--scenarios", "2", "--rt-scenarios", "1"])
    assert (rt_args.scenarios, rt_args.rt_scenarios) == (2, 1)


def test_scenario_generation_and_probability_validation():
    instance = small_instance()
    scenarios = ScenarioSet.generate(instance, n=2, seed=7)
    real_time = RealTimeSet.generate(instance, scenarios.n, n_w=1, seed=9)

    assert scenarios.prob.sum() == pytest.approx(1.0)
    assert real_time.prob.shape == (2, 1)
    assert np.all(real_time.load_rel_dev == 0.0)

    invalid = ScenarioSet(
        prob=np.array([0.4, 0.4]),
        id_buy_eur_mwh=np.ones((2, 2)),
        id_sell_eur_mwh=np.zeros((2, 2)),
        load_dev_mw=np.zeros((2, 2)),
    )
    with pytest.raises(InstanceValidationError, match="sum to 1"):
        invalid.validate(instance)


def test_models_share_stage_one_and_expand_recourse_by_scenario():
    instance = small_instance()
    scenarios = ScenarioSet.generate(instance, n=2, seed=5)
    rt_scenarios = RealTimeSet.generate(instance, scenarios.n, n_w=1, seed=6)
    config = SchedulerConfig(start_step_h=1.0)
    market = IntradayMarket()
    candidates = build_candidates(instance, step=1.0)

    day_ahead = build_model(instance, config, candidates)
    intraday = build_intraday_model(instance, market, scenarios, config, candidates)
    realtime = build_realtime_model(
        instance,
        market,
        scenarios,
        rt_scenarios,
        BalancingMarket(),
        config,
        candidates,
    )

    assert len(day_ahead.s) == len(candidates)
    assert len(intraday.Ibuy) == scenarios.n * instance.horizon_h
    assert len(realtime.Dp) == scenarios.n * rt_scenarios.n_w * instance.horizon_h
    assert len(realtime.c_power_rt) == len(realtime.Dp)


def test_small_realtime_solve_is_verified_and_plots_all_assets(tmp_path):
    instance = small_instance()
    scenarios = ScenarioSet.generate(instance, n=1, seed=3)
    rt_scenarios = RealTimeSet.constant(scenarios.n, 1, instance.horizon_h)
    config = SchedulerConfig(
        start_step_h=1.0,
        solver=SolverSettings(time_limit_s=30.0),
    )

    result = optimize_realtime(
        instance,
        scenarios,
        rt_scenarios,
        IntradayMarket(),
        BalancingMarket(),
        config,
    )

    assert result.verification["passed"]
    assert plot_realtime(instance, result, tmp_path / "schedule_rt.png")
    assert {
        "schedule_rt.png",
        "market_positions_rt.png",
        "market_prices_rt.png",
        "bess_dispatch_rt.png",
        "microturbine_dispatch_rt.png",
        "realtime_imbalance.png",
    } <= {path.name for path in tmp_path.iterdir()}