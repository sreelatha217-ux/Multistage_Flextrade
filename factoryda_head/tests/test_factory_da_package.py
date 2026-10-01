import factory_da
import factory_da_scheduler


def test_public_api_and_legacy_compatibility():
    assert factory_da_scheduler.Instance is factory_da.Instance
    assert factory_da_scheduler.optimize_day_ahead is factory_da.optimize_day_ahead
    assert factory_da.make_benchmark_instance().validate()


def test_time_indexed_model_builds():
    instance = factory_da.make_benchmark_instance()
    config = factory_da.SchedulerConfig(start_step_h=6.0)
    candidates = factory_da.build_candidates(instance, config.start_step_h)
    model = factory_da.build_model(instance, config, candidates)

    assert len(candidates) > 0
    assert len(model.s) == len(candidates)


def test_end_to_end_schedule_is_verified():
    instance = factory_da.Instance(
        machines=["furnace"],
        tasks=["batch"],
        power_mw=[[5.0]],
        duration_h=[[1.0]],
        yield_units=[10.0],
        base_load_mw=[2.0] * 4,
        price_eur_mwh=[50.0, 20.0, 30.0, 40.0],
        demand_units=[0.0, 0.0, 0.0, 10.0],
        horizon_h=4,
        buffer_h=0.0,
        grid_limit_mw=20.0,
        max_batches_per_machine=1,
        inventory_init=0.0,
        inventory_max=20.0,
    )

    result = factory_da.optimize_day_ahead(
        instance, factory_da.SchedulerConfig(start_step_h=1.0)
    )

    assert result.status == "optimal"
    assert result.verification["passed"] is True
    assert len(result.jobs) == 1