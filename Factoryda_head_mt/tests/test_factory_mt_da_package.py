import factory_mt_da
import factory_mt_da_scheduler
import factory_mt_da_scheduler_v2


def test_benchmark_and_time_indexed_model_build():
    assert factory_mt_da_scheduler.Instance is factory_mt_da_scheduler_v2.Instance
    assert factory_mt_da_scheduler.__version__ == factory_mt_da_scheduler_v2.__version__
    instance = factory_mt_da.make_benchmark_instance()
    config = factory_mt_da.SchedulerConfig(start_step_h=6.0)
    candidates = factory_mt_da.build_candidates(instance, config.start_step_h)
    model = factory_mt_da.build_model(instance, config, candidates)

    assert candidates
    assert len(model.s) == len(candidates)
    assert len(model.Pmt) == instance.horizon_h


def test_small_schedule_dispatches_mt_and_writes_tables_and_figure(tmp_path):
    instance = factory_mt_da.Instance(
        machines=["furnace"],
        tasks=["batch"],
        power_mw=[[0.0]],
        duration_h=[[1.0]],
        yield_units=[0.0],
        base_load_mw=[20.0] * 4,
        price_eur_mwh=[100.0] * 4,
        demand_units=[0.0] * 4,
        horizon_h=4,
        buffer_h=0.0,
        grid_limit_mw=30.0,
        max_batches_per_machine=1,
        inventory_init=0.0,
        inventory_max=20.0,
    )

    result = factory_mt_da.optimize_day_ahead(
        instance,
        factory_mt_da.SchedulerConfig(start_step_h=1.0),
    )

    assert result.status == "optimal"
    assert result.verification["passed"] is True
    assert result.kpis["mt_energy_mwh"] == 80.0
    assert {"mt_fuel_cost_eur", "grid_cost_eur", "mt_startstop_cost_eur"}.issubset(
        result.hourly.columns
    )
    output = result.save(tmp_path / "results")
    assert (output / "jobs.csv").is_file()
    assert (output / "hourly.csv").is_file()
    assert (output / "summary.json").is_file()
    assert factory_mt_da.plot_schedule(instance, result, output / "schedule.png")
    assert (output / "schedule.png").stat().st_size > 0