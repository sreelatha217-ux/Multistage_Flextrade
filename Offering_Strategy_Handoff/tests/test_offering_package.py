import numpy as np
import pandas as pd

import factory_mt_da_scheduler as da
import factory_mt_offering_strategy as strategy
from factory_mt_offering_cli import _parse
from factory_mt_production_rescheduler import reschedule_production_by_id_scenario


def test_monotone_chain_separates_ordered_and_tied_prices():
    ordered, tied = strategy.monotone_chain(np.array([30.0, 10.0, 20.0, 20.0]))

    assert ordered == [(1, 2), (3, 0)]
    assert tied == [(2, 3)]


def test_cli_delegates_shared_options_and_keeps_offering_options():
    args = _parse(["--beta", "0.4", "--scenarios", "3", "--no-reserve",
                   "--id-reschedule-hour", "14"], strategy)

    assert args.beta == 0.4
    assert args.scenarios == 3
    assert args.no_reserve
    assert args.id_reschedule_hour == 14.0


def test_intraday_reschedule_keeps_jobs_started_before_cutoff():
    inst = da.make_benchmark_instance(seed=2024, n_tasks=30, max_batches=1, with_mt=False, with_bess=False)
    inst = strategy.shrink_instance(inst, n_tasks=2, demand_frac=0.55, max_batches=1)
    scenarios = strategy.OfferingScenarioSet.generate(inst, n=2, seed=17)
    cfg = da.SchedulerConfig(start_step_h=1.0,
                             solver=da.SolverSettings("appsi_highs", 1e-3, 60.0))
    baseline_jobs = pd.DataFrame([
        dict(machine=inst.machines[0], task=inst.tasks[0], start_h=0.0),
        dict(machine=inst.machines[1], task=inst.tasks[1], start_h=2.0),
    ])

    jobs, summary = reschedule_production_by_id_scenario(
        inst, baseline_jobs, np.zeros(inst.horizon_h), scenarios, cfg, reschedule_hour=1.0
    )

    assert set(jobs.id_scenario) == {0, 1}
    frozen = jobs[jobs.task == inst.tasks[0]]
    assert len(frozen) == 2
    assert np.all(frozen.start_h == 0.0)
    assert (summary.production_reschedule_savings_eur >= -1e-6).all()
    assert summary.verification_passed.all()