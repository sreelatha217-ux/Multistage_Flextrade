"""Command-line interface for the scheduler."""
import argparse
import logging
import sys

from .benchmark import make_benchmark_instance
from .data import Instance, SchedulerConfig, SolverSettings
from .exceptions import (InfeasibleScheduleError, InstanceValidationError, SolveFailedError,
                         SolverUnavailableError, VerificationError)
from ._internal import log
from .optimizer import optimize_day_ahead
from .plotting import plot_schedule


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Factory day-ahead batch scheduling (MILP)")
    ap.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    ap.add_argument("--seed", type=int, default=2024, help="benchmark generator seed")
    ap.add_argument("--max-batches", type=int, default=10, help="N, max batches per machine")
    ap.add_argument("--start-step", type=float, default=0.5, help="start-time grid in hours")
    ap.add_argument("--allow-repeat-tasks", action="store_true", help="a task may run more than once")
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="da_results")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--export-instance", help="write the instance used to this JSON path")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)

def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        inst = Instance.load(a.instance) if a.instance else make_benchmark_instance(a.seed, max_batches=a.max_batches)
        if a.export_instance:
            inst.save(a.export_instance)
        cfg = SchedulerConfig(a.start_step, not a.allow_repeat_tasks,
                              SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_day_ahead(inst, cfg)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw",
                        "units_out", "energy_mwh", "energy_cost_eur"]].round(2).to_string(index=False))
        print(res.hourly[["hour", "price_eur_mwh", "base_load_mw", "batch_load_mw", "p_da_mw",
                          "inventory_units"]].round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_schedule(inst, res, out / "schedule.png")
        print(f"\nResults written to {out.resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, InfeasibleScheduleError, SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130
