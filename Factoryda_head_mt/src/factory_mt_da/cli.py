"""Command-line interface for the factory microturbine scheduler."""

import argparse
import logging

from ._internal import log
from .benchmark import make_benchmark_instance
from .data import Instance, SchedulerConfig, SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .optimizer import optimize_day_ahead
from .parameters import (
    BENCHMARK_SEED,
    DEFAULT_CHARGE_MIN_POWER_FUEL,
    DEFAULT_MAX_BATCHES_PER_MACHINE,
    DEFAULT_MIP_GAP,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOLVER,
    DEFAULT_START_STEP_H,
    DEFAULT_TIME_LIMIT_S,
)
from .plotting import plot_schedule


def _parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factory + microturbine day-ahead scheduling (MILP)"
    )
    parser.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES_PER_MACHINE)
    parser.add_argument("--start-step", type=float, default=DEFAULT_START_STEP_H)
    parser.add_argument("--allow-repeat-tasks", action="store_true")
    parser.add_argument("--no-mt", action="store_true")
    parser.add_argument(
        "--mt-charge-min",
        action="store_true",
        default=DEFAULT_CHARGE_MIN_POWER_FUEL,
        help="charge fuel for the minimum MT output",
    )
    parser.add_argument("--mt-initial-on", action="store_true")
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export-instance", help="write the benchmark/input instance as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        instance = (
            Instance.load(args.instance)
            if args.instance
            else make_benchmark_instance(args.seed, max_batches=args.max_batches)
        )
        if args.no_mt:
            instance.mt = None
        if args.mt_initial_on and instance.mt is not None:
            instance.mt.initial_on = True
            instance.mt.initial_power_mw = instance.mt.p_min_mw
            instance.mt.validate()
        if args.export_instance:
            instance.save(args.export_instance)

        config = SchedulerConfig(
            start_step_h=args.start_step,
            unique_tasks=not args.allow_repeat_tasks,
            charge_min_power_fuel=args.mt_charge_min,
            solver=SolverSettings(
                name=args.solver,
                mip_gap=args.mip_gap,
                time_limit_s=args.time_limit,
                threads=args.threads,
                verbose=args.verbose,
            ),
        )
        result = optimize_day_ahead(instance, config)
        print(result.summary())
        print(result.jobs[[
            "machine", "position_n", "task", "start_time", "end_time", "power_mw",
            "units_out", "energy_mwh", "energy_cost_eur",
        ]].round(2).to_string(index=False))
        print(result.hourly[[
            "hour", "price_eur_mwh", "total_load_mw", "p_mt_mw", "mt_on", "p_da_mw",
            "grid_cost_eur", "mt_fuel_cost_eur", "mt_startstop_cost_eur", "cost_eur",
            "inventory_units",
        ]].round(2).to_string(index=False))
        output = result.save(args.out)
        if args.plot:
            plot_schedule(instance, result, output / "schedule.png")
        print(f"\nResults written to {output.resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, InfeasibleScheduleError, SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130