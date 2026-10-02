"""Command-line interface for the factory MT+BESS scheduler."""

import argparse
import logging
from pathlib import Path

from .config import RunConfig
from .data import Instance, SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .optimizer import optimize_day_ahead
from .parameters import DEFAULT_OUTPUT_DIR
from .plotting import plot_schedule

log = logging.getLogger("factory_mt_bess_da")


def _parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factory + microturbine + BESS day-ahead scheduler")
    parser.add_argument("--config", help="JSON run configuration")
    parser.add_argument("--instance", help="JSON instance file; takes precedence over benchmark generation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--start-step", type=float)
    parser.add_argument("--allow-repeat-tasks", action="store_true")
    parser.add_argument("--no-mt", action="store_true")
    parser.add_argument("--no-bess", action="store_true")
    parser.add_argument("--bess-degradation", type=float)
    parser.add_argument("--bess-degradation-basis", choices=["throughput", "discharge"])
    parser.add_argument("--bess-power", type=float)
    parser.add_argument("--bess-energy", type=float)
    parser.add_argument("--bess-relax-exclusive", action="store_true")
    parser.add_argument("--mt-ramp-up", type=float)
    parser.add_argument("--mt-ramp-down", type=float)
    parser.add_argument("--mt-initial-on", action="store_true")
    parser.add_argument("--solver")
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export-instance", help="write the generated/input instance to JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _override_solver(config: RunConfig, args: argparse.Namespace) -> None:
    settings = config.solver
    config.solver = SolverSettings(
        name=args.solver if args.solver is not None else settings.name,
        mip_gap=args.mip_gap if args.mip_gap is not None else settings.mip_gap,
        time_limit_s=args.time_limit if args.time_limit is not None else settings.time_limit_s,
        threads=args.threads if args.threads is not None else settings.threads,
        verbose=args.verbose or settings.verbose,
    )


def _apply_asset_overrides(instance: Instance, args: argparse.Namespace) -> None:
    if args.no_mt:
        instance.mt = None
    elif instance.mt is not None:
        if args.mt_ramp_up is not None:
            instance.mt.ramp_up_mw_h = args.mt_ramp_up
        if args.mt_ramp_down is not None:
            instance.mt.ramp_down_mw_h = args.mt_ramp_down
        if args.mt_initial_on:
            instance.mt.initial_on = True
            instance.mt.initial_power_mw = instance.mt.p_min_mw
        instance.mt.validate()
    if args.no_bess:
        instance.bess = None
    elif instance.bess is not None:
        if args.bess_degradation is not None:
            instance.bess.degradation_eur_mwh = args.bess_degradation
        if args.bess_degradation_basis is not None:
            instance.bess.degradation_basis = args.bess_degradation_basis
        if args.bess_power is not None:
            instance.bess.p_max_mw = args.bess_power
        if args.bess_energy is not None:
            instance.bess.e_max_mwh = args.bess_energy
        if args.bess_relax_exclusive:
            instance.bess.enforce_exclusive = False
        instance.bess.validate()


def main(argv=None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run_config = RunConfig.load(args.config) if args.config else RunConfig.bundled()
        if args.seed is not None:
            run_config.seed = args.seed
        if args.max_batches is not None:
            run_config.max_batches = args.max_batches
        if args.start_step is not None:
            run_config.start_step_h = args.start_step
        if args.allow_repeat_tasks:
            run_config.unique_tasks = False
        if args.no_mt:
            run_config.with_mt = False
        if args.no_bess:
            run_config.with_bess = False
        _override_solver(run_config, args)

        instance = Instance.load(args.instance) if args.instance else run_config.make_instance()
        _apply_asset_overrides(instance, args)
        if args.mt_initial_on and instance.mt is not None:
            instance.mt.initial_on = True
            instance.mt.initial_power_mw = instance.mt.p_min_mw
            instance.mt.validate()
        if args.export_instance:
            instance.save(args.export_instance)

        scheduler_config = run_config.scheduler_config()
        result = optimize_day_ahead(instance, scheduler_config)
        print(result.summary())
        print(result.jobs[[
            "machine", "position_n", "task", "start_time", "end_time", "power_mw",
            "units_out", "energy_mwh", "energy_cost_eur",
        ]].round(2).to_string(index=False))
        print(result.hourly[[
            "hour", "price_buy_eur_mwh", "price_sell_eur_mwh", "total_load_mw",
            "p_mt_mw", "mt_on", "p_buy_mw", "p_sell_mw", "p_bess_ch_mw",
            "p_bess_dis_mw", "soc_mwh", "cost_eur", "inventory_units",
        ]].round(2).to_string(index=False))
        output = result.save(args.out or DEFAULT_OUTPUT_DIR)
        if args.plot:
            plot_schedule(instance, result, output / "schedule.png")
        print(f"\nResults written to {Path(output).resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, InfeasibleScheduleError, SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130
