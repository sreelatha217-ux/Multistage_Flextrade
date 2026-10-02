"""Command-line interface for stochastic day-ahead and intraday scheduling."""

import argparse
import logging
from pathlib import Path

from .cli import _apply_asset_overrides, _override_solver
from .config import RunConfig
from .data import Instance, SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .intraday_optimizer import optimize_intraday
from .intraday_plotting import plot_bess_scenarios, plot_intraday
from .parameters import DEFAULT_INTRADAY_OUTPUT_DIR
from .scenarios import IntradayMarket, ScenarioSet

log = logging.getLogger("factory_twostage")


def _parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factory + microturbine + BESS day-ahead / intraday two-stage scheduler"
    )
    instance = parser.add_argument_group("instance")
    instance.add_argument("--config", help="JSON run configuration")
    instance.add_argument("--instance", help="JSON instance file")
    instance.add_argument("--seed", type=int)
    instance.add_argument("--max-batches", type=int)
    instance.add_argument("--start-step", type=float)
    instance.add_argument("--allow-repeat-tasks", action="store_true")
    instance.add_argument("--no-mt", action="store_true")
    instance.add_argument("--no-bess", action="store_true")
    instance.add_argument("--mt-ramp-up", type=float)
    instance.add_argument("--mt-ramp-down", type=float)
    instance.add_argument("--mt-initial-on", action="store_true")
    instance.add_argument("--bess-degradation", type=float)
    instance.add_argument("--bess-degradation-basis", choices=["throughput", "discharge"])
    instance.add_argument("--bess-power", type=float)
    instance.add_argument("--bess-energy", type=float)
    instance.add_argument("--bess-relax-exclusive", action="store_true")
    instance.add_argument("--export-instance")

    intraday = parser.add_argument_group("intraday market and scenarios")
    intraday.add_argument("--id-cap-buy", type=float)
    intraday.add_argument("--id-cap-sell", type=float)
    intraday.add_argument("--scenarios", type=int)
    intraday.add_argument("--scenario-seed", type=int)
    intraday.add_argument("--price-sigma", type=float)
    intraday.add_argument("--level-sigma", type=float)
    intraday.add_argument("--load-sigma", type=float)
    intraday.add_argument("--scenario-file", help="JSON scenario set; overrides generation")
    intraday.add_argument("--vss", action="store_true", help="compute value of the stochastic solution")

    solver = parser.add_argument_group("solver and output")
    solver.add_argument("--solver")
    solver.add_argument("--mip-gap", type=float)
    solver.add_argument("--time-limit", type=float)
    solver.add_argument("--threads", type=int)
    solver.add_argument("--out")
    solver.add_argument("--plot", action="store_true")
    solver.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


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
        for option, setting in (
            (args.id_cap_buy, "id_cap_buy_mw"),
            (args.id_cap_sell, "id_cap_sell_mw"),
            (args.scenarios, "scenario_count"),
            (args.scenario_seed, "scenario_seed"),
            (args.price_sigma, "price_sigma"),
            (args.level_sigma, "level_sigma"),
            (args.load_sigma, "load_sigma_mw"),
        ):
            if option is not None:
                setattr(run_config, setting, option)
        _override_solver(run_config, args)

        instance = Instance.load(args.instance) if args.instance else run_config.make_instance()
        _apply_asset_overrides(instance, args)
        if args.export_instance:
            instance.save(args.export_instance)
        market = IntradayMarket(run_config.id_cap_buy_mw, run_config.id_cap_sell_mw)
        scenarios = (
            ScenarioSet.load(args.scenario_file)
            if args.scenario_file
            else ScenarioSet.generate(
                instance,
                n=run_config.scenario_count,
                seed=run_config.scenario_seed,
                price_sigma=run_config.price_sigma,
                level_sigma=run_config.level_sigma,
                load_sigma_mw=run_config.load_sigma_mw,
            )
        )
        result = optimize_intraday(
            instance,
            scenarios,
            market,
            run_config.scheduler_config(),
            compute_vss=args.vss,
        )
        print(result.summary())
        print(result.jobs[[
            "machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"
        ]].round(2).to_string(index=False))
        print(result.da_position.round(2).to_string(index=False))
        print(result.scenarios.round(2).to_string(index=False))
        output = result.save(args.out or DEFAULT_INTRADAY_OUTPUT_DIR)
        if args.plot:
            plot_intraday(instance, result, output / "schedule.png")
            if instance.bess is not None:
                plot_bess_scenarios(instance, result, output / "bess_scenarios.png")
        print(f"\nResults written to {Path(output).resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (
        SolverUnavailableError,
        InfeasibleScheduleError,
        SolveFailedError,
        VerificationError,
    ) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130