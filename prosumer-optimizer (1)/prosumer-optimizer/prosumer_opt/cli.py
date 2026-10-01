"""
Command line interface.

    python -m prosumer_opt                                   # built-in defaults
    python -m prosumer_opt --config configs/default.yaml
    python -m prosumer_opt --config my.yaml --mip-gap 1e-3 --out results/run1
    python -m prosumer_opt --dump-config                     # print the resolved configuration

Flags override the config file, which overrides the built-in defaults.
Exit codes: 0 ok, 2 invalid input/configuration, 3 optimisation failure, 130 interrupted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Optional, Sequence

from ._version import __version__
from .config import AppConfig
from .exceptions import (ConfigurationError, DataValidationError, ModelInfeasibleError,
                         SolveFailedError, SolverUnavailableError)
from .logging_setup import configure_logging
from .parameters import BESSParams, TimeGrid
from .pipeline import run_pipeline

log = logging.getLogger("prosumer_opt")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="prosumer-opt",
        description="Multi-stage DA + intraday + real-time optimisation for an industrial prosumer")
    ap.add_argument("--config", help="YAML/JSON configuration file")
    ap.add_argument("--dump-config", action="store_true", help="print the resolved configuration and exit")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    g = ap.add_argument_group("overrides (take precedence over the config file)")
    g.add_argument("--id-scenarios", type=int, help="number of intraday price scenarios")
    g.add_argument("--rt-branches", type=int, help="real-time branches per scenario")
    g.add_argument("--dt", type=float, help="time step in hours")
    g.add_argument("--seed", type=int, help="master seed: factory=SEED, scenarios=SEED+4, intraday=SEED+92")
    g.add_argument("--shift-window", type=float, help="intraday batch shift window in hours, 0 disables")
    g.add_argument("--intraday-step", type=int, help="re-optimise from this step (0 = skip)")
    g.add_argument("--spike", type=float, help="ID price multiplier after the re-optimisation step")
    g.add_argument("--solver", help="e.g. appsi_highs, appsi_gurobi, gurobi, cplex, cbc")
    g.add_argument("--mip-gap", type=float, help="relative MIP gap (e.g. 1e-4 = 0.01 %%)")
    g.add_argument("--time-limit", type=float, help="solver time limit per stage, seconds")
    g.add_argument("--threads", type=int, help="solver threads")
    g.add_argument("--bess", choices=["large", "medium"], help="battery size profile")
    g.add_argument("--out", help="output directory")
    g.add_argument("--no-save", action="store_true", help="do not write result files")
    g.add_argument("--verbose", action="store_true", help="DEBUG logging and solver output")
    return ap


def apply_overrides(cfg: AppConfig, a: argparse.Namespace) -> AppConfig:
    """Apply every flag that was actually given to ``cfg`` (in place) and return it."""
    if a.dt is not None:
        cfg.time = TimeGrid(cfg.time.horizon_h, a.dt)
    if a.bess is not None:
        cfg.bess = BESSParams.from_profile(a.bess)
    if a.id_scenarios is not None or a.rt_branches is not None:
        cfg.scenarios = replace(
            cfg.scenarios,
            n_id_scenarios=a.id_scenarios if a.id_scenarios is not None else cfg.scenarios.n_id_scenarios,
            n_rt_branches=a.rt_branches if a.rt_branches is not None else cfg.scenarios.n_rt_branches)
    if a.seed is not None:
        cfg.demo_factory = replace(cfg.demo_factory, seed=a.seed)
        cfg.scenarios = replace(cfg.scenarios, seed=a.seed + 4)
        cfg.intraday = replace(cfg.intraday, seed=a.seed + 92)
    if a.intraday_step is not None:
        cfg.intraday = replace(cfg.intraday, step=a.intraday_step)
    if a.spike is not None:
        cfg.intraday = replace(cfg.intraday, spike=a.spike)
    if a.shift_window is not None:
        cfg.demo_factory = replace(cfg.demo_factory, shift_window_h=a.shift_window)
        if cfg.factory is not None:
            cfg.factory = replace(cfg.factory, intraday_shift_window_h=a.shift_window)
    if a.solver is not None:
        cfg.solver.name = a.solver
    if a.mip_gap is not None:
        cfg.solver.mip_gap = a.mip_gap
    if a.time_limit is not None:
        cfg.solver.time_limit_s = a.time_limit
    if a.threads is not None:
        cfg.solver.threads = a.threads
    if a.verbose:
        cfg.solver.verbose = True
    if a.out is not None:
        cfg.output = replace(cfg.output, directory=a.out)
    if a.no_save:
        cfg.output = replace(cfg.output, save=False)
    return cfg


def _print_report(res) -> None:
    da, idr, t0 = res.day_ahead, res.intraday, res.config.intraday.step
    print(da.summary())
    print(da.batch_plan[["job_id", "machine", "start_hour", "end_hour", "power_mw"]].to_string(index=False))
    print(da.schedule[["hour", "da_net_mw", "mt_on", "exp_mt_mw", "exp_bess_ch_mw",
                       "exp_bess_dis_mw", "exp_soc_mwh", "exp_batch_load_mw"]].round(2).to_string())
    if idr is not None:
        print(idr.summary())
        print(idr.schedule.loc[t0:, ["hour", "da_net_mw", "exp_id_net_mw", "exp_mt_mw", "exp_bess_ch_mw",
                                     "exp_bess_dis_mw", "exp_soc_mwh", "exp_batch_load_mw",
                                     "exp_imb_surplus_mw", "exp_imb_shortfall_mw"]].round(2).to_string())
    if res.output_dir is not None:
        print(f"\nResults written to {res.output_dir.resolve()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = AppConfig.from_file(args.config) if args.config else AppConfig()
        cfg = apply_overrides(cfg, args)
        configure_logging(verbose=args.verbose, level=cfg.log_level)
        if args.dump_config:
            print(cfg.to_json())
            return 0
        res = run_pipeline(cfg)
        _print_report(res)
        return 0
    except (ConfigurationError, DataValidationError) as err:
        log.error("Invalid input: %s", err)
        return 2
    except (SolverUnavailableError, ModelInfeasibleError, SolveFailedError) as err:
        log.error("Optimisation failed: %s", err)
        return 3
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
