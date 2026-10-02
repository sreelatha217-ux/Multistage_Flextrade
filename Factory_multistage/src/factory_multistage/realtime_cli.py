from __future__ import annotations

import argparse
import logging
import sys

from . import day_ahead as da
from . import intraday_cli as idm
from ._internal import log
from .config import load_config_arg
from .day_ahead import Instance, SchedulerConfig, SolverSettings, VerificationError
from .exceptions import InstanceValidationError
from .realtime_optimizer import optimize_realtime
from .realtime_plotting import plot_realtime
from .realtime_scenarios import MODES, BalancingMarket, RealTimeSet
from .realtime_testing import selftest
from .scenarios import IntradayMarket, ScenarioSet


def _rt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = ap.add_argument_group("real-time balancing (Stage 3)")
    g.add_argument("--rt-scenarios", type=int, default=5, help="RT scenarios per ID scenario (odd n adds a zero path)")
    g.add_argument("--rt-seed", type=int, default=11)
    g.add_argument("--rt-load-sigma", type=float, default=0.03, help="std of the relative factory-load deviation eta")
    g.add_argument("--rt-load-rho", type=float, default=0.6, help="hourly AR(1) persistence of eta")
    g.add_argument("--r-premium", type=float, default=0.25, help="r+ = 1 + premium * g")
    g.add_argument("--r-discount", type=float, default=0.25, help="r- = 1 - discount * g")
    g.add_argument("--r-sigma", type=float, default=0.3, help="lognormal sigma of g")
    g.add_argument("--rt-file", help="JSON RT scenario set (overrides the generator)")
    g.add_argument("--imbalance-mode", choices=MODES, default="strategic",
                   help="strategic: plan may be deliberately unbalanced; passive: plan balances the expected RT load")
    g.add_argument("--bal-exclusive", action="store_true", help="add the z_BAL binaries (redundant for lam+ >= lam-)")
    g.add_argument("--max-imbalance", type=float, default=None, help="Delta_bar for both directions in MW (default Q_buy + Q_sell)")
    g.add_argument("--vss-rt", action="store_true", help="also compute the value of modelling RT risk (one extra MILP + one re-solve)")
    g.add_argument("--selftest", action="store_true", help="run the built-in consistency checks and exit")
    return ap


def _parse(argv=None) -> argparse.Namespace:
    """RT options are parsed here; every other option (instance, MT, BESS, ID market, solver, output) is delegated
    to factory_mt_id_scheduler._parse so the two programs can never drift apart."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config_arg(argv)
    rp = _rt_parser()
    if any(a in ("-h", "--help") for a in argv):
        argparse.ArgumentParser(parents=[rp], description="Stage 3 options (all Stage 1/2 options follow)").print_help()
        print()
        idm._parse(["--help"])                      # prints the remaining options and exits
    rt_args, rest = rp.parse_known_args(argv)
    if config is not None:
        rt_defaults = config.cli_defaults("rt")
        rt_names = {
            "rt_scenarios", "rt_seed", "rt_load_sigma", "rt_load_rho", "r_premium",
            "r_discount", "r_sigma", "imbalance_mode", "bal_exclusive", "max_imbalance",
        }
        rp.set_defaults(**{name: value for name, value in rt_defaults.items() if name in rt_names})
        rt_args, rest = rp.parse_known_args(argv)
    a = idm._parse(rest)
    for key, val in vars(rt_args).items():
        setattr(a, key, val)
    if a.out == "da_id_results":
        a.out = "da_id_rt_results"
    return a


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        if a.selftest:
            return selftest(a.solver)
        inst = (Instance.load(a.instance) if a.instance else
                da.make_benchmark_instance(a.seed, max_batches=a.max_batches, with_mt=not a.no_mt, with_bess=not a.no_bess))
        da.apply_instance_overrides(inst, a)
        mkt = IntradayMarket(a.id_cap_buy, a.id_cap_sell).validate()
        scen = (ScenarioSet.load(a.scenario_file) if a.scenario_file else
                ScenarioSet.generate(inst, a.scenarios, a.scenario_seed, a.price_sigma, a.level_sigma, a.load_sigma))
        rt = (RealTimeSet.load(a.rt_file) if a.rt_file else
              RealTimeSet.generate(inst, scen.n, a.rt_scenarios, a.rt_seed, a.rt_load_sigma, a.rt_load_rho,
                                   a.r_premium, a.r_discount, a.r_sigma))
        bal = BalancingMarket(a.imbalance_mode, a.bal_exclusive, a.max_imbalance, a.max_imbalance).validate()
        cfg = SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                              solver=SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_realtime(inst, scen, rt, mkt, bal, cfg, compute_vss_rt=a.vss_rt)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"]]
              .round(2).to_string(index=False))
        print(res.da_position.round(2).to_string(index=False))
        print(res.scenarios.round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_realtime(inst, res, out / "schedule_rt.png")
        print(f"\nResults written to {out.resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (da.SolverUnavailableError, da.InfeasibleScheduleError, da.SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130
