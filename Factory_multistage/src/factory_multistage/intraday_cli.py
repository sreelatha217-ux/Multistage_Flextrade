from __future__ import annotations

import argparse
import logging
import sys

from . import day_ahead as da
from ._internal import log
from .config import load_config_arg
from .day_ahead import Instance, SchedulerConfig, SolverSettings, VerificationError
from .exceptions import InstanceValidationError
from .intraday_optimizer import optimize_intraday
from .intraday_plotting import plot_intraday
from .scenarios import IntradayMarket, ScenarioSet


def _parse(argv=None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config_arg(argv)
    ap = argparse.ArgumentParser(description="Factory + MT + BESS day-ahead / intraday two-stage stochastic MILP")
    ap.add_argument("--config", help="JSON run configuration")
    g = ap.add_argument_group("instance")
    g.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    g.add_argument("--seed", type=int, default=2024, help="benchmark generator seed")
    g.add_argument("--max-batches", type=int, default=10)
    g.add_argument("--start-step", type=float, default=0.5)
    g.add_argument("--allow-repeat-tasks", action="store_true")
    g.add_argument("--no-mt", action="store_true")
    g.add_argument("--no-bess", action="store_true")
    g.add_argument("--mt-ramp-up", type=float, default=None)
    g.add_argument("--mt-ramp-down", type=float, default=None)
    g.add_argument("--mt-initial-on", action="store_true")
    g.add_argument("--bess-degradation", type=float, default=None)
    g.add_argument("--bess-degradation-basis", choices=["throughput", "discharge"], default=None)
    g.add_argument("--bess-power", type=float, default=None)
    g.add_argument("--bess-energy", type=float, default=None)
    g.add_argument("--bess-relax-exclusive", action="store_true", help="drop the per-scenario charge/discharge binaries")
    s = ap.add_argument_group("intraday market and scenarios")
    s.add_argument("--id-cap-buy", type=float, default=100.0, help="ID purchase depth in MW per hour")
    s.add_argument("--id-cap-sell", type=float, default=100.0, help="ID sale depth in MW per hour")
    s.add_argument("--scenarios", type=int, default=10, help="number of generated scenarios")
    s.add_argument("--scenario-seed", type=int, default=7)
    s.add_argument("--price-sigma", type=float, default=0.15, help="hourly relative ID price volatility")
    s.add_argument("--level-sigma", type=float, default=0.05, help="common daily relative ID price shift")
    s.add_argument("--load-sigma", type=float, default=0.0, help="std of the additive base-load deviation in MW")
    s.add_argument("--scenario-file", help="JSON scenario set (overrides the generator)")
    s.add_argument("--vss", action="store_true", help="also compute the value of the stochastic solution")
    o = ap.add_argument_group("solver / output")
    o.add_argument("--solver", default="appsi_highs")
    o.add_argument("--mip-gap", type=float, default=1e-3)
    o.add_argument("--time-limit", type=float, default=300.0)
    o.add_argument("--threads", type=int, default=None)
    o.add_argument("--out", default="da_id_results")
    o.add_argument("--plot", action="store_true")
    o.add_argument("--verbose", action="store_true")
    if config is not None:
        ap.set_defaults(**config.cli_defaults("id"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        inst = (Instance.load(a.instance) if a.instance else
                da.make_benchmark_instance(a.seed, max_batches=a.max_batches, with_mt=not a.no_mt, with_bess=not a.no_bess))
        da.apply_instance_overrides(inst, a)
        mkt = IntradayMarket(a.id_cap_buy, a.id_cap_sell).validate()
        scen = (ScenarioSet.load(a.scenario_file) if a.scenario_file else
                ScenarioSet.generate(inst, a.scenarios, a.scenario_seed, a.price_sigma, a.level_sigma, a.load_sigma))
        cfg = SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                              solver=SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_intraday(inst, scen, mkt, cfg, compute_vss=a.vss)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"]]
              .round(2).to_string(index=False))
        print(res.da_position.round(2).to_string(index=False))
        print(res.scenarios.round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_intraday(inst, res, out / "schedule.png")
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
