from __future__ import annotations

import argparse
import logging
import sys

from ._internal import log
from .benchmark import make_benchmark_instance
from .config import load_config_arg
from .data import Instance, SchedulerConfig, SolverSettings
from .exceptions import (
    InfeasibleScheduleError,
    InstanceValidationError,
    SolveFailedError,
    SolverUnavailableError,
    VerificationError,
)
from .optimizer import optimize_day_ahead
from .plotting import plot_schedule


def _parse(argv=None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config_arg(argv)
    ap = argparse.ArgumentParser(description="Factory + microturbine day-ahead scheduling (MILP)")
    ap.add_argument("--config", help="JSON run configuration")
    ap.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    ap.add_argument("--seed", type=int, default=2024, help="benchmark generator seed")
    ap.add_argument("--max-batches", type=int, default=10, help="N, max batches per machine")
    ap.add_argument("--start-step", type=float, default=0.5, help="start-time grid in hours")
    ap.add_argument("--allow-repeat-tasks", action="store_true", help="a task may run more than once")
    ap.add_argument("--no-mt", action="store_true", help="benchmark without the microturbine")
    ap.add_argument("--no-bess", action="store_true", help="benchmark without the battery")
    ap.add_argument("--bess-degradation", type=float, default=None, help="C_TP in EUR/MWh throughput (default 35)")
    ap.add_argument("--bess-degradation-basis", choices=["throughput", "discharge"], default=None,
                    help="charge C_TP on charge+discharge energy (default, skill) or on discharge energy only")
    ap.add_argument("--bess-power", type=float, default=None, help="BESS max charge/discharge power in MW")
    ap.add_argument("--bess-energy", type=float, default=None, help="BESS nominal energy capacity in MWh")
    ap.add_argument("--bess-relax-exclusive", action="store_true",
                    help="drop the charge/discharge exclusivity binaries (LP relaxation of eq. 3.3.4)")
    ap.add_argument("--mt-ramp-up", type=float, default=None, help="MT ramp-up limit RU in MW/h (default 20)")
    ap.add_argument("--mt-ramp-down", type=float, default=None, help="MT ramp-down limit RD in MW/h (default 20)")
    ap.add_argument("--mt-initial-on", action="store_true", help="MT is online before t=0 at Pmin")
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="da_mt_results")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--export-instance", help="write the instance used to this JSON path")
    ap.add_argument("--verbose", action="store_true")
    if config is not None:
        ap.set_defaults(**config.cli_defaults("da"))
    return ap.parse_args(argv)


def apply_instance_overrides(inst: Instance, a: argparse.Namespace) -> Instance:
    """Apply the MT / BESS command-line overrides to an instance (missing CLI attributes are ignored)."""
    g = lambda name, default=None: getattr(a, name, default)
    if g("no_mt"):
        inst.mt = None
    if inst.mt is not None:
        if g("mt_ramp_up") is not None:
            inst.mt.ramp_up_mw_h = a.mt_ramp_up
        if g("mt_ramp_down") is not None:
            inst.mt.ramp_down_mw_h = a.mt_ramp_down
        if g("mt_initial_on"):
            inst.mt.initial_on, inst.mt.initial_power_mw = True, inst.mt.p_min_mw
        inst.mt.validate()
    if g("no_bess"):
        inst.bess = None
    if inst.bess is not None:
        if g("bess_degradation") is not None:
            inst.bess.degradation_eur_mwh = a.bess_degradation
        if g("bess_degradation_basis") is not None:
            inst.bess.degradation_basis = a.bess_degradation_basis
        if g("bess_power") is not None:
            inst.bess.p_max_mw = a.bess_power
        if g("bess_energy") is not None:
            inst.bess.e_max_mwh = a.bess_energy
        if g("bess_relax_exclusive"):
            inst.bess.enforce_exclusive = False
        inst.bess.validate()
    return inst


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        if a.instance:
            inst = Instance.load(a.instance)
        else:
            inst = make_benchmark_instance(a.seed, max_batches=a.max_batches, with_mt=not a.no_mt,
                                       with_bess=not a.no_bess)
        apply_instance_overrides(inst, a)
        if a.export_instance:
            inst.save(a.export_instance)
        cfg = SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                              solver=SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_day_ahead(inst, cfg)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw",
                        "units_out", "energy_mwh", "energy_cost_eur"]].round(2).to_string(index=False))
        print(res.hourly[["hour", "price_buy_eur_mwh", "price_sell_eur_mwh", "total_load_mw", "p_mt_mw", "mt_on",
                          "p_buy_mw", "p_sell_mw", "p_bess_ch_mw", "p_bess_dis_mw", "soc_mwh",
                          "cost_eur", "inventory_units"]].round(2).to_string(index=False))
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
