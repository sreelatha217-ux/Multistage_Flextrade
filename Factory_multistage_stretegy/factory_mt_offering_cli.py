"""Command-line interface for the strategic offering optimizer."""
from __future__ import annotations

import argparse
import logging
import sys
from types import ModuleType

import factory_mt_rt_scheduler as rtm


def _offer_parser(api: ModuleType) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = ap.add_argument_group("offering strategy and risk")
    g.add_argument("--beta", type=float, default=0.0, help="risk aversion in [0,1]: (1-beta)*E[profit] + beta*CVaR")
    g.add_argument("--alpha", type=float, default=0.95, help="CVaR confidence level")
    g.add_argument("--bidding", choices=api.BIDDING, default="curve", help="curve: monotone price-quantity curves; fixed: price-taker")
    g.add_argument("--no-reserve-curves", action="store_true", help="do not force the reserve offer to be monotone in lam_SR")
    g.add_argument("--soc-terminal-baseline", action="store_true", help="SoC_T >= SoC_0 only on the no-call path")
    p = ap.add_argument_group("intraday production rescheduling")
    p.add_argument("--id-reschedule-hour", type=float, default=0.0,
                   help="delivery-hour cutoff; Stage-1 jobs already started are frozen (default 0 for next-day scheduling)")
    r = ap.add_argument_group("spinning reserve")
    r.add_argument("--no-reserve", action="store_true", help="energy-only offering")
    r.add_argument("--reserve-assets", default=",".join(api.ASSETS), help=f"comma list from {api.ASSETS}")
    r.add_argument("--mt-rr-minutes", type=float, default=10.0, help="MT reserve = ramp-up * minutes / 60")
    r.add_argument("--dr-fraction", type=float, default=0.10, help="curtailable share of the running batch load")
    r.add_argument("--dr-cost", type=float, default=150.0, help="DR activation cost in EUR/MWh")
    r.add_argument("--act-ratio", type=float, default=1.0, help="activation energy price / DA sell price")
    r.add_argument("--max-reserve", type=float, default=None, help="cap on the total reserve offer in MW")
    r.add_argument("--bess-reserve-hours", type=float, default=1.0, help="energy duration behind the BESS reserve")
    s = ap.add_argument_group("scenarios (override the ID/RT defaults; --price-sigma/--level-sigma drive the DA price)")
    s.add_argument("--scenarios", type=int, default=6, help="number of DA/SR/ID price scenarios")
    s.add_argument("--rt-scenarios", type=int, default=3, help="real-time branches per price scenario")
    s.add_argument("--id-spread-sigma", type=float, default=0.08, help="ID price volatility around the DA price")
    s.add_argument("--sr-ratio", type=float, default=0.12, help="mean SR price / DA buy price")
    s.add_argument("--sr-sigma", type=float, default=0.20, help="SR price volatility")
    s.add_argument("--rho-mean", type=float, default=0.08, help="mean fraction of the cleared reserve that is called")
    s.add_argument("--rho-sigma", type=float, default=0.8, help="lognormal sigma of the call fraction")
    s.add_argument("--rho-seed", type=int, default=13)
    s.add_argument("--offering-scenario-file", help="JSON OfferingScenarioSet (overrides the generator)")
    s.add_argument("--deploy-file", help="JSON deployment set (overrides the generator)")
    a = ap.add_argument_group("analysis")
    a.add_argument("--compare", action="store_true", help="self-schedule vs curves vs curves + reserve")
    a.add_argument("--frontier", help="comma list of beta values, e.g. 0,0.3,0.6,0.9")
    a.add_argument("--shrink", type=int, default=None, help="use only the first N tasks (fast demo plant)")
    a.add_argument("--selftest", action="store_true", help="run the built-in consistency checks and exit")
    return ap


def _parse(argv, api):
    """Parse offering options, delegating shared scheduler options to the RT CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    op = _offer_parser(api)
    if any(x in ("-h", "--help") for x in argv):
        argparse.ArgumentParser(parents=[op], description="Strategic offering options (all Stage 1-3 options follow)").print_help()
        print()
        rtm._parse(["--help"])
    own, rest = op.parse_known_args(argv)
    args = rtm._parse(rest)
    for key, value in vars(own).items():
        setattr(args, key, value)
    if args.out in ("da_id_results", "da_id_rt_results"):
        args.out = "offering_results"
    return args


def main(argv=None, api=None) -> int:
    """Run the CLI using the loaded strategy module as its domain API."""
    if api is None:
        import factory_mt_offering_strategy as api

    args = _parse(argv, api)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.selftest:
            return api.selftest(args.solver)
        inst = (api.Instance.load(args.instance) if args.instance else
                api.da.make_benchmark_instance(args.seed, max_batches=args.max_batches,
                                               with_mt=not args.no_mt, with_bess=not args.no_bess))
        api.da.apply_instance_overrides(inst, args)
        if args.shrink:
            inst = api.shrink_instance(inst, args.shrink, max_batches=min(args.max_batches, 3))
        mkt = api.IntradayMarket(args.id_cap_buy, args.id_cap_sell).validate()
        scen = (api.OfferingScenarioSet.load(args.offering_scenario_file).validate(inst)
                if args.offering_scenario_file else
                api.OfferingScenarioSet.generate(inst, args.scenarios, args.scenario_seed, args.price_sigma,
                                                 args.level_sigma, args.id_spread_sigma, args.sr_ratio,
                                                 args.sr_sigma, args.load_sigma))
        rt = (api.RealTimeSet.load(args.rt_file) if args.rt_file else
              api.RealTimeSet.generate(inst, scen.n, args.rt_scenarios, args.rt_seed, args.rt_load_sigma,
                                       args.rt_load_rho, args.r_premium, args.r_discount, args.r_sigma))
        dep = (api.DeploymentSet.load(args.deploy_file) if args.deploy_file else
               api.DeploymentSet.generate(scen.n, rt.n_w, inst.horizon_h, args.rho_seed,
                                          args.rho_mean, args.rho_sigma))
        assets = tuple(x.strip() for x in args.reserve_assets.split(",") if x.strip())
        rsv = api.ReserveMarket(not args.no_reserve, assets, args.mt_rr_minutes, args.dr_fraction,
                                args.dr_cost, args.act_ratio, args.max_reserve,
                                args.bess_reserve_hours).validate()
        bal = api.BalancingMarket(args.imbalance_mode, args.bal_exclusive,
                                  args.max_imbalance, args.max_imbalance).validate()
        strat = api.StrategyConfig(args.beta, args.alpha, args.bidding, not args.no_reserve_curves,
                                   not args.soc_terminal_baseline).validate()
        cfg = api.SchedulerConfig(start_step_h=args.start_step, unique_tasks=not args.allow_repeat_tasks,
                                  solver=api.SolverSettings(args.solver, args.mip_gap, args.time_limit,
                                                           args.threads, args.verbose))
        result = api.optimize_offering(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg)
        from factory_mt_production_rescheduler import reschedule_production_by_id_scenario

        id_jobs, id_summary = reschedule_production_by_id_scenario(
            inst, result.jobs, result.da_position.mt_on.to_numpy(), scen, cfg, args.id_reschedule_hour
        )
        market_stages = None
        if args.compare:
            result.comparison = api.compare_strategies(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg, main=result)
            market_stages = api.compare_market_stages(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg, main=result)
        if args.frontier:
            result.frontier = api.risk_frontier(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg,
                                                [float(x) for x in args.frontier.split(",")])
        print(result.summary())
        print("\nStage-2 production rescheduling by ID scenario:")
        print(id_summary[["id_scenario", "probability", "operating_cost_eur", "batches",
                  "units_produced", "jobs_changed", "frozen_jobs"]].round(2).to_string(index=False))
        if market_stages is not None:
            print("\nMarket participation ladder (same scenario tree):")
            print(market_stages[["market_stage", "expected_net_cost_eur", "savings_vs_da_only_eur"]]
                  .round(2).to_string(index=False))
        print(result.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"]]
              .round(2).to_string(index=False))
        print(result.da_position[["hour", "exp_price_da_buy_eur_mwh", "mt_on", "exp_p_da_buy_mw",
                                 "exp_p_da_sell_mw", "exp_r_total_mw", "exp_price_sr_up_eur_mw_h",
                                 "exp_soc_mwh", "exp_profit_eur"]].round(2).to_string(index=False))
        out = result.save(args.out)
        id_jobs.to_csv(out / "intraday_rescheduled_jobs.csv", index=False)
        id_summary.to_csv(out / "intraday_reschedule_summary.csv", index=False)
        if market_stages is not None:
            market_stages.to_csv(out / "market_stage_comparison.csv", index=False)
        if args.plot:
            api.plot_offering(inst, result, out / "offering.png")
            from factory_mt_offering_plotting import plot_intraday_reschedules
            from factory_mt_offering_plotting import plot_intraday_reschedule_analysis

            plot_intraday_reschedules(inst, result.jobs, id_jobs, id_summary, out,
                                      args.id_reschedule_hour)
            plot_intraday_reschedule_analysis(inst, result.jobs, id_jobs, id_summary, scen, out)
            if market_stages is not None:
                from factory_mt_offering_plotting import plot_owner_dashboard

                plot_owner_dashboard(inst, result, id_summary, market_stages, out)
        print(f"\nResults written to {out.resolve()}")
        return 0
    except api.InstanceValidationError as err:
        logging.getLogger(api.__name__).error("Invalid input: %s", err)
        return 2
    except (api.da.SolverUnavailableError, api.da.InfeasibleScheduleError,
            api.da.SolveFailedError, api.VerificationError) as err:
        logging.getLogger(api.__name__).error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130