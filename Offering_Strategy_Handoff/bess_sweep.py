"""Battery-size sweep of the production-coupling value.

For each battery scale k (1.0 = 40 MW / 200 MWh as in the paper, 0 = no battery):
  coupled   : production + DA/ID + reserve optimised together
  decoupled : production plan fixed first from the energy-only (DA+ID, no reserve) optimum,
              then DA/ID/reserve re-optimised around that fixed plan
  coupling value = decoupled cost - coupled cost
Usage:  uv run python bess_sweep.py            (scales 0, 0.25, 0.5, 1.0)
        uv run python bess_sweep.py 0.5 1.0    (custom scales)
"""
import os
import sys
from dataclasses import replace

HANDOFF = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HANDOFF, "..", "Factory_multistage_stretegy", "offering_owner_report_results")
sys.path.insert(0, HANDOFF)

import factory_mt_offering_cli as cli
import factory_mt_offering_strategy as api

scales = [float(x) for x in sys.argv[1:]] or [0.0, 0.25, 0.5, 1.0]
argv = ["--seed", "2024", "--max-batches", "10", "--bess-degradation", "5",
        "--bess-degradation-basis", "throughput", "--mip-gap", "0.001", "--time-limit", "900",
        "--offering-scenario-file", f"{RES}/scenario_set.json",
        "--rt-file", f"{RES}/rt_set.json", "--deploy-file", f"{RES}/deployment_set.json"]
args = cli._parse(argv, api)

base_inst = api.da.make_benchmark_instance(args.seed, max_batches=args.max_batches,
                                           with_mt=not args.no_mt, with_bess=not args.no_bess)
api.da.apply_instance_overrides(base_inst, args)
mkt = api.IntradayMarket(args.id_cap_buy, args.id_cap_sell).validate()
scen = api.OfferingScenarioSet.load(args.offering_scenario_file).validate(base_inst)
rt = api.RealTimeSet.load(args.rt_file)
dep = api.DeploymentSet.load(args.deploy_file)
assets = tuple(x.strip() for x in args.reserve_assets.split(",") if x.strip())
rsv = api.ReserveMarket(True, assets, args.mt_rr_minutes, args.dr_fraction, args.dr_cost,
                        args.act_ratio, args.max_reserve, args.bess_reserve_hours).validate()
bal = api.BalancingMarket(args.imbalance_mode, args.bal_exclusive, args.max_imbalance, args.max_imbalance).validate()
strat = api.StrategyConfig(args.beta, args.alpha, args.bidding, not args.no_reserve_curves,
                           not args.soc_terminal_baseline).validate()
cfg = api.SchedulerConfig(start_step_h=args.start_step, unique_tasks=not args.allow_repeat_tasks,
                          solver=api.SolverSettings(args.solver, args.mip_gap, args.time_limit,
                                                    args.threads, args.verbose))
no_res = replace(rsv, enabled=False)
cands = api.da.build_candidates(base_inst, cfg.start_step_h)
tix = {t: i for i, t in enumerate(base_inst.tasks)}
mix = {m: i for i, m in enumerate(base_inst.machines)}


def cost(r):
    return -r.expected_profit_eur


def scaled(k):
    if k <= 0:
        return replace(base_inst, bess=None).validate()
    b = base_inst.bess
    return replace(base_inst, bess=replace(b, p_max_mw=b.p_max_mw * k, e_max_mwh=b.e_max_mwh * k,
                                           soc_init_mwh=b.soc_init_mwh * k)).validate()


def fixed_run(inst, jobs):
    keep = set()
    for j in jobs.itertuples():
        hit = [i for i, c in enumerate(cands)
               if c.m == mix[j.machine] and c.p == tix[j.task] and abs(c.start_h - float(j.start_h)) < 1e-6]
        assert len(hit) == 1, j
        keep.add(hit[0])
    model = api.build_offering_model(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, cands)
    for i in range(len(cands)):
        model.s[i].fix(1 if i in keep else 0)
    info = api.da.solve_model(model, cfg.solver)
    return api.extract_offering(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, model, cands, info)


rows = []
for k in scales:
    print(f"\n>>> battery scale {k:g}: coupled", flush=True)
    inst = scaled(k)
    cpl = api.optimize_offering(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg)
    print(f">>> battery scale {k:g}: energy-only plan", flush=True)
    eo = api.optimize_offering(inst, scen, rt, dep, mkt, no_res, bal, strat, cfg)
    print(f">>> battery scale {k:g}: decoupled", flush=True)
    dec = fixed_run(inst, eo.jobs)
    rows.append((k, cost(eo), cost(cpl), cost(dec), cpl.mip_gap or 0.0, dec.mip_gap or 0.0,
                 cpl.kpis["avg_reserve_offer_mw"]))
    print(f"    done: coupled {cost(cpl):.2f}  decoupled {cost(dec):.2f}", flush=True)

print("\n=== RESULT (battery size sweep) ===")
print(f"{'scale':>6s}{'MW/MWh':>10s}{'no-reserve':>13s}{'coupled':>12s}{'decoupled':>12s}"
      f"{'coupling EUR':>14s}{'gap cpl %':>10s}{'gap dec %':>10s}{'res MW':>8s}")
for k, eo, c, d, gc, gd, rmw in rows:
    print(f"{k:6.2f}{f'{40*k:g}/{200*k:g}':>10s}{eo:13.2f}{c:12.2f}{d:12.2f}{d-c:14.2f}{100*gc:10.3f}{100*gd:10.3f}{rmw:8.1f}")
