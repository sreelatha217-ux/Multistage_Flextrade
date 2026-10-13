"""Value-of-coupling test.

Coupled   : production plan is optimised together with DA/ID/reserve (the paper's Level 3).
Decoupled : production plan is fixed FIRST by a model that does not know about reserve
            (plan A = energy-only DA+ID optimum, plan B = DA-only optimum), then DA/ID/reserve
            decisions are re-optimised around that fixed plan.
Coupling value = decoupled cost - coupled cost.
"""
import os
import sys
from dataclasses import replace

HANDOFF = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HANDOFF, "..", "Factory_multistage_stretegy", "offering_owner_report_results")
sys.path.insert(0, HANDOFF)

import factory_mt_offering_cli as cli
import factory_mt_offering_strategy as api
import pyomo.environ as pyo

argv = ["--seed", "2024", "--max-batches", "10", "--bess-degradation", "5",
        "--bess-degradation-basis", "throughput", "--mip-gap", "0.001", "--time-limit", "900",
        "--offering-scenario-file", f"{RES}/scenario_set.json",
        "--rt-file", f"{RES}/rt_set.json", "--deploy-file", f"{RES}/deployment_set.json"]
args = cli._parse(argv, api)

inst = api.da.make_benchmark_instance(args.seed, max_batches=args.max_batches,
                                      with_mt=not args.no_mt, with_bess=not args.no_bess)
api.da.apply_instance_overrides(inst, args)
mkt = api.IntradayMarket(args.id_cap_buy, args.id_cap_sell).validate()
scen = api.OfferingScenarioSet.load(args.offering_scenario_file).validate(inst)
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
no_id = api.IntradayMarket(0.0, 0.0).validate()
cands = api.da.build_candidates(inst, cfg.start_step_h)
tix = {t: i for i, t in enumerate(inst.tasks)}
mix = {m: i for i, m in enumerate(inst.machines)}


def cost(r):
    return -r.expected_profit_eur


def fixed_production_run(jobs, label):
    """Re-optimise market + reserve decisions with the production plan `jobs` held fixed."""
    keep = set()
    for j in jobs.itertuples():
        hit = [i for i, c in enumerate(cands)
               if c.m == mix[j.machine] and c.p == tix[j.task] and abs(c.start_h - float(j.start_h)) < 1e-6]
        assert len(hit) == 1, (label, j)
        keep.add(hit[0])
    model = api.build_offering_model(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, cands)
    for i in range(len(cands)):
        model.s[i].fix(1 if i in keep else 0)
    info = api.da.solve_model(model, cfg.solver)
    return api.extract_offering(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, model, cands, info)


out = {}
print(">>> coupled (full model, reserve on)", flush=True)
coupled = api.optimize_offering(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg)
out["coupled"] = (cost(coupled), coupled.mip_gap, coupled.kpis["avg_reserve_offer_mw"], int(coupled.kpis["n_batches"]))

print(">>> plan A: energy-only DA+ID optimum", flush=True)
planA = api.optimize_offering(inst, scen, rt, dep, mkt, no_res, bal, strat, cfg)
print(">>> decoupled A: fix plan A, re-optimise reserve/market", flush=True)
decA = fixed_production_run(planA.jobs, "A")
out["decoupled A (plan from energy-only DA+ID)"] = (cost(decA), decA.mip_gap, decA.kpis["avg_reserve_offer_mw"], int(decA.kpis["n_batches"]))

print(">>> plan B: DA-only optimum", flush=True)
planB = api.optimize_offering(inst, scen, rt, dep, no_id, no_res, bal, strat, cfg)
print(">>> decoupled B: fix plan B, re-optimise reserve/market", flush=True)
decB = fixed_production_run(planB.jobs, "B")
out["decoupled B (plan from DA-only)"] = (cost(decB), decB.mip_gap, decB.kpis["avg_reserve_offer_mw"], int(decB.kpis["n_batches"]))

base = out["coupled"][0]
print("\n=== RESULT ===")
print(f"{'case':46s}{'cost EUR':>12s}{'gap %':>8s}{'res MW':>8s}{'jobs':>6s}{'coupling value EUR':>20s}")
for k, (c, g, r, n) in out.items():
    print(f"{k:46s}{c:12.2f}{100*(g or 0):8.3f}{r:8.1f}{n:6d}{(c-base):20.2f}")
