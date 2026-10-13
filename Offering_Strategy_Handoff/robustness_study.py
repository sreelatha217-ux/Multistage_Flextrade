"""Robustness studies for the offering model.

  uv run python robustness_study.py reserve                  # reserve price x0.5 / x1 / x1.5 on the paper's 3x3 tree
  uv run python robustness_study.py oos   --sizes 10 30 --oos 40
  uv run python robustness_study.py frontier --n 12 --w 3    # CVaR beta frontier on a larger tree

Common options: --gap 0.001 --time-limit 1800 --start-step 0.5
"""
import argparse
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Factory_multistage_stretegy", "offering_owner_report_results")
sys.path.insert(0, HERE)

import numpy as np
import factory_mt_offering_cli as cli
import factory_mt_offering_strategy as api

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["reserve", "oos", "frontier"])
ap.add_argument("--gap", type=float, default=0.001)
ap.add_argument("--time-limit", type=float, default=1800)
ap.add_argument("--start-step", type=float, default=0.5)
ap.add_argument("--solver", default="appsi_highs", help="appsi_highs (default) or gurobi (needs a licence)")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--sizes", type=int, nargs="+", default=[10, 30])
ap.add_argument("--oos", type=int, default=40, help="number of out-of-sample price scenarios")
ap.add_argument("--oos-gap", type=float, default=0.005)
ap.add_argument("--w", type=int, default=2, help="real-time branches per price scenario (oos / frontier)")
ap.add_argument("--n", type=int, default=12, help="price scenarios for the frontier")
ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 1.0, 1.5])
opt = ap.parse_args()

base_argv = ["--seed", "2024", "--max-batches", "10", "--bess-degradation", "5",
             "--bess-degradation-basis", "throughput", "--mip-gap", str(opt.gap),
             "--time-limit", str(opt.time_limit), "--start-step", str(opt.start_step),
             "--solver", opt.solver, "--threads", str(opt.threads)]
args = cli._parse(base_argv, api)

inst = api.da.make_benchmark_instance(args.seed, max_batches=args.max_batches,
                                      with_mt=not args.no_mt, with_bess=not args.no_bess)
api.da.apply_instance_overrides(inst, args)
mkt = api.IntradayMarket(args.id_cap_buy, args.id_cap_sell).validate()
assets = tuple(x.strip() for x in args.reserve_assets.split(",") if x.strip())
rsv = api.ReserveMarket(True, assets, args.mt_rr_minutes, args.dr_fraction, args.dr_cost,
                        args.act_ratio, args.max_reserve, args.bess_reserve_hours).validate()
bal = api.BalancingMarket(args.imbalance_mode, args.bal_exclusive, args.max_imbalance, args.max_imbalance).validate()
strat = api.StrategyConfig(args.beta, args.alpha, args.bidding, not args.no_reserve_curves,
                           not args.soc_terminal_baseline).validate()


def make_cfg(gap):
    return api.SchedulerConfig(start_step_h=args.start_step, unique_tasks=not args.allow_repeat_tasks,
                               solver=api.SolverSettings(args.solver, gap, opt.time_limit, args.threads, args.verbose))


cfg = make_cfg(opt.gap)
cands = api.da.build_candidates(inst, cfg.start_step_h)
tix = {t: i for i, t in enumerate(inst.tasks)}
mix = {m: i for i, m in enumerate(inst.machines)}


def tree(n, w, seed):
    sc = api.OfferingScenarioSet.generate(inst, n, seed, args.price_sigma, args.level_sigma, args.id_spread_sigma,
                                          args.sr_ratio, args.sr_sigma, args.load_sigma)
    r = api.RealTimeSet.generate(inst, sc.n, w, seed + 1, args.rt_load_sigma, args.rt_load_rho,
                                 args.r_premium, args.r_discount, args.r_sigma)
    d = api.DeploymentSet.generate(sc.n, w, inst.horizon_h, seed + 2, args.rho_mean, args.rho_sigma)
    return sc, r, d


def paper_tree():
    sc = api.OfferingScenarioSet.load(f"{RES}/scenario_set.json").validate(inst)
    return sc, api.RealTimeSet.load(f"{RES}/rt_set.json"), api.DeploymentSet.load(f"{RES}/deployment_set.json")


def solve(sc, r, d, reserve=True, strategy=strat, config=None, fix=None):
    config = config or cfg
    rs = rsv if reserve else replace(rsv, enabled=False)
    model = api.build_offering_model(inst, mkt, sc, r, d, rs, bal, strategy, config, cands)
    if fix:
        fix(model)
    info = api.da.solve_model(model, config.solver)
    res = api.extract_offering(inst, mkt, sc, r, d, rs, bal, strategy, config, model, cands, info)
    return model, res


# ------------------------------------------------------------------ reserve price sensitivity
if opt.mode == "reserve":
    sc, r, d = paper_tree()
    t0 = time.time()
    _, none = solve(sc, r, d, reserve=False)
    print(f"no-reserve cost {-none.expected_profit_eur:.2f}  gap {100*(none.mip_gap or 0):.3f}%", flush=True)
    rows = []
    for f in opt.factors:
        sc_f = replace(sc, sr_up=sc.sr_up * f)
        _, res = solve(sc_f, r, d)
        c = -res.expected_profit_eur
        rows.append((f, float(sc_f.sr_up.mean()), c, -none.expected_profit_eur - c, res.mip_gap or 0.0,
                     res.kpis["avg_reserve_offer_mw"], res.kpis["exp_rev_sr_eur"] + res.kpis["exp_rev_act_eur"]))
        print(f"  factor {f:g}: cost {c:.2f}", flush=True)
    base = -none.expected_profit_eur
    print("\n=== RESULT (reserve price sensitivity) ===")
    print(f"{'factor':>7s}{'mean EUR/MW/h':>15s}{'cost EUR':>12s}{'saving EUR':>12s}{'saving %':>10s}{'gap %':>8s}{'res MW':>8s}{'res revenue':>13s}")
    for f, p, c, s_, g, mw, rv in rows:
        print(f"{f:7.2f}{p:15.2f}{c:12.2f}{s_:12.2f}{100*s_/base:10.2f}{100*g:8.3f}{mw:8.1f}{rv:13.2f}")

# ------------------------------------------------------------------ frontier
elif opt.mode == "frontier":
    sc, r, d = tree(opt.n, opt.w, 4242)
    print(f"tree: {sc.n} x {opt.w} = {sc.n*opt.w} paths; alpha={args.alpha}", flush=True)
    fr = api.risk_frontier(inst, sc, r, d, mkt, rsv, bal, strat, cfg, opt.betas)
    print("\n=== RESULT (CVaR frontier) ===")
    print(fr.round(2).to_string(index=False))

# ------------------------------------------------------------------ out-of-sample
else:
    def step_curve(price_in, q_in, price_out, kind):
        order = np.argsort(price_in)
        p, q = price_in[order], q_in[order]
        if kind == "sell":      # offer curve: quantity of the highest in-sample price not above the new price
            idx = np.searchsorted(p, price_out, side="right") - 1
        else:                   # bid curve: quantity of the lowest in-sample price not below the new price
            idx = np.searchsorted(p, price_out, side="left")
        return q[np.clip(idx, 0, len(p) - 1)]

    oos_sc, oos_r, oos_d = tree(opt.oos, opt.w, 987654)
    print(f"out-of-sample tree: {oos_sc.n} x {opt.w} paths", flush=True)
    T = inst.horizon_h
    cfg_oos = make_cfg(opt.oos_gap)
    rows = []

    def evaluate(label, sc, r, d):
        t0 = time.time()
        model, res = solve(sc, r, d)
        t_in = time.time() - t0
        S = sc.n
        Pbuy, Psell = api._arr2(model, "Pbuy", S, T), api._arr2(model, "Psell", S, T)
        u = [int(round(float(x))) for x in res.da_position.mt_on.to_numpy()]
        keep = set()
        for j in res.jobs.itertuples():
            hit = [i for i, c in enumerate(cands) if c.m == mix[j.machine] and c.p == tix[j.task]
                   and abs(c.start_h - float(j.start_h)) < 1e-6]
            keep.add(hit[0])
        qs = np.zeros((oos_sc.n, T))
        qb = np.zeros((oos_sc.n, T))
        for t in range(T):
            qs[:, t] = step_curve(sc.da_sell[:, t], Psell[:, t], oos_sc.da_sell[:, t], "sell")
            qb[:, t] = step_curve(sc.da_buy[:, t], Pbuy[:, t], oos_sc.da_buy[:, t], "buy")

        def fix(m2):
            for i in range(len(cands)):
                m2.s[i].fix(1 if i in keep else 0)
            for t in range(T):
                m2.u[t].fix(u[t])
                for k in range(oos_sc.n):
                    m2.Psell[k, t].fix(float(qs[k, t]))
                    m2.Pbuy[k, t].fix(float(qb[k, t]))

        _, res2 = solve(oos_sc, oos_r, oos_d, config=cfg_oos, fix=fix)
        cin, cout = -res.expected_profit_eur, -res2.expected_profit_eur
        rows.append((label, sc.n * r.n_w, cin, cout, cout - cin, res.mip_gap or 0.0, t_in))
        print(f"  {label}: in-sample {cin:.2f}  out-of-sample {cout:.2f}", flush=True)

    sc3, r3, d3 = paper_tree()
    evaluate("paper 3x3 tree", sc3, r3, d3)
    for n in opt.sizes:
        sc, r, d = tree(n, opt.w, 1000 + n)
        evaluate(f"{n} x {opt.w} tree", sc, r, d)
    print("\n=== RESULT (out-of-sample test) ===")
    print(f"{'training tree':18s}{'paths':>7s}{'in-sample':>12s}{'out-of-sample':>15s}{'optimism EUR':>14s}{'gap %':>8s}{'time s':>9s}")
    for lab, p, ci, co, op, g, tt in rows:
        print(f"{lab:18s}{p:7d}{ci:12.2f}{co:15.2f}{op:14.2f}{100*g:8.3f}{tt:9.0f}")
