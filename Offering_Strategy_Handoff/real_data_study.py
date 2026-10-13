"""Re-run the paper's study on REAL DK1 day-ahead prices (dk1_hourly_da.csv).

Put this file + real_data_tree.py + dk1_hourly_da.csv in Offering_Strategy_Handoff/ and run (uv run ...):

  # 1) one real-data tree (day types from the last 90 days) -> full run + market ladder (Table III analogue)
  uv run python real_data_study.py ladder --end 2026-10-11 --window 90 --scenarios 3 --rt 3

  # 2) rolling backtest over real days: train tree on the previous `window` days, fix Stage-1,
  #    evaluate on the ACTUAL next-day prices (Table IX analogue).  Compares full model vs no-reserve.
  uv run python real_data_study.py backtest --start 2026-04-01 --end 2026-10-12 --step 14 --window 90

Results: real_data_results/ladder/* and real_data_results/backtest.csv
Assumptions you can change from the CLI: --sell-ratio (paper: 0.55 x buy), --sr-ratio (paper: 0.12 x DA),
--price-floor (negative prices are rejected by the repo validator, so they are floored, default 0).
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import real_data_tree as rdt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["ladder", "backtest"])
ap.add_argument("--csv", default=os.path.join(HERE, "dk1_hourly_da.csv"))
ap.add_argument("--out", default=os.path.join(HERE, "real_data_results"))
ap.add_argument("--window", type=int, default=90, help="history days used to build the tree")
ap.add_argument("--scenarios", type=int, default=3, help="DA price scenarios (day types); keep <= 24 for HiGHS")
ap.add_argument("--rt", type=int, default=3, help="real-time branches per price scenario")
ap.add_argument("--end", default=None, help="ladder: last history day; backtest: last test day")
ap.add_argument("--start", default=None, help="backtest: first test day")
ap.add_argument("--step", type=int, default=14, help="backtest: test every N-th day")
ap.add_argument("--sell-ratio", type=float, default=0.55)
ap.add_argument("--sr-ratio", type=float, default=0.12)
ap.add_argument("--price-floor", type=float, default=0.0)
ap.add_argument("--gap", type=float, default=0.01)
ap.add_argument("--oos-gap", type=float, default=0.005)
ap.add_argument("--time-limit", type=float, default=300)
ap.add_argument("--solver", default="appsi_highs")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--seed", type=int, default=2024)
opt = ap.parse_args()

import factory_mt_offering_cli as cli  # noqa: E402  (needs pyomo/highspy: `uv sync`)
import factory_mt_offering_strategy as api  # noqa: E402

# Same plant as the paper (seed 2024, 10 batches, BESS degradation 5 EUR/MWh throughput)
args = cli._parse(["--seed", str(opt.seed), "--max-batches", "10", "--bess-degradation", "5",
                   "--bess-degradation-basis", "throughput", "--mip-gap", str(opt.gap),
                   "--time-limit", str(opt.time_limit), "--solver", opt.solver,
                   "--threads", str(opt.threads)], api)
base_inst = api.da.make_benchmark_instance(args.seed, max_batches=args.max_batches,
                                           with_mt=not args.no_mt, with_bess=not args.no_bess)
api.da.apply_instance_overrides(base_inst, args)
mkt = api.IntradayMarket(args.id_cap_buy, args.id_cap_sell).validate()
assets = tuple(x.strip() for x in args.reserve_assets.split(",") if x.strip())
rsv = api.ReserveMarket(True, assets, args.mt_rr_minutes, args.dr_fraction, args.dr_cost,
                        args.act_ratio, args.max_reserve, args.bess_reserve_hours).validate()
bal = api.BalancingMarket(args.imbalance_mode, args.bal_exclusive, args.max_imbalance, args.max_imbalance).validate()
strat = api.StrategyConfig(args.beta, args.alpha, args.bidding, not args.no_reserve_curves,
                           not args.soc_terminal_baseline).validate()


def make_cfg(gap):
    return api.SchedulerConfig(start_step_h=args.start_step, unique_tasks=not args.allow_repeat_tasks,
                               solver=api.SolverSettings(opt.solver, gap, opt.time_limit, opt.threads, args.verbose))


cfg = make_cfg(opt.gap)
T = base_inst.horizon_h
df = rdt.load_prices(opt.csv)


def scen_from(arrs):
    return api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                   arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])


def instance_for(scen):
    """Plant DA tariff = probability-weighted scenario mean (as in the paper, where lam_DA = tariff * shock)."""
    buy = (scen.prob[:, None] * scen.da_buy).sum(0)
    sell = (scen.prob[:, None] * scen.da_sell).sum(0)
    return replace(base_inst, price_buy_eur_mwh=buy, price_sell_eur_mwh=sell).validate()


def sets_for(inst, scen, seed):
    rt = api.RealTimeSet.generate(inst, scen.n, opt.rt, seed + 1, args.rt_load_sigma, args.rt_load_rho,
                                  args.r_premium, args.r_discount, args.r_sigma)
    dep = api.DeploymentSet.generate(scen.n, rt.n_w, inst.horizon_h, seed + 2, args.rho_mean, args.rho_sigma)
    return rt, dep


def tree_for(end_date, seed):
    P, pi = rdt.day_types(df, end_date, opt.window, opt.scenarios, seed=opt.seed)
    arrs = rdt.tree_arrays(P, pi, floor=opt.price_floor, sell_ratio=opt.sell_ratio, sr_ratio=opt.sr_ratio, seed=seed)
    scen = scen_from(arrs)
    inst = instance_for(scen)
    scen.validate(inst)
    rt, dep = sets_for(inst, scen, seed)
    return inst, scen, rt, dep


# ============================================================== 1) ladder on one real-data tree
if opt.mode == "ladder":
    end = opt.end or str(df.index[-1].date())
    inst, scen, rt, dep = tree_for(end, opt.seed)
    print(f"real-data tree: {scen.n} day types x {rt.n_w} RT branches, history = {opt.window} days to {end}, "
          f"probs = {np.round(scen.prob, 3)}", flush=True)
    res = api.optimize_offering(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg)
    stages = api.compare_market_stages(inst, scen, rt, dep, mkt, rsv, bal, strat, cfg, main=res)
    print(res.summary())
    print("\nMarket participation ladder (REAL DK1 prices):")
    print(stages[["market_stage", "expected_net_cost_eur", "savings_vs_da_only_eur"]].round(2).to_string(index=False))
    out = res.save(os.path.join(opt.out, "ladder"))
    stages.to_csv(out / "market_stage_comparison.csv", index=False)
    scen.save(out / "scenario_set.json")
    print(f"\nwritten to {out.resolve()}")
    sys.exit(0)

# ============================================================== 2) rolling backtest on actual prices
cands = api.da.build_candidates(base_inst, cfg.start_step_h)
tix = {t: i for i, t in enumerate(base_inst.tasks)}
mix = {m: i for i, m in enumerate(base_inst.machines)}


def solve(inst, sc, r, d, reserve=True, config=None, fix=None):
    config = config or cfg
    rs = rsv if reserve else replace(rsv, enabled=False)
    model = api.build_offering_model(inst, mkt, sc, r, d, rs, bal, strat, config, cands)
    if fix:
        fix(model)
    info = api.da.solve_model(model, config.solver)
    return model, api.extract_offering(inst, mkt, sc, r, d, rs, bal, strat, config, model, cands, info)


def step_curve(price_in, q_in, price_out, kind):   # identical to robustness_study.py
    order = np.argsort(price_in)
    p, q = price_in[order], q_in[order]
    idx = (np.searchsorted(p, price_out, side="right") - 1 if kind == "sell"
           else np.searchsorted(p, price_out, side="left"))
    return q[np.clip(idx, 0, len(p) - 1)]


def oos_cost(inst, model, res, sc, day_prices, reserve, seed):
    """Fix Stage-1 (jobs, MT commitment, DA offer/bid curves as step functions) and re-optimise Stage 2/3
    on the ACTUAL day (one real DA price path, 3 ID/SR shocks around it, same RT recipe)."""
    arrs = rdt.actual_day_tree(day_prices, 3, floor=opt.price_floor, sell_ratio=opt.sell_ratio,
                               sr_ratio=opt.sr_ratio, seed=seed)
    osc = scen_from(arrs)
    oinst = replace(inst, price_buy_eur_mwh=osc.da_buy[0], price_sell_eur_mwh=osc.da_sell[0]).validate()
    osc.validate(oinst)
    ort, odep = sets_for(oinst, osc, seed)
    S = sc.n
    Pbuy, Psell = api._arr2(model, "Pbuy", S, T), api._arr2(model, "Psell", S, T)
    u = [int(round(float(x))) for x in res.da_position.mt_on.to_numpy()]
    keep = set()
    for j in res.jobs.itertuples():
        keep.add([i for i, c in enumerate(cands) if c.m == mix[j.machine] and c.p == tix[j.task]
                  and abs(c.start_h - float(j.start_h)) < 1e-6][0])
    qs, qb = np.zeros((osc.n, T)), np.zeros((osc.n, T))
    for t in range(T):
        qs[:, t] = step_curve(sc.da_sell[:, t], Psell[:, t], osc.da_sell[:, t], "sell")
        qb[:, t] = step_curve(sc.da_buy[:, t], Pbuy[:, t], osc.da_buy[:, t], "buy")

    def fix(m2):
        for i in range(len(cands)):
            m2.s[i].fix(1 if i in keep else 0)
        for t in range(T):
            m2.u[t].fix(u[t])
            for k in range(osc.n):
                m2.Psell[k, t].fix(float(qs[k, t]))
                m2.Pbuy[k, t].fix(float(qb[k, t]))

    _, r2 = solve(oinst, osc, ort, odep, reserve=reserve, config=make_cfg(opt.oos_gap), fix=fix)
    return -r2.expected_profit_eur


lo = pd.Timestamp(opt.start or df.index[opt.window])
hi = pd.Timestamp(opt.end or df.index[-1])
test_days = [d for d in df.index[(df.index >= lo) & (df.index <= hi)]][::opt.step]
os.makedirs(opt.out, exist_ok=True)
rows = []
for k, day in enumerate(test_days):
    t0 = time.time()
    prev = day - pd.Timedelta(days=1)                     # history strictly BEFORE the test day
    inst, sc, r, d = tree_for(prev, opt.seed + k)
    actual = df.loc[day].values
    row = dict(day=str(day.date()), actual_mean_price=float(actual.mean()))
    for label, reserve in (("energy_only", False), ("full", True)):
        model, res = solve(inst, sc, r, d, reserve=reserve)
        row[f"{label}_in_sample"] = -res.expected_profit_eur
        row[f"{label}_oos"] = oos_cost(inst, model, res, sc, actual, reserve, opt.seed + 1000 + k)
        row[f"{label}_gap"] = res.mip_gap or 0.0
    row["oos_saving_eur"] = row["energy_only_oos"] - row["full_oos"]
    row["oos_saving_pct"] = 100 * row["oos_saving_eur"] / row["energy_only_oos"]
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(opt.out, "backtest.csv"), index=False)   # saved after every day
    print(f"[{k+1}/{len(test_days)}] {row['day']}  energy-only {row['energy_only_oos']:.0f}  "
          f"full {row['full_oos']:.0f}  saving {row['oos_saving_eur']:.0f} ({row['oos_saving_pct']:.2f}%)  "
          f"{time.time()-t0:.0f}s", flush=True)

bt = pd.DataFrame(rows)
print("\n=== REAL-DATA BACKTEST (out-of-sample on actual DK1 prices) ===")
print(f"days tested: {len(bt)}   mean saving {bt.oos_saving_eur.mean():.0f} EUR ({bt.oos_saving_pct.mean():.2f}%)   "
      f"median {bt.oos_saving_pct.median():.2f}%   days with saving<0: {(bt.oos_saving_eur < 0).sum()}")
print(f"mean in-sample vs out-of-sample (full): {bt.full_in_sample.mean():.0f} vs {bt.full_oos.mean():.0f}")
