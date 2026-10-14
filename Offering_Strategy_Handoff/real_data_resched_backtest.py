"""Does re-scheduling the jobs AFTER day-ahead clearing pay, once the DA position is already booked?

Rolling test on REAL DK1 days (same idea as `real_data_study.py backtest`):
  1. Train a tree on the `window` days before the test day, solve the full model, take Stage 1
     (jobs, MT on/off, DA offer/bid curves; reserve capacity as the model decides).
  2. The test day's REAL DA prices are published (~13:00 D-1).  The DA curves are applied as step functions,
     so the DA quantities are now BOOKED (fixed).  Stage 2/3 (intraday, BESS, MT output, reserve split,
     imbalance) is re-optimised on the actual day in three ways:
        fixed : jobs stay as in Stage 1                      (= what real_data_study.py backtest does)
        free  : jobs may be re-planned completely
        shift : jobs may move at most --shift-hours, only the tasks already in the Stage-1 plan
  3. saving = cost(fixed) - cost(variant), both evaluated with the same booked DA quantities, so any mismatch
     between the booked DA energy and the new load is paid for (intraday / imbalance / BESS / MT).

Reserve prices: real mFRR up-capacity prices if mfrr_capacity_dk1.csv exists (training: cluster means;
test day: that day's real prices), otherwise the synthetic --sr-ratio x DA.  Add --reserve-stage1 to commit
reserve capacity as one Stage-1 number.  Does not modify any existing file.

  uv run python real_data_resched_backtest.py --start 2026-04-01 --end 2026-10-12 --step 28 --gap 0.01 --time-limit 600

Result: real_data_results/resched_backtest/backtest.csv (rewritten after every day).
A saving smaller than about oos-gap x cost is not resolved.  Not added to the market-ladder saving.
"""
import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import real_data_common as rc
import real_data_mfrr_ladder as ml

rdt = rc.rdt
EPS = 1e-6


def step_curve(price_in, q_in, price_out, kind):   # identical to real_data_study.py / robustness_study.py
    order = np.argsort(price_in)
    p, q = price_in[order], q_in[order]
    idx = (np.searchsorted(p, price_out, side="right") - 1 if kind == "sell"
           else np.searchsorted(p, price_out, side="left"))
    return q[np.clip(idx, 0, len(p) - 1)]


def baseline_candidates(cands, tix, mix, jobs):
    """Indices of the candidate start slots that are exactly the Stage-1 jobs."""
    keep = set()
    for j in jobs.itertuples():
        hit = [i for i, c in enumerate(cands) if c.m == mix[j.machine] and c.p == tix[j.task]
               and abs(c.start_h - float(j.start_h)) < EPS]
        assert len(hit) == 1, f"cannot map job {j.task} on {j.machine} at {j.start_h}"
        keep.add(hit[0])
    return keep


def allowed_by_shift(cands, tix, jobs, shift_h):
    """Candidate slots whose task is in the Stage-1 plan and whose start is within +-shift_h of its Stage-1 start."""
    base = [(tix[j.task], float(j.start_h)) for j in jobs.itertuples()]
    return {i for i, c in enumerate(cands)
            if any(c.p == p and abs(c.start_h - s) <= shift_h + EPS for p, s in base)}


def changed_jobs(base_jobs, new_jobs):
    a = {(j.machine, j.task, round(float(j.start_h), 4)) for j in base_jobs.itertuples()}
    b = {(j.machine, j.task, round(float(j.start_h), 4)) for j in new_jobs.itertuples()}
    return len(a - b)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    rc.add_common_args(ap)
    ap.add_argument("--mfrr", default=ml.MfrrCtx.mfrr_csv, help="csv from fetch_mfrr_capacity.py (optional)")
    ap.add_argument("--oos-gap", type=float, default=0.002, help="MIP gap of the three evaluation solves")
    ap.add_argument("--shift-hours", type=float, default=3.0, help="max start shift in the 'shift' variant")
    ap.set_defaults(gap=0.01, time_limit=600, step=28)
    opt, extra = ap.parse_known_args()
    if not opt.start:
        ap.error("--start is required (first test day); --end = last test day")

    use_mfrr = os.path.exists(opt.mfrr)
    ml.MfrrCtx.mfrr_csv = opt.mfrr
    ctx = (ml.MfrrCtx if use_mfrr else rc.Ctx)(opt, extra)
    api, a = ctx.api, ctx.args
    mfrr_tab = ml.load_mfrr_hourly(opt.mfrr) if use_mfrr else None
    T = ctx.base_inst.horizon_h
    print(f"reserve prices: {'REAL mFRR' if use_mfrr else 'synthetic ' + str(opt.sr_ratio) + ' x DA'}   "
          f"reserve capacity: {'Stage-1 quantity' if ctx.strat.reserve_stage1 else 'price-quantity curve'}", flush=True)

    def solve(inst, sc, r, d, fix=None, gap=None):
        cfg = ctx.make_cfg(gap) if gap else ctx.cfg
        model = api.build_offering_model(inst, ctx.mkt, sc, r, d, ctx.rsv, ctx.bal, ctx.strat, cfg, ctx.cands)
        if fix:
            fix(model)
        info = api.da.solve_model(model, cfg.solver)
        return model, api.extract_offering(inst, ctx.mkt, sc, r, d, ctx.rsv, ctx.bal, ctx.strat, cfg, model, ctx.cands, info)

    def actual_day(inst, day, seed):
        arrs = rdt.actual_day_tree(ctx.df.loc[day].values, 3, floor=opt.price_floor, sell_ratio=opt.sell_ratio,
                                   sr_ratio=opt.sr_ratio, seed=seed)
        real = None
        if mfrr_tab is not None and day in mfrr_tab.index and not mfrr_tab.loc[day].isna().any():
            real = mfrr_tab.loc[day].to_numpy(float)
            arrs["sr_up"] = np.tile(real, (3, 1))
        osc = api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                      arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])
        oinst = replace(inst, price_buy_eur_mwh=osc.da_buy[0], price_sell_eur_mwh=osc.da_sell[0]).validate()
        osc.validate(oinst)
        ort = api.RealTimeSet.generate(oinst, osc.n, opt.rt, seed + 1, a.rt_load_sigma, a.rt_load_rho,
                                       a.r_premium, a.r_discount, a.r_sigma)
        odep = api.DeploymentSet.generate(osc.n, ort.n_w, oinst.horizon_h, seed + 2, a.rho_mean, a.rho_sigma)
        return oinst, osc, ort, odep, real is not None

    out_dir = os.path.join(opt.out, "resched_backtest")
    rows = []
    for k, end in enumerate(ctx.history_ends()):
        day = end + pd.Timedelta(days=1)
        if day not in ctx.df.index:
            print(f"skip {day.date()}: no DA prices for the test day")
            continue
        t0 = time.time()
        inst, sc, r, d = ctx.tree(end, seed=opt.seed + k)
        model, res = solve(inst, sc, r, d)                              # Stage 1 on the training tree
        gap_train = res.mip_gap or 0.0
        oinst, osc, ort, odep, real_sr = actual_day(inst, day, opt.seed + 1000 + k)
        S = sc.n
        Pbuy, Psell = api._arr2(model, "Pbuy", S, T), api._arr2(model, "Psell", S, T)
        u = [int(round(float(x))) for x in res.da_position.mt_on.to_numpy()]
        qs, qb = np.zeros((osc.n, T)), np.zeros((osc.n, T))
        for t in range(T):
            qs[:, t] = step_curve(sc.da_sell[:, t], Psell[:, t], osc.da_sell[:, t], "sell")
            qb[:, t] = step_curve(sc.da_buy[:, t], Pbuy[:, t], osc.da_buy[:, t], "buy")
        keep = baseline_candidates(ctx.cands, ctx.tix, ctx.mix, res.jobs)
        near = allowed_by_shift(ctx.cands, ctx.tix, res.jobs, opt.shift_hours)

        def make_fix(allowed):
            def fix(m2):
                for i in range(len(ctx.cands)):
                    if allowed is None:
                        continue                                        # jobs completely free
                    if len(allowed) == len(keep) and allowed == keep:
                        m2.s[i].fix(1 if i in keep else 0)              # jobs fixed as in Stage 1
                    elif i not in allowed:
                        m2.s[i].fix(0)                                  # outside the allowed slots
                for t in range(T):
                    if hasattr(m2, "u"):
                        m2.u[t].fix(u[t])
                    for kk in range(osc.n):
                        m2.Psell[kk, t].fix(float(qs[kk, t]))
                        m2.Pbuy[kk, t].fix(float(qb[kk, t]))
            return fix

        row = dict(day=str(day.date()), actual_mean_price=float(ctx.df.loc[day].mean()),
                   real_mfrr_used=real_sr, train_gap=gap_train)
        results = {}
        for label, allowed in (("fixed", keep), ("shift", near), ("free", None)):
            _, r2 = solve(oinst, osc, ort, odep, fix=make_fix(allowed), gap=opt.oos_gap)
            results[label] = r2
            row[f"{label}_cost"] = -r2.expected_profit_eur
            row[f"{label}_gap"] = r2.mip_gap or 0.0
            row[f"{label}_jobs_changed"] = changed_jobs(res.jobs, r2.jobs)
        for label in ("shift", "free"):
            row[f"saving_{label}_eur"] = row["fixed_cost"] - row[f"{label}_cost"]
            row[f"saving_{label}_pct"] = 100 * row[f"saving_{label}_eur"] / row["fixed_cost"]
        row["noise_eur"] = (row["fixed_gap"] + row["free_gap"]) * row["fixed_cost"]
        row["seconds"] = time.time() - t0
        rows.append(row)
        rc.save_rows(rows, out_dir, "backtest.csv")
        print(f"[{len(rows)}] {row['day']}  fixed {row['fixed_cost']:.0f}  "
              f"shift<={opt.shift_hours:g}h saves {row['saving_shift_eur']:.0f} ({row['saving_shift_pct']:.2f}%, "
              f"{row['shift_jobs_changed']} jobs moved)  free saves {row['saving_free_eur']:.0f} "
              f"({row['saving_free_pct']:.2f}%, {row['free_jobs_changed']} jobs moved)  "
              f"noise ~{row['noise_eur']:.0f}  {row['seconds']:.0f}s", flush=True)

    if rows:
        bt = pd.DataFrame(rows)
        print("\n=== PRODUCTION RECOURSE AFTER DA CLEARING (DA quantities booked, real DK1 days) ===")
        print(f"days: {len(bt)}")
        for label in ("shift", "free"):
            s = bt[f"saving_{label}_eur"]
            p = bt[f"saving_{label}_pct"]
            print(f"{label:5s}: mean {s.mean():.0f} EUR ({p.mean():.2f}%)  median {p.median():.2f}%  "
                  f"days below noise: {(s < bt.noise_eur).sum()}  negative days: {(s < 0).sum()}")
        print(f"mean noise level ~{bt.noise_eur.mean():.0f} EUR;  results in {out_dir}/backtest.csv")


if __name__ == "__main__":
    main()
