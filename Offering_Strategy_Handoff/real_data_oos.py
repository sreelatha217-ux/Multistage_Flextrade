"""Table IX analogue on REAL DK1 days: in-sample vs out-of-sample cost for different scenario-tree sizes.

For every test day (history = the `window` days before it) and every tree size:
  1. build the tree from the history, solve the full model  -> in-sample expected cost
  2. fix Stage 1 (jobs, MT on/off, DA curves applied as step functions to the ACTUAL day's real DA prices)
     and re-optimise Stage 2/3 on the actual day (1 real DA path, 3 intraday / reserve shocks) -> out-of-sample cost
Reserve prices: real mFRR if mfrr_capacity_dk1.csv exists (training: cluster means, test day: that day's real
prices), else synthetic --sr-ratio x DA.  Add --reserve-stage1 for the reserve-fixed model.
--compare-energy-only / --compare-da-only also train and evaluate the NO-reserve (DA + ID) plan and the DA-only plan on the
same days, so the REALIZED saving of the full model on real prices is reported (not only the expected saving on the tree).
--fix-reserve additionally fixes the reserve quantity of the training solution on the test day (needs
--reserve-stage1; if the fixed reserve cannot be delivered the day is recorded as NaN).

  uv run python real_data_oos.py --start 2026-04-01 --end 2026-10-12 --step 28 --trees 3x3 6x2 12x2 \\
        --reserve-stage1 --gap 0.01 --time-limit 600 --oos-gap 0.005

Result: real_data_results/oos/oos.csv (rewritten after every row).  The same seeds are used for every tree size
of a day, so the comparison between trees is paired.  Differences below about oos-gap x cost are not resolved.
Slow part: the 12x2 training solve (24 paths) can use the whole --time-limit.
"""
import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import real_data_common as rc
import real_data_mfrr_ladder as ml
import real_data_resched_backtest as rb

rdt = rc.rdt


def parse_trees(items):
    out = []
    for it in items:
        n, w = it.lower().split("x")
        out.append((int(n), int(w)))
    return out


class Evaluator:
    """Solve the full model on a training tree and evaluate its Stage 1 on one actual day."""

    def __init__(self, ctx, mfrr_tab=None, oos_gap=0.005, oos_rt=3, fix_reserve=False):
        self.ctx, self.mfrr_tab = ctx, mfrr_tab
        self.oos_gap, self.oos_rt, self.fix_reserve = oos_gap, oos_rt, fix_reserve
        self.T = ctx.base_inst.horizon_h
        self.unverified = False            # True if the LAST solve failed the independent verification

    def solve(self, inst, sc, r, d, fix=None, gap=None, mkt=None, rsv=None):
        ctx, api = self.ctx, self.ctx.api
        mkt = ctx.mkt if mkt is None else mkt
        rsv = ctx.rsv if rsv is None else rsv
        cfg = ctx.make_cfg(gap) if gap else ctx.cfg
        model = api.build_offering_model(inst, mkt, sc, r, d, rsv, ctx.bal, ctx.strat, cfg, ctx.cands)
        if fix:
            fix(model)
        info = api.da.solve_model(model, cfg.solver)
        self.unverified = False
        try:
            res = api.extract_offering(inst, mkt, sc, r, d, rsv, ctx.bal, ctx.strat, cfg, model, ctx.cands, info)
        except Exception as e:
            if type(e).__name__ != "VerificationError":
                raise
            # Do not hide it: print the size of the violation, then continue with the (valid) solution and flag the row.
            import pyomo.environ as pyo
            S, W, T = sc.n, r.n_w, inst.horizon_h
            dp = np.array([[[pyo.value(model.Dp[k, w, t]) for t in range(T)] for w in range(W)] for k in range(S)])
            dm = np.array([[[pyo.value(model.Dm[k, w, t]) for t in range(T)] for w in range(W)] for k in range(S)])
            both = np.minimum(dp, dm)
            print(f"   VERIFICATION FAILED: {str(e)[:150]}\n   largest simultaneous deficit/surplus = {both.max():.2e} MW "
                  f"in {int((both > 1e-6).sum())} of {both.size} hour-branches; continuing with the solution, row flagged", flush=True)
            res = api.extract_offering(inst, mkt, sc, r, d, rsv, ctx.bal, ctx.strat, replace(cfg, strict_verification=False),
                                       model, ctx.cands, info)
            self.unverified = True
        return model, res

    def actual_day(self, inst, day, seed):
        """Evaluation tree: ONE real DA price path, 3 ID/SR shocks, real mFRR prices of that day if available."""
        from dataclasses import replace
        ctx, api, o, a = self.ctx, self.ctx.api, self.ctx.opt, self.ctx.args
        arrs = rdt.actual_day_tree(ctx.df.loc[day].values, 3, floor=o.price_floor, sell_ratio=o.sell_ratio,
                                   sr_ratio=o.sr_ratio, seed=seed)
        mt = self.mfrr_tab
        if mt is not None and day in mt.index and not mt.loc[day].isna().any():
            arrs["sr_up"] = np.tile(mt.loc[day].to_numpy(float), (3, 1))
        osc = api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                      arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])
        oinst = replace(inst, price_buy_eur_mwh=osc.da_buy[0], price_sell_eur_mwh=osc.da_sell[0]).validate()
        osc.validate(oinst)
        ort = api.RealTimeSet.generate(oinst, osc.n, self.oos_rt, seed + 1, a.rt_load_sigma, a.rt_load_rho,
                                       a.r_premium, a.r_discount, a.r_sigma)
        odep = api.DeploymentSet.generate(osc.n, ort.n_w, oinst.horizon_h, seed + 2, a.rho_mean, a.rho_sigma)
        return oinst, osc, ort, odep

    def oos(self, model, res, sc, inst, day, seed, mkt=None, rsv=None):
        """(cost, mip_gap) of the Stage-1 plan of `model`/`res` on the actual `day`."""
        ctx, api, T = self.ctx, self.ctx.api, self.T
        oinst, osc, ort, odep = self.actual_day(inst, day, seed)
        S = sc.n
        Pbuy, Psell = api._arr2(model, "Pbuy", S, T), api._arr2(model, "Psell", S, T)
        u = [int(round(float(x))) for x in res.da_position.mt_on.to_numpy()]
        qs, qb = np.zeros((osc.n, T)), np.zeros((osc.n, T))
        for t in range(T):
            qs[:, t] = rb.step_curve(sc.da_sell[:, t], Psell[:, t], osc.da_sell[:, t], "sell")
            qb[:, t] = rb.step_curve(sc.da_buy[:, t], Pbuy[:, t], osc.da_buy[:, t], "buy")
        keep = rb.baseline_candidates(ctx.cands, ctx.tix, ctx.mix, res.jobs)
        R0 = (api._arr2(model, "Rtot", S, T)[0]
              if (self.fix_reserve and ctx.strat.reserve_stage1 and rsv is None) else None)

        def fix(m2):
            for i in range(len(ctx.cands)):
                m2.s[i].fix(1 if i in keep else 0)
            for t in range(T):
                if hasattr(m2, "u"):
                    m2.u[t].fix(u[t])
                for k in range(osc.n):
                    m2.Psell[k, t].fix(float(qs[k, t]))
                    m2.Pbuy[k, t].fix(float(qb[k, t]))
                    if R0 is not None:
                        m2.Rtot[k, t].fix(float(R0[t]))

        try:
            _, r2 = self.solve(oinst, osc, ort, odep, fix=fix, gap=self.oos_gap, mkt=mkt, rsv=rsv)
        except Exception as e:                       # fixed reserve may be undeliverable on the actual day
            if R0 is None:
                raise
            print(f"   OOS solve failed with the fixed reserve ({type(e).__name__}: {str(e)[:80]})", flush=True)
            return float("nan"), float("nan")
        return -r2.expected_profit_eur, r2.mip_gap or 0.0


def summarize(rows):
    df = pd.DataFrame(rows)
    g = df.groupby("tree", sort=False).agg(days=("day", "count"), paths=("paths", "first"),
                                           in_sample=("in_sample", "mean"), oos=("oos", "mean"),
                                           diff=("diff_eur", "mean"), diff_pct=("diff_pct", "mean"),
                                           train_gap=("train_gap", "mean"))
    txt = ["", "=== IN-SAMPLE vs OUT-OF-SAMPLE (mean over test days) ===", g.round(2).to_string()]
    first = df.tree.iloc[0]
    wide = df.pivot(index="day", columns="tree", values="oos")
    if first in wide.columns and wide.shape[1] > 1:
        txt.append(f"\npaired out-of-sample cost difference vs {first} (negative = better than {first}):")
        for c in wide.columns:
            if c == first:
                continue
            d = (wide[c] - wide[first]).dropna()
            txt.append(f"  {c:>6s}: mean {d.mean():8.0f} EUR   better on {(d < 0).sum()} of {len(d)} days")
    return "\n".join(txt)


def summarize_savings(rows, names):
    df = pd.DataFrame(rows)
    txt = ["", "=== REALIZED SAVING of the full model on the actual days (real prices, Stage 1 fixed) ==="]
    for name in names:
        for tree, g in df.groupby("tree", sort=False):
            s, p, nz = g[f"saving_{name}_eur"], g[f"saving_{name}_pct"], g[f"noise_{name}_eur"]
            txt.append(f"vs {name:11s} {tree:>5s}: mean {s.mean():7.0f} EUR ({p.mean():5.2f}%)  median {p.median():5.2f}%  "
                       f"min {s.min():7.0f}  max {s.max():7.0f}  negative days {int((s < 0).sum())}  "
                       f"below noise {int((s < nz).sum())}  | expected saving on the tree (mean) "
                       f"{g[f'ins_saving_{name}_eur'].mean():7.0f} EUR")
    return "\n".join(txt)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    rc.add_common_args(ap)
    ap.add_argument("--mfrr", default=ml.MfrrCtx.mfrr_csv, help="csv from fetch_mfrr_capacity.py (optional)")
    ap.add_argument("--trees", nargs="+", default=["3x3", "6x2", "12x2"], help="price scenarios x RT branches")
    ap.add_argument("--oos-gap", type=float, default=0.005)
    ap.add_argument("--oos-rt", type=int, default=3, help="RT branches of the evaluation tree")
    ap.add_argument("--fix-reserve", action="store_true")
    ap.add_argument("--compare-energy-only", action="store_true", help="also evaluate the DA + ID plan without reserve")
    ap.add_argument("--compare-da-only", action="store_true", help="also evaluate the DA-only plan (no ID, no reserve)")
    ap.set_defaults(gap=0.01, time_limit=600, step=28)
    opt, extra = ap.parse_known_args()
    if not opt.start:
        ap.error("--start is required (first test day); --end = last test day")
    use_mfrr = os.path.exists(opt.mfrr)
    ml.MfrrCtx.mfrr_csv = opt.mfrr
    ctx = (ml.MfrrCtx if use_mfrr else rc.Ctx)(opt, extra)
    mfrr_tab = ml.load_mfrr_hourly(opt.mfrr) if use_mfrr else None
    ev = Evaluator(ctx, mfrr_tab, opt.oos_gap, opt.oos_rt, opt.fix_reserve)
    trees = parse_trees(opt.trees)
    variants = []
    if opt.compare_da_only:
        variants.append(("da_only", ctx.no_id, ctx.no_res))
    if opt.compare_energy_only:
        variants.append(("energy_only", ctx.mkt, ctx.no_res))
    print(f"reserve prices: {'REAL mFRR' if use_mfrr else 'synthetic ' + str(opt.sr_ratio) + ' x DA'}   reserve capacity: "
          f"{'Stage-1 quantity' if ctx.strat.reserve_stage1 else 'price-quantity curve'}   trees: {opt.trees}", flush=True)
    out_dir = os.path.join(opt.out, "oos")
    rows = []
    for k, end in enumerate(ctx.history_ends()):
        day = end + pd.Timedelta(days=1)
        if day not in ctx.df.index:
            print(f"skip {day.date()}: no DA prices for the test day")
            continue
        for n, w in trees:
            t0 = time.time()
            inst, sc, r, d = ctx.tree(end, n=n, w=w, seed=opt.seed + k)
            model, res = ev.solve(inst, sc, r, d)
            ins = -res.expected_profit_eur
            train_unv = ev.unverified
            cost, gap = ev.oos(model, res, sc, inst, day, opt.seed + 1000 + k)
            row = dict(day=str(day.date()), tree=f"{n}x{w}", paths=n * w, in_sample=ins, oos=cost,
                       unverified=int(train_unv or ev.unverified),
                       diff_eur=cost - ins, diff_pct=100 * (cost - ins) / ins, train_gap=res.mip_gap or 0.0,
                       oos_gap=gap)
            print(f"[{day.date()}] {n}x{w:<2d} full model: in-sample {ins:9.0f}  out-of-sample {cost:9.0f}  "
                  f"diff {cost - ins:8.0f} ({100 * (cost - ins) / ins:5.1f}%)  train gap {100 * (res.mip_gap or 0):.2f}%",
                  flush=True)
            for name, vm, vr in variants:               # same tree, same actual day, fewer markets
                mv, rv = ev.solve(inst, sc, r, d, mkt=vm, rsv=vr)
                ins_v, unv = -rv.expected_profit_eur, ev.unverified
                cost_v, gap_v = ev.oos(mv, rv, sc, inst, day, opt.seed + 1000 + k, mkt=vm, rsv=vr)
                row[f"in_{name}"], row[f"oos_{name}"] = ins_v, cost_v
                row[f"saving_{name}_eur"] = cost_v - cost
                row[f"saving_{name}_pct"] = 100 * (cost_v - cost) / cost_v
                row[f"ins_saving_{name}_eur"] = ins_v - ins
                row[f"noise_{name}_eur"] = (gap + gap_v) * cost_v
                row["unverified"] = int(row["unverified"] or unv or ev.unverified)
                print(f"      vs {name:11s}: out-of-sample {cost_v:9.0f}  REALIZED saving {cost_v - cost:7.0f} EUR "
                      f"({100 * (cost_v - cost) / cost_v:5.2f}%)  expected on tree {ins_v - ins:7.0f}  "
                      f"noise ~{(gap + gap_v) * cost_v:.0f}", flush=True)
            row["seconds"] = time.time() - t0
            rows.append(row)
            rc.save_rows(rows, out_dir, "oos.csv")
            print(f"      {time.time() - t0:.0f}s", flush=True)
    if rows:
        print(summarize(rows))
        if variants:
            print(summarize_savings(rows, [v[0] for v in variants]))
        print(f"\nresults in {out_dir}/oos.csv")


if __name__ == "__main__":
    main()
