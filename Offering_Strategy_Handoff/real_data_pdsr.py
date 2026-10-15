"""Pilot: problem-driven scenario reduction (PDSR, Zhuang et al., arXiv 2404.07810) vs k-means day types.

Question answered: with the SAME number K of price scenarios, does choosing K REAL days by decision applicability
(PDSR) give a Stage-1 plan with a lower cost on the actual next day than k-means day types (cluster averages)?

For every test day (history = the `--window` days before it, N days):
  1. For each history day i solve the single-day problem -> its optimal Stage 1 z_i (jobs, MT on/off, DA quantities).
  2. F[i][j] = cost of day j when Stage 1 = z_i is fixed and Stage 2/3 is re-optimised  (N + N(N-1) solves).
     F depends only on the pair of days, so it is cached in a csv and re-used by later windows / reruns.
  3. d(i,k) = F[k][i] - F[i][i] + F[i][k] - F[k][k] ; a p-median MILP picks K real days minimising
     sum_i gamma_i d(i, rep(i))  (paper eq. 12 with K fixed).  Weights = share of days represented.
  4. Train the full model on the K chosen days (their real DA and real mFRR prices), evaluate on the actual next day.
  5. Same with the k-means tree (ctx.tree, same window, same K), evaluate on the same day.

  uv run python real_data_pdsr.py --dry-run --end 2026-10-11 --window 30 --scenarios 3
  uv run python real_data_pdsr.py --start 2026-09-01 --end 2026-10-12 --step 14 --window 30 --scenarios 3 --rt 3 \\
        --reserve-stage1 --gap 0.01 --time-limit 600

Approximations: single-day problems have one DA quantity per hour instead of an offer curve; the reserve quantity is
re-optimised (not fixed) when z_i is applied to day j; CVaR (beta > 0) is outside PDSR's scope (paper, limitation ii).
Result: real_data_results/pdsr/pdsr_pilot.csv.   Pilot only: a few test days cannot prove anything on their own.
"""
import argparse
import csv
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import real_data_common as rc
import real_data_mfrr_ladder as ml
import real_data_oos as oo

rdt = rc.rdt


def pmedian(D, gamma, K, time_limit=60.0):
    """Choose K representatives minimising sum_i gamma_i * D[i, rep(i)].  Returns (reps, assign, objective)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    N = len(gamma)
    nv = N * N + N                                    # v_ij (i assigned to j), u_j (j is a representative)
    c = np.concatenate([(np.asarray(gamma)[:, None] * D).ravel(), np.zeros(N)])
    r, cidx, val, lb, ub = [], [], [], [], []
    row = 0
    for i in range(N):                                # every scenario assigned to exactly one representative
        for j in range(N):
            r.append(row); cidx.append(i * N + j); val.append(1.0)
        lb.append(1.0); ub.append(1.0); row += 1
    for i in range(N):                                # v_ij <= u_j
        for j in range(N):
            r += [row, row]; cidx += [i * N + j, N * N + j]; val += [1.0, -1.0]
            lb.append(-np.inf); ub.append(0.0); row += 1
    for j in range(N):
        r.append(row); cidx.append(N * N + j); val.append(1.0)
    lb.append(float(K)); ub.append(float(K)); row += 1   # exactly K representatives
    for j in range(N):                                # a representative represents itself: v_jj = u_j
        r += [row, row]; cidx += [j * N + j, N * N + j]; val += [1.0, -1.0]
        lb.append(0.0); ub.append(0.0); row += 1
    A = coo_matrix((val, (r, cidx)), shape=(row, nv)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"p-median MILP failed: {res.message}")
    x = res.x
    u = x[N * N:] > 0.5
    v = x[:N * N].reshape(N, N)
    reps = [j for j in range(N) if u[j]]
    assign = np.array([max(reps, key=lambda j: v[i, j]) for i in range(N)])
    return reps, assign, float(res.fun)


def symmetric_distance(F):
    """d(i,k) = F[k][i] - F[i][i] + F[i][k] - F[k][k]   (F[a][b] = cost of scenario b under the plan of a)."""
    diag = np.diag(F)
    D = F.T - diag[:, None] + F - diag[None, :]
    D = np.maximum(D, 0.0)                            # solver-gap noise can make a regret slightly negative
    np.fill_diagonal(D, 0.0)
    return D


class Pdsr:
    def __init__(self, ctx, ev, mfrr_tab, opt, cache_path):
        self.ctx, self.ev, self.mfrr_tab, self.opt = ctx, ev, mfrr_tab, opt
        self.cache_path = cache_path
        self.cache = {}
        self._prob, self._opt = {}, {}
        if os.path.exists(cache_path):
            for rec in pd.read_csv(cache_path).itertuples():
                self.cache[(rec.i, rec.j)] = float(rec.cost)
        self.fill = np.nanmean(mfrr_tab.to_numpy(float), axis=0) if mfrr_tab is not None else None

    # ---------------------------------------------------------------- scenario construction
    def mfrr_vec(self, day):
        v = self.mfrr_tab.loc[day].to_numpy(float) if day in self.mfrr_tab.index else np.full(24, np.nan)
        return np.where(np.isnan(v), self.fill, v)

    def _build(self, P, pi, days, w, seed):
        ctx, api, o, a = self.ctx, self.ctx.api, self.opt, self.ctx.args
        arrs = rdt.tree_arrays(P, pi, floor=o.price_floor, sell_ratio=o.sell_ratio, sr_ratio=o.sr_ratio, seed=seed)
        if self.mfrr_tab is not None:
            arrs["sr_up"] = np.vstack([self.mfrr_vec(d) for d in days])
        scen = api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                       arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])
        buy = (scen.prob[:, None] * scen.da_buy).sum(0)
        sell = (scen.prob[:, None] * scen.da_sell).sum(0)
        inst = replace(ctx.base_inst, price_buy_eur_mwh=buy, price_sell_eur_mwh=sell).validate()
        scen.validate(inst)
        rt = api.RealTimeSet.generate(inst, scen.n, w, seed + 1, a.rt_load_sigma, a.rt_load_rho,
                                      a.r_premium, a.r_discount, a.r_sigma)
        dep = api.DeploymentSet.generate(scen.n, rt.n_w, inst.horizon_h, seed + 2, a.rho_mean, a.rho_sigma)
        return inst, scen, rt, dep

    def single(self, day):
        if day not in self._prob:
            self._prob[day] = self._build(self.ctx.df.loc[[day]].values, np.array([1.0]), [day], 1, self.opt.seed)
        return self._prob[day]

    def tree_from_days(self, days, probs, w, seed):
        return self._build(self.ctx.df.loc[days].values, np.asarray(probs, float), days, w, seed)

    # ---------------------------------------------------------------- single-day optimum and cross evaluation
    def optimum(self, day):
        if day not in self._opt:
            api, T = self.ctx.api, self.ev.T
            inst, sc, rt, dep = self.single(day)
            model, res = self.ev.solve(inst, sc, rt, dep, gap=self.opt.f_gap)
            self._opt[day] = dict(
                cost=-res.expected_profit_eur,
                keep=rb_keep(self.ctx, res.jobs),
                u=[int(round(float(x))) for x in res.da_position.mt_on.to_numpy()],
                Pbuy=api._arr2(model, "Pbuy", 1, T)[0], Psell=api._arr2(model, "Psell", 1, T)[0])
        return self._opt[day]

    def cross(self, z, day):
        T, ncand = self.ev.T, len(self.ctx.cands)
        inst, sc, rt, dep = self.single(day)

        def fix(m2):
            for i in range(ncand):
                m2.s[i].fix(1 if i in z["keep"] else 0)
            for t in range(T):
                if hasattr(m2, "u"):
                    m2.u[t].fix(z["u"][t])
                m2.Psell[0, t].fix(float(z["Psell"][t]))
                m2.Pbuy[0, t].fix(float(z["Pbuy"][t]))

        _, res = self.ev.solve(inst, sc, rt, dep, fix=fix, gap=self.opt.f_gap)
        return -res.expected_profit_eur

    # ---------------------------------------------------------------- F matrix with cache
    def missing(self, days):
        return sum(1 for a in days for b in days if (a, b) not in self.cache)

    def fmatrix(self, days):
        keys = [str(d.date()) for d in days]
        N = len(keys)
        todo = self.missing(keys)
        print(f"   F matrix {N}x{N}: {N * N - todo} cached, {todo} solves to do", flush=True)
        done, t0 = 0, time.time()
        new = not os.path.exists(self.cache_path)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a", newline="") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["i", "j", "cost"])
            for a, da in zip(keys, days):
                if all((a, b) in self.cache for b in keys):
                    continue
                z = self.optimum(da)
                if (a, a) not in self.cache:
                    self.cache[(a, a)] = z["cost"]
                    wr.writerow([a, a, z["cost"]]); fh.flush(); done += 1
                for b, db in zip(keys, days):
                    if (a, b) in self.cache:
                        continue
                    c = self.cross(z, db)
                    self.cache[(a, b)] = c
                    wr.writerow([a, b, c]); fh.flush(); done += 1
                    if done % 25 == 0:
                        el = time.time() - t0
                        print(f"   {done}/{todo} solves, {el / 60:.1f} min elapsed, ~{(todo - done) * el / done / 60:.1f} min left", flush=True)
        return np.array([[self.cache[(a, b)] for b in keys] for a in keys])

    def select(self, days, K):
        F = self.fmatrix(days)
        D = symmetric_distance(F)
        gamma = np.full(len(days), 1.0 / len(days))
        reps, assign, obj = pmedian(D, gamma, K)
        w = [float(gamma[assign == j].sum()) for j in reps]
        return [days[j] for j in reps], w, obj


def rb_keep(ctx, jobs):
    import real_data_resched_backtest as rb
    return rb.baseline_candidates(ctx.cands, ctx.tix, ctx.mix, jobs)


def summarize(rows):
    df = pd.DataFrame(rows)
    g = df.groupby("method", sort=False).agg(days=("day", "count"), in_sample=("in_sample", "mean"), oos=("oos", "mean"),
                                             diff_pct=("diff_pct", "mean"))
    txt = ["", "=== PDSR vs k-means (same history, same K) ===", g.round(2).to_string()]
    w = df.pivot(index="day", columns="method", values="oos")
    if {"kmeans", "pdsr"} <= set(w.columns):
        d = (w["kmeans"] - w["pdsr"]).dropna()
        txt.append(f"\nk-means minus PDSR out-of-sample cost (positive = PDSR better): mean {d.mean():.0f} EUR "
                   f"({100 * d.mean() / w['kmeans'].mean():.2f}%), PDSR better on {(d > 0).sum()} of {len(d)} days")
        txt.append("Worth continuing only if PDSR wins on most days AND the mean gain exceeds the evaluation noise "
                   "(about oos-gap x cost, ~0.7k EUR at 0.5%).")
    return "\n".join(txt)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    rc.add_common_args(ap)
    ap.add_argument("--mfrr", default=ml.MfrrCtx.mfrr_csv)
    ap.add_argument("--oos-gap", type=float, default=0.005)
    ap.add_argument("--oos-rt", type=int, default=3)
    ap.add_argument("--f-gap", type=float, default=0.005, help="MIP gap of the single-day / cross solves")
    ap.add_argument("--dry-run", action="store_true", help="only print the number of solves for the first test day")
    ap.set_defaults(window=30, gap=0.01, time_limit=600, step=14)
    opt, extra = ap.parse_known_args()
    use_mfrr = os.path.exists(opt.mfrr)
    ml.MfrrCtx.mfrr_csv = opt.mfrr
    ctx = (ml.MfrrCtx if use_mfrr else rc.Ctx)(opt, extra)
    mfrr_tab = ml.load_mfrr_hourly(opt.mfrr) if use_mfrr else None
    ev = oo.Evaluator(ctx, mfrr_tab, opt.oos_gap, opt.oos_rt)
    tag = f"{'mfrr' if use_mfrr else 'synth'}_{'stage1' if ctx.strat.reserve_stage1 else 'curve'}"
    out_dir = os.path.join(opt.out, "pdsr")
    pd_ = Pdsr(ctx, ev, mfrr_tab, opt, os.path.join(out_dir, f"pdsr_F_cache_{tag}.csv"))
    K = opt.scenarios
    print(f"N = {opt.window} history days, K = {K} scenarios; reserve prices "
          f"{'REAL mFRR' if use_mfrr else 'synthetic'}, reserve "
          f"{'Stage-1 quantity' if ctx.strat.reserve_stage1 else 'curve'}", flush=True)
    ends = ctx.history_ends()
    if opt.dry_run:
        days = list(ctx.df.loc[:ends[0]].iloc[-opt.window:].index)
        todo = pd_.missing([str(d.date()) for d in days])
        print(f"first test day: {todo} single-day / cross solves still to do "
              f"(N + N(N-1) = {opt.window + opt.window * (opt.window - 1)} without cache). "
              f"Assume ~2-4 s per solve to estimate the time.")
        return
    rows = []
    for k, end in enumerate(ends):
        day = end + pd.Timedelta(days=1)
        if day not in ctx.df.index:
            print(f"skip {day.date()}: no DA prices for the test day")
            continue
        hist = list(ctx.df.loc[:end].iloc[-opt.window:].index)
        t0 = time.time()
        sel, wts, obj = pd_.select(hist, K)
        print(f"[{day.date()}] PDSR picked {[str(d.date()) for d in sel]} weights {np.round(wts, 3)} "
              f"(SPDD {obj:.0f}, {time.time() - t0:.0f}s)", flush=True)
        seed = opt.seed + k
        trees = {"pdsr": pd_.tree_from_days(sel, wts, opt.rt, seed),
                 "kmeans": ctx.tree(end, n=K, w=opt.rt, seed=seed)}
        for method, (inst, sc, rt, dep) in trees.items():
            t1 = time.time()
            model, res = ev.solve(inst, sc, rt, dep)
            ins = -res.expected_profit_eur
            cost, gap = ev.oos(model, res, sc, inst, day, opt.seed + 1000 + k)
            rows.append(dict(day=str(day.date()), method=method, in_sample=ins, oos=cost, diff_eur=cost - ins,
                             diff_pct=100 * (cost - ins) / ins, train_gap=res.mip_gap or 0.0, oos_gap=gap,
                             selected=";".join(str(d.date()) for d in sel) if method == "pdsr" else "",
                             weights=";".join(f"{x:.3f}" for x in wts) if method == "pdsr" else "",
                             seconds=time.time() - t1))
            rc.save_rows(rows, out_dir, "pdsr_pilot.csv")
            print(f"[{day.date()}] {method:6s} in-sample {ins:9.0f}  out-of-sample {cost:9.0f}  "
                  f"diff {100 * (cost - ins) / ins:5.1f}%  train gap {100 * (res.mip_gap or 0):.2f}%", flush=True)
    if rows:
        print(summarize(rows))
        print(f"\nresults in {out_dir}/pdsr_pilot.csv")


if __name__ == "__main__":
    main()
