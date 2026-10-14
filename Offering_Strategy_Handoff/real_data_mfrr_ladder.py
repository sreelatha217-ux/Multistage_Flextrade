"""Market ladder on REAL DK1 prices with REAL mFRR up-capacity prices (replaces sr_up = 0.12 x DA x noise).

  uv run python real_data_mfrr_ladder.py --end 2026-10-11 --window 90 --scenarios 3 --rt 3

Needs mfrr_capacity_dk1.csv (from fetch_mfrr_capacity.py) next to this file.
Day types are exactly the same k-means clusters as real_data_study.py ladder; the only change is the reserve
capacity price: for each day type, sr_up[t] = mean real mFRR up price (EUR/MW/h) of the days in that cluster, hour t.
Does not modify any existing file.  Results: real_data_results/ladder_mfrr/<history_end>/
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import real_data_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mfrr_hourly(path):
    """day x 24 table (EUR/MW/h), index = local date, columns 0..23 (DST duplicates averaged, gaps = NaN)."""
    d = pd.read_csv(path)
    t = pd.to_datetime(d["TimeDK"])
    d = pd.DataFrame({"date": t.dt.normalize(), "hour": t.dt.hour, "p": d["UpPriceEUR"].astype(float)})
    tab = d.pivot_table(index="date", columns="hour", values="p", aggfunc="mean")
    return tab.reindex(columns=range(24))


def cluster_mfrr(df, mfrr, end_date, window, n, seed):
    """Same k-means as real_data_tree.day_types, plus the cluster-mean mFRR price.
    Returns (da_prices [n,24], probs [n], sr_up [n,24]) sorted cheap -> expensive."""
    hist = df.loc[:pd.Timestamp(end_date)].iloc[-window:]
    if len(hist) < max(n, 5):
        raise ValueError(f"only {len(hist)} history days before {end_date}")
    if n == 1:
        labels, P, pi = np.zeros(len(hist), int), hist.values.mean(0, keepdims=True), np.array([1.0])
    else:
        km = KMeans(n_clusters=n, n_init=20, random_state=seed).fit(hist.values)
        order = np.argsort(km.cluster_centers_.mean(1))
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        labels = rank[km.labels_]
        P = km.cluster_centers_[order]
        pi = (np.bincount(km.labels_, minlength=n) / len(hist))[order]
    mh = mfrr.reindex(hist.index).to_numpy(float)                  # [days, 24], NaN where no mFRR data
    if np.isnan(mh).all():
        raise ValueError("no mFRR prices for the history window: download them first (fetch_mfrr_capacity.py)")
    overall = np.nanmean(mh, axis=0)
    sr = np.zeros((n, 24))
    for k in range(n):
        rows = mh[labels == k]
        with np.errstate(all="ignore"):
            m = np.nanmean(rows, axis=0) if len(rows) else np.full(24, np.nan)
        sr[k] = np.where(np.isnan(m), overall, m)
    if np.isnan(sr).any():
        raise ValueError("mFRR prices missing for some hours of the day in the history window")
    return P, pi, sr


class MfrrCtx(rc.Ctx):
    mfrr_csv = os.path.join(HERE, "mfrr_capacity_dk1.csv")

    def tree(self, end_date, n=None, w=None, sr_ratio=None, seed=None):
        api, o, a = self.api, self.opt, self.args
        n = o.scenarios if n is None else n
        w = o.rt if w is None else w
        seed = o.seed if seed is None else seed
        if not hasattr(self, "_mfrr"):
            self._mfrr = load_mfrr_hourly(self.mfrr_csv)
        P, pi, sr = cluster_mfrr(self.df, self._mfrr, end_date, o.window, n, o.seed)
        arrs = rc.rdt.tree_arrays(P, pi, floor=o.price_floor, sell_ratio=o.sell_ratio, sr_ratio=o.sr_ratio, seed=seed)
        scale = 1.0 if sr_ratio is None else sr_ratio / o.sr_ratio     # --ratios 0.06 0.12 0.18 -> x0.5 x1 x1.5 of real mFRR
        arrs["sr_up"] = sr * scale                                  # real mFRR up-capacity price per day type
        scen = api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                       arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])
        from dataclasses import replace
        buy = (scen.prob[:, None] * scen.da_buy).sum(0)
        sell = (scen.prob[:, None] * scen.da_sell).sum(0)
        inst = replace(self.base_inst, price_buy_eur_mwh=buy, price_sell_eur_mwh=sell).validate()
        scen.validate(inst)
        rt = api.RealTimeSet.generate(inst, scen.n, w, seed + 1, a.rt_load_sigma, a.rt_load_rho,
                                      a.r_premium, a.r_discount, a.r_sigma)
        dep = api.DeploymentSet.generate(scen.n, rt.n_w, inst.horizon_h, seed + 2, a.rho_mean, a.rho_sigma)
        return inst, scen, rt, dep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    rc.add_common_args(ap)
    ap.add_argument("--mfrr", default=MfrrCtx.mfrr_csv, help="csv from fetch_mfrr_capacity.py")
    ap.set_defaults(gap=0.005, time_limit=900)
    opt, extra = ap.parse_known_args()
    MfrrCtx.mfrr_csv = opt.mfrr
    ctx = MfrrCtx(opt, extra)
    api = ctx.api
    for end in ctx.history_ends():
        day = str(end.date())
        inst, scen, rt, dep = ctx.tree(end)
        print(f"real-data tree + REAL mFRR prices: {scen.n} day types x {rt.n_w} RT, history to {day}, "
              f"probs = {np.round(scen.prob, 3)}, reserve capacity = "
              f"{'ONE Stage-1 quantity (same in all scenarios)' if ctx.strat.reserve_stage1 else 'price-quantity curve'}", flush=True)
        for k in range(scen.n):
            print(f"  day type {k}: mean DA {scen.da_buy[k].mean():6.1f} EUR/MWh | real mFRR mean {scen.sr_up[k].mean():6.2f} "
                  f"(max {scen.sr_up[k].max():6.2f}) vs old 0.12 x DA = {0.12 * scen.da_buy[k].mean():6.2f} EUR/MW/h", flush=True)
        res = ctx.optimize(inst, scen, rt, dep)
        stages = api.compare_market_stages(inst, scen, rt, dep, ctx.mkt, ctx.rsv, ctx.bal, ctx.strat, ctx.cfg, main=res)
        print(res.summary())
        print("\nMarket participation ladder (REAL DA + REAL mFRR capacity prices):")
        print(stages[["market_stage", "expected_net_cost_eur", "savings_vs_da_only_eur"]].round(2).to_string(index=False))
        out = res.save(os.path.join(opt.out, "ladder_mfrr_stage1" if ctx.strat.reserve_stage1 else "ladder_mfrr", day))
        stages.to_csv(out / "market_stage_comparison.csv", index=False)
        scen.save(out / "scenario_set.json")
        print(f"\nwritten to {out.resolve()}", flush=True)
