"""Shared setup for the real-data studies. Does NOT modify real_data_study.py / real_data_tree.py.

Used by: real_data_coupling.py, real_data_sensitivity.py, real_data_resched.py
(same plant, same tree recipe, same assumptions as real_data_study.py).
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import real_data_tree as rdt  # noqa: E402


def add_common_args(ap):
    ap.add_argument("--csv", default=os.path.join(HERE, "dk1_hourly_da.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "real_data_results"))
    ap.add_argument("--window", type=int, default=90, help="history days used to build the tree")
    ap.add_argument("--scenarios", type=int, default=3, help="DA price scenarios (day types)")
    ap.add_argument("--rt", type=int, default=3, help="real-time branches per price scenario")
    ap.add_argument("--end", default=None,
                    help="single-day mode: last HISTORY day. With --start: last TEST day")
    ap.add_argument("--start", default=None, help="range mode: first test day (history = day before each test day)")
    ap.add_argument("--step", type=int, default=14, help="range mode: every N-th day")
    ap.add_argument("--sell-ratio", type=float, default=0.55)
    ap.add_argument("--sr-ratio", type=float, default=0.12)
    ap.add_argument("--price-floor", type=float, default=0.0)
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2024)
    return ap


class Ctx:
    """Plant + markets + solver config + price table, built exactly like real_data_study.py."""

    def __init__(self, opt, extra=()):
        import factory_mt_offering_cli as cli  # needs pyomo/highspy (uv sync)
        import factory_mt_offering_strategy as api
        self.opt, self.api = opt, api
        a = self.args = cli._parse(
            ["--seed", str(opt.seed), "--max-batches", "10", "--bess-degradation", "5",
             "--bess-degradation-basis", "throughput", "--mip-gap", str(opt.gap),
             "--time-limit", str(opt.time_limit), "--solver", opt.solver,
             "--threads", str(opt.threads), *extra], api)
        self.base_inst = api.da.make_benchmark_instance(a.seed, max_batches=a.max_batches,
                                                        with_mt=not a.no_mt, with_bess=not a.no_bess)
        api.da.apply_instance_overrides(self.base_inst, a)
        self.mkt = api.IntradayMarket(a.id_cap_buy, a.id_cap_sell).validate()
        self.no_id = api.IntradayMarket(0.0, 0.0).validate()
        self.all_assets = tuple(x.strip() for x in a.reserve_assets.split(",") if x.strip())
        self.rsv = self.reserve(self.all_assets)
        self.no_res = replace(self.rsv, enabled=False)
        self.bal = api.BalancingMarket(a.imbalance_mode, a.bal_exclusive, a.max_imbalance, a.max_imbalance).validate()
        self.strat = api.StrategyConfig(a.beta, a.alpha, a.bidding, not a.no_reserve_curves,
                                        not a.soc_terminal_baseline).validate()
        self.cfg = self.make_cfg(opt.gap)
        self.df = rdt.load_prices(opt.csv)
        self.cands = api.da.build_candidates(self.base_inst, self.cfg.start_step_h)
        self.tix = {t: i for i, t in enumerate(self.base_inst.tasks)}
        self.mix = {m: i for i, m in enumerate(self.base_inst.machines)}

    # ------------------------------------------------------------------ builders
    def reserve(self, assets):
        a = self.args
        return self.api.ReserveMarket(True, tuple(assets), a.mt_rr_minutes, a.dr_fraction, a.dr_cost,
                                      a.act_ratio, a.max_reserve, a.bess_reserve_hours).validate()

    def make_cfg(self, gap):
        a, o = self.args, self.opt
        return self.api.SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                                        solver=self.api.SolverSettings(o.solver, gap, o.time_limit, o.threads, a.verbose))

    def history_ends(self):
        """Single mode: [--end or last day].  Range mode: day before each test day."""
        o = self.opt
        if o.start:
            lo = pd.Timestamp(o.start)
            hi = pd.Timestamp(o.end) if o.end else self.df.index[-1]
            days = list(self.df.index[(self.df.index >= lo) & (self.df.index <= hi)])[::o.step]
            return [d - pd.Timedelta(days=1) for d in days]
        return [pd.Timestamp(o.end) if o.end else self.df.index[-1]]

    def tree(self, end_date, n=None, w=None, sr_ratio=None, seed=None):
        """(inst, scen, rt, dep) from k-means day types of the `window` days up to end_date."""
        api, o, a = self.api, self.opt, self.args
        n = o.scenarios if n is None else n
        w = o.rt if w is None else w
        seed = o.seed if seed is None else seed
        sr = o.sr_ratio if sr_ratio is None else sr_ratio
        P, pi = rdt.day_types(self.df, end_date, o.window, n, seed=o.seed)
        arrs = rdt.tree_arrays(P, pi, floor=o.price_floor, sell_ratio=o.sell_ratio, sr_ratio=sr, seed=seed)
        scen = api.OfferingScenarioSet(arrs["prob"], arrs["da_buy"], arrs["da_sell"], arrs["sr_up"],
                                       arrs["id_buy"], arrs["id_sell"], arrs["load_dev_mw"])
        buy = (scen.prob[:, None] * scen.da_buy).sum(0)
        sell = (scen.prob[:, None] * scen.da_sell).sum(0)
        inst = replace(self.base_inst, price_buy_eur_mwh=buy, price_sell_eur_mwh=sell).validate()
        scen.validate(inst)
        rt = api.RealTimeSet.generate(inst, scen.n, w, seed + 1, a.rt_load_sigma, a.rt_load_rho,
                                      a.r_premium, a.r_discount, a.r_sigma)
        dep = api.DeploymentSet.generate(scen.n, rt.n_w, inst.horizon_h, seed + 2, a.rho_mean, a.rho_sigma)
        return inst, scen, rt, dep

    # ------------------------------------------------------------------ solves
    def optimize(self, inst, scen, rt, dep, mkt=None, rsv=None, strat=None):
        return self.api.optimize_offering(inst, scen, rt, dep, mkt or self.mkt, rsv or self.rsv,
                                          self.bal, strat or self.strat, self.cfg)

    def fixed_production_run(self, inst, scen, rt, dep, jobs):
        """Re-optimise market + reserve with the production plan `jobs` held fixed (full market, reserve on)."""
        api = self.api
        keep = set()
        for j in jobs.itertuples():
            hit = [i for i, c in enumerate(self.cands)
                   if c.m == self.mix[j.machine] and c.p == self.tix[j.task]
                   and abs(c.start_h - float(j.start_h)) < 1e-6]
            assert len(hit) == 1, j
            keep.add(hit[0])
        model = api.build_offering_model(inst, self.mkt, scen, rt, dep, self.rsv, self.bal, self.strat,
                                         self.cfg, self.cands)
        for i in range(len(self.cands)):
            model.s[i].fix(1 if i in keep else 0)
        info = api.da.solve_model(model, self.cfg.solver)
        return api.extract_offering(inst, self.mkt, scen, rt, dep, self.rsv, self.bal, self.strat,
                                    self.cfg, model, self.cands, info)

    def scaled_bess(self, k):
        """Battery scale k (1 = paper 40 MW / 200 MWh, 0 = none) -- same as bess_sweep.py."""
        b = self.base_inst.bess
        if k <= 0:
            return replace(self.base_inst, bess=None).validate()
        return replace(self.base_inst, bess=replace(b, p_max_mw=b.p_max_mw * k, e_max_mwh=b.e_max_mwh * k,
                                                    soc_init_mwh=b.soc_init_mwh * k)).validate()


def cost(res):
    return -res.expected_profit_eur


def save_rows(rows, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    pd.DataFrame(rows).to_csv(path, index=False)   # rewritten after every row
    return path
