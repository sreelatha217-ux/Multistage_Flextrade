"""Tier-2 additions on REAL DK1 prices (real DA + real mFRR capacity, reserve = Stage-1 quantity).

  # A. Expected-value benchmark / Value of the Stochastic Solution on the eight ladder trees
  uv run python tier2_real.py vss --solver appsi_gurobi --gap 0.005 --time-limit 900
  uv run python tier2_real.py vss --ends 2026-10-11 --levels 3            # one tree only
  uv run python tier2_real.py vss --winter                                 # + the three winter trees

  # B. Battery round-trip efficiency and degradation-cost sensitivity, reference tree
  uv run python tier2_real.py battery --gap 0.002 --time-limit 900
  uv run python tier2_real.py battery --grid                               # full 3 x 3 grid

VSS definition (three-stage, risk neutral, beta = 0)
  RP  = optimum of the full 3x3 tree.
  EV  = same model on ONE day type x ONE real-time branch whose DA/ID/mFRR prices, load deviation,
        imbalance ratios and reserve deployment are the probability-weighted means of the full tree.
        Its Stage-1 decisions (batch starts s, MT u/x/y, DA bid/offer quantities, hourly reserve Rtot)
        are kept.
  EEV = full 3x3 tree with those Stage-1 decisions fixed (Stage 2/3 re-optimised).
  VSS = EEV - RP >= 0 up to solver gap.  noise = (gap_RP + gap_EEV) * RP; VSS below noise is not resolved.
Uses the same Ctx / tree recipe / seeds / plant as real_data_mfrr_ladder.py, so RP equals Table II.
"""
import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import real_data_common as rc
import real_data_mfrr_ladder as ml
import real_data_oos as oos

HISTORY_ENDS = ["2026-03-31", "2026-04-28", "2026-05-26", "2026-06-23",
                "2026-07-21", "2026-08-18", "2026-09-15", "2026-10-11"]
WINTER_ENDS = ["2025-12-31", "2026-01-28", "2026-02-25"]
REFERENCE = "2026-10-11"


# ----------------------------------------------------------------------------- helpers
def level_markets(ctx, lv):
    """(mkt, rsv) of market level 1 = DA only, 2 = DA+ID, 3 = DA+ID+up-reserve."""
    return (ctx.no_id if lv == 1 else ctx.mkt), (ctx.rsv if lv == 3 else ctx.no_res)


def collapse_to_ev(api, inst, scen, rt, dep):
    """One day type, one RT branch; probability-weighted means of every tree array."""
    pi = scen.prob
    joint = pi[:, None] * rt.prob                                    # [S, W]
    wm_s = lambda a: (pi[:, None] * a).sum(0, keepdims=True)         # [S,T] -> [1,T]
    wm_sw = lambda a: (joint[:, :, None] * a).sum((0, 1))[None, None, :]   # [S,W,T] -> [1,1,T]
    scen1 = api.OfferingScenarioSet(np.array([1.0]), wm_s(scen.da_buy), wm_s(scen.da_sell), wm_s(scen.sr_up),
                                    wm_s(scen.id_buy), wm_s(scen.id_sell), wm_s(scen.load_dev_mw))
    rt1 = api.RealTimeSet(np.ones((1, 1)), wm_sw(rt.load_rel_dev), wm_sw(rt.r_plus), wm_sw(rt.r_minus))
    dep1 = api.DeploymentSet(wm_sw(dep.rho))
    scen1.validate(inst)
    api.validate_tree(inst, scen1, rt1, dep1)
    return scen1, rt1, dep1


def stage1_from(model):
    """Stage-1 decisions of a solved model (scenario 0 values for the scenario-indexed ones)."""
    T = len(model.H)
    d = dict(s={i: int(round(pyo.value(model.s[i]))) for i in model.s},
             Pbuy=np.array([pyo.value(model.Pbuy[0, t]) for t in range(T)]),
             Psell=np.array([pyo.value(model.Psell[0, t]) for t in range(T)]))
    for nm in ("u", "x", "y"):
        if hasattr(model, nm):
            d[nm] = np.array([int(round(pyo.value(getattr(model, nm)[t]))) for t in range(T)])
    d["R"] = np.array([pyo.value(model.Rtot[0, t]) for t in range(T)]) if hasattr(model, "Rtot") else None
    return d


def make_fixer(st):
    def fix(m):
        T = len(m.H)
        for i in m.s:
            m.s[i].fix(st["s"][i])
        for nm in ("u", "x", "y"):
            if nm in st:
                for t in range(T):
                    getattr(m, nm)[t].fix(int(st[nm][t]))
        for k in m.S:
            for t in range(T):
                m.Pbuy[k, t].fix(float(np.clip(st["Pbuy"][t], m.Pbuy[k, t].lb, m.Pbuy[k, t].ub)))
                m.Psell[k, t].fix(float(np.clip(st["Psell"][t], m.Psell[k, t].lb, m.Psell[k, t].ub)))
        if st["R"] is not None and hasattr(m, "Rtot"):
            m.c_fixR = pyo.ConstraintList()
            for k in m.S:
                for t in range(T):
                    m.c_fixR.add(m.Rtot[k, t] == float(max(st["R"][t], 0.0)))
    return fix


def gap_of(res):
    return float(res.mip_gap or 0.0)


# ----------------------------------------------------------------------------- A. VSS
def run_vss(ctx, ev, opt):
    api = ctx.api
    ends = opt.ends or (HISTORY_ENDS + (WINTER_ENDS if opt.winter else []))
    rows = []
    for end in ends:
        inst, scen, rt, dep = ctx.tree(pd.Timestamp(end))
        scen1, rt1, dep1 = collapse_to_ev(api, inst, scen, rt, dep)
        for lv in opt.levels:
            mkt, rsv = level_markets(ctx, lv)
            t0 = time.time()
            print(f"\n>>> {end} level {lv}: RP (full tree)", flush=True)
            _, rp = ev.solve(inst, scen, rt, dep, mkt=mkt, rsv=rsv)
            print(f">>> {end} level {lv}: EV problem (1 x 1 tree)", flush=True)
            m_ev, r_ev = ev.solve(inst, scen1, rt1, dep1, mkt=mkt, rsv=rsv)
            st = stage1_from(m_ev)
            print(f">>> {end} level {lv}: EEV (Stage 1 of EV fixed in full tree)", flush=True)
            _, eev = ev.solve(inst, scen, rt, dep, fix=make_fixer(st), mkt=mkt, rsv=rsv)
            c_rp, c_eev = rc.cost(rp), rc.cost(eev)
            noise = (gap_of(rp) + gap_of(eev)) * c_rp
            vss = c_eev - c_rp
            row = dict(history_end=end, level=lv, RP=c_rp, EV_model_cost=rc.cost(r_ev), EEV=c_eev,
                       VSS=vss, VSS_pct=100 * vss / c_rp, noise=noise, resolved=bool(vss > noise),
                       gap_RP=gap_of(rp), gap_EEV=gap_of(eev),
                       offer_RP_mw=rp.kpis["avg_reserve_offer_mw"], offer_EV_mw=r_ev.kpis["avg_reserve_offer_mw"],
                       jobs_RP=len(rp.jobs), jobs_EV=len(r_ev.jobs), seconds=time.time() - t0)
            rows.append(row)
            rc.save_rows(rows, opt.out_t2, "vss.csv")
            print(f"    RP {c_rp:,.0f}  EEV {c_eev:,.0f}  VSS {vss:,.0f} ({row['VSS_pct']:.2f}%)  "
                  f"noise ~{noise:,.0f}  resolved={row['resolved']}", flush=True)
    df = pd.DataFrame(rows)
    write_vss_tex(df, opt.out_t2)
    return df


def write_vss_tex(df, out):
    d3 = df[df.level == df.level.max()]
    lines = []
    for r in d3.itertuples():
        mark = "" if r.resolved else r"$^{\ast}$"
        lines.append(f"{r.history_end} & {r.RP:,.0f} & {r.EEV:,.0f} & {r.VSS:,.0f}{mark} ({r.VSS_pct:.1f}) "
                     f"& {r.noise:,.0f} \\\\")
    lines += [r"\midrule", f"Mean & & & {d3.VSS.mean():,.0f} ({d3.VSS_pct.mean():.1f}) & \\\\"]
    with open(os.path.join(out, "vss_table.tex"), "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------- B. battery
def battery_cases(grid):
    base = (0.80, 0.95, 5.0)
    sym = lambda rte: (rte ** 0.5, rte ** 0.5)
    if grid:
        out = []
        for rte in (0.76, 0.85, 0.90):
            ec, ed = (0.80, 0.95) if rte == 0.76 else sym(rte)
            out += [(f"RTE {rte:.0%}, C_TP {c:g}", ec, ed, c) for c in (2.5, 5.0, 10.0)]
        return out
    return [("baseline RTE 76%, C_TP 5", *base),
            ("RTE 85%, C_TP 5", *sym(0.85), 5.0), ("RTE 90%, C_TP 5", *sym(0.90), 5.0),
            ("RTE 76%, C_TP 2.5", 0.80, 0.95, 2.5), ("RTE 76%, C_TP 10", 0.80, 0.95, 10.0)]


def run_battery(ctx, ev, opt):
    inst0, scen, rt, dep = ctx.tree(pd.Timestamp(opt.ref))
    rows = []
    for label, ec, ed, c in battery_cases(opt.grid):
        inst = replace(inst0, bess=replace(inst0.bess, eta_ch=ec, eta_dis=ed, degradation_eur_mwh=c)).validate()
        res = {}
        for lv in (1, 2, 3):
            mkt, rsv = level_markets(ctx, lv)
            print(f"\n>>> {label}: level {lv}", flush=True)
            _, res[lv] = ev.solve(inst, scen, rt, dep, mkt=mkt, rsv=rsv)
        l1, l2, l3 = (rc.cost(res[i]) for i in (1, 2, 3))
        noise = (gap_of(res[1]) + gap_of(res[3])) * l1
        k3 = res[3].kpis
        row = dict(case=label, rte=ec * ed, eta_ch=ec, eta_dis=ed, C_TP=c, L1=l1, L2=l2, L3=l3,
                   ID_saving=l1 - l2, reserve_saving=l2 - l3, total_saving=l1 - l3,
                   total_pct=100 * (l1 - l3) / l1, offer_mw=k3["avg_reserve_offer_mw"],
                   bess_offer_mw=k3["avg_r_bess_dis_mw"] + k3["avg_r_bess_ch_mw"],
                   implied_throughput_mwh=(abs(k3["exp_cost_bess_eur"]) / c if c > 0 else np.nan),
                   noise=noise, gap_L1=gap_of(res[1]), gap_L2=gap_of(res[2]), gap_L3=gap_of(res[3]))
        rows.append(row)
        rc.save_rows(rows, opt.out_t2, "battery.csv")
        print(f"    total saving {row['total_saving']:,.0f} ({row['total_pct']:.2f}%), reserve {row['reserve_saving']:,.0f}, "
              f"offer {row['offer_mw']:.1f} MW, noise ~{noise:,.0f}", flush=True)
    df = pd.DataFrame(rows)
    lines = [f"{r.case} & {r.L1:,.0f} & {r.L2:,.0f} & {r.L3:,.0f} & {r.ID_saving:,.0f} & {r.reserve_saving:,.0f} & "
             f"{r.total_saving:,.0f} ({r.total_pct:.1f}) & {r.offer_mw:.1f} & {r.noise:,.0f} \\\\"
             for r in df.itertuples()]
    with open(os.path.join(opt.out_t2, "battery_table.tex"), "w") as f:
        f.write("\n".join(lines))
    return df


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=["vss", "battery"])
    rc.add_common_args(ap)
    ap.add_argument("--mfrr", default=ml.MfrrCtx.mfrr_csv)
    ap.add_argument("--ends", nargs="+", default=None, help="history ends (default: the eight Table II trees)")
    ap.add_argument("--levels", type=int, nargs="+", default=[3], help="market levels for VSS (1,2,3)")
    ap.add_argument("--winter", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--ref", default=REFERENCE)
    ap.set_defaults(gap=0.005, time_limit=900)
    opt, extra = ap.parse_known_args()
    extra = list(extra)
    if "--reserve-stage1" not in extra:
        extra.append("--reserve-stage1")               # the paper's Stage-1 reserve configuration
    ml.MfrrCtx.mfrr_csv = opt.mfrr
    ctx = ml.MfrrCtx(opt, extra)
    assert ctx.strat.reserve_stage1 and ctx.strat.beta == 0.0, "expected Stage-1 reserve and beta = 0"
    opt.out_t2 = os.path.join(opt.out, "tier2")
    os.makedirs(opt.out_t2, exist_ok=True)
    ev = oos.Evaluator(ctx)                             # solve() with verification handling, as in the OOS runs
    df = run_vss(ctx, ev, opt) if opt.mode == "vss" else run_battery(ctx, ev, opt)
    print("\n=== RESULT ===")
    print(df.round(2).to_string(index=False))
    print(f"\nwritten to {opt.out_t2}")
