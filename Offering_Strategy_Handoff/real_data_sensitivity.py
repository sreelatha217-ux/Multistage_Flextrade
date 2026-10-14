"""Sensitivity studies on REAL DK1 prices (paper Tables VI, VIII, X).

  # Table VIII analogue: reserve price ratio (paper: 0.5x / 1x / 1.5x of 0.12)
  uv run python real_data_sensitivity.py reserve --end 2026-10-11 --ratios 0.06 0.12 0.18

  # Table X analogue: CVaR frontier on a 12 x 2 tree
  uv run python real_data_sensitivity.py frontier --end 2026-10-11 --betas 0 0.25 0.5 0.75 0.9

  # Table VI analogue: reserve by asset (none / MT+DR / BESS only / all four)
  uv run python real_data_sensitivity.py assets --end 2026-10-11

Several days: add  --start 2026-04-01 --end 2026-10-12 --step 28  (history = day before each test day).
"""
import argparse
import os
import time

import pandas as pd

import real_data_common as rc

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("mode", choices=["reserve", "frontier", "assets"])
rc.add_common_args(ap)
ap.add_argument("--ratios", type=float, nargs="+", default=[0.06, 0.12, 0.18])
ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
ap.add_argument("--fr-scenarios", type=int, default=12, help="frontier: price scenarios")
ap.add_argument("--fr-rt", type=int, default=2, help="frontier: RT branches")
ap.set_defaults(gap=0.01, time_limit=900)
ap.add_argument("--real-mfrr", action="store_true",
                help="use REAL mFRR up-capacity prices (mfrr_capacity_dk1.csv) instead of --sr-ratio x DA")
opt, extra = ap.parse_known_args()
if opt.real_mfrr:
    import real_data_mfrr_ladder as ml
    ctx = ml.MfrrCtx(opt, extra)
else:
    ctx = rc.Ctx(opt, extra)
print("reserve prices:", "REAL mFRR" if opt.real_mfrr else f"synthetic {opt.sr_ratio} x DA", "|",
      "reserve capacity:", "Stage-1 quantity" if ctx.strat.reserve_stage1 else "price-quantity curve", flush=True)
api = ctx.api
out_dir = os.path.join(opt.out, f"sens_{opt.mode}")
rows = []

for end in ctx.history_ends():
    day = str(end.date())
    t0 = time.time()
    if opt.mode == "reserve":
        for r in opt.ratios:
            inst, scen, rt, dep = ctx.tree(end, sr_ratio=r)
            res = ctx.optimize(inst, scen, rt, dep)
            st = api.compare_market_stages(inst, scen, rt, dep, ctx.mkt, ctx.rsv, ctx.bal, ctx.strat, ctx.cfg, main=res)
            da_only, full = float(st.expected_net_cost_eur.iloc[0]), float(st.expected_net_cost_eur.iloc[-1])
            rows.append(dict(history_end=day, sr_ratio=r, da_only_cost=da_only, full_cost=full,
                             saving_eur=da_only - full, saving_pct=100 * (da_only - full) / da_only,
                             avg_reserve_mw=res.kpis["avg_reserve_offer_mw"], gap=res.mip_gap or 0.0))
            rc.save_rows(rows, out_dir, "reserve_ratio.csv")
            print(f"[{day}] sr_ratio {r}: saving {rows[-1]['saving_eur']:.0f} ({rows[-1]['saving_pct']:.2f}%)  "
                  f"offer {rows[-1]['avg_reserve_mw']:.1f} MW", flush=True)
    elif opt.mode == "frontier":
        inst, scen, rt, dep = ctx.tree(end, n=opt.fr_scenarios, w=opt.fr_rt)
        fr = api.risk_frontier(inst, scen, rt, dep, ctx.mkt, ctx.rsv, ctx.bal, ctx.strat, ctx.cfg, opt.betas)
        fr.insert(0, "history_end", day)
        rows.extend(fr.to_dict("records"))
        rc.save_rows(rows, out_dir, "cvar_frontier.csv")
        print(fr.round(1).to_string(index=False), flush=True)
    else:  # assets
        inst, scen, rt, dep = ctx.tree(end)
        cases = [("none (DA+ID only)", None), ("MT + DR only", ("mt", "dr")),
                 ("BESS only (dis + ch)", ("bess_dis", "bess_ch")), ("all four", ctx.all_assets)]
        base = None
        for name, assets in cases:
            rsv = ctx.no_res if assets is None else ctx.reserve(assets)
            res = ctx.optimize(inst, scen, rt, dep, rsv=rsv)
            c = rc.cost(res)
            base = c if base is None else base
            rows.append(dict(history_end=day, reserve_from=name, net_cost=c, saving_vs_none=base - c,
                             avg_reserve_mw=res.kpis["avg_reserve_offer_mw"], gap=res.mip_gap or 0.0))
            rc.save_rows(rows, out_dir, "reserve_by_asset.csv")
            print(f"[{day}] {name:24s} cost {c:10.0f}  saving {base - c:8.0f}  gap {100*(res.mip_gap or 0):.2f}%", flush=True)
    print(f"[{day}] {opt.mode} done in {time.time()-t0:.0f}s", flush=True)

print("\n=== RESULT ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nNote: differences smaller than about gap x cost are not resolved.")
