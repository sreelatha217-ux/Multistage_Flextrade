"""Stage-2 intraday production rescheduling on REAL DK1 day types (paper Table V analogue).

  uv run python real_data_resched.py --end 2026-10-11 --scenarios 3 --rt 3 --reschedule-hour 0

Solves the integrated model on the real-data tree, then for each day type (the 'ID scenario') re-optimises the
production plan under fixed Stage-1 MT commitment (factory_mt_production_rescheduler.py).
This is a perfect-foresight standalone study: do NOT add its saving to the market-ladder saving.
Several days: --start 2026-04-01 --end 2026-10-12 --step 28
"""
import argparse
import os

import pandas as pd

import real_data_common as rc
from factory_mt_production_rescheduler import reschedule_production_by_id_scenario

ap = argparse.ArgumentParser(allow_abbrev=False)
rc.add_common_args(ap)
ap.add_argument("--reschedule-hour", type=float, default=0.0)
ap.set_defaults(gap=0.01, time_limit=600)
opt, extra = ap.parse_known_args()
ctx = rc.Ctx(opt, extra)
out_dir = os.path.join(opt.out, "resched")
os.makedirs(out_dir, exist_ok=True)

summ, wrows = [], []
for end in ctx.history_ends():
    day = str(end.date())
    inst, scen, rt, dep = ctx.tree(end)
    res = ctx.optimize(inst, scen, rt, dep)
    jobs, s = reschedule_production_by_id_scenario(inst, res.jobs, res.da_position.mt_on.to_numpy(), scen,
                                                   ctx.cfg, opt.reschedule_hour)
    s = s.copy()
    s.insert(0, "history_end", day)
    summ.append(s)
    w = float((s.probability * s.production_reschedule_savings_eur).sum())
    base = float((s.probability * s.fixed_plan_cost_eur).sum())
    wrows.append(dict(history_end=day, weighted_saving_eur=w, weighted_fixed_cost_eur=base, saving_pct=100 * w / base))
    jobs.to_csv(os.path.join(out_dir, f"jobs_{day}.csv"), index=False)
    pd.concat(summ).to_csv(os.path.join(out_dir, "resched_summary.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(out_dir, "resched_weighted.csv"), index=False)
    print(f"\n[{day}] rescheduling by day type (hour {opt.reschedule_hour:g}):")
    print(s[["id_scenario", "probability", "fixed_plan_cost_eur", "operating_cost_eur",
             "production_reschedule_savings_eur", "jobs_changed", "units_produced", "mip_gap"]].round(3).to_string(index=False))
    print(f"[{day}] probability-weighted saving {w:.0f} EUR ({wrows[-1]['saving_pct']:.2f}%)", flush=True)

print("\n=== RESULT ===")
print(pd.DataFrame(wrows).round(2).to_string(index=False))
