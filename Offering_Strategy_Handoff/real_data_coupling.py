"""Production-coupling tests on REAL DK1 prices (paper Table VII and the decoupling test).

  # decoupling test at the paper battery (plan A = energy-only optimum, plan B = DA-only optimum)
  uv run python real_data_coupling.py decouple --end 2026-10-11

  # battery-size sweep: reserve benefit + coupling value for 0 / 0.25 / 0.5 / 1.0 x (40 MW / 200 MWh)
  uv run python real_data_coupling.py bess --end 2026-10-11 --scales 0 0.25 0.5 1.0

  # several days (history = day before each test day)
  uv run python real_data_coupling.py decouple --start 2026-04-01 --end 2026-10-12 --step 28

coupling value = decoupled cost - integrated cost.  Decoupled is a restriction of the integrated model, so a
NEGATIVE value (or one below `noise_eur`) is solver-gap noise, not a real effect.
Paper used gap 0.1%; defaults here are --gap 0.002 --time-limit 900 (slow: 3-4 solves per case).
"""
import argparse
import os
import time

import numpy as np

import real_data_common as rc

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("mode", choices=["decouple", "bess"])
rc.add_common_args(ap)
ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
ap.set_defaults(gap=0.002, time_limit=900)
opt, extra = ap.parse_known_args()
ctx = rc.Ctx(opt, extra)
api = ctx.api
out_dir = os.path.join(opt.out, f"coupling_{opt.mode}")

scales = [1.0] if opt.mode == "decouple" else opt.scales
rows = []
for end in ctx.history_ends():
    inst0, scen, rt, dep = ctx.tree(end)
    print(f"\n##### history to {end.date()}  ({scen.n} day types x {rt.n_w} RT)", flush=True)
    for k in scales:
        t0 = time.time()
        inst = ctx.scaled_bess(k)
        print(f">>> scale {k:g}: integrated", flush=True)
        cpl = ctx.optimize(inst, scen, rt, dep)
        print(f">>> scale {k:g}: plan A (energy-only DA+ID, no reserve)", flush=True)
        eo = ctx.optimize(inst, scen, rt, dep, rsv=ctx.no_res)
        print(f">>> scale {k:g}: decoupled A", flush=True)
        decA = ctx.fixed_production_run(inst, scen, rt, dep, eo.jobs)
        decB, planB = None, None
        if opt.mode == "decouple":
            print(">>> plan B (DA-only)", flush=True)
            planB = ctx.optimize(inst, scen, rt, dep, mkt=ctx.no_id, rsv=ctx.no_res)
            print(">>> decoupled B", flush=True)
            decB = ctx.fixed_production_run(inst, scen, rt, dep, planB.jobs)
        c = rc.cost(cpl)
        gap_c, gap_a = cpl.mip_gap or 0.0, decA.mip_gap or 0.0
        row = dict(history_end=str(end.date()), bess_scale=k, bess_mw=40 * k, bess_mwh=200 * k,
                   no_reserve_cost=rc.cost(eo), integrated_cost=c, decoupled_A_cost=rc.cost(decA),
                   decoupled_B_cost=rc.cost(decB) if decB is not None else np.nan,
                   reserve_benefit=rc.cost(eo) - c,
                   coupling_A=rc.cost(decA) - c,
                   coupling_B=(rc.cost(decB) - c) if decB is not None else np.nan,
                   noise_eur=(gap_c + gap_a) * c,
                   gap_integrated=gap_c, gap_decoupled_A=gap_a,
                   avg_reserve_mw=cpl.kpis["avg_reserve_offer_mw"], seconds=time.time() - t0)
        rows.append(row)
        rc.save_rows(rows, out_dir, "coupling.csv")
        print(f"    done {time.time()-t0:.0f}s: integrated {c:.0f}  decoupled A {row['decoupled_A_cost']:.0f}  "
              f"coupling A {row['coupling_A']:.0f} (noise ~{row['noise_eur']:.0f})", flush=True)

print("\n=== RESULT ===")
cols = ["history_end", "bess_mw", "bess_mwh", "no_reserve_cost", "integrated_cost", "decoupled_A_cost",
        "coupling_A", "coupling_B", "reserve_benefit", "noise_eur", "avg_reserve_mw"]
import pandas as pd  # noqa: E402
print(pd.DataFrame(rows)[cols].round(1).to_string(index=False))
print(f"\nwritten to {os.path.join(out_dir, 'coupling.csv')}")
