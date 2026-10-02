from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from . import day_ahead as da
from . import intraday as idm
from .day_ahead import SchedulerConfig, SolverSettings
from .realtime_optimizer import optimize_realtime
from .realtime_results import RealTimeResult
from .realtime_scenarios import BalancingMarket, RealTimeSet
from .scenarios import IntradayMarket, ScenarioSet


def selftest(solver: str = "appsi_highs") -> int:
    """Four consistency checks that follow from theory (tolerance covers the MIP gap):
       T1  prohibitive imbalance prices + no RT noise  ->  objective equals the Stage-2 (ID) model
       T2  RT noise (same prices)                      ->  objective >= noise-free objective (Jensen)
       T3  passive mode                                ->  objective >= strategic mode (restricted feasible set)
       T4  every solution passes the independent verification (strict mode raises otherwise)"""
    inst = da.make_benchmark_instance(2024, max_batches=5)
    cfg = SchedulerConfig(solver=SolverSettings(solver, 1e-4, 240.0))
    mkt, S = IntradayMarket(), 3
    scen = ScenarioSet.generate(inst, S, 7)
    T = inst.horizon_h
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    results: List[RealTimeResult] = []

    def run(rts: RealTimeSet, mode: str) -> float:
        results.append(optimize_realtime(inst, scen, rts, mkt, BalancingMarket(mode), cfg))
        return results[-1].objective_eur

    id_obj = idm.optimize_intraday(inst, scen, mkt, cfg).objective_eur
    o1 = run(RealTimeSet.constant(S, 1, T, r_plus=50.0, r_minus=0.0), "strategic")
    tol = lambda x: max(1.0, 3e-3 * abs(x))
    check("T1 reduces to the ID model", abs(o1 - id_obj) <= tol(id_obj), f"RT={o1:,.2f}  ID={id_obj:,.2f} EUR "
          "(a deviation can come from a binding ID cap: surplus is dumped for free at r-=0)")

    noisy = RealTimeSet.generate(inst, S, 4, seed=3)
    rp, rm = np.full_like(noisy.r_plus, 1.25), np.full_like(noisy.r_minus, 0.75)
    calm = run(replace(noisy, load_rel_dev=np.zeros_like(noisy.load_rel_dev), r_plus=rp, r_minus=rm), "strategic")
    noisy = replace(noisy, r_plus=rp, r_minus=rm)
    strat = run(noisy, "strategic")
    check("T2 RT noise does not lower the cost", strat >= calm - tol(calm), f"noisy={strat:,.2f}  noise-free={calm:,.2f} EUR")
    pas = run(noisy, "passive")
    check("T3 passive >= strategic", pas >= strat - tol(strat), f"passive={pas:,.2f}  strategic={strat:,.2f} EUR")
    check("T4 independent verification", all(r.verification["passed"] for r in results),
          f"{len(results)} RT solves verified")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
