"""Orchestration for the two-stage stochastic DA + intraday optimization."""

import logging

import pyomo.environ as pyo

from .candidates import build_candidates
from .data import Instance, SchedulerConfig
from .exceptions import InfeasibleScheduleError
from .intraday_model import build_intraday_model
from .intraday_results import IntradayResult, extract_intraday, _value
from .scenarios import IntradayMarket, ScenarioSet
from .solver import solve_model

log = logging.getLogger("factory_twostage")


def value_of_stochastic_solution(
    inst: Instance,
    market: IntradayMarket,
    scenarios: ScenarioSet,
    cfg: SchedulerConfig,
    candidates,
    rp_eur: float,
) -> dict[str, float]:
    """Compute VSS by fixing the expected-value plan and resolving recourse."""
    expected_model = build_intraday_model(
        inst, market, scenarios.expected_value(), cfg, candidates
    )
    solve_model(expected_model, cfg.solver)
    recourse_model = build_intraday_model(inst, market, scenarios, cfg, candidates)
    for candidate in recourse_model.C:
        recourse_model.s[candidate].fix(round(_value(expected_model.s[candidate])))
    for hour in range(inst.horizon_h):
        recourse_model.Pbuy[hour].fix(_value(expected_model.Pbuy[hour]))
        recourse_model.Psell[hour].fix(_value(expected_model.Psell[hour]))
        if inst.mt is not None:
            for name in ("u", "x", "y"):
                getattr(recourse_model, name)[hour].fix(
                    round(_value(getattr(expected_model, name)[hour]))
                )
    try:
        solve_model(recourse_model, cfg.solver)
        expected_value_cost = float(pyo.value(recourse_model.obj))
    except InfeasibleScheduleError:
        log.warning("expected-value plan is infeasible in at least one scenario; EEV = inf")
        expected_value_cost = float("inf")
    expected_plan_cost = float(pyo.value(expected_model.obj))
    value = expected_value_cost - rp_eur
    return {
        "ev_objective_eur": expected_plan_cost,
        "eev_eur": expected_value_cost,
        "rp_eur": rp_eur,
        "vss_eur": value,
        "vss_pct": 100 * value / abs(rp_eur) if rp_eur else float("nan"),
    }


def optimize_intraday(
    inst: Instance,
    scenarios: ScenarioSet,
    market: IntradayMarket | None = None,
    cfg: SchedulerConfig | None = None,
    compute_vss: bool = False,
) -> IntradayResult:
    """Validate, build, solve, verify, and package the stochastic schedule."""
    cfg = cfg or SchedulerConfig()
    market = (market or IntradayMarket()).validate()
    inst.validate()
    cfg.check(inst)
    scenarios.validate(inst)
    candidates = build_candidates(inst, cfg.start_step_h)
    log.info(
        "%d candidate starts, %d scenarios, MT=%s, BESS=%s",
        len(candidates), scenarios.n, "yes" if inst.mt else "no", "yes" if inst.bess else "no",
    )
    model = build_intraday_model(inst, market, scenarios, cfg, candidates)
    solve_info = solve_model(model, cfg.solver)
    result = extract_intraday(
        inst, market, scenarios, cfg, model, candidates, solve_info
    )
    if compute_vss:
        result.vss = value_of_stochastic_solution(
            inst, market, scenarios, cfg, candidates, result.objective_eur
        )
    return result