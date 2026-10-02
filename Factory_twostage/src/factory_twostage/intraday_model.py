"""Extensive-form Pyomo model for the DA + intraday stochastic scheduler."""

import pyomo.environ as pyo

from .candidates import Candidate
from .data import Instance, SchedulerConfig
from .model_blocks import add_batch_block, add_mt_commitment
from .parameters import DT_H
from .scenarios import IntradayMarket, ScenarioSet


def build_intraday_model(
    inst: Instance,
    market: IntradayMarket,
    scenarios: ScenarioSet,
    cfg: SchedulerConfig,
    candidates: list[Candidate],
) -> pyo.ConcreteModel:
    """Build the deterministic equivalent with shared Stage-1 decisions."""
    hours, scenario_count = inst.horizon_h, scenarios.n
    mt, bess = inst.mt, inst.bess
    model = pyo.ConcreteModel("FactoryMicroturbineIntraday")
    model.H = pyo.RangeSet(0, hours - 1)
    model.S = pyo.RangeSet(0, scenario_count - 1)

    model.Pbuy = pyo.Var(model.H, bounds=(0, inst.grid_limit_mw))
    model.Psell = pyo.Var(model.H, bounds=(0, inst.grid_sell_limit_mw))
    by_hour = add_batch_block(model, inst, cfg, candidates)
    model.Lbatch = pyo.Expression(
        model.H,
        rule=lambda m, hour: pyo.quicksum(power * m.s[index] for index, power, _ in by_hour[hour]),
    )
    if mt is not None:
        add_mt_commitment(model, inst)

    model.Ibuy = pyo.Var(model.S, model.H, bounds=(0, market.cap_buy_mw))
    model.Isell = pyo.Var(model.S, model.H, bounds=(0, market.cap_sell_mw))
    if mt is not None:
        model.B = pyo.RangeSet(0, len(mt.block_width_mw) - 1)
        model.Pmt = pyo.Var(model.S, model.H, bounds=(0, mt.p_max_mw))
        model.Pb = pyo.Var(
            model.B,
            model.S,
            model.H,
            bounds=lambda _, block, __, ___: (0, mt.block_width_mw[block]),
        )
        initial_on = 1.0 if mt.initial_on else 0.0
        initial_power = float(mt.initial_power_mw) if mt.initial_on else 0.0
        previous_on = lambda m, hour: m.u[hour - 1] if hour > 0 else initial_on
        previous_power = lambda m, scenario, hour: (
            m.Pmt[scenario, hour - 1] if hour > 0 else initial_power
        )
        model.c_mt_sum = pyo.Constraint(
            model.S,
            model.H,
            rule=lambda m, scenario, hour: m.Pmt[scenario, hour]
            == mt.p_min_mw * m.u[hour]
            + pyo.quicksum(m.Pb[block, scenario, hour] for block in m.B),
        )
        model.c_mt_block = pyo.Constraint(
            model.B,
            model.S,
            model.H,
            rule=lambda m, block, scenario, hour: m.Pb[block, scenario, hour]
            <= mt.block_width_mw[block] * m.u[hour],
        )
        model.c_ramp_up = pyo.Constraint(
            model.S,
            model.H,
            rule=lambda m, scenario, hour: m.Pmt[scenario, hour]
            - previous_power(m, scenario, hour)
            <= mt.ramp_up_mw_h * previous_on(m, hour) + mt.startup_ramp_mw_h * m.x[hour],
        )
        model.c_ramp_down = pyo.Constraint(
            model.S,
            model.H,
            rule=lambda m, scenario, hour: previous_power(m, scenario, hour)
            - m.Pmt[scenario, hour]
            <= mt.ramp_down_mw_h * m.u[hour] + mt.shutdown_ramp_mw_h * m.y[hour],
        )

    if bess is not None:
        model.Pch = pyo.Var(model.S, model.H, bounds=(0, bess.p_max_mw))
        model.Pdis = pyo.Var(model.S, model.H, bounds=(0, bess.p_max_mw))
        model.SoC = pyo.Var(model.S, model.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))
        previous_soc = lambda m, scenario, hour: (
            m.SoC[scenario, hour - 1] if hour > 0 else bess.soc_init_mwh
        )
        model.c_soc = pyo.Constraint(
            model.S,
            model.H,
            rule=lambda m, scenario, hour: m.SoC[scenario, hour]
            == previous_soc(m, scenario, hour)
            + bess.eta_ch * m.Pch[scenario, hour] * DT_H
            - m.Pdis[scenario, hour] * DT_H / bess.eta_dis,
        )
        model.c_soc_terminal = pyo.Constraint(
            model.S, rule=lambda m, scenario: m.SoC[scenario, hours - 1] >= bess.soc_init_mwh
        )
        if bess.enforce_exclusive:
            model.vch = pyo.Var(model.S, model.H, domain=pyo.Binary)
            model.vdis = pyo.Var(model.S, model.H, domain=pyo.Binary)
            model.c_charge_power = pyo.Constraint(
                model.S,
                model.H,
                rule=lambda m, scenario, hour: m.Pch[scenario, hour]
                <= bess.p_max_mw * m.vch[scenario, hour],
            )
            model.c_discharge_power = pyo.Constraint(
                model.S,
                model.H,
                rule=lambda m, scenario, hour: m.Pdis[scenario, hour]
                <= bess.p_max_mw * m.vdis[scenario, hour],
            )
            model.c_bess_exclusive = pyo.Constraint(
                model.S,
                model.H,
                rule=lambda m, scenario, hour: m.vch[scenario, hour]
                + m.vdis[scenario, hour] <= 1,
            )

    mt_output = lambda m, scenario, hour: (
        m.Pmt[scenario, hour] if mt is not None else 0.0
    )
    bess_output = lambda m, scenario, hour: (
        m.Pdis[scenario, hour] - m.Pch[scenario, hour] if bess is not None else 0.0
    )
    net_position = lambda m, scenario, hour: (
        m.Pbuy[hour] - m.Psell[hour] + m.Ibuy[scenario, hour] - m.Isell[scenario, hour]
    )
    model.c_power_balance = pyo.Constraint(
        model.S,
        model.H,
        rule=lambda m, scenario, hour: net_position(m, scenario, hour)
        + mt_output(m, scenario, hour)
        + bess_output(m, scenario, hour)
        == inst.base_load_mw[hour]
        + scenarios.load_dev_mw[scenario, hour]
        + m.Lbatch[hour],
    )
    model.c_net_import = pyo.Constraint(
        model.S,
        model.H,
        rule=lambda m, scenario, hour: net_position(m, scenario, hour) <= inst.grid_limit_mw,
    )
    model.c_net_export = pyo.Constraint(
        model.S,
        model.H,
        rule=lambda m, scenario, hour: -net_position(m, scenario, hour)
        <= inst.grid_sell_limit_mw,
    )

    objective = pyo.quicksum(
        inst.price_buy_eur_mwh[hour] * model.Pbuy[hour]
        - inst.price_sell_eur_mwh[hour] * model.Psell[hour]
        for hour in range(hours)
    )
    objective += pyo.quicksum(
        scenarios.prob[scenario]
        * (
            scenarios.id_buy_eur_mwh[scenario, hour] * model.Ibuy[scenario, hour]
            - scenarios.id_sell_eur_mwh[scenario, hour] * model.Isell[scenario, hour]
        )
        for scenario in range(scenario_count)
        for hour in range(hours)
    )
    if mt is not None:
        objective += pyo.quicksum(
            mt.base_cost_eur_mwh * mt.p_min_mw * model.u[hour]
            + mt.startup_cost_eur * model.x[hour]
            + mt.shutdown_cost_eur * model.y[hour]
            for hour in range(hours)
        )
        objective += pyo.quicksum(
            scenarios.prob[scenario]
            * mt.block_cost_eur_mwh[block]
            * model.Pb[block, scenario, hour]
            for block in range(len(mt.block_width_mw))
            for scenario in range(scenario_count)
            for hour in range(hours)
        )
    if bess is not None:
        charge_weight = 1.0 if bess.degradation_basis == "throughput" else 0.0
        objective += pyo.quicksum(
            scenarios.prob[scenario]
            * bess.degradation_eur_mwh
            * (charge_weight * model.Pch[scenario, hour] + model.Pdis[scenario, hour])
            for scenario in range(scenario_count)
            for hour in range(hours)
        )
    model.obj = pyo.Objective(expr=objective, sense=pyo.minimize)
    return model