"""Extraction, scenario verification, KPIs, and persisted stochastic results."""

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ._internal import __version__
from .candidates import Candidate
from .data import Instance, SchedulerConfig
from .exceptions import VerificationError
from .results import build_hourly, jobs_from_starts, verify
from .scenarios import IntradayMarket, ScenarioSet

log = logging.getLogger("factory_twostage")


@dataclass
class IntradayResult:
    status: str
    objective_eur: float
    mip_gap: float | None
    solve_time_s: float
    model_stats: dict[str, int]
    jobs: pd.DataFrame
    da_position: pd.DataFrame
    scenarios: pd.DataFrame
    hourly: pd.DataFrame
    kpis: dict[str, float]
    verification: dict[str, object]
    scenario_set: ScenarioSet
    vss: dict[str, float] | None = None

    def summary(self) -> str:
        kpis = self.kpis
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        lines = [
            f"status={self.status}  expected cost={self.objective_eur:,.2f} EUR  "
            f"gap={gap}  time={self.solve_time_s:.1f}s  scenarios={int(kpis['n_scenarios'])}",
            f"Stage 1: DA net cost={kpis['da_cost_eur']:,.2f} EUR "
            f"(buy {kpis['da_buy_mwh']:.1f} MWh, sell {kpis['da_sell_mwh']:.1f} MWh)  "
            f"MT on {int(kpis['mt_on_hours'])} h, {int(kpis['mt_starts'])} start(s)  "
            f"batches={int(kpis['n_batches'])}  units={kpis['units_produced']:.0f}",
            f"Stage 2 (expected): ID net cost={kpis['exp_id_cost_eur']:,.2f} EUR "
            f"(buy {kpis['exp_id_buy_mwh']:.1f} MWh, sell {kpis['exp_id_sell_mwh']:.1f} MWh)  "
            f"MT cost={kpis['exp_mt_cost_eur']:,.2f}  "
            f"BESS degradation={kpis['exp_bess_deg_eur']:,.2f} EUR",
            f"scenario cost: mean={kpis['cost_mean_eur']:,.2f}  "
            f"std={kpis['cost_std_eur']:,.2f}  min={kpis['cost_min_eur']:,.2f}  "
            f"max={kpis['cost_max_eur']:,.2f} EUR",
            f"verification={'PASS' if self.verification['passed'] else 'FAIL'}",
        ]
        if self.vss:
            lines.append(
                f"VSS: EEV={self.vss['eev_eur']:,.2f}  RP={self.vss['rp_eur']:,.2f}  "
                f"VSS={self.vss['vss_eur']:,.2f} EUR ({self.vss['vss_pct']:.3f}%)"
            )
        return "\n".join(lines)

    def save(self, outdir: str | Path) -> Path:
        output = Path(outdir)
        output.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(output / "jobs.csv", index=False)
        self.da_position.to_csv(output / "da_position.csv", index=False)
        self.scenarios.to_csv(output / "scenarios.csv", index=False)
        self.hourly.to_csv(output / "hourly_scenarios.csv", index=False)
        self.scenario_set.save(output / "scenario_set.json")
        metadata = {
            "version": __version__,
            "status": self.status,
            "expected_cost_eur": self.objective_eur,
            "mip_gap": self.mip_gap,
            "solve_time_s": self.solve_time_s,
            "model_stats": self.model_stats,
            "kpis": self.kpis,
            "verification": self.verification,
            "vss": self.vss,
        }
        (output / "summary.json").write_text(
            json.dumps(metadata, indent=2, default=float), encoding="utf-8"
        )
        return output


def _value(variable, default: float = 0.0) -> float:
    return default if variable.value is None else float(variable.value)


def extract_intraday(
    inst: Instance,
    market: IntradayMarket,
    scenarios: ScenarioSet,
    cfg: SchedulerConfig,
    model: pyo.ConcreteModel,
    candidates: list[Candidate],
    solve_info: dict[str, object],
) -> IntradayResult:
    hours, scenario_count = inst.horizon_h, scenarios.n
    jobs = jobs_from_starts(inst, candidates, model)
    hour_index = np.arange(hours, dtype=float)
    jobs["energy_cost_eur"] = [
        float(
            (
                np.clip(
                    np.minimum(row.end_h, hour_index + 1)
                    - np.maximum(row.start_h, hour_index),
                    0,
                    1,
                )
                * inst.price_buy_eur_mwh
            ).sum()
            * row.power_mw
        )
        for row in jobs.itertuples()
    ]

    commitment = (
        np.array([round(_value(model.u[hour])) for hour in range(hours)], dtype=float)
        if inst.mt is not None else np.zeros(hours)
    )
    da_buy = np.array([max(0.0, _value(model.Pbuy[hour])) for hour in range(hours)])
    da_sell = np.array([max(0.0, _value(model.Psell[hour])) for hour in range(hours)])
    frames: list[pd.DataFrame] = []
    issues: list[str] = []
    seen_issues: set[str] = set()

    for scenario in range(scenario_count):
        scenario_instance = replace(
            inst, base_load_mw=inst.base_load_mw + scenarios.load_dev_mw[scenario]
        )
        mt_power = (
            np.array([
                max(0.0, _value(model.Pmt[scenario, hour])) for hour in range(hours)
            ]) * commitment
            if inst.mt is not None else np.zeros(hours)
        )
        if inst.bess is not None:
            charge = np.array([
                max(0.0, _value(model.Pch[scenario, hour])) for hour in range(hours)
            ])
            discharge = np.array([
                max(0.0, _value(model.Pdis[scenario, hour])) for hour in range(hours)
            ])
            charge[charge < 1e-9] = 0.0
            discharge[discharge < 1e-9] = 0.0
        else:
            charge, discharge = np.zeros(hours), np.zeros(hours)
        id_buy = np.array([
            max(0.0, _value(model.Ibuy[scenario, hour])) for hour in range(hours)
        ])
        id_sell = np.array([
            max(0.0, _value(model.Isell[scenario, hour])) for hour in range(hours)
        ])

        hourly = build_hourly(
            scenario_instance, jobs, mt_power, commitment, charge, discharge
        )
        hourly.insert(0, "scenario", scenario)
        hourly.insert(1, "prob", scenarios.prob[scenario])
        hourly["price_id_buy_eur_mwh"] = scenarios.id_buy_eur_mwh[scenario]
        hourly["price_id_sell_eur_mwh"] = scenarios.id_sell_eur_mwh[scenario]
        hourly["p_da_buy_mw"], hourly["p_da_sell_mw"] = da_buy, da_sell
        hourly["p_id_buy_mw"], hourly["p_id_sell_mw"] = id_buy, id_sell
        hourly["da_cost_eur"] = (
            inst.price_buy_eur_mwh * da_buy - inst.price_sell_eur_mwh * da_sell
        )
        hourly["id_cost_eur"] = (
            scenarios.id_buy_eur_mwh[scenario] * id_buy
            - scenarios.id_sell_eur_mwh[scenario] * id_sell
        )
        hourly["grid_buy_cost_eur"] = (
            inst.price_buy_eur_mwh * da_buy + scenarios.id_buy_eur_mwh[scenario] * id_buy
        )
        hourly["grid_sell_revenue_eur"] = (
            inst.price_sell_eur_mwh * da_sell
            + scenarios.id_sell_eur_mwh[scenario] * id_sell
        )
        hourly["grid_cost_eur"] = hourly.da_cost_eur + hourly.id_cost_eur
        hourly["cost_eur"] = (
            hourly.grid_cost_eur
            + hourly.mt_fuel_cost_eur
            + hourly.mt_startstop_cost_eur
            + hourly.bess_degradation_cost_eur
        )
        hourly["soc_model_mwh"] = (
            [_value(model.SoC[scenario, hour]) for hour in range(hours)]
            if inst.bess is not None else np.zeros(hours)
        )
        frames.append(hourly)

        report = verify(scenario_instance, cfg, jobs, hourly, None)
        for message in report["issues"]:
            if message not in seen_issues:
                seen_issues.add(message)
                issues.append(f"[scenario {scenario}] {message}")
        net_position = da_buy - da_sell + id_buy - id_sell
        if np.abs(net_position - hourly.p_net_grid_mw.to_numpy()).max() > 1e-3:
            issues.append(f"[scenario {scenario}] modeled and recomputed net exchange differ")
        if da_buy.max() > inst.grid_limit_mw + 1e-4:
            issues.append("DA purchase exceeds its substation limit")
        if da_sell.max() > inst.grid_sell_limit_mw + 1e-4:
            issues.append("DA sale exceeds its substation limit")
        if id_buy.max() > market.cap_buy_mw + 1e-4:
            issues.append(f"[scenario {scenario}] intraday purchase cap exceeded")
        if id_sell.max() > market.cap_sell_mw + 1e-4:
            issues.append(f"[scenario {scenario}] intraday sale cap exceeded")
        if inst.bess is not None and np.abs(
            hourly.soc_model_mwh.to_numpy() - hourly.soc_mwh.to_numpy()
        ).max() > 1e-3:
            issues.append(f"[scenario {scenario}] modeled and recomputed BESS SoC differ")

    objective = float(pyo.value(model.obj))
    scenario_costs = np.array([frame.cost_eur.sum() for frame in frames], dtype=float)
    expected_cost = float(scenarios.prob @ scenario_costs)
    if abs(expected_cost - objective) > max(1e-4, 1e-6 * abs(objective)):
        issues.append(
            f"expected cost mismatch: recomputed {expected_cost:.4f} vs solver {objective:.4f}"
        )
    verification = {
        "passed": not issues,
        "issues": issues,
        "recomputed_cost_eur": expected_cost,
    }
    if issues:
        message = "; ".join(issues)
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {message}")
        log.error("verification failed: %s", message)

    hourly = pd.concat(frames, ignore_index=True)
    probabilities = scenarios.prob

    def expectation(column: str) -> np.ndarray:
        return sum(
            float(probabilities[index]) * frames[index][column].to_numpy()
            for index in range(scenario_count)
        )

    positions = pd.DataFrame({
        "hour": np.arange(hours),
        "price_da_buy_eur_mwh": inst.price_buy_eur_mwh,
        "price_da_sell_eur_mwh": inst.price_sell_eur_mwh,
        "p_da_buy_mw": da_buy,
        "p_da_sell_mw": da_sell,
        "mt_on": commitment.astype(int),
        "mt_startup": frames[0].mt_startup.to_numpy(),
        "mt_shutdown": frames[0].mt_shutdown.to_numpy(),
        "exp_p_id_buy_mw": expectation("p_id_buy_mw"),
        "exp_p_id_sell_mw": expectation("p_id_sell_mw"),
        "exp_p_mt_mw": expectation("p_mt_mw"),
        "exp_bess_ch_mw": expectation("p_bess_ch_mw"),
        "exp_bess_dis_mw": expectation("p_bess_dis_mw"),
        "exp_soc_mwh": expectation("soc_mwh"),
        "exp_price_id_buy_eur_mwh": expectation("price_id_buy_eur_mwh"),
    })
    scenario_table = pd.DataFrame([
        {
            "scenario": index,
            "prob": float(probabilities[index]),
            "cost_eur": float(scenario_costs[index]),
            "da_cost_eur": float(frames[index].da_cost_eur.sum()),
            "id_cost_eur": float(frames[index].id_cost_eur.sum()),
            "mt_cost_eur": float((
                frames[index].mt_fuel_cost_eur + frames[index].mt_startstop_cost_eur
            ).sum()),
            "bess_deg_eur": float(frames[index].bess_degradation_cost_eur.sum()),
            "id_buy_mwh": float(frames[index].p_id_buy_mw.sum()),
            "id_sell_mwh": float(frames[index].p_id_sell_mw.sum()),
            "mt_mwh": float(frames[index].p_mt_mw.sum()),
            "bess_throughput_mwh": float((
                frames[index].p_bess_ch_mw + frames[index].p_bess_dis_mw
            ).sum()),
        }
        for index in range(scenario_count)
    ])
    mean_cost = float(probabilities @ scenario_costs)
    kpis = {
        "n_scenarios": float(scenario_count),
        "n_batches": float(len(jobs)),
        "units_produced": float(jobs.units_out.sum()),
        "da_cost_eur": float(frames[0].da_cost_eur.sum()),
        "da_buy_mwh": float(da_buy.sum()),
        "da_sell_mwh": float(da_sell.sum()),
        "exp_id_cost_eur": float(probabilities @ scenario_table.id_cost_eur.to_numpy()),
        "exp_id_buy_mwh": float(probabilities @ scenario_table.id_buy_mwh.to_numpy()),
        "exp_id_sell_mwh": float(probabilities @ scenario_table.id_sell_mwh.to_numpy()),
        "exp_mt_cost_eur": float(probabilities @ scenario_table.mt_cost_eur.to_numpy()),
        "exp_bess_deg_eur": float(probabilities @ scenario_table.bess_deg_eur.to_numpy()),
        "exp_mt_energy_mwh": float(probabilities @ scenario_table.mt_mwh.to_numpy()),
        "exp_bess_throughput_mwh": float(
            probabilities @ scenario_table.bess_throughput_mwh.to_numpy()
        ),
        "mt_on_hours": float(commitment.sum()),
        "mt_starts": float(frames[0].mt_startup.sum()),
        "cost_mean_eur": mean_cost,
        "cost_std_eur": float(math.sqrt(max(
            0.0, probabilities @ ((scenario_costs - mean_cost) ** 2)
        ))),
        "cost_min_eur": float(scenario_costs.min()),
        "cost_max_eur": float(scenario_costs.max()),
    }
    return IntradayResult(
        solve_info["status"],
        objective,
        solve_info["gap"],
        solve_info["time"],
        solve_info["stats"],
        jobs,
        positions,
        scenario_table,
        hourly,
        kpis,
        verification,
        scenarios,
    )