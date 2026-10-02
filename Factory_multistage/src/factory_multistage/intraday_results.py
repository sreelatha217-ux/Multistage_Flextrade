from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from . import day_ahead as da
from ._internal import __version__, log
from .day_ahead import Instance, SchedulerConfig
from .exceptions import VerificationError
from .intraday_model import _val
from .scenarios import IntradayMarket, ScenarioSet


@dataclass
class IntradayResult:
    status: str
    objective_eur: float                 # expected total cost
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    jobs: pd.DataFrame
    da_position: pd.DataFrame            # Stage-1 decisions + scenario expectations, per hour
    scenarios: pd.DataFrame              # one row per scenario
    hourly: pd.DataFrame                 # long table: scenario x hour
    kpis: Dict[str, float]
    verification: Dict[str, object]
    scenario_set: ScenarioSet
    vss: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        k = self.kpis
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        lines = [
            f"status={self.status}  expected cost={self.objective_eur:,.2f} EUR  gap={gap}  time={self.solve_time_s:.1f}s  "
            f"scenarios={int(k['n_scenarios'])}",
            f"Stage 1: DA net cost={k['da_cost_eur']:,.2f} EUR (buy {k['da_buy_mwh']:.1f} MWh, sell {k['da_sell_mwh']:.1f} MWh)  "
            f"MT on {int(k['mt_on_hours'])} h, {int(k['mt_starts'])} start(s)  batches={int(k['n_batches'])}  "
            f"units={k['units_produced']:.0f}",
            f"Stage 2 (expected): ID net cost={k['exp_id_cost_eur']:,.2f} EUR (buy {k['exp_id_buy_mwh']:.1f} MWh, "
            f"sell {k['exp_id_sell_mwh']:.1f} MWh)  MT cost={k['exp_mt_cost_eur']:,.2f}  "
            f"BESS degradation={k['exp_bess_deg_eur']:,.2f} EUR  BESS throughput={k['exp_bess_throughput_mwh']:.1f} MWh",
            f"scenario cost: mean={k['cost_mean_eur']:,.2f}  std={k['cost_std_eur']:,.2f}  "
            f"min={k['cost_min_eur']:,.2f}  max={k['cost_max_eur']:,.2f} EUR",
            f"verification={'PASS' if self.verification['passed'] else 'FAIL'}"]
        if self.vss:
            v = self.vss
            lines.append(f"VSS: EV-plan cost in scenarios (EEV)={v['eev_eur']:,.2f}  RP={v['rp_eur']:,.2f}  "
                         f"VSS={v['vss_eur']:,.2f} EUR ({v['vss_pct']:.3f}%)")
        return "\n".join(lines)

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(out / "jobs.csv", index=False)
        self.da_position.to_csv(out / "da_position.csv", index=False)
        self.scenarios.to_csv(out / "scenarios.csv", index=False)
        self.hourly.to_csv(out / "hourly_scenarios.csv", index=False)
        self.scenario_set.save(out / "scenario_set.json")
        meta = dict(version=__version__, status=self.status, expected_cost_eur=self.objective_eur,
                    mip_gap=self.mip_gap, solve_time_s=self.solve_time_s, model_stats=self.model_stats,
                    kpis=self.kpis, verification=self.verification, vss=self.vss)
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
        return out


def extract_intraday(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, cfg: SchedulerConfig,
                     model: pyo.ConcreteModel, cands: List[da.Candidate], info: Dict[str, object]) -> IntradayResult:
    T, S = inst.horizon_h, scen.n
    jobs = da.jobs_from_starts(inst, cands, model)
    hrs = np.arange(T, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_buy_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]
    u = np.array([round(_val(model.u[t])) for t in range(T)], dtype=float) if inst.mt is not None else np.zeros(T)
    pda_b = np.array([max(0.0, _val(model.Pbuy[t])) for t in range(T)])
    pda_s = np.array([max(0.0, _val(model.Psell[t])) for t in range(T)])

    frames: List[pd.DataFrame] = []
    issues: List[str] = []
    seen = set()
    for k in range(S):
        inst_k = replace(inst, base_load_mw=inst.base_load_mw + scen.load_dev_mw[k])
        p_mt = (np.array([max(0.0, _val(model.Pmt[k, t])) for t in range(T)]) * u) if inst.mt is not None else np.zeros(T)
        if inst.bess is not None:
            ch = np.array([max(0.0, _val(model.Pch[k, t])) for t in range(T)])
            dis = np.array([max(0.0, _val(model.Pdis[k, t])) for t in range(T)])
            ch[ch < 1e-9], dis[dis < 1e-9] = 0.0, 0.0
        else:
            ch, dis = np.zeros(T), np.zeros(T)
        ib = np.array([max(0.0, _val(model.Ibuy[k, t])) for t in range(T)])
        isl = np.array([max(0.0, _val(model.Isell[k, t])) for t in range(T)])
        h = da._build_hourly(inst_k, cfg, jobs, p_mt, u, ch, dis)
        h.insert(0, "scenario", k)
        h.insert(1, "prob", scen.prob[k])
        h["price_id_buy_eur_mwh"], h["price_id_sell_eur_mwh"] = scen.id_buy_eur_mwh[k], scen.id_sell_eur_mwh[k]
        h["p_da_buy_mw"], h["p_da_sell_mw"], h["p_id_buy_mw"], h["p_id_sell_mw"] = pda_b, pda_s, ib, isl
        h["da_cost_eur"] = inst.price_buy_eur_mwh * pda_b - inst.price_sell_eur_mwh * pda_s
        h["id_cost_eur"] = scen.id_buy_eur_mwh[k] * ib - scen.id_sell_eur_mwh[k] * isl
        h["grid_buy_cost_eur"] = inst.price_buy_eur_mwh * pda_b + scen.id_buy_eur_mwh[k] * ib
        h["grid_sell_revenue_eur"] = inst.price_sell_eur_mwh * pda_s + scen.id_sell_eur_mwh[k] * isl
        h["grid_cost_eur"] = h.da_cost_eur + h.id_cost_eur
        h["cost_eur"] = h.grid_cost_eur + h.mt_fuel_cost_eur + h.mt_startstop_cost_eur + h.bess_degradation_cost_eur
        h["soc_model_mwh"] = [_val(model.SoC[k, t]) for t in range(T)] if inst.bess is not None else np.zeros(T)
        frames.append(h)

        # independent re-check of every constraint of this scenario (factory, MT, BESS, limits)
        for msg in da.verify(inst_k, cfg, jobs, h, None)["issues"]:
            if msg not in seen:
                seen.add(msg)
                issues.append(f"[scenario {k}] {msg}")
        model_net = pda_b - pda_s + ib - isl
        if np.abs(model_net - h.p_net_grid_mw.values).max() > 1e-3:
            issues.append(f"[scenario {k}] model net grid exchange differs from the load-balance recomputation")
        if pda_b.max() > inst.grid_limit_mw + 1e-4 or pda_s.max() > inst.grid_sell_limit_mw + 1e-4:
            issues.append("DA position exceeds the substation limits")
        if ib.max() > mkt.cap_buy_mw + 1e-4 or isl.max() > mkt.cap_sell_mw + 1e-4:
            issues.append(f"[scenario {k}] intraday cap exceeded")
        if inst.bess is not None and np.abs(h.soc_model_mwh.values - h.soc_mwh.values).max() > 1e-3:
            issues.append(f"[scenario {k}] model SoC differs from the independent SoC recomputation")

    obj = float(pyo.value(model.obj))
    cost_k = np.array([f.cost_eur.sum() for f in frames])
    expected = float(scen.prob @ cost_k)
    if abs(expected - obj) > max(1e-4, 1e-6 * abs(obj)):
        issues.append(f"expected cost mismatch: recomputed {expected:.4f} vs solver {obj:.4f}")
    ver = dict(passed=not issues, issues=issues, recomputed_cost_eur=expected)
    if issues:
        msg = "; ".join(issues)
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)

    hourly = pd.concat(frames, ignore_index=True)
    pi = scen.prob
    E = lambda col: sum(float(pi[k]) * frames[k][col].values for k in range(S))       # expectation over scenarios
    pos = pd.DataFrame(dict(hour=np.arange(T), price_da_buy_eur_mwh=inst.price_buy_eur_mwh,
                            price_da_sell_eur_mwh=inst.price_sell_eur_mwh, p_da_buy_mw=pda_b, p_da_sell_mw=pda_s,
                            mt_on=u.astype(int), mt_startup=frames[0].mt_startup.values,
                            mt_shutdown=frames[0].mt_shutdown.values,
                            exp_p_id_buy_mw=E("p_id_buy_mw"), exp_p_id_sell_mw=E("p_id_sell_mw"),
                            exp_p_mt_mw=E("p_mt_mw"), exp_bess_ch_mw=E("p_bess_ch_mw"),
                            exp_bess_dis_mw=E("p_bess_dis_mw"), exp_soc_mwh=E("soc_mwh"),
                            exp_price_id_buy_eur_mwh=E("price_id_buy_eur_mwh")))
    sc = pd.DataFrame([dict(scenario=k, prob=float(pi[k]), cost_eur=float(cost_k[k]),
                            da_cost_eur=float(frames[k].da_cost_eur.sum()), id_cost_eur=float(frames[k].id_cost_eur.sum()),
                            mt_cost_eur=float((frames[k].mt_fuel_cost_eur + frames[k].mt_startstop_cost_eur).sum()),
                            bess_deg_eur=float(frames[k].bess_degradation_cost_eur.sum()),
                            id_buy_mwh=float(frames[k].p_id_buy_mw.sum()), id_sell_mwh=float(frames[k].p_id_sell_mw.sum()),
                            mt_mwh=float(frames[k].p_mt_mw.sum()),
                            bess_throughput_mwh=float((frames[k].p_bess_ch_mw + frames[k].p_bess_dis_mw).sum()))
                       for k in range(S)])
    mean = float(pi @ cost_k)
    kpis = dict(
        n_scenarios=float(S), n_batches=float(len(jobs)), units_produced=float(jobs.units_out.sum()),
        da_cost_eur=float(frames[0].da_cost_eur.sum()), da_buy_mwh=float(pda_b.sum()), da_sell_mwh=float(pda_s.sum()),
        exp_id_cost_eur=float(pi @ sc.id_cost_eur.values), exp_id_buy_mwh=float(pi @ sc.id_buy_mwh.values),
        exp_id_sell_mwh=float(pi @ sc.id_sell_mwh.values), exp_mt_cost_eur=float(pi @ sc.mt_cost_eur.values),
        exp_bess_deg_eur=float(pi @ sc.bess_deg_eur.values), exp_mt_energy_mwh=float(pi @ sc.mt_mwh.values),
        exp_bess_throughput_mwh=float(pi @ sc.bess_throughput_mwh.values),
        mt_on_hours=float(u.sum()), mt_starts=float(frames[0].mt_startup.sum()),
        cost_mean_eur=mean, cost_std_eur=float(math.sqrt(max(0.0, pi @ (cost_k - mean) ** 2))),
        cost_min_eur=float(cost_k.min()), cost_max_eur=float(cost_k.max()))
    return IntradayResult(info["status"], obj, info["gap"], info["time"], info["stats"], jobs, pos, sc, hourly,
                          kpis, ver, scen)
