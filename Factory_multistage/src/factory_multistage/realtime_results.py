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
from .realtime_scenarios import BalancingMarket, RealTimeSet
from .scenarios import IntradayMarket, ScenarioSet


@dataclass
class RealTimeResult:
    status: str
    objective_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    jobs: pd.DataFrame
    da_position: pd.DataFrame            # Stage-1 decisions + expectations of Stage 2 / Stage 3, per hour
    scenarios: pd.DataFrame              # one row per ID scenario (conditional expectation over RT)
    joint: pd.DataFrame                  # one row per (ID scenario, RT scenario)
    hourly: pd.DataFrame                 # long table: scenario x rt x hour
    kpis: Dict[str, float]
    verification: Dict[str, object]
    scenario_set: ScenarioSet
    rt_set: RealTimeSet
    vss_rt: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        k = self.kpis
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        lines = [
            f"status={self.status}  expected cost={self.objective_eur:,.2f} EUR  gap={gap}  time={self.solve_time_s:.1f}s  "
            f"tree={int(k['n_scenarios'])} ID x {int(k['n_rt'])} RT = {int(k['n_joint'])} joint scenarios",
            f"Stage 1: DA net cost={k['da_cost_eur']:,.2f} EUR (buy {k['da_buy_mwh']:.1f} MWh, sell {k['da_sell_mwh']:.1f} MWh)  "
            f"MT on {int(k['mt_on_hours'])} h, {int(k['mt_starts'])} start(s)  batches={int(k['n_batches'])}  "
            f"units={k['units_produced']:.0f}",
            f"Stage 2 (expected): ID net cost={k['exp_id_cost_eur']:,.2f} EUR (buy {k['exp_id_buy_mwh']:.1f} MWh, "
            f"sell {k['exp_id_sell_mwh']:.1f} MWh)  MT cost={k['exp_mt_cost_eur']:,.2f}  "
            f"BESS degradation={k['exp_bess_deg_eur']:,.2f} EUR",
            f"Stage 3 (expected): imbalance cost={k['exp_bal_cost_eur']:,.2f} EUR ({k['bal_cost_share_pct']:.2f}% of total)  "
            f"deficit {k['exp_deficit_mwh']:.1f} MWh, surplus {k['exp_surplus_mwh']:.1f} MWh  "
            f"deliberate (planned) imbalance {k['planned_imbalance_mwh']:.2f} MWh",
            f"joint cost: mean={k['cost_mean_eur']:,.2f}  std={k['cost_std_eur']:,.2f}  "
            f"CVaR95={k['cost_cvar95_eur']:,.2f}  min={k['cost_min_eur']:,.2f}  max={k['cost_max_eur']:,.2f} EUR",
            f"verification={'PASS' if self.verification['passed'] else 'FAIL'}"]
        if self.vss_rt:
            v = self.vss_rt
            lines.append(f"RT-blind Stage-1 plan (ID model) evaluated in the RT tree: EEV_RT={v['eev_rt_eur']:,.2f}  "
                         f"RP={v['rp_eur']:,.2f}  value of modelling RT risk={v['vss_rt_eur']:,.2f} EUR ({v['vss_rt_pct']:.3f}%)")
        return "\n".join(lines)

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(out / "jobs.csv", index=False)
        self.da_position.to_csv(out / "da_position.csv", index=False)
        self.scenarios.to_csv(out / "scenarios.csv", index=False)
        self.joint.to_csv(out / "joint_scenarios.csv", index=False)
        self.hourly.to_csv(out / "hourly_rt.csv", index=False)
        self.scenario_set.save(out / "scenario_set.json")
        self.rt_set.save(out / "rt_set.json")
        meta = dict(version=__version__, status=self.status, expected_cost_eur=self.objective_eur,
                    mip_gap=self.mip_gap, solve_time_s=self.solve_time_s, model_stats=self.model_stats,
                    kpis=self.kpis, verification=self.verification, vss_rt=self.vss_rt)
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
        return out


def _cvar(cost: np.ndarray, prob: np.ndarray, alpha: float = 0.95) -> float:
    """Discrete CVaR: probability-weighted mean of the worst (1 - alpha) tail."""
    order = np.argsort(-cost)
    c, p = cost[order], prob[order]
    tail = 1.0 - alpha
    take = np.minimum(p, np.maximum(0.0, tail - (np.cumsum(p) - p)))
    return float((c * take).sum() / take.sum())


def _apply_rt_load(h: pd.DataFrame, eta: np.ndarray) -> None:
    """Scale the Stage-2 load columns by (1 + eta) and recompute the physical grid exchange in place."""
    f = 1.0 + eta
    for col in ("base_load_mw", "batch_load_mw", "total_load_mw"):
        h[col] = h[col].values * f
    net = h.total_load_mw - h.p_mt_mw - h.p_bess_dis_mw + h.p_bess_ch_mw
    h["p_net_grid_mw"] = net
    h["p_buy_mw"] = np.maximum(net, 0.0)
    h["p_sell_mw"] = np.maximum(-net, 0.0)


def extract_realtime(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, rt: RealTimeSet, bal: BalancingMarket,
                     cfg: SchedulerConfig, model: pyo.ConcreteModel, cands: List[da.Candidate],
                     info: Dict[str, object]) -> RealTimeResult:
    T, S, W = inst.horizon_h, scen.n, rt.n_w
    pi, pw = scen.prob, rt.prob
    jobs = da.jobs_from_starts(inst, cands, model)
    hrs = np.arange(T, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_buy_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]
    u = np.array([round(_val(model.u[t])) for t in range(T)], dtype=float) if inst.mt is not None else np.zeros(T)
    pda_b = np.array([max(0.0, _val(model.Pbuy[t])) for t in range(T)])
    pda_s = np.array([max(0.0, _val(model.Psell[t])) for t in range(T)])
    lam_p, lam_m = rt.lam_plus(inst), rt.lam_minus(inst)

    frames: List[List[pd.DataFrame]] = []
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
        market = pda_b - pda_s + ib - isl
        if market.max() > inst.grid_limit_mw + 1e-4 or -market.min() > inst.grid_sell_limit_mw + 1e-4:
            issues.append(f"[scenario {k}] market position exceeds the substation limits")
        if ib.max() > mkt.cap_buy_mw + 1e-4 or isl.max() > mkt.cap_sell_mw + 1e-4:
            issues.append(f"[scenario {k}] intraday cap exceeded")
        row: List[pd.DataFrame] = []
        for w in range(W):
            h = da._build_hourly(inst_k, cfg, jobs, p_mt, u, ch, dis)            # Stage-2 view of the factory
            _apply_rt_load(h, rt.load_rel_dev[k, w])
            delta = h.p_net_grid_mw.values - market                              # v2 5.1: physical - market
            d_plus, d_minus = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
            bal_cost = lam_p[k, w] * d_plus - lam_m[k, w] * d_minus
            h.insert(0, "scenario", k)
            h.insert(1, "rt", w)
            h.insert(2, "prob_joint", float(pi[k] * pw[k, w]))
            h["rt_load_rel_dev"] = rt.load_rel_dev[k, w]
            h["price_id_buy_eur_mwh"], h["price_id_sell_eur_mwh"] = scen.id_buy_eur_mwh[k], scen.id_sell_eur_mwh[k]
            h["r_plus"], h["r_minus"] = rt.r_plus[k, w], rt.r_minus[k, w]
            h["price_bal_deficit_eur_mwh"], h["price_bal_surplus_eur_mwh"] = lam_p[k, w], lam_m[k, w]
            h["p_da_buy_mw"], h["p_da_sell_mw"], h["p_id_buy_mw"], h["p_id_sell_mw"] = pda_b, pda_s, ib, isl
            h["p_market_net_mw"] = market
            h["imb_net_mw"], h["imb_deficit_mw"], h["imb_surplus_mw"] = delta, d_plus, d_minus
            h["da_cost_eur"] = inst.price_buy_eur_mwh * pda_b - inst.price_sell_eur_mwh * pda_s
            h["id_cost_eur"] = scen.id_buy_eur_mwh[k] * ib - scen.id_sell_eur_mwh[k] * isl
            h["bal_cost_eur"] = bal_cost
            h["grid_buy_cost_eur"] = inst.price_buy_eur_mwh * pda_b + scen.id_buy_eur_mwh[k] * ib + lam_p[k, w] * d_plus
            h["grid_sell_revenue_eur"] = inst.price_sell_eur_mwh * pda_s + scen.id_sell_eur_mwh[k] * isl + lam_m[k, w] * d_minus
            h["grid_cost_eur"] = h.da_cost_eur + h.id_cost_eur + h.bal_cost_eur
            h["cost_eur"] = h.grid_cost_eur + h.mt_fuel_cost_eur + h.mt_startstop_cost_eur + h.bess_degradation_cost_eur
            h["soc_model_mwh"] = [_val(model.SoC[k, t]) for t in range(T)] if inst.bess is not None else np.zeros(T)
            row.append(h)

            # independent re-check of the physical scenario (factory, MT, BESS, PHYSICAL substation limits)
            for msg in da.verify(inst_k, cfg, jobs, h, None)["issues"]:
                tag = f"[scenario {k}/{w}] {msg}"
                if msg not in seen:
                    seen.add(msg)
                    issues.append(tag)
            dp_m = np.array([_val(model.Dp[k, w, t]) for t in range(T)])
            dm_m = np.array([_val(model.Dm[k, w, t]) for t in range(T)])
            if np.abs((dp_m - dm_m) - delta).max() > 1e-3:
                issues.append(f"[scenario {k}/{w}] model imbalance differs from the physical-minus-market recomputation")
            if (not bal.enforce_exclusive) and np.any(np.minimum(dp_m, dm_m) > 1e-6) and np.any(lam_p[k, w] - lam_m[k, w] > 1e-9):
                issues.append(f"[scenario {k}/{w}] deficit and surplus bought/sold in the same hour")
            if inst.bess is not None and np.abs(h.soc_model_mwh.values - h.soc_mwh.values).max() > 1e-3:
                issues.append(f"[scenario {k}/{w}] model SoC differs from the independent SoC recomputation")
        frames.append(row)
    if pda_b.max() > inst.grid_limit_mw + 1e-4 or pda_s.max() > inst.grid_sell_limit_mw + 1e-4:
        issues.append("DA position exceeds the substation limits")

    cost = np.array([[frames[k][w].cost_eur.sum() for w in range(W)] for k in range(S)])       # [S, W]
    joint_p = pi[:, None] * pw
    obj = float(pyo.value(model.obj))
    expected = float((joint_p * cost).sum())
    if abs(expected - obj) > max(1e-4, 1e-6 * abs(obj)):
        issues.append(f"expected cost mismatch: recomputed {expected:.4f} vs solver {obj:.4f}")

    # planned (deliberate) imbalance: expectation over RT of the imbalance, per ID scenario and hour
    planned = np.stack([pw[k] @ np.stack([frames[k][w].imb_net_mw.values for w in range(W)]) for k in range(S)])   # [S, T]
    if bal.mode == "passive" and np.abs(planned).max() > 1e-3:        # D = (eta - E[eta]) * load, so E_w[D] must vanish
        issues.append("passive mode: expected imbalance is not zero")
    ver = dict(passed=not issues, issues=issues, recomputed_cost_eur=expected)
    if issues:
        msg = "; ".join(issues)
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)

    flat = [frames[k][w] for k in range(S) for w in range(W)]
    hourly = pd.concat(flat, ignore_index=True)
    Ej = lambda col: sum(float(joint_p[k, w]) * frames[k][w][col].values for k in range(S) for w in range(W))
    pos = pd.DataFrame(dict(hour=np.arange(T), price_da_buy_eur_mwh=inst.price_buy_eur_mwh,
                            price_da_sell_eur_mwh=inst.price_sell_eur_mwh, p_da_buy_mw=pda_b, p_da_sell_mw=pda_s,
                            mt_on=u.astype(int), mt_startup=frames[0][0].mt_startup.values,
                            mt_shutdown=frames[0][0].mt_shutdown.values,
                            exp_p_id_buy_mw=Ej("p_id_buy_mw"), exp_p_id_sell_mw=Ej("p_id_sell_mw"),
                            exp_p_mt_mw=Ej("p_mt_mw"), exp_bess_ch_mw=Ej("p_bess_ch_mw"),
                            exp_bess_dis_mw=Ej("p_bess_dis_mw"), exp_soc_mwh=Ej("soc_mwh"),
                            exp_imb_deficit_mw=Ej("imb_deficit_mw"), exp_imb_surplus_mw=Ej("imb_surplus_mw"),
                            exp_planned_imbalance_mw=pi @ planned,
                            exp_price_id_buy_eur_mwh=Ej("price_id_buy_eur_mwh")))
    col = lambda name: np.array([[frames[k][w][name].sum() for w in range(W)] for k in range(S)])
    sc = pd.DataFrame([dict(scenario=k, prob=float(pi[k]), cost_eur=float(pw[k] @ cost[k]),
                            cost_worst_rt_eur=float(cost[k].max()),
                            id_cost_eur=float(pw[k] @ col("id_cost_eur")[k]), bal_cost_eur=float(pw[k] @ col("bal_cost_eur")[k]),
                            deficit_mwh=float(pw[k] @ col("imb_deficit_mw")[k]), surplus_mwh=float(pw[k] @ col("imb_surplus_mw")[k]),
                            planned_imbalance_mwh=float(np.abs(planned[k]).sum()),
                            id_buy_mwh=float(frames[k][0].p_id_buy_mw.sum()), id_sell_mwh=float(frames[k][0].p_id_sell_mw.sum()),
                            mt_mwh=float(frames[k][0].p_mt_mw.sum()),
                            bess_throughput_mwh=float((frames[k][0].p_bess_ch_mw + frames[k][0].p_bess_dis_mw).sum()))
                       for k in range(S)])
    jt = pd.DataFrame([dict(scenario=k, rt=w, prob=float(joint_p[k, w]), cost_eur=float(cost[k, w]),
                            da_cost_eur=float(frames[k][w].da_cost_eur.sum()), id_cost_eur=float(frames[k][w].id_cost_eur.sum()),
                            bal_cost_eur=float(frames[k][w].bal_cost_eur.sum()),
                            deficit_mwh=float(frames[k][w].imb_deficit_mw.sum()), surplus_mwh=float(frames[k][w].imb_surplus_mw.sum()))
                       for k in range(S) for w in range(W)])
    flat_cost, flat_p = cost.ravel(), joint_p.ravel()
    mean = float(flat_p @ flat_cost)
    exp_bal = float(flat_p @ col("bal_cost_eur").ravel())
    kpis = dict(
        n_scenarios=float(S), n_rt=float(W), n_joint=float(S * W), n_batches=float(len(jobs)),
        units_produced=float(jobs.units_out.sum()),
        da_cost_eur=float(frames[0][0].da_cost_eur.sum()), da_buy_mwh=float(pda_b.sum()), da_sell_mwh=float(pda_s.sum()),
        exp_id_cost_eur=float(pi @ sc.id_cost_eur.values), exp_id_buy_mwh=float(pi @ sc.id_buy_mwh.values),
        exp_id_sell_mwh=float(pi @ sc.id_sell_mwh.values),
        exp_mt_cost_eur=float(flat_p @ np.array([[(frames[k][w].mt_fuel_cost_eur + frames[k][w].mt_startstop_cost_eur).sum()
                                                  for w in range(W)] for k in range(S)]).ravel()),
        exp_bess_deg_eur=float(flat_p @ col("bess_degradation_cost_eur").ravel()),
        exp_bess_throughput_mwh=float(pi @ sc.bess_throughput_mwh.values),
        exp_bal_cost_eur=exp_bal, bal_cost_share_pct=100 * exp_bal / abs(obj) if obj else float("nan"),
        exp_deficit_mwh=float(flat_p @ col("imb_deficit_mw").ravel()), exp_surplus_mwh=float(flat_p @ col("imb_surplus_mw").ravel()),
        exp_abs_imbalance_mwh=float(flat_p @ (col("imb_deficit_mw") + col("imb_surplus_mw")).ravel()),
        planned_imbalance_mwh=float(pi @ np.abs(planned).sum(axis=1)),
        mt_on_hours=float(u.sum()), mt_starts=float(frames[0][0].mt_startup.sum()),
        cost_mean_eur=mean, cost_std_eur=float(math.sqrt(max(0.0, flat_p @ (flat_cost - mean) ** 2))),
        cost_cvar95_eur=_cvar(flat_cost, flat_p), cost_min_eur=float(flat_cost.min()), cost_max_eur=float(flat_cost.max()))
    return RealTimeResult(info["status"], obj, info["gap"], info["time"], info["stats"], jobs, pos, sc, jt, hourly,
                          kpis, ver, scen, rt)
