#!/usr/bin/env python3
"""
factory_mt_id_scheduler.py
==========================

Two-stage stochastic Day-Ahead (DA) + Intraday (ID) scheduling of an industrial batch plant with an
on-site Microturbine (MT) and a Battery Energy Storage System (BESS).  Extension of
factory_mt_da_scheduler.py (imported, not copied) following "factory_multistage_intraday_formulation.md".
All money is in EUR.

    Stage 1 (here-and-now, one decision for all scenarios)
        P_DA,buy[t], P_DA,sell[t]            day-ahead position
        u[t], x[t], y[t]                      MT unit commitment
        s[m,p,k]                              batch start times (master production schedule)
    Stage 2 (wait-and-see, one decision set per intraday price scenario k)
        P_ID,buy[k,t], P_ID,sell[k,t]         intraday re-trading
        P_MT[k,t], P_MT,b[k,t]                MT re-dispatch inside the Stage-1 commitment
        P_ch[k,t], P_dis[k,t], SoC[k,t]       BESS recourse dispatch

    min   sum_t [ lam_DAbuy*P_DAbuy - lam_DAsell*P_DAsell + C0*Pmin*u + SUC*x + SDC*y ]
        + sum_k pi_k * sum_t [ lam_IDbuy[k,t]*P_IDbuy - lam_IDsell[k,t]*P_IDsell
                               + sum_b C_b*P_MT,b[k,t] + C_TP*(w*P_ch + P_dis) ]

    (P_DAbuy - P_DAsell) + (P_IDbuy - P_IDsell) + P_MT + P_dis - P_ch = l_base[t] + dl[k,t] + L_batch[t]
    -Q_sell <= (P_DAbuy - P_DAsell) + (P_IDbuy - P_IDsell) <= Q_buy           physical substation limit
    0 <= P_ID,buy <= Cap_buy ; 0 <= P_ID,sell <= Cap_sell                      intraday market depth

Review of the formulation file - inconsistencies found and how this program resolves them
-----------------------------------------------------------------------------------------
 1. Scenario-indexed factory variables are redundant.  s[m,p,k] is Stage 1, so L_batch, nprod and the
    inventory are identical in every scenario; the file still indexes them by s although its Stage-2 box
    promises "flexible batch load shifting", which no equation implements.  Here they are single
    deterministic expressions (|S| times fewer variables).  Real load shifting would need scenario-indexed
    start binaries (a much larger model) and is not included.
 2. Section 8.2 (bid-curve monotonicity / non-anticipativity) contradicts Section 2.  DA prices
    lam_DA are deterministic and P_DA has no scenario index, so non-anticipativity holds by construction
    and monotonicity has nothing to act on.  Both would only make sense with scenario-dependent DA prices.
    This program uses quantity bids at known DA prices and one DA decision for all scenarios.
 3. Q_buy / Q_sell are described as DA limits only.  The physical substation limit applies to the NET
    exchange DA + ID; without it the plant could import Q_buy + Cap_buy.  Both the per-market caps and the
    net limit are enforced.
 4. Unit mismatch: C0 is declared in EUR/h but used as C0*Pmin*u (EUR/h * MW).  As in the DA model, C0
    is EUR/MWh (48.41), so C0*Pmin*u = 484.10 EUR/h.
 5. MT commitment costs (C0*Pmin*u, SUC*x, SDC*y) sit inside the scenario sum.  Because sum(pi) = 1 this
    equals counting them once; they are written once.
 6. Cap_buy^ID / Cap_sell^ID, the scenario count, pi and the ID prices have no values.  Defaults: caps
    100 MW, 10 equiprobable scenarios, ID price = TOU price * lognormal factor (mean one, antithetic
    sampling, price_sigma 15 % hourly + level_sigma 5 % daily).  ID buy/sell use the same factor, so
    sell <= buy holds in every scenario.  Scenario files can be supplied instead.
 7. "Operational fluctuations" are mentioned but only ID prices are random.  An optional additive base-load
    deviation dl[k,t] (--load-sigma, default 0) is provided so the ID market can balance volume risk.
 8. Perfect foresight inside a scenario: Stage-2 decisions see the whole 24 h price path of their
    scenario, which overstates the value of MT / BESS re-dispatch (a real ID market reveals prices
    gradually).  A multi-stage scenario tree would be needed to remove it.
 9. Virtual DA/ID arbitrage: with independent DA / ID positions the model buys DA and sells ID (or the
    reverse) whenever E[lam_ID] differs from lam_DA by more than the spread; the positions are then only
    bounded by Q and the caps.  The scenario set is checked and a warning is logged if this can happen.
10. The BESS is purely Stage 2 here, while the multistage skill places a BESS day-ahead baseline in
    Stage 1.  A net DA position makes a separate baseline unnecessary, so this follows the formulation
    file.  The skill's real-time balancing stage is not part of this file; it is implemented in
    factory_mt_rt_scheduler.py (formulation v3).
11. Per-scenario BESS exclusivity binaries multiply the binary count by |S|; they are redundant under the
    conditions in the v6 skill (use --bess-relax-exclusive to drop them).
12. Parameter drift between the two skills: MT blocks (5.3 / 7.2 / 7.2 / 5.3 MW, 35 MW total) vs the
    corrected 5.3 / 4.7 / 5.0 MW (15 MW); BESS SoC_max 200 vs 180 MWh; MUT/MDT only for t >= MUT.  The
    corrected DA-model values are used.
13. Risk neutral: expected cost only (no CVaR / risk term), so a high-variance Stage-1 plan is not penalised.

Outputs (IntradayResult.save): jobs.csv, da_position.csv, scenarios.csv, hourly_scenarios.csv,
scenario_set.json, summary.json (+ schedule.png with --plot).

Requires: numpy, pandas, pyomo, highspy and factory_mt_da_scheduler.py in the same folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import factory_mt_da_scheduler as da
from factory_mt_da_scheduler import (DT_H, EPS, Instance, InstanceValidationError, SchedulerConfig,
                                     SchedulingError, SolverSettings, VerificationError)

__version__ = "1.0.1"
log = logging.getLogger("factory_mt_id")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class IntradayMarket:
    """Depth of the intraday market (MW per hour, each direction)."""
    cap_buy_mw: float = 100.0
    cap_sell_mw: float = 100.0

    def validate(self) -> "IntradayMarket":
        if self.cap_buy_mw < 0 or self.cap_sell_mw < 0:
            raise InstanceValidationError("intraday caps must be >= 0")
        return self


@dataclass
class ScenarioSet:
    """Intraday scenarios: probabilities, ID buy/sell prices [EUR/MWh] and base-load deviation [MW], all [S, T]."""
    prob: np.ndarray
    id_buy_eur_mwh: np.ndarray
    id_sell_eur_mwh: np.ndarray
    load_dev_mw: np.ndarray

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])

    # ------------------------------------------------------------------ checks
    def validate(self, inst: Instance) -> "ScenarioSet":
        T = inst.horizon_h
        self.prob = da._arr(self.prob, None, "scenario prob", 1)
        S = self.prob.shape[0]
        if S == 0:
            raise InstanceValidationError("at least one scenario is required")
        for nm in ("id_buy_eur_mwh", "id_sell_eur_mwh", "load_dev_mw"):
            a = da._arr(getattr(self, nm), None, nm, 2)
            if a.shape != (S, T):
                raise InstanceValidationError(f"{nm}: expected shape {(S, T)}, got {a.shape}")
            setattr(self, nm, a)
        if np.any(self.prob < 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise InstanceValidationError("scenario probabilities must be >= 0 and sum to 1")
        if np.any(self.id_buy_eur_mwh < 0) or np.any(self.id_sell_eur_mwh < 0):
            raise InstanceValidationError("negative intraday prices are not supported")
        if np.any(self.id_sell_eur_mwh > self.id_buy_eur_mwh + EPS):
            raise InstanceValidationError("intraday sell price must not exceed the buy price in any scenario/hour")
        if np.any(inst.base_load_mw[None, :] + self.load_dev_mw < -EPS):
            raise InstanceValidationError("base load plus deviation must stay >= 0")
        # virtual DA/ID arbitrage (formulation issue 9)
        e_buy, e_sell = self.prob @ self.id_buy_eur_mwh, self.prob @ self.id_sell_eur_mwh
        bad = np.nonzero((e_sell > inst.price_buy_eur_mwh + 1e-6) | (e_buy < inst.price_sell_eur_mwh - 1e-6))[0]
        if bad.size:
            log.warning("expected ID prices leave the DA buy/sell band in hours %s: DA/ID virtual arbitrage is "
                        "profitable there and is limited only by Q and the ID caps", bad.tolist())
        return self

    def expected_value(self) -> "ScenarioSet":
        """Single scenario with the probability-weighted prices and deviation (for the EV problem)."""
        return ScenarioSet(np.ones(1), (self.prob @ self.id_buy_eur_mwh)[None, :],
                           (self.prob @ self.id_sell_eur_mwh)[None, :], (self.prob @ self.load_dev_mw)[None, :])

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return dict(prob=self.prob.tolist(), id_buy_eur_mwh=self.id_buy_eur_mwh.tolist(),
                    id_sell_eur_mwh=self.id_sell_eur_mwh.tolist(), load_dev_mw=self.load_dev_mw.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSet":
        try:
            return cls(np.asarray(d["prob"], float), np.asarray(d["id_buy_eur_mwh"], float),
                       np.asarray(d["id_sell_eur_mwh"], float), np.asarray(d["load_dev_mw"], float))
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad scenario file schema: {err}") from err

    @classmethod
    def load(cls, path) -> "ScenarioSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read scenario file '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---------------------------------------------------------------- generator
    @classmethod
    def generate(cls, inst: Instance, n: int = 10, seed: int = 7, price_sigma: float = 0.15,
                 level_sigma: float = 0.05, load_sigma_mw: float = 0.0) -> "ScenarioSet":
        """ID price = DA tariff * exp(eps - var/2); eps = hourly noise + common daily level shock.
        Antithetic sampling (eps and -eps) keeps the sample mean close to the DA tariff."""
        if n < 1 or min(price_sigma, level_sigma, load_sigma_mw) < 0:
            raise InstanceValidationError("need n >= 1 and non-negative volatilities")
        T = inst.horizon_h
        rng = np.random.default_rng(seed)
        half = n // 2                                   # antithetic pairs; an odd n adds one zero (central) path
        eps = price_sigma * rng.standard_normal((half, T)) + level_sigma * rng.standard_normal((half, 1))
        dev = load_sigma_mw * rng.standard_normal((half, T))
        dev = np.clip(dev, -inst.base_load_mw[None, :], inst.base_load_mw[None, :])   # symmetric clip keeps the +/- pairs
        zero = np.zeros((n - 2 * half, T))
        eps, dev = np.vstack([eps, -eps, zero]), np.vstack([dev, -dev, zero])
        factor = np.exp(eps - (price_sigma ** 2 + level_sigma ** 2) / 2.0)
        dev = load_sigma_mw * rng.standard_normal((half, T))
        dev = np.clip(dev, -inst.base_load_mw[None, :], inst.base_load_mw[None, :])   # symmetric: keeps the +/- pairs, so the mean stays 0
        dev = np.vstack([dev, -dev])[:n]
        return cls(np.full(n, 1.0 / n), inst.price_buy_eur_mwh[None, :] * factor,
                   inst.price_sell_eur_mwh[None, :] * factor, dev).validate(inst)


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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_intraday_model(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, cfg: SchedulerConfig,
                         cands: List[da.Candidate]) -> pyo.ConcreteModel:
    """Deterministic equivalent of the two-stage problem (extensive form)."""
    T, S = inst.horizon_h, scen.n
    mt, bess = inst.mt, inst.bess
    m = pyo.ConcreteModel("FactoryMicroturbineIntraday")
    m.H = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, S - 1)

    # ---------------- Stage 1 ----------------
    m.Pbuy = pyo.Var(m.H, bounds=(0, inst.grid_limit_mw))                   # DA purchase (MW)
    m.Psell = pyo.Var(m.H, bounds=(0, inst.grid_sell_limit_mw))             # DA sale (MW)
    by_hour = da.add_batch_block(m, inst, cfg, cands)                       # s, N, inventory, sequencing
    m.Lbatch = pyo.Expression(m.H, rule=lambda mm, t: pyo.quicksum(mw * mm.s[c] for c, mw, _ in by_hour[t]))
    if mt is not None:
        da.add_mt_commitment(m, inst)                                       # u, x, y, MUT / MDT

    # ---------------- Stage 2 ----------------
    m.Ibuy = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_buy_mw))                  # ID purchase (MW)
    m.Isell = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_sell_mw))                # ID sale (MW)
    if mt is not None:
        m.B = pyo.RangeSet(0, len(mt.block_width_mw) - 1)
        m.Pmt = pyo.Var(m.S, m.H, bounds=(0, mt.p_max_mw))
        m.Pb = pyo.Var(m.B, m.S, m.H, bounds=lambda mm, b, k, t: (0, mt.block_width_mw[b]))
        u0 = 1.0 if mt.initial_on else 0.0
        p0 = float(mt.initial_power_mw) if mt.initial_on else 0.0
        up = lambda mm, t: mm.u[t - 1] if t > 0 else u0
        pp = lambda mm, k, t: mm.Pmt[k, t - 1] if t > 0 else p0
        m.c_mt_sum = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] == mt.p_min_mw * mm.u[t]
                                    + pyo.quicksum(mm.Pb[b, k, t] for b in mm.B))
        m.c_mt_blk = pyo.Constraint(m.B, m.S, m.H, rule=lambda mm, b, k, t: mm.Pb[b, k, t] <= mt.block_width_mw[b] * mm.u[t])
        m.c_ru = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] - pp(mm, k, t)
                                <= mt.ramp_up_mw_h * up(mm, t) + mt.startup_ramp_mw_h * mm.x[t])
        m.c_rd = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: pp(mm, k, t) - mm.Pmt[k, t]
                                <= mt.ramp_down_mw_h * mm.u[t] + mt.shutdown_ramp_mw_h * mm.y[t])
    if bess is not None:
        m.Pch = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.Pdis = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.SoC = pyo.Var(m.S, m.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))
        soc_prev = lambda mm, k, t: mm.SoC[k, t - 1] if t > 0 else bess.soc_init_mwh
        m.c_soc = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.SoC[k, t] == soc_prev(mm, k, t)
                                 + bess.eta_ch * mm.Pch[k, t] * DT_H - mm.Pdis[k, t] * DT_H / bess.eta_dis)
        m.c_soc_term = pyo.Constraint(m.S, rule=lambda mm, k: mm.SoC[k, T - 1] >= bess.soc_init_mwh)
        if bess.enforce_exclusive:
            m.vch = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.vdis = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.c_ch = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pch[k, t] <= bess.p_max_mw * mm.vch[k, t])
            m.c_dis = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pdis[k, t] <= bess.p_max_mw * mm.vdis[k, t])
            m.c_excl = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.vch[k, t] + mm.vdis[k, t] <= 1)

    # ---------------- balance and substation limits ----------------
    mt_out = (lambda mm, k, t: mm.Pmt[k, t]) if mt is not None else (lambda mm, k, t: 0.0)
    bess_out = (lambda mm, k, t: mm.Pdis[k, t] - mm.Pch[k, t]) if bess is not None else (lambda mm, k, t: 0.0)
    net = lambda mm, k, t: mm.Pbuy[t] - mm.Psell[t] + mm.Ibuy[k, t] - mm.Isell[k, t]
    m.c_power = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: net(mm, k, t) + mt_out(mm, k, t) + bess_out(mm, k, t)
                               == inst.base_load_mw[t] + scen.load_dev_mw[k, t] + mm.Lbatch[t])
    m.c_net_imp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: net(mm, k, t) <= inst.grid_limit_mw)
    m.c_net_exp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: -net(mm, k, t) <= inst.grid_sell_limit_mw)

    # ---------------- objective ----------------
    pi = scen.prob
    cost = pyo.quicksum(inst.price_buy_eur_mwh[t] * m.Pbuy[t] - inst.price_sell_eur_mwh[t] * m.Psell[t] for t in range(T))
    cost += pyo.quicksum(pi[k] * (scen.id_buy_eur_mwh[k, t] * m.Ibuy[k, t] - scen.id_sell_eur_mwh[k, t] * m.Isell[k, t])
                         for k in range(S) for t in range(T))
    if mt is not None:
        cost += pyo.quicksum(mt.base_cost_eur_mwh * mt.p_min_mw * m.u[t] + mt.startup_cost_eur * m.x[t]
                             + mt.shutdown_cost_eur * m.y[t] for t in range(T))          # Stage-1 commitment cost
        cost += pyo.quicksum(pi[k] * mt.block_cost_eur_mwh[b] * m.Pb[b, k, t]
                             for b in range(len(mt.block_width_mw)) for k in range(S) for t in range(T))
    if bess is not None:
        w_ch = 1.0 if bess.degradation_basis == "throughput" else 0.0
        cost += pyo.quicksum(pi[k] * bess.degradation_eur_mwh * (w_ch * m.Pch[k, t] + m.Pdis[k, t])
                             for k in range(S) for t in range(T))
    m.obj = pyo.Objective(expr=cost, sense=pyo.minimize)
    return m


# --------------------------------------------------------------------------- #
# Post-processing and verification
# --------------------------------------------------------------------------- #
def _val(var, default: float = 0.0) -> float:
    v = var.value
    return default if v is None else float(v)


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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def value_of_stochastic_solution(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, cfg: SchedulerConfig,
                                 cands: List[da.Candidate], rp_eur: float) -> Dict[str, float]:
    """VSS = EEV - RP.  EEV: solve the expected-value problem, fix its Stage-1 decisions, re-optimise the
    recourse in every scenario.  RP is the stochastic solution value already computed."""
    ev_model = build_intraday_model(inst, mkt, scen.expected_value(), cfg, cands)
    da.solve_model(ev_model, cfg.solver)
    full = build_intraday_model(inst, mkt, scen, cfg, cands)
    for c in full.C:
        full.s[c].fix(round(_val(ev_model.s[c])))
    for t in range(inst.horizon_h):
        full.Pbuy[t].fix(_val(ev_model.Pbuy[t]))
        full.Psell[t].fix(_val(ev_model.Psell[t]))
        if inst.mt is not None:
            for nm in ("u", "x", "y"):
                getattr(full, nm)[t].fix(round(_val(getattr(ev_model, nm)[t])))
    try:
        da.solve_model(full, cfg.solver)
        eev = float(pyo.value(full.obj))
    except da.InfeasibleScheduleError:
        log.warning("the expected-value plan is infeasible in at least one scenario: EEV = inf")
        eev = float("inf")
    return dict(ev_objective_eur=float(pyo.value(ev_model.obj)), eev_eur=eev, rp_eur=rp_eur, vss_eur=eev - rp_eur,
                vss_pct=100 * (eev - rp_eur) / abs(rp_eur) if rp_eur else float("nan"))


def optimize_intraday(inst: Instance, scen: ScenarioSet, mkt: Optional[IntradayMarket] = None,
                      cfg: Optional[SchedulerConfig] = None, compute_vss: bool = False) -> IntradayResult:
    """Validate -> candidates -> extensive-form MILP -> solve -> verify -> package results."""
    cfg, mkt = cfg or SchedulerConfig(), (mkt or IntradayMarket()).validate()
    inst.validate()
    cfg.check(inst)
    scen.validate(inst)
    cands = da.build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts, %d scenarios, MT=%s, BESS=%s", len(cands), scen.n,
             "yes" if inst.mt else "no", "yes" if inst.bess else "no")
    model = build_intraday_model(inst, mkt, scen, cfg, cands)
    info = da.solve_model(model, cfg.solver)
    res = extract_intraday(inst, mkt, scen, cfg, model, cands, info)
    if compute_vss:
        res.vss = value_of_stochastic_solution(inst, mkt, scen, cfg, cands, res.objective_eur)
    return res


def plot_intraday(inst: Instance, res: IntradayResult, path) -> bool:
    """Gantt chart, DA position vs expected ID trade with the ID price fan, and the BESS SoC fan."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    T = inst.horizon_h
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 2.4, 2]))
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Stage-1 batch schedule | expected cost {res.objective_eur:,.0f} EUR | {int(res.kpis['n_scenarios'])} scenarios")
    p = res.da_position
    a2.bar(p.hour + 0.3, p.p_da_buy_mw - p.p_da_sell_mw, width=0.4, color="tab:blue", label="DA net import (MW)")
    a2.bar(p.hour + 0.7, p.exp_p_id_buy_mw - p.exp_p_id_sell_mw, width=0.4, color="tab:orange", label="E[ID net import] (MW)")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("MW")
    pr = a2.twinx()
    for k in range(res.scenario_set.n):
        pr.step(np.append(np.arange(T), T), np.append(res.scenario_set.id_buy_eur_mwh[k], res.scenario_set.id_buy_eur_mwh[k][-1]),
                where="post", color="gray", lw=0.5, alpha=0.5)
    pr.step(np.append(p.hour, T), np.append(p.price_da_buy_eur_mwh, p.price_da_buy_eur_mwh.iloc[-1]), where="post",
            color="tab:red", lw=1.4, label="DA buy price")
    pr.set_ylabel("EUR/MWh (grey = ID buy scenarios)")
    a2.legend(loc="upper left", ncol=2)
    pr.legend(loc="upper right")
    if inst.bess is not None:
        for k in range(res.scenario_set.n):
            hk = res.hourly[res.hourly.scenario == k]
            a3.plot(hk.hour + 0.5, hk.soc_mwh, color="tab:purple", lw=0.8, alpha=0.6)
        a3.axhline(inst.bess.soc_min_mwh, color="gray", ls=":")
        a3.axhline(inst.bess.soc_max_mwh, color="gray", ls=":")
        a3.set_ylabel("BESS SoC (MWh), one line per scenario")
    if inst.mt is not None:
        a4 = a3.twinx()
        a4.step(p.hour, p.mt_on * inst.mt.p_max_mw, where="post", color="tab:green", lw=1.2, label="MT committed (x Pmax)")
        a4.plot(p.hour + 0.5, p.exp_p_mt_mw, color="tab:olive", lw=1.2, label="E[P_MT] (MW)")
        a4.set_ylabel("MT MW")
        a4.legend(loc="upper right")
    a3.set_xlabel("hour of day")
    a3.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Factory + MT + BESS day-ahead / intraday two-stage stochastic MILP")
    g = ap.add_argument_group("instance")
    g.add_argument("--instance", help="JSON instance file (default: built-in benchmark)")
    g.add_argument("--seed", type=int, default=2024, help="benchmark generator seed")
    g.add_argument("--max-batches", type=int, default=10)
    g.add_argument("--start-step", type=float, default=0.5)
    g.add_argument("--allow-repeat-tasks", action="store_true")
    g.add_argument("--no-mt", action="store_true")
    g.add_argument("--no-bess", action="store_true")
    g.add_argument("--mt-ramp-up", type=float, default=None)
    g.add_argument("--mt-ramp-down", type=float, default=None)
    g.add_argument("--mt-initial-on", action="store_true")
    g.add_argument("--bess-degradation", type=float, default=None)
    g.add_argument("--bess-degradation-basis", choices=["throughput", "discharge"], default=None)
    g.add_argument("--bess-power", type=float, default=None)
    g.add_argument("--bess-energy", type=float, default=None)
    g.add_argument("--bess-relax-exclusive", action="store_true", help="drop the per-scenario charge/discharge binaries")
    s = ap.add_argument_group("intraday market and scenarios")
    s.add_argument("--id-cap-buy", type=float, default=100.0, help="ID purchase depth in MW per hour")
    s.add_argument("--id-cap-sell", type=float, default=100.0, help="ID sale depth in MW per hour")
    s.add_argument("--scenarios", type=int, default=10, help="number of generated scenarios")
    s.add_argument("--scenario-seed", type=int, default=7)
    s.add_argument("--price-sigma", type=float, default=0.15, help="hourly relative ID price volatility")
    s.add_argument("--level-sigma", type=float, default=0.05, help="common daily relative ID price shift")
    s.add_argument("--load-sigma", type=float, default=0.0, help="std of the additive base-load deviation in MW")
    s.add_argument("--scenario-file", help="JSON scenario set (overrides the generator)")
    s.add_argument("--vss", action="store_true", help="also compute the value of the stochastic solution")
    o = ap.add_argument_group("solver / output")
    o.add_argument("--solver", default="appsi_highs")
    o.add_argument("--mip-gap", type=float, default=1e-3)
    o.add_argument("--time-limit", type=float, default=300.0)
    o.add_argument("--threads", type=int, default=None)
    o.add_argument("--out", default="da_id_results")
    o.add_argument("--plot", action="store_true")
    o.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        inst = (Instance.load(a.instance) if a.instance else
                da.make_benchmark_instance(a.seed, max_batches=a.max_batches, with_mt=not a.no_mt, with_bess=not a.no_bess))
        da.apply_instance_overrides(inst, a)
        mkt = IntradayMarket(a.id_cap_buy, a.id_cap_sell).validate()
        scen = (ScenarioSet.load(a.scenario_file) if a.scenario_file else
                ScenarioSet.generate(inst, a.scenarios, a.scenario_seed, a.price_sigma, a.level_sigma, a.load_sigma))
        cfg = SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                              solver=SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_intraday(inst, scen, mkt, cfg, compute_vss=a.vss)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"]]
              .round(2).to_string(index=False))
        print(res.da_position.round(2).to_string(index=False))
        print(res.scenarios.round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_intraday(inst, res, out / "schedule.png")
        print(f"\nResults written to {out.resolve()}")
        return 0
    except InstanceValidationError as err:
        log.error("Invalid input: %s", err)
        return 2
    except (da.SolverUnavailableError, da.InfeasibleScheduleError, da.SolveFailedError, VerificationError) as err:
        log.error("%s: %s", type(err).__name__, err)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
