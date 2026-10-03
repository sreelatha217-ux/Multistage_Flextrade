#!/usr/bin/env python3
"""
factory_mt_rt_scheduler.py
==========================

Three-stage stochastic Day-Ahead (DA) -> Intraday (ID) -> Real-Time (RT) balancing scheduling of an
industrial batch plant with an on-site Microturbine (MT) and a Battery (BESS).  Stage 3 extension of
factory_mt_da_scheduler.py and factory_mt_id_scheduler.py (both imported, neither modified) following
"factory_multistage_formulation-v3.md" (v2 with the corrections listed below) and the corrected skill
"industrial-large-consumer-multistage-optimization".  All money is in EUR.

Information structure (scenario tree: ID scenario s, then RT scenario w | s)
---------------------------------------------------------------------------
    Stage 1  here-and-now      P_DAbuy[t], P_DAsell[t]; u, x, y (MT commitment); s[m,p,k] (batch starts)
    Stage 2  1st wait-and-see  indexed by ID scenario s only:
                               P_IDbuy[s,t], P_IDsell[s,t], P_MT[s,t], P_ch[s,t], P_dis[s,t], SoC[s,t]
    Stage 3  2nd wait-and-see  indexed by (s, w):  D+[s,w,t] >= 0 deficit bought from the TSO,
                               D-[s,w,t] >= 0 surplus sold to the TSO.  The RT load deviation eta[s,w,t] and
                               the imbalance price ratios r+[s,w,t], r-[s,w,t] are revealed here.

    min   sum_t [ lamDAbuy*PDAbuy - lamDAsell*PDAsell + C0*Pmin*u + SUC*x + SDC*y ]
        + sum_s pi_s sum_t [ lamIDbuy*PIDbuy - lamIDsell*PIDsell + sum_b C_b*P_MT,b + C_TP*(w*P_ch + P_dis) ]
        + sum_s pi_s sum_w pi_w|s sum_t [ r+ * lamDAbuy * D+  -  r- * lamDAsell * D- ]

    (PDAbuy - PDAsell) + (PIDbuy - PIDsell) + P_MT + P_dis - P_ch  +  D+ - D-
                                                  = (1 + eta[s,w,t]) * (l_base + dl[s,t] + L_batch)
    physical exchange  (PDA + PID) + D+ - D-  in [-Q_sell, Q_buy]       market net (PDA + PID) in [-Q_sell, Q_buy]

    imbalance modes   "strategic": the Stage-2 position may be deliberately unbalanced (literal v2 model, a
                                   newsvendor hedge against eta);
                      "passive"  : the Stage-2 plan must balance the expected RT load, so D+ - D- only carries
                                   the realised deviation (the "passive physical settlement" of v2 section 10).

Review of the Stage-3 text (formulation v2 and skill) - inconsistencies and how this program resolves them
------------------------------------------------------------------------------------------------------------
 1. SIGN ERROR in v2 section 4.  The balance reads  supply = load + (D+ - D-)  but section 5.1/5.3 and the
    cost term define D+ as a DEFICIT bought at the penalty price (D = physical - market = D+ - D-).  With the
    section-4 sign a deficit would be bought while the plant has a surplus.  The consistent balance is
    supply + D+ - D- = load, used here.
 2. NAMING MIRRORED between the skill and v2.  Skill: D- = shortage at r- >= 1, D+ = surplus at r+ <= 1.
    v2: D+ = shortage at r+ >= 1, D- = surplus at r- <= 1.  v2 naming is used (skill D- == v2 D+).
 3. Skill prices imbalance at ONE price lam^DA (and indexes it by s although DA is deterministic).  The TOU
    tariff has sell = 55 % of buy; surplus sold at r*lam_buy would earn more than any DA/ID sale (loophole).
    v2 uses lam_buy for deficits and lam_sell for surplus; kept.
 4. STAGE 3 IS NOT A SEPARATE STAGE in v2: D is indexed by the same s as every Stage-2 variable, so the Stage-2
    decisions already know what D "reveals" and D becomes a free trading channel (deliberate imbalance),
    contradicting "passive settlement" (v2 section 10).  A scenario tree (s, w) is used: Stage-2 decisions
    cannot depend on w.
 5. WITHOUT RT RANDOMNESS Stage 3 IS VACUOUS: the only uncertainty in the DA/ID files is the ID price.  RT
    uncertainty is added: relative load deviation eta (AR(1), antithetic, zero mean) and random ratios.
    NOTE: with a free slack and no RT uncertainty the model only uses D when an ID price is worse than the
    imbalance price - that is price arbitrage, not balancing.  A log warning flags such price sets.
 6. z_BAL binaries (v2 5.2) are redundant: r+ >= 1 >= r- and lam_buy >= lam_sell give lam+ >= lam-, so buying and
    selling imbalance at once never pays.  They would add |S||W|T binaries; default OFF (--bal-exclusive adds
    them).  The ordering lam+ >= lam- is validated instead.  Delta_bar_buy/sell are unspecified: default
    Q_buy + Q_sell (a valid bound, never restricting).
 7. Q_buy / Q_sell must bind the PHYSICAL exchange (market + D), not only the market position.  Both enforced.
 8. r+, r-, RT volatility and the number of RT scenarios have no values in the skill.  Defaults: eta sigma 3 %,
    AR(1) rho 0.6, r+ = 1 + 0.25 g, r- = 1 - 0.25 g with g lognormal (mean 1, sigma 0.3), 5 RT scenarios per ID
    scenario (1 zero path + 2 antithetic pairs).
 9. Still open from the earlier notes: skill MT blocks (35 MW) and SoC_max (200) differ from the corrected DA
    values; skill Stage 1 lists a BESS DA baseline that no equation implements; skill 2.6 monotonicity vs
    non-anticipativity; v2 mentions renewables that are not modelled; risk-neutral objective; within-scenario
    perfect foresight of ID prices.
10. RT price ratios are drawn independently of the sign of eta.  Real imbalance prices depend on the system
    direction (and correlate with ID prices); supply a scenario file (--rt-file) for a calibrated joint law.

Outputs (RealTimeResult.save): jobs.csv, da_position.csv, scenarios.csv, joint_scenarios.csv, hourly_rt.csv,
scenario_set.json, rt_set.json, summary.json (+ schedule_rt.png with --plot).

Requires: numpy, pandas, pyomo, highspy and the two scheduler files in the same folder.
Self-check (needs a solver):  python factory_mt_rt_scheduler.py --selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import factory_mt_da_scheduler as da
import factory_mt_id_scheduler as idm
from factory_mt_da_scheduler import (EPS, Instance, InstanceValidationError, SchedulerConfig, SolverSettings,
                                     VerificationError)
from factory_mt_id_scheduler import IntradayMarket, ScenarioSet, _val

__version__ = "1.0.0"
log = logging.getLogger("factory_mt_rt")
MODES = ("strategic", "passive")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class BalancingMarket:
    """Settings of the imbalance settlement (v2 section 5)."""
    mode: str = "strategic"                    # "strategic" | "passive" (see module docstring)
    enforce_exclusive: bool = False            # z_BAL binaries (redundant while lam+ >= lam-)
    max_deficit_mw: Optional[float] = None     # Delta_bar_buy  (default Q_buy + Q_sell)
    max_surplus_mw: Optional[float] = None     # Delta_bar_sell (default Q_buy + Q_sell)

    def validate(self) -> "BalancingMarket":
        if self.mode not in MODES:
            raise InstanceValidationError(f"imbalance mode must be one of {MODES}")
        for nm in ("max_deficit_mw", "max_surplus_mw"):
            v = getattr(self, nm)
            if v is not None and v < 0:
                raise InstanceValidationError(f"{nm} must be >= 0")
        return self

    def bounds(self, inst: Instance) -> tuple:
        full = inst.grid_limit_mw + inst.grid_sell_limit_mw
        return (full if self.max_deficit_mw is None else self.max_deficit_mw,
                full if self.max_surplus_mw is None else self.max_surplus_mw)


@dataclass
class RealTimeSet:
    """Real-time scenarios for every ID scenario s: conditional probabilities, relative factory-load
    deviation eta and imbalance price ratios, all [S, W, T] (prob is [S, W])."""
    prob: np.ndarray
    load_rel_dev: np.ndarray        # eta: actual factory load = (1 + eta) * planned load
    r_plus: np.ndarray              # deficit (buy) penalty ratio, >= 1
    r_minus: np.ndarray             # surplus (sell) discount ratio, in [0, 1]

    @property
    def n_s(self) -> int:
        return int(self.prob.shape[0])

    @property
    def n_w(self) -> int:
        return int(self.prob.shape[1])

    # ------------------------------------------------------------------ prices
    def lam_plus(self, inst: Instance) -> np.ndarray:
        """Deficit price lam+ = r+ * lam_DA,buy  [S, W, T]."""
        return self.r_plus * inst.price_buy_eur_mwh[None, None, :]

    def lam_minus(self, inst: Instance) -> np.ndarray:
        """Surplus price lam- = r- * lam_DA,sell  [S, W, T]."""
        return self.r_minus * inst.price_sell_eur_mwh[None, None, :]

    # ------------------------------------------------------------------ checks
    def validate(self, inst: Instance, scen: ScenarioSet) -> "RealTimeSet":
        T = inst.horizon_h
        self.prob = da._arr(self.prob, None, "rt prob", 2)
        S, W = self.prob.shape
        if S != scen.n:
            raise InstanceValidationError(f"rt set has {S} ID scenarios, the ID scenario set has {scen.n}")
        if W == 0:
            raise InstanceValidationError("at least one RT scenario per ID scenario is required")
        for nm in ("load_rel_dev", "r_plus", "r_minus"):
            a = da._arr(getattr(self, nm), None, nm, 3)
            if a.shape != (S, W, T):
                raise InstanceValidationError(f"{nm}: expected shape {(S, W, T)}, got {a.shape}")
            setattr(self, nm, a)
        if np.any(self.prob < 0) or np.abs(self.prob.sum(axis=1) - 1.0).max() > 1e-9:
            raise InstanceValidationError("rt probabilities must be >= 0 and sum to 1 for every ID scenario")
        if np.any(self.load_rel_dev <= -1.0):
            raise InstanceValidationError("relative load deviation must stay > -1 (load would become negative)")
        if np.any(self.r_plus < 1.0 - 1e-9):
            raise InstanceValidationError("r_plus must be >= 1 (deficit may not be cheaper than the DA buy price)")
        if np.any(self.r_minus < -1e-9) or np.any(self.r_minus > 1.0 + 1e-9):
            raise InstanceValidationError("r_minus must lie in [0, 1] (surplus may not earn more than the DA sell price)")
        if np.any(self.lam_plus(inst) < self.lam_minus(inst) - EPS):      # cannot happen for valid ratios; kept as guard
            raise InstanceValidationError("deficit price below surplus price: simultaneous buy/sell would be an arbitrage loop")
        # price arbitrage ID <-> imbalance (issue 5): imbalance cheaper than the ID market in expectation
        e_lp = np.einsum("sw,swt->st", self.prob, self.lam_plus(inst))
        e_lm = np.einsum("sw,swt->st", self.prob, self.lam_minus(inst))
        n_bad = int(np.sum((scen.id_buy_eur_mwh > e_lp + 1e-6) | (scen.id_sell_eur_mwh < e_lm - 1e-6)))
        if n_bad:
            log.warning("in %d of %d (scenario, hour) cells the expected imbalance price beats the ID price: the model "
                        "may buy/sell imbalance deliberately instead of trading intraday (use --imbalance-mode passive "
                        "or larger ratio spreads to suppress)", n_bad, scen.n * T)
        return self

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return dict(prob=self.prob.tolist(), load_rel_dev=self.load_rel_dev.tolist(),
                    r_plus=self.r_plus.tolist(), r_minus=self.r_minus.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "RealTimeSet":
        try:
            return cls(np.asarray(d["prob"], float), np.asarray(d["load_rel_dev"], float),
                       np.asarray(d["r_plus"], float), np.asarray(d["r_minus"], float))
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad RT file schema: {err}") from err

    @classmethod
    def load(cls, path) -> "RealTimeSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read RT file '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---------------------------------------------------------------- builders
    @classmethod
    def constant(cls, n_s: int, n_w: int, horizon_h: int, r_plus: float = 1.25, r_minus: float = 0.75,
                 eta: float = 0.0) -> "RealTimeSet":
        """Equiprobable RT scenarios with constant ratios and a constant deviation (deterministic RT stage)."""
        shape = (n_s, n_w, horizon_h)
        return cls(np.full((n_s, n_w), 1.0 / n_w), np.full(shape, float(eta)), np.full(shape, float(r_plus)),
                   np.full(shape, float(r_minus)))

    @classmethod
    def generate(cls, inst: Instance, n_s: int, n_w: int = 5, seed: int = 11, rel_sigma: float = 0.03,
                 rho: float = 0.6, premium: float = 0.25, discount: float = 0.25,
                 ratio_sigma: float = 0.3) -> "RealTimeSet":
        """eta: stationary AR(1) with std rel_sigma, antithetic pairs (+eta, -eta) plus one zero path when n_w is odd,
        so E[eta] = 0 exactly.  r+ = 1 + premium*g, r- = clip(1 - discount*g, 0, 1), g lognormal with mean one."""
        if n_s < 1 or n_w < 1:
            raise InstanceValidationError("need n_s >= 1 and n_w >= 1")
        if min(rel_sigma, premium, discount, ratio_sigma) < 0 or not (0 <= rho < 1):
            raise InstanceValidationError("need non-negative volatilities and 0 <= rho < 1")
        T = inst.horizon_h
        rng = np.random.default_rng(seed)
        pairs = n_w // 2
        z = rng.standard_normal((n_s, pairs, T))
        e = np.empty_like(z)
        if pairs:
            e[..., 0] = z[..., 0]
            a = math.sqrt(1.0 - rho ** 2)
            for t in range(1, T):
                e[..., t] = rho * e[..., t - 1] + a * z[..., t]
        e = np.clip(rel_sigma * e, -0.5, 0.5)                       # symmetric clip keeps the zero mean
        parts = [e, -e] + ([np.zeros((n_s, 1, T))] if n_w % 2 else [])
        eta = np.concatenate(parts, axis=1)
        g = np.exp(ratio_sigma * rng.standard_normal((n_s, n_w, T)) - ratio_sigma ** 2 / 2.0)
        return cls(np.full((n_s, n_w), 1.0 / n_w), eta, 1.0 + premium * g, np.clip(1.0 - discount * g, 0.0, 1.0))


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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_realtime_model(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, rt: RealTimeSet,
                         bal: BalancingMarket, cfg: SchedulerConfig, cands: List[da.Candidate]) -> pyo.ConcreteModel:
    """Extensive form of the three-stage problem.  The Stage 1/2 model of factory_mt_id_scheduler is reused; its
    scenario balance (which has no Stage-3 slack) and its objective are replaced."""
    T, S, W = inst.horizon_h, scen.n, rt.n_w
    mt, bess = inst.mt, inst.bess
    m = idm.build_intraday_model(inst, mkt, scen, cfg, cands)
    stage12_cost = m.obj.expr                      # DA + ID + MT + BESS cost, unchanged
    m.del_component(m.c_power)
    m.del_component(m.obj)
    m.W = pyo.RangeSet(0, W - 1)

    # ---------------- Stage 3 variables ----------------
    d_buy_max, d_sell_max = bal.bounds(inst)
    m.Dp = pyo.Var(m.S, m.W, m.H, bounds=(0, d_buy_max))                    # deficit bought (v2 Delta+)
    m.Dm = pyo.Var(m.S, m.W, m.H, bounds=(0, d_sell_max))                   # surplus sold   (v2 Delta-)
    if bal.enforce_exclusive:
        m.zb = pyo.Var(m.S, m.W, m.H, domain=pyo.Binary)
        m.c_dp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dp[k, w, t] <= d_buy_max * mm.zb[k, w, t])
        m.c_dm = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dm[k, w, t] <= d_sell_max * (1 - mm.zb[k, w, t]))

    # ---------------- balance, physical limits ----------------
    mt_out = (lambda mm, k, t: mm.Pmt[k, t]) if mt is not None else (lambda mm, k, t: 0.0)
    bess_out = (lambda mm, k, t: mm.Pdis[k, t] - mm.Pch[k, t]) if bess is not None else (lambda mm, k, t: 0.0)
    market = lambda mm, k, t: mm.Pbuy[t] - mm.Psell[t] + mm.Ibuy[k, t] - mm.Isell[k, t]
    supply = lambda mm, k, t: market(mm, k, t) + mt_out(mm, k, t) + bess_out(mm, k, t)
    load2 = lambda mm, k, t: inst.base_load_mw[t] + scen.load_dev_mw[k, t] + mm.Lbatch[t]     # Stage-2 factory load
    m.c_power_rt = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: supply(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]
                                  == (1.0 + rt.load_rel_dev[k, w, t]) * load2(mm, k, t))
    phys = lambda mm, k, w, t: market(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]             # physical grid exchange
    m.c_phys_imp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: phys(mm, k, w, t) <= inst.grid_limit_mw)
    m.c_phys_exp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: -phys(mm, k, w, t) <= inst.grid_sell_limit_mw)
    if bal.mode == "passive":                      # plan must balance the expected RT load
        e_eta = np.einsum("sw,swt->st", rt.prob, rt.load_rel_dev)
        m.c_plan = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: supply(mm, k, t) == (1.0 + e_eta[k, t]) * load2(mm, k, t))

    # ---------------- objective ----------------
    lam_p, lam_m = rt.lam_plus(inst), rt.lam_minus(inst)
    bal_cost = pyo.quicksum(scen.prob[k] * rt.prob[k, w] * (lam_p[k, w, t] * m.Dp[k, w, t] - lam_m[k, w, t] * m.Dm[k, w, t])
                            for k in range(S) for w in range(W) for t in range(T))
    m.obj = pyo.Objective(expr=stage12_cost + bal_cost, sense=pyo.minimize)
    return m


# --------------------------------------------------------------------------- #
# Post-processing and verification
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def value_of_rt_modelling(inst: Instance, mkt: IntradayMarket, scen: ScenarioSet, rt: RealTimeSet,
                          bal: BalancingMarket, cfg: SchedulerConfig, cands: List[da.Candidate],
                          rp_eur: float) -> Dict[str, float]:
    """EEV_RT - RP.  Solve the Stage 1/2 model WITHOUT real-time risk, fix its Stage-1 decisions (batch starts, DA
    position, MT commitment) in the three-stage model and re-optimise Stage 2 / 3.  The gap to the full solution
    (RP) is the value of anticipating real-time imbalance when committing in Stage 1."""
    id_model = idm.build_intraday_model(inst, mkt, scen, cfg, cands)
    da.solve_model(id_model, cfg.solver)
    full = build_realtime_model(inst, mkt, scen, rt, bal, cfg, cands)
    for c in full.C:
        full.s[c].fix(round(_val(id_model.s[c])))
    for t in range(inst.horizon_h):
        full.Pbuy[t].fix(_val(id_model.Pbuy[t]))
        full.Psell[t].fix(_val(id_model.Psell[t]))
        if inst.mt is not None:
            for nm in ("u", "x", "y"):
                getattr(full, nm)[t].fix(round(_val(getattr(id_model, nm)[t])))
    try:
        da.solve_model(full, cfg.solver)
        eev = float(pyo.value(full.obj))
    except da.InfeasibleScheduleError:
        log.warning("the RT-blind Stage-1 plan is infeasible in at least one RT scenario: EEV_RT = inf")
        eev = float("inf")
    return dict(id_model_objective_eur=float(pyo.value(id_model.obj)), eev_rt_eur=eev, rp_eur=rp_eur,
                vss_rt_eur=eev - rp_eur, vss_rt_pct=100 * (eev - rp_eur) / abs(rp_eur) if rp_eur else float("nan"))


def optimize_realtime(inst: Instance, scen: ScenarioSet, rt: RealTimeSet, mkt: Optional[IntradayMarket] = None,
                      bal: Optional[BalancingMarket] = None, cfg: Optional[SchedulerConfig] = None,
                      compute_vss_rt: bool = False) -> RealTimeResult:
    """Validate -> candidates -> extensive-form MILP -> solve -> verify -> package results."""
    cfg, mkt, bal = cfg or SchedulerConfig(), (mkt or IntradayMarket()).validate(), (bal or BalancingMarket()).validate()
    inst.validate()
    cfg.check(inst)
    scen.validate(inst)
    rt.validate(inst, scen)
    cands = da.build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts, %d ID x %d RT scenarios, mode=%s, MT=%s, BESS=%s", len(cands), scen.n, rt.n_w,
             bal.mode, "yes" if inst.mt else "no", "yes" if inst.bess else "no")
    model = build_realtime_model(inst, mkt, scen, rt, bal, cfg, cands)
    info = da.solve_model(model, cfg.solver)
    res = extract_realtime(inst, mkt, scen, rt, bal, cfg, model, cands, info)
    if compute_vss_rt:
        res.vss_rt = value_of_rt_modelling(inst, mkt, scen, rt, bal, cfg, cands, res.objective_eur)
    return res


def plot_realtime(inst: Instance, res: RealTimeResult, path) -> bool:
    """Gantt chart, Stage-1/2 positions with expected imbalance, and the fan of Stage-3 imbalances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    T = inst.horizon_h
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 2.4, 2.4]))
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Stage-1 batch schedule | expected cost {res.objective_eur:,.0f} EUR | "
                 f"{int(res.kpis['n_scenarios'])} ID x {int(res.kpis['n_rt'])} RT scenarios")
    p = res.da_position
    a2.bar(p.hour + 0.2, p.p_da_buy_mw - p.p_da_sell_mw, width=0.2, color="tab:blue", label="DA net import")
    a2.bar(p.hour + 0.4, p.exp_p_id_buy_mw - p.exp_p_id_sell_mw, width=0.2, color="tab:orange", label="E[ID net import]")
    a2.bar(p.hour + 0.6, p.exp_imb_deficit_mw - p.exp_imb_surplus_mw, width=0.2, color="tab:red", label="E[imbalance] (+ deficit)")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("MW")
    a2.legend(loc="upper left", ncol=3)
    h = res.hourly
    for (_, _), g in h.groupby(["scenario", "rt"]):
        a3.plot(g.hour + 0.5, g.imb_net_mw, color="gray", lw=0.5, alpha=0.5)
    a3.plot(p.hour + 0.5, p.exp_imb_deficit_mw - p.exp_imb_surplus_mw, color="tab:red", lw=1.6, label="expected")
    a3.axhline(0, color="k", lw=0.6)
    a3.set_ylabel("imbalance MW (+ = deficit bought), one line per (s, w)")
    a3.set_xlabel("hour of day")
    a3.set_xlim(0, T)
    a3.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Self-check (needs a solver)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _rt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = ap.add_argument_group("real-time balancing (Stage 3)")
    g.add_argument("--rt-scenarios", type=int, default=5, help="RT scenarios per ID scenario (odd n adds a zero path)")
    g.add_argument("--rt-seed", type=int, default=11)
    g.add_argument("--rt-load-sigma", type=float, default=0.03, help="std of the relative factory-load deviation eta")
    g.add_argument("--rt-load-rho", type=float, default=0.6, help="hourly AR(1) persistence of eta")
    g.add_argument("--r-premium", type=float, default=0.25, help="r+ = 1 + premium * g")
    g.add_argument("--r-discount", type=float, default=0.25, help="r- = 1 - discount * g")
    g.add_argument("--r-sigma", type=float, default=0.3, help="lognormal sigma of g")
    g.add_argument("--rt-file", help="JSON RT scenario set (overrides the generator)")
    g.add_argument("--imbalance-mode", choices=MODES, default="strategic",
                   help="strategic: plan may be deliberately unbalanced; passive: plan balances the expected RT load")
    g.add_argument("--bal-exclusive", action="store_true", help="add the z_BAL binaries (redundant for lam+ >= lam-)")
    g.add_argument("--max-imbalance", type=float, default=None, help="Delta_bar for both directions in MW (default Q_buy + Q_sell)")
    g.add_argument("--vss-rt", action="store_true", help="also compute the value of modelling RT risk (one extra MILP + one re-solve)")
    g.add_argument("--selftest", action="store_true", help="run the built-in consistency checks and exit")
    return ap


def _parse(argv=None) -> argparse.Namespace:
    """RT options are parsed here; every other option (instance, MT, BESS, ID market, solver, output) is delegated
    to factory_mt_id_scheduler._parse so the two programs can never drift apart."""
    argv = list(sys.argv[1:] if argv is None else argv)
    rp = _rt_parser()
    if any(a in ("-h", "--help") for a in argv):
        argparse.ArgumentParser(parents=[rp], description="Stage 3 options (all Stage 1/2 options follow)").print_help()
        print()
        idm._parse(["--help"])                      # prints the remaining options and exits
    rt_args, rest = rp.parse_known_args(argv)
    a = idm._parse(rest)
    for key, val in vars(rt_args).items():
        setattr(a, key, val)
    if a.out == "da_id_results":
        a.out = "da_id_rt_results"
    return a


def main(argv=None) -> int:
    a = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        if a.selftest:
            return selftest(a.solver)
        inst = (Instance.load(a.instance) if a.instance else
                da.make_benchmark_instance(a.seed, max_batches=a.max_batches, with_mt=not a.no_mt, with_bess=not a.no_bess))
        da.apply_instance_overrides(inst, a)
        mkt = IntradayMarket(a.id_cap_buy, a.id_cap_sell).validate()
        scen = (ScenarioSet.load(a.scenario_file) if a.scenario_file else
                ScenarioSet.generate(inst, a.scenarios, a.scenario_seed, a.price_sigma, a.level_sigma, a.load_sigma))
        rt = (RealTimeSet.load(a.rt_file) if a.rt_file else
              RealTimeSet.generate(inst, scen.n, a.rt_scenarios, a.rt_seed, a.rt_load_sigma, a.rt_load_rho,
                                   a.r_premium, a.r_discount, a.r_sigma))
        bal = BalancingMarket(a.imbalance_mode, a.bal_exclusive, a.max_imbalance, a.max_imbalance).validate()
        cfg = SchedulerConfig(start_step_h=a.start_step, unique_tasks=not a.allow_repeat_tasks,
                              solver=SolverSettings(a.solver, a.mip_gap, a.time_limit, a.threads, a.verbose))
        res = optimize_realtime(inst, scen, rt, mkt, bal, cfg, compute_vss_rt=a.vss_rt)
        print(res.summary())
        print(res.jobs[["machine", "position_n", "task", "start_time", "end_time", "power_mw", "units_out"]]
              .round(2).to_string(index=False))
        print(res.da_position.round(2).to_string(index=False))
        print(res.scenarios.round(2).to_string(index=False))
        out = res.save(a.out)
        if a.plot:
            plot_realtime(inst, res, out / "schedule_rt.png")
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
