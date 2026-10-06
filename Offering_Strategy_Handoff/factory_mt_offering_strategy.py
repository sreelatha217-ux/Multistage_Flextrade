#!/usr/bin/env python3
"""
factory_mt_offering_strategy.py
===============================

Strategic offering and bidding extension to the three-stage formulation in
factory_multistage_formulation-v3.md for the industrial prosumer: batch factory + Microturbine (MT) + BESS.
It sits on top of
factory_mt_da_scheduler.py, factory_mt_id_scheduler.py and factory_mt_rt_scheduler.py (all three are imported,
none is modified) and adds what those programs deliberately leave out:

    (E1) Price-quantity DAY-AHEAD OFFER / BID CURVES   P_sell,t,s ~ lam_DA,t,s (non-decreasing),
         P_buy,t,s ~ lam_DA,t,s (non-increasing), identical quantities for identical prices (non-anticipativity).
    (E2) Co-optimised UP-SPINNING-RESERVE offers from MT (10-min ramp), BESS (discharge headroom AND charge
         curtailment) and factory demand response (DR), with scenario SR prices, a monotone reserve curve and
         a Stage-3 activation model.
    (E3) Risk control:  max (1-beta)*E[Profit] + beta*CVaR_alpha(Profit)  on the joint (s, w) scenario tree.
    (E4) Analytics: bid package (JSON), self-schedule benchmark (value of price-responsive offering),
         value of reserve, E[profit]-CVaR efficient frontier, independent verification, self-test.

All money is in EUR.  The program is a profit MAXIMISATION as in the formulation; internally the negative is
minimised so that factory_mt_da_scheduler.solve_model (gap, status handling) can be reused unchanged.

Information structure (scenario s = joint DA/SR/ID price path, w | s = real-time branch)
----------------------------------------------------------------------------------------
    Stage 1  here-and-now      u,x,y (MT commitment); s[m,p,k] (batches); the BID PACKAGE: for every hour a curve
                               P_DAbuy[s,t], P_DAsell[s,t], R_total[s,t] that is a function of the price only
    Stage 2  wait-and-see (s)  P_IDbuy/sell, P_MT, P_ch, P_dis, SoC and the split of R_total over MT / BESS / DR
    Stage 3  wait-and-see (s,w) deployment fraction rho, load deviation eta, imbalance ratios r+, r-
                               -> imbalance D+, D-, activated reserve energy, SoC paths

    Profit(s,w) = sum_t [ lamDAsell*Psell - lamDAbuy*Pbuy                       day-ahead energy
                          + lamIDsell*Isell - lamIDbuy*Ibuy                     intraday re-trading
                          + lamSR*R                                             reserve CAPACITY payment
                          + lamACT*rho*R                                        reserve ACTIVATION energy
                          - (r+*lamDAbuy*D+ - r-*lamDAsell*D-)                  imbalance settlement
                          - C0*Pmin*u - SUC*x - SDC*y - sum_b C_b*[(1-rho)*P_b + rho*P_b^up]   MT
                          - C_TP*[w*(Pch - rho*Rch) + Pdis + rho*Rdis]          BESS ageing (incl. activation)
                          - c_DR*rho*R_DR ]                                     DR compensation

    supply + D+ - D-  = (1+eta) * (l_base + dl + L_batch)         (same balance as the RT program)
    physical import   = market net - rho*R_total + D+ - D-  in [-Q_sell, Q_buy]
    SoC[s,w,t]        = SoC[s,w,t-1] + eta_ch*(Pch - rho*Rch) - (Pdis + rho*Rdis)/eta_dis   in [SoC_min, SoC_max]

Why the activation terms cancel in the imbalance (derivation used in the code)
-----------------------------------------------------------------------------
    The TSO instructs every cleared reserve asset pro rata (fraction rho).  Physical import
        I = (1+eta)*L - [P_MT + rho*R_MT + P_dis + rho*R_dis - P_ch + rho*R_ch] - rho*R_DR
          = (1+eta)*L - (P_MT + P_dis - P_ch) - rho*R_total.
    The schedule the TSO expects is  M - rho*R_total  (market net M minus instructed activation), hence
        imbalance = I - (M - rho*R_total) = (1+eta)*L - (P_MT + P_dis - P_ch) - M,
    independent of rho.  Activation therefore changes (i) the physical exchange limits, (ii) the SoC paths,
    (iii) MT fuel / BESS ageing / DR cost and (iv) the activation revenue - never the imbalance volume.

Review of the v3 formulation file and the skill - inconsistencies and how this program resolves them
-----------------------------------------------------------------------------------------------------
 1. Pairwise conditions (md 2.1) are O(S^2) per hour.  Sorting the scenarios by price and constraining ADJACENT
    pairs is equivalent (transitivity) and O(S); equal prices (|dprice| <= tie_tol) get an equality.
 2. Skill 2.6 literally demands P_DA identical in ALL scenarios (non-anticipativity), which collapses the curve to
    one quantity (= price-taker self-schedule).  The md (2.3) applies it only to EQUAL prices.  The md reading is
    the strategic model; the skill reading is implemented as bidding="fixed" (benchmark).
 3. The md prices energy with ONE lam_DA, the data have a TOU buy/sell spread (sell = 55 % of buy).  Both tariffs
    are scaled by a common scenario factor; the sell curve is ordered by the sell price, the buy curve by the buy price.
 4. The md applies curves to energy only.  A reserve quantity that depends on the realised lam_SR is as anticipative
    as an energy quantity that depends on the realised lam_DA, so R_total[s,t] is made non-decreasing in lam_SR
    (reserve_curves=True).  The split of R_total over MT / BESS / DR is a Stage-2 decision.
 5. md 3.1-3.3 define downward reserve R^dn but nothing pays for it (revenue uses lam_SR^up only): dropped.
 6. md 3.3  R_DR <= L_batch (interrupt "flexible" batches) contradicts the skill (tasks are non-interruptible).
    R_DR <= dr_fraction * L_batch (default 10 %: temporary set-point reduction, batch not aborted) and every
    activated MWh costs dr_activation_cost (lost throughput, re-heating).
 7. RR_10min is not given: RR = RU * 10/60 = 3.33 MW for the benchmark MT (capped at Pmax - Pmin).
 8. md 3.2 ignores stored energy.  Added: end-of-hour deliverability  SoC - R_dis*tau/eta_dis >= SoC_min, and the
    SoC PATH in every (s,w) branch including activation (a 40 MW offer called 8 % of the time drains ~70 MWh/day).
 9. md Stage 3 "reserve deployment" has no deployment law, activation price or settlement.  Added: rho[s,w,t]
    (lognormal, mean 8 %), activation energy paid at act_price_ratio * lam_DA,sell[s,t] (default ratio 1.0).
10. md 5 balance has the sign error already documented in factory_mt_rt_scheduler.py (item 1): supply + D+ - D- = load.
11. md 4.3 CVaR is over "s"; with a (s,w) tree the tail must be taken over the JOINT scenarios.  A 5 % tail needs
    >= 20 joint scenarios to be more than one scenario wide - a warning is logged otherwise.
12. md 3.1 headroom P_MT + R_up <= Pmax*u is implied by the fuel-block bounds (sum of blocks = Pmax - Pmin) of the
    up-dispatch point and is not repeated.  MT ramp feasibility of the FULLY called point is added.
13. Fuel cost of activated MT energy: F is convex, so F(P+rho*R) <= (1-rho)*F(P) + rho*F(P+R) - the chord is an exact
    linear UPPER bound (exact for rho in {0,1}); the cost deviation is below the spread of the block costs (7 %).
14. Parameter drift between skill and DA program (MT blocks 35 vs 15 MW, SoC_max 200 vs 180) is resolved as before:
    the corrected DA-program values are used.  BESS exclusivity binaries are kept per (s,t) (md 3.2 uses v_dis).
15. Still open: perfect foresight of the ID price path inside a scenario, independence of rho and eta, no
    correlation of rho with prices (supply files with --offering-scenario-file / --deploy-file / --rt-file).

Outputs (OfferingResult.save): jobs.csv, offer_curves.csv, da_position.csv, scenarios.csv, joint_scenarios.csv,
hourly_offering.csv, bid_package.json, scenario_set.json, deployment_set.json, rt_set.json, summary.json
(with --plot: offering overview, schedule, market positions/prices, BESS, MT, imbalance, DA curves, reserve and profit figures;
with --compare: comparison.csv; with --frontier: frontier.csv).
The CLI additionally writes per-ID-scenario production reschedules; these are standalone schedule analyses and do not
change the offering-profit objective.

Requires: numpy, pandas, pyomo, highspy and the three scheduler modules in the same project.
Self-check (needs a solver):  uv run factory-mt-offering --selftest
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import factory_mt_da_scheduler as da
import factory_mt_rt_scheduler as rtm
from factory_mt_da_scheduler import (DT_H, EPS, Instance, InstanceValidationError, SchedulerConfig,
                                     SolverSettings, VerificationError)
from factory_mt_id_scheduler import IntradayMarket, _val
from factory_mt_rt_scheduler import BalancingMarket, RealTimeSet

__version__ = "1.0.0"
log = logging.getLogger("factory_mt_offering")
ASSETS = ("mt", "bess_dis", "bess_ch", "dr")
BIDDING = ("curve", "fixed")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ReserveMarket:
    """Up-spinning-reserve market and the physical limits of the three reserve providers."""
    enabled: bool = True
    assets: Tuple[str, ...] = ASSETS
    mt_ramp_minutes: float = 10.0            # RR_10min = ramp_up * minutes / 60
    dr_fraction: float = 0.10                # R_DR <= dr_fraction * L_batch
    dr_activation_cost_eur_mwh: float = 150.0
    act_price_ratio: float = 1.0             # activation energy price = ratio * lam_DA,sell[s,t]
    max_offer_mw: Optional[float] = None     # optional cap on R_total per hour
    bess_reserve_h: float = 1.0              # hours of stored energy that must stand behind R_dis

    def uses(self, asset: str) -> bool:
        return self.enabled and asset in self.assets

    def mt_rr_mw(self, mt: da.Microturbine) -> float:
        return min(mt.ramp_up_mw_h * self.mt_ramp_minutes / 60.0, mt.p_max_mw - mt.p_min_mw)

    def validate(self) -> "ReserveMarket":
        self.assets = tuple(self.assets)
        if not set(self.assets) <= set(ASSETS):
            raise InstanceValidationError(f"reserve assets must be a subset of {ASSETS}")
        if self.mt_ramp_minutes <= 0 or not (0 <= self.dr_fraction <= 1) or self.dr_activation_cost_eur_mwh < 0:
            raise InstanceValidationError("need mt_ramp_minutes > 0, 0 <= dr_fraction <= 1, DR cost >= 0")
        if self.act_price_ratio < 0 or self.bess_reserve_h <= 0 or (self.max_offer_mw is not None and self.max_offer_mw < 0):
            raise InstanceValidationError("need act_price_ratio >= 0, bess_reserve_h > 0, max_offer_mw >= 0")
        return self


@dataclass
class StrategyConfig:
    """Bidding rule and risk preference."""
    beta: float = 0.0                 # risk aversion in [0, 1]; 0 = risk neutral
    alpha: float = 0.95               # CVaR confidence level
    bidding: str = "curve"            # "curve": md 2.3 | "fixed": one quantity for every price scenario (skill 2.6 literal)
    reserve_curves: bool = True       # monotone reserve offer (issue 4)
    terminal_soc_paths: bool = True   # SoC_T >= SoC_0 in every (s,w) branch, not only on the no-call path
    tie_tol: float = 1e-6             # EUR/MWh: prices closer than this count as identical

    def validate(self) -> "StrategyConfig":
        if not (0 <= self.beta <= 1) or not (0 < self.alpha < 1):
            raise InstanceValidationError("need 0 <= beta <= 1 and 0 < alpha < 1")
        if self.bidding not in BIDDING:
            raise InstanceValidationError(f"bidding must be one of {BIDDING}")
        if self.tie_tol < 0:
            raise InstanceValidationError("tie_tol must be >= 0")
        return self


@dataclass
class OfferingScenarioSet:
    """Price scenarios, every array [S, T]: DA buy / sell price, up-SR capacity price, ID buy / sell price
    (EUR/MWh, EUR/MW/h) and additive base-load deviation (MW).  The DA price is what the offer curve is a function of."""
    prob: np.ndarray
    da_buy: np.ndarray
    da_sell: np.ndarray
    sr_up: np.ndarray
    id_buy: np.ndarray
    id_sell: np.ndarray
    load_dev_mw: np.ndarray
    _FIELDS = ("da_buy", "da_sell", "sr_up", "id_buy", "id_sell", "load_dev_mw")

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])

    def validate(self, inst: Instance) -> "OfferingScenarioSet":
        T = inst.horizon_h
        self.prob = da._arr(self.prob, None, "scenario prob", 1)
        S = self.prob.shape[0]
        if S == 0:
            raise InstanceValidationError("at least one scenario is required")
        for nm in self._FIELDS:
            a = da._arr(getattr(self, nm), None, nm, 2)
            if a.shape != (S, T):
                raise InstanceValidationError(f"{nm}: expected shape {(S, T)}, got {a.shape}")
            setattr(self, nm, a)
        if np.any(self.prob < 0) or abs(float(self.prob.sum()) - 1.0) > 1e-9:
            raise InstanceValidationError("scenario probabilities must be >= 0 and sum to 1")
        if any(np.any(getattr(self, nm) < 0) for nm in ("da_buy", "da_sell", "sr_up", "id_buy", "id_sell")):
            raise InstanceValidationError("negative prices are not supported")
        if np.any(self.da_sell > self.da_buy + EPS) or np.any(self.id_sell > self.id_buy + EPS):
            raise InstanceValidationError("sell price must not exceed the buy price (arbitrage loop) in DA and ID")
        if np.any(inst.base_load_mw[None, :] + self.load_dev_mw < -EPS):
            raise InstanceValidationError("base load plus deviation must stay >= 0")
        n_arb = int(np.sum((self.id_sell > self.da_buy + 1e-6) | (self.id_buy < self.da_sell - 1e-6)))
        if n_arb:
            log.warning("in %d (scenario, hour) cells the ID price leaves the DA buy/sell band: virtual DA/ID arbitrage "
                        "is profitable there and is limited only by Q and the ID caps", n_arb)
        return self

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return dict(prob=self.prob.tolist(), **{nm: getattr(self, nm).tolist() for nm in self._FIELDS})

    @classmethod
    def from_dict(cls, d: dict) -> "OfferingScenarioSet":
        try:
            return cls(np.asarray(d["prob"], float), *(np.asarray(d[nm], float) for nm in cls._FIELDS))
        except (KeyError, TypeError, ValueError) as err:
            raise InstanceValidationError(f"bad offering scenario schema: {err}") from err

    @classmethod
    def load(cls, path) -> "OfferingScenarioSet":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as err:
            raise InstanceValidationError(f"cannot read scenario file '{path}': {err}") from err

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---------------------------------------------------------------- generator
    @classmethod
    def generate(cls, inst: Instance, n: int = 6, seed: int = 7, da_sigma: float = 0.15, da_level_sigma: float = 0.05,
                 id_sigma: float = 0.08, sr_ratio: float = 0.12, sr_sigma: float = 0.20,
                 load_sigma_mw: float = 0.0) -> "OfferingScenarioSet":
        """lam_DA[s,t] = tariff[t] * f_DA (lognormal, mean 1, hourly + common daily shock);
        lam_ID = lam_DA[s,t] * g_ID (ID spread around the realised DA price);
        lam_SR[s,t] = sr_ratio * lam_DA,buy[s,t] * f_SR  (reserve is dearer when energy is dear).
        All shocks are antithetic (eps, -eps) so the sample means stay at the tariff; an odd n adds a central path."""
        if n < 1 or min(da_sigma, da_level_sigma, id_sigma, sr_ratio, sr_sigma, load_sigma_mw) < 0:
            raise InstanceValidationError("need n >= 1 and non-negative volatilities / ratio")
        T = inst.horizon_h
        rng = np.random.default_rng(seed)
        half = n // 2

        def anti(sig_h: float, sig_lvl: float) -> np.ndarray:
            e = sig_h * rng.standard_normal((half, T)) + sig_lvl * rng.standard_normal((half, 1))
            return np.vstack([e, -e, np.zeros((n - 2 * half, T))])

        f_da = np.exp(anti(da_sigma, da_level_sigma) - (da_sigma ** 2 + da_level_sigma ** 2) / 2.0)
        g_id = np.exp(anti(id_sigma, 0.0) - id_sigma ** 2 / 2.0)
        f_sr = np.exp(anti(sr_sigma, 0.0) - sr_sigma ** 2 / 2.0)
        dev = np.clip(load_sigma_mw * rng.standard_normal((half, T)), -inst.base_load_mw[None, :], inst.base_load_mw[None, :])
        dev = np.vstack([dev, -dev, np.zeros((n - 2 * half, T))])
        da_buy, da_sell = inst.price_buy_eur_mwh[None, :] * f_da, inst.price_sell_eur_mwh[None, :] * f_da
        return cls(np.full(n, 1.0 / n), da_buy, da_sell, sr_ratio * da_buy * f_sr, da_buy * g_id, da_sell * g_id,
                   dev).validate(inst)


@dataclass
class DeploymentSet:
    """Fraction rho in [0, 1] of the cleared up-reserve that the TSO actually calls, [S, W, T]."""
    rho: np.ndarray

    def validate(self, n_s: int, n_w: int, horizon: int) -> "DeploymentSet":
        self.rho = da._arr(self.rho, None, "rho", 3)
        if self.rho.shape != (n_s, n_w, horizon):
            raise InstanceValidationError(f"rho: expected shape {(n_s, n_w, horizon)}, got {self.rho.shape}")
        if np.any(self.rho < 0) or np.any(self.rho > 1):
            raise InstanceValidationError("deployment fraction rho must lie in [0, 1]")
        return self

    @classmethod
    def generate(cls, n_s: int, n_w: int, horizon: int, seed: int = 13, mean: float = 0.08,
                 sigma: float = 0.8) -> "DeploymentSet":
        if min(mean, sigma) < 0 or mean > 1:
            raise InstanceValidationError("need 0 <= mean <= 1 and sigma >= 0")
        rng = np.random.default_rng(seed)
        g = np.exp(sigma * rng.standard_normal((n_s, n_w, horizon)) - sigma ** 2 / 2.0)
        return cls(np.clip(mean * g, 0.0, 1.0))

    @classmethod
    def constant(cls, n_s: int, n_w: int, horizon: int, value: float = 0.0) -> "DeploymentSet":
        return cls(np.full((n_s, n_w, horizon), float(value)))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dict(rho=self.rho.tolist())))

    @classmethod
    def load(cls, path) -> "DeploymentSet":
        try:
            return cls(np.asarray(json.loads(Path(path).read_text())["rho"], float))
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as err:
            raise InstanceValidationError(f"cannot read deployment file '{path}': {err}") from err


def validate_tree(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet) -> None:
    """Shapes and value ranges of the (s, w) tree (the RT program's own check needs ID-only scenario objects)."""
    T, S = inst.horizon_h, scen.n
    rt.prob = da._arr(rt.prob, None, "rt prob", 2)
    if rt.prob.shape[0] != S or rt.prob.shape[1] < 1:
        raise InstanceValidationError(f"rt set has {rt.prob.shape[0]} ID scenarios, the price set has {S}")
    W = rt.prob.shape[1]
    for nm in ("load_rel_dev", "r_plus", "r_minus"):
        a = da._arr(getattr(rt, nm), None, nm, 3)
        if a.shape != (S, W, T):
            raise InstanceValidationError(f"{nm}: expected shape {(S, W, T)}, got {a.shape}")
        setattr(rt, nm, a)
    if np.any(rt.prob < 0) or np.abs(rt.prob.sum(axis=1) - 1.0).max() > 1e-9:
        raise InstanceValidationError("rt probabilities must be >= 0 and sum to 1 for every price scenario")
    if np.any(rt.load_rel_dev <= -1.0) or np.any(rt.r_plus < 1.0 - 1e-9) or np.any(rt.r_minus < -1e-9) \
            or np.any(rt.r_minus > 1.0 + 1e-9):
        raise InstanceValidationError("need eta > -1, r_plus >= 1 and 0 <= r_minus <= 1")
    dep.validate(S, W, T)


def shrink_instance(inst: Instance, n_tasks: int, demand_frac: float = 0.55, max_batches: int = 3) -> Instance:
    """Smaller plant for tests / demos: first n_tasks tasks, demand = demand_frac * total yield (spread evenly)."""
    n = min(n_tasks, len(inst.tasks))
    y = inst.yield_units[:n]
    return replace(inst, tasks=inst.tasks[:n], power_mw=inst.power_mw[:, :n], duration_h=inst.duration_h[:, :n],
                   yield_units=y, demand_units=np.full(inst.horizon_h, demand_frac * float(y.sum()) / inst.horizon_h),
                   max_batches_per_machine=max_batches).validate()


# --------------------------------------------------------------------------- #
# Offer-curve mathematics
# --------------------------------------------------------------------------- #
def monotone_chain(price: np.ndarray, tol: float = 1e-6) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Scenarios sorted by price -> (strictly ordered adjacent pairs (lo, hi), tied adjacent pairs).
    Constraining only adjacent pairs implies the all-pairs conditions of md 2.1 by transitivity."""
    order = np.argsort(price, kind="stable")
    ordered, tied = [], []
    for a, b in zip(order[:-1], order[1:]):
        (tied if price[b] - price[a] <= tol else ordered).append((int(a), int(b)))
    return ordered, tied


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_offering_model(inst: Instance, mkt: IntradayMarket, scen: OfferingScenarioSet, rt: RealTimeSet,
                         dep: DeploymentSet, rsv: ReserveMarket, bal: BalancingMarket, strat: StrategyConfig,
                         cfg: SchedulerConfig, cands: List[da.Candidate]) -> pyo.ConcreteModel:
    """Extensive form of the three-stage offering problem (minimises the negative risk-adjusted profit)."""
    T, S, W = inst.horizon_h, scen.n, rt.n_w
    mt, bess = inst.mt, inst.bess
    pi, pw, rho = scen.prob, rt.prob, dep.rho
    use_mt = rsv.uses("mt") and mt is not None
    use_dis = rsv.uses("bess_dis") and bess is not None
    use_ch = rsv.uses("bess_ch") and bess is not None
    use_dr = rsv.uses("dr")
    has_res = use_mt or use_dis or use_ch or use_dr
    m = pyo.ConcreteModel("FactoryOfferingStrategy")
    m.H, m.S, m.W = pyo.RangeSet(0, T - 1), pyo.RangeSet(0, S - 1), pyo.RangeSet(0, W - 1)

    # ================= Stage 1: commitment, batches, bid package =================
    m.Pbuy = pyo.Var(m.S, m.H, bounds=(0, inst.grid_limit_mw))              # DA bid curve (MW) at scenario price s
    m.Psell = pyo.Var(m.S, m.H, bounds=(0, inst.grid_sell_limit_mw))        # DA offer curve (MW)
    by_hour = da.add_batch_block(m, inst, cfg, cands)
    m.Lbatch = pyo.Expression(m.H, rule=lambda mm, t: pyo.quicksum(mw * mm.s[c] for c, mw, _ in by_hour[t]))
    if mt is not None:
        da.add_mt_commitment(m, inst)
    m.c_curve = pyo.ConstraintList()
    for t in range(T):
        if strat.bidding == "fixed":                                         # skill 2.6 literal: one quantity for all s
            for k in range(1, S):
                m.c_curve.add(m.Psell[k, t] == m.Psell[0, t])
                m.c_curve.add(m.Pbuy[k, t] == m.Pbuy[0, t])
            continue
        up, tie = monotone_chain(scen.da_sell[:, t], strat.tie_tol)          # offer curve: non-decreasing in lam_sell
        for a, b in up:
            m.c_curve.add(m.Psell[a, t] <= m.Psell[b, t])
        for a, b in tie:
            m.c_curve.add(m.Psell[a, t] == m.Psell[b, t])
        up, tie = monotone_chain(scen.da_buy[:, t], strat.tie_tol)           # bid curve: non-increasing in lam_buy
        for a, b in up:
            m.c_curve.add(m.Pbuy[a, t] >= m.Pbuy[b, t])
        for a, b in tie:
            m.c_curve.add(m.Pbuy[a, t] == m.Pbuy[b, t])

    # ================= Stage 2: re-trading, dispatch, reserve split =================
    m.Ibuy = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_buy_mw))
    m.Isell = pyo.Var(m.S, m.H, bounds=(0, mkt.cap_sell_mw))
    zero = lambda mm, k, t: 0.0
    Rmt = Rdis = Rch = Rdr = zero
    if mt is not None:
        nb = len(mt.block_width_mw)
        m.B = pyo.RangeSet(0, nb - 1)
        m.Pmt = pyo.Var(m.S, m.H, bounds=(0, mt.p_max_mw))
        m.Pb = pyo.Var(m.B, m.S, m.H, bounds=lambda mm, b, k, t: (0, mt.block_width_mw[b]))
        u0 = 1.0 if mt.initial_on else 0.0
        p0 = float(mt.initial_power_mw) if mt.initial_on else 0.0
        up_u = lambda mm, t: mm.u[t - 1] if t > 0 else u0
        pp = lambda mm, k, t: mm.Pmt[k, t - 1] if t > 0 else p0
        m.c_mt_sum = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] == mt.p_min_mw * mm.u[t]
                                    + pyo.quicksum(mm.Pb[b, k, t] for b in mm.B))
        m.c_mt_blk = pyo.Constraint(m.B, m.S, m.H, rule=lambda mm, b, k, t: mm.Pb[b, k, t] <= mt.block_width_mw[b] * mm.u[t])
        m.c_ru = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] - pp(mm, k, t)
                                <= mt.ramp_up_mw_h * up_u(mm, t) + mt.startup_ramp_mw_h * mm.x[t])
        m.c_rd = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: pp(mm, k, t) - mm.Pmt[k, t]
                                <= mt.ramp_down_mw_h * mm.u[t] + mt.shutdown_ramp_mw_h * mm.y[t])
        if use_mt:                                                           # MT up-reserve = 10-min ramp headroom
            rr = rsv.mt_rr_mw(mt)
            m.Rmt = pyo.Var(m.S, m.H, bounds=(0, rr))
            m.Pbu = pyo.Var(m.B, m.S, m.H, bounds=lambda mm, b, k, t: (0, mt.block_width_mw[b]))   # fuel blocks at P+R
            Rmt = lambda mm, k, t: mm.Rmt[k, t]
            m.c_rmt_on = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Rmt[k, t] <= rr * mm.u[t])
            m.c_mtu_sum = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] + mm.Rmt[k, t]
                                         == mt.p_min_mw * mm.u[t] + pyo.quicksum(mm.Pbu[b, k, t] for b in mm.B))
            m.c_mtu_blk = pyo.Constraint(m.B, m.S, m.H, rule=lambda mm, b, k, t: mm.Pbu[b, k, t]
                                         <= mt.block_width_mw[b] * mm.u[t])                          # => P + R <= Pmax*u
            m.c_ru_up = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pmt[k, t] + mm.Rmt[k, t] - pp(mm, k, t)
                                       <= mt.ramp_up_mw_h * up_u(mm, t) + mt.startup_ramp_mw_h * mm.x[t])
            m.c_rd_up = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: pyo.Constraint.Skip if t == 0 else
                                       mm.Pmt[k, t - 1] + mm.Rmt[k, t - 1] - mm.Pmt[k, t]
                                       <= mt.ramp_down_mw_h * mm.u[t] + mt.shutdown_ramp_mw_h * mm.y[t])
    if bess is not None:
        m.Pch = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.Pdis = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
        m.SoC = pyo.Var(m.S, m.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))      # no-call path
        soc_prev = lambda mm, k, t: mm.SoC[k, t - 1] if t > 0 else bess.soc_init_mwh
        m.c_soc = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.SoC[k, t] == soc_prev(mm, k, t)
                                 + bess.eta_ch * mm.Pch[k, t] * DT_H - mm.Pdis[k, t] * DT_H / bess.eta_dis)
        m.c_soc_term = pyo.Constraint(m.S, rule=lambda mm, k: mm.SoC[k, T - 1] >= bess.soc_init_mwh)
        if use_dis:
            m.Rdis = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
            Rdis = lambda mm, k, t: mm.Rdis[k, t]
            m.c_rdis_energy = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.SoC[k, t]
                                             - mm.Rdis[k, t] * rsv.bess_reserve_h / bess.eta_dis >= bess.soc_min_mwh)
        if use_ch:
            m.Rch = pyo.Var(m.S, m.H, bounds=(0, bess.p_max_mw))
            Rch = lambda mm, k, t: mm.Rch[k, t]
            m.c_rch = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Rch[k, t] <= mm.Pch[k, t])
        if bess.enforce_exclusive:
            m.vch = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.vdis = pyo.Var(m.S, m.H, domain=pyo.Binary)
            m.c_ch = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pch[k, t] <= bess.p_max_mw * mm.vch[k, t])
            m.c_dis = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pdis[k, t] + Rdis(mm, k, t)
                                     <= bess.p_max_mw * mm.vdis[k, t])
            m.c_excl = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.vch[k, t] + mm.vdis[k, t] <= 1)
        elif use_dis:
            m.c_dis_head = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Pdis[k, t] + mm.Rdis[k, t] <= bess.p_max_mw)
    if use_dr:
        m.Rdr = pyo.Var(m.S, m.H, bounds=(0, None))
        Rdr = lambda mm, k, t: mm.Rdr[k, t]
        m.c_rdr = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Rdr[k, t] <= rsv.dr_fraction * mm.Lbatch[t])

    # ---- reserve offer: total, cap and curve ----
    if has_res:
        m.Rtot = pyo.Expression(m.S, m.H, rule=lambda mm, k, t: 0.0 + Rmt(mm, k, t) + Rdis(mm, k, t)
                                + Rch(mm, k, t) + Rdr(mm, k, t))
        if rsv.max_offer_mw is not None:
            m.c_rcap = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: mm.Rtot[k, t] <= rsv.max_offer_mw)
        m.c_rcurve = pyo.ConstraintList()
        for t in range(T):
            if strat.bidding == "fixed":
                for k in range(1, S):
                    m.c_rcurve.add(m.Rtot[k, t] == m.Rtot[0, t])
            elif strat.reserve_curves:
                up, tie = monotone_chain(scen.sr_up[:, t], strat.tie_tol)
                for a, b in up:
                    m.c_rcurve.add(m.Rtot[a, t] <= m.Rtot[b, t])
                for a, b in tie:
                    m.c_rcurve.add(m.Rtot[a, t] == m.Rtot[b, t])
    Rtot = (lambda mm, k, t: mm.Rtot[k, t]) if has_res else zero

    # ================= Stage 3: imbalance, physical limits, SoC paths =================
    d_buy_max, d_sell_max = bal.bounds(inst)
    m.Dp = pyo.Var(m.S, m.W, m.H, bounds=(0, d_buy_max))                    # deficit bought
    m.Dm = pyo.Var(m.S, m.W, m.H, bounds=(0, d_sell_max))                   # surplus sold
    if bal.enforce_exclusive:
        m.zb = pyo.Var(m.S, m.W, m.H, domain=pyo.Binary)
        m.c_dp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dp[k, w, t] <= d_buy_max * mm.zb[k, w, t])
        m.c_dm = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.Dm[k, w, t] <= d_sell_max * (1 - mm.zb[k, w, t]))
    mt_out = (lambda mm, k, t: mm.Pmt[k, t]) if mt is not None else zero
    bess_out = (lambda mm, k, t: mm.Pdis[k, t] - mm.Pch[k, t]) if bess is not None else zero
    market = lambda mm, k, t: mm.Pbuy[k, t] - mm.Psell[k, t] + mm.Ibuy[k, t] - mm.Isell[k, t]
    supply = lambda mm, k, t: market(mm, k, t) + mt_out(mm, k, t) + bess_out(mm, k, t)
    load2 = lambda mm, k, t: inst.base_load_mw[t] + scen.load_dev_mw[k, t] + mm.Lbatch[t]
    m.c_power = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: supply(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]
                               == (1.0 + rt.load_rel_dev[k, w, t]) * load2(mm, k, t))
    m.c_mkt_imp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: market(mm, k, t) <= inst.grid_limit_mw)
    m.c_mkt_exp = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: -market(mm, k, t) <= inst.grid_sell_limit_mw)
    phys = lambda mm, k, w, t: market(mm, k, t) - rho[k, w, t] * Rtot(mm, k, t) + mm.Dp[k, w, t] - mm.Dm[k, w, t]
    m.c_phys_imp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: phys(mm, k, w, t) <= inst.grid_limit_mw)
    m.c_phys_exp = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: -phys(mm, k, w, t) <= inst.grid_sell_limit_mw)
    if has_res:                                                              # a full call must be deliverable (D = 0)
        m.c_full_call = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: -(market(mm, k, t) - mm.Rtot[k, t])
                                       <= inst.grid_sell_limit_mw)
    if bal.mode == "passive":
        e_eta = np.einsum("sw,swt->st", rt.prob, rt.load_rel_dev)
        m.c_plan = pyo.Constraint(m.S, m.H, rule=lambda mm, k, t: supply(mm, k, t) == (1.0 + e_eta[k, t]) * load2(mm, k, t))
    if bess is not None:                                                     # SoC of every branch incl. activation
        m.SoCw = pyo.Var(m.S, m.W, m.H, bounds=(bess.soc_min_mwh, bess.soc_max_mwh))
        socw_prev = lambda mm, k, w, t: mm.SoCw[k, w, t - 1] if t > 0 else bess.soc_init_mwh
        m.c_socw = pyo.Constraint(m.S, m.W, m.H, rule=lambda mm, k, w, t: mm.SoCw[k, w, t] == socw_prev(mm, k, w, t)
                                  + bess.eta_ch * (mm.Pch[k, t] - rho[k, w, t] * Rch(mm, k, t)) * DT_H
                                  - (mm.Pdis[k, t] + rho[k, w, t] * Rdis(mm, k, t)) * DT_H / bess.eta_dis)
        if strat.terminal_soc_paths:
            m.c_socw_term = pyo.Constraint(m.S, m.W, rule=lambda mm, k, w: mm.SoCw[k, w, T - 1] >= bess.soc_init_mwh)

    # ================= profit per joint scenario =================
    lam_act = rsv.act_price_ratio * scen.da_sell                             # [S, T]
    lam_p = rt.r_plus * scen.da_buy[:, None, :]                              # deficit price [S, W, T]
    lam_m = rt.r_minus * scen.da_sell[:, None, :]                            # surplus price [S, W, T]
    w_ch = 1.0 if (bess is not None and bess.degradation_basis == "throughput") else 0.0
    if mt is not None:
        m.commit = pyo.Expression(expr=pyo.quicksum(mt.base_cost_eur_mwh * mt.p_min_mw * m.u[t]
                                                    + mt.startup_cost_eur * m.x[t] + mt.shutdown_cost_eur * m.y[t]
                                                    for t in range(T)))
    else:
        m.commit = pyo.Expression(expr=0.0)

    def profit_rule(mm, k, w):
        terms = []
        for t in range(T):
            r = float(rho[k, w, t])
            terms.append(scen.da_sell[k, t] * mm.Psell[k, t] - scen.da_buy[k, t] * mm.Pbuy[k, t])
            terms.append(scen.id_sell[k, t] * mm.Isell[k, t] - scen.id_buy[k, t] * mm.Ibuy[k, t])
            if has_res:
                terms.append((scen.sr_up[k, t] + lam_act[k, t] * r) * mm.Rtot[k, t])
            terms.append(lam_m[k, w, t] * mm.Dm[k, w, t] - lam_p[k, w, t] * mm.Dp[k, w, t])
            if mt is not None:
                for b in range(len(mt.block_width_mw)):
                    c_b = mt.block_cost_eur_mwh[b]
                    terms.append(-c_b * ((1.0 - r) * mm.Pb[b, k, t] + (r * mm.Pbu[b, k, t] if use_mt else r * mm.Pb[b, k, t])))
            if bess is not None:
                terms.append(-bess.degradation_eur_mwh * (w_ch * (mm.Pch[k, t] - r * Rch(mm, k, t))
                                                          + mm.Pdis[k, t] + r * Rdis(mm, k, t)))
            if use_dr:
                terms.append(-rsv.dr_activation_cost_eur_mwh * r * mm.Rdr[k, t])
        return pyo.quicksum(terms) - mm.commit

    m.profit = pyo.Expression(m.S, m.W, rule=profit_rule)
    m.exp_profit = pyo.Expression(expr=pyo.quicksum(pi[k] * pw[k, w] * m.profit[k, w] for k in range(S) for w in range(W)))
    if strat.beta > 0:
        m.zeta = pyo.Var(domain=pyo.Reals)                                   # VaR_alpha of the profit
        m.eta = pyo.Var(m.S, m.W, domain=pyo.NonNegativeReals)              # shortfall below VaR
        m.c_cvar = pyo.Constraint(m.S, m.W, rule=lambda mm, k, w: mm.eta[k, w] >= mm.zeta - mm.profit[k, w])
        m.cvar = pyo.Expression(expr=m.zeta - (1.0 / (1.0 - strat.alpha))
                                * pyo.quicksum(pi[k] * pw[k, w] * m.eta[k, w] for k in range(S) for w in range(W)))
        risk_adj = (1.0 - strat.beta) * m.exp_profit + strat.beta * m.cvar
    else:
        risk_adj = m.exp_profit
    m.obj = pyo.Objective(expr=-risk_adj, sense=pyo.minimize)
    return m


# --------------------------------------------------------------------------- #
# Independent evaluation of a plan (used for extraction AND verification)
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    """Stage-1/2 decisions extracted from the model.  Arrays [S, T] unless noted."""
    u: np.ndarray                 # [T]
    Lbatch: np.ndarray            # [T]
    Pbuy: np.ndarray
    Psell: np.ndarray
    Ibuy: np.ndarray
    Isell: np.ndarray
    Pmt: np.ndarray
    Rmt: np.ndarray
    Pch: np.ndarray
    Pdis: np.ndarray
    Rdis: np.ndarray
    Rch: np.ndarray
    Rdr: np.ndarray

    @property
    def Rtot(self) -> np.ndarray:
        return self.Rmt + self.Rdis + self.Rch + self.Rdr


def _arr2(model: pyo.ConcreteModel, name: str, S: int, T: int) -> np.ndarray:
    if not hasattr(model, name):
        return np.zeros((S, T))
    var = getattr(model, name)
    a = np.array([[max(0.0, _val(var[k, t])) for t in range(T)] for k in range(S)])
    a[a < 1e-9] = 0.0
    return a


def extract_plan(inst: Instance, jobs: pd.DataFrame, model: pyo.ConcreteModel, S: int) -> Plan:
    T = inst.horizon_h
    u = np.array([round(_val(model.u[t])) for t in range(T)], dtype=float) if inst.mt is not None else np.zeros(T)
    lb = da._load_from_jobs(inst, jobs).batch_load_mw.values
    g = lambda nm: _arr2(model, nm, S, T)
    pmt = g("Pmt") * u[None, :]
    return Plan(u, lb, g("Pbuy"), g("Psell"), g("Ibuy"), g("Isell"), pmt, g("Rmt"), g("Pch"), g("Pdis"),
                g("Rdis"), g("Rch"), g("Rdr"))


def evaluate_plan(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                  rsv: ReserveMarket, p: Plan) -> Dict[str, np.ndarray]:
    """Recompute imbalance, activation, SoC paths, physical exchange and every profit component, [S, W, T],
    from the decisions only (no solver values) - the arithmetic of the module docstring."""
    S, W, T = scen.n, rt.n_w, inst.horizon_h
    mt, bess = inst.mt, inst.bess
    b3 = lambda a: np.broadcast_to(np.asarray(a)[:, None, :], (S, W, T))
    rho, eta = dep.rho, rt.load_rel_dev
    r_tot = p.Rtot
    load2 = inst.base_load_mw[None, :] + scen.load_dev_mw + p.Lbatch[None, :]
    market = p.Pbuy - p.Psell + p.Ibuy - p.Isell
    delta = (1.0 + eta) * b3(load2) - b3(p.Pmt + p.Pdis - p.Pch) - b3(market)       # imbalance, + = deficit
    d_plus, d_minus = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    lam_p, lam_m = rt.r_plus * scen.da_buy[:, None, :], rt.r_minus * scen.da_sell[:, None, :]
    # MT: chord bound between the planned point and the fully called point (exact for rho in {0, 1})
    if mt is not None:
        f0, left0 = da.mt_fuel_cost_by_hour(mt, p.Pmt, p.u[None, :])
        f1, left1 = da.mt_fuel_cost_by_hour(mt, p.Pmt + p.Rmt, p.u[None, :])
        f0, f1 = np.broadcast_to(f0, (S, T)), np.broadcast_to(f1, (S, T))
        fuel = (1.0 - rho) * b3(f0) + rho * b3(f1)
        u_prev = np.concatenate(([1.0 if mt.initial_on else 0.0], p.u[:-1]))
        ss = mt.startup_cost_eur * np.maximum(p.u - u_prev, 0.0) + mt.shutdown_cost_eur * np.maximum(u_prev - p.u, 0.0)
        ss = np.broadcast_to(ss[None, None, :], (S, W, T))
        mt_leftover = float(max(np.max(left0), np.max(left1)))
    else:
        fuel, ss, mt_leftover = np.zeros((S, W, T)), np.zeros((S, W, T)), 0.0
    if bess is not None:
        w_ch = 1.0 if bess.degradation_basis == "throughput" else 0.0
        c_bess = bess.degradation_eur_mwh * (w_ch * (b3(p.Pch) - rho * b3(p.Rch)) + b3(p.Pdis) + rho * b3(p.Rdis))
        soc = np.zeros((S, W, T))
        prev = np.full((S, W), bess.soc_init_mwh)
        for t in range(T):
            prev = (prev + bess.eta_ch * (p.Pch[:, None, t] - rho[:, :, t] * p.Rch[:, None, t]) * DT_H
                    - (p.Pdis[:, None, t] + rho[:, :, t] * p.Rdis[:, None, t]) * DT_H / bess.eta_dis)
            soc[:, :, t] = prev
        soc0 = np.zeros((S, T))
        prev0 = np.full(S, bess.soc_init_mwh)
        for t in range(T):
            prev0 = prev0 + bess.eta_ch * p.Pch[:, t] * DT_H - p.Pdis[:, t] * DT_H / bess.eta_dis
            soc0[:, t] = prev0
    else:
        c_bess, soc, soc0 = np.zeros((S, W, T)), np.zeros((S, W, T)), np.zeros((S, T))
    rev = dict(
        rev_da=b3(scen.da_sell * p.Psell - scen.da_buy * p.Pbuy),
        rev_id=b3(scen.id_sell * p.Isell - scen.id_buy * p.Ibuy),
        rev_sr=b3(scen.sr_up * r_tot),
        rev_act=rsv.act_price_ratio * b3(scen.da_sell) * rho * b3(r_tot),
        cost_bal=lam_p * d_plus - lam_m * d_minus,
        cost_mt=fuel + ss,
        cost_bess=c_bess,
        cost_dr=rsv.dr_activation_cost_eur_mwh * rho * b3(p.Rdr))
    profit_t = (rev["rev_da"] + rev["rev_id"] + rev["rev_sr"] + rev["rev_act"] - rev["cost_bal"] - rev["cost_mt"]
                - rev["cost_bess"] - rev["cost_dr"])
    return dict(rev, profit_t=profit_t, delta=delta, d_plus=d_plus, d_minus=d_minus, lam_p=lam_p, lam_m=lam_m,
                phys=b3(market) - rho * b3(r_tot) + delta, market=market, soc_path=soc, soc_base=soc0,
                load2=load2, load_rt=(1.0 + eta) * b3(load2), p_mt_rt=b3(p.Pmt) + rho * b3(p.Rmt),
                mt_leftover=np.array(mt_leftover))


def discrete_cvar_profit(profit: np.ndarray, prob: np.ndarray, alpha: float) -> float:
    """Lower-tail CVaR of the profit = minus the upper-tail CVaR of the loss (reuses the RT program's routine)."""
    return -rtm._cvar(-profit.ravel(), prob.ravel(), alpha)


def discrete_var_profit(profit: np.ndarray, prob: np.ndarray, alpha: float) -> float:
    """Largest v such that P(profit <= v) >= 1 - alpha (the VaR level the LP variable zeta converges to)."""
    order = np.argsort(profit.ravel())
    cum = np.cumsum(prob.ravel()[order])
    return float(profit.ravel()[order][np.searchsorted(cum, 1.0 - alpha - 1e-12)])


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_offering(inst: Instance, mkt: IntradayMarket, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                    rsv: ReserveMarket, bal: BalancingMarket, strat: StrategyConfig, cfg: SchedulerConfig,
                    jobs: pd.DataFrame, plan: Plan, ev: Dict[str, np.ndarray], model: pyo.ConcreteModel,
                    risk_adj_model: float, tol: float = 1e-4) -> Dict[str, object]:
    """Re-check every constraint family of the model from the extracted decisions only."""
    issues: List[str] = []
    S, W, T = scen.n, rt.n_w, inst.horizon_h
    mt, bess = inst.mt, inst.bess
    pi, pw = scen.prob, rt.prob
    # When (almost) all weight sits on the CVaR tail (beta -> 1) the solver may waste money in scenarios outside the tail
    # (fuel blocks out of merit order, simultaneous D+/D-).  The recomputation below always dispatches optimally, so it
    # can only be BETTER than the solver's own number; equality is then replaced by the one-sided check.
    tail_only = (1.0 - strat.beta) < 0.05
    # ---- factory, MT and BESS (no-call path) per price scenario, via the DA program's checker ----
    seen = set()
    for k in range(S):
        inst_k = replace(inst, base_load_mw=inst.base_load_mw + scen.load_dev_mw[k])
        h = da._build_hourly(inst_k, cfg, jobs, plan.Pmt[k], plan.u, plan.Pch[k], plan.Pdis[k])
        for msg in da.verify(inst_k, cfg, jobs, h, None)["issues"]:
            if msg.startswith("substation") or msg in seen:     # physical exchange is checked below on the call tree
                continue
            seen.add(msg)
            issues.append(f"[scenario {k}] {msg}")
    # ---- model imbalance equals the recomputed one ----
    dp_m = np.array([[[_val(model.Dp[k, w, t]) for t in range(T)] for w in range(W)] for k in range(S)])
    dm_m = np.array([[[_val(model.Dm[k, w, t]) for t in range(T)] for w in range(W)] for k in range(S)])
    if np.abs((dp_m - dm_m) - ev["delta"]).max() > 1e-3:
        issues.append("model imbalance differs from the recomputed (1+eta)*load - supply - market")
    if (not bal.enforce_exclusive) and (not tail_only) \
            and np.any((np.minimum(dp_m, dm_m) > 1e-6) & (ev["lam_p"] - ev["lam_m"] > 1e-9)):
        issues.append("deficit and surplus bought / sold in the same hour")
    if bal.mode == "passive":
        exp_delta = np.einsum("sw,swt->st", pw, ev["delta"])
        if np.abs(exp_delta).max() > 1e-3:
            issues.append("passive mode: expected imbalance is not zero")
    # ---- market and physical limits ----
    for nm, a, lim in (("DA buy", plan.Pbuy, inst.grid_limit_mw), ("DA sell", plan.Psell, inst.grid_sell_limit_mw),
                       ("ID buy", plan.Ibuy, mkt.cap_buy_mw), ("ID sell", plan.Isell, mkt.cap_sell_mw)):
        if a.max() > lim + tol:
            issues.append(f"{nm} above its limit ({a.max():.3f} > {lim:.3f} MW)")
    mk = ev["market"]
    if mk.max() > inst.grid_limit_mw + tol or -mk.min() > inst.grid_sell_limit_mw + tol:
        issues.append("market net position exceeds the substation limits")
    if ev["phys"].max() > inst.grid_limit_mw + 1e-3 or -ev["phys"].min() > inst.grid_sell_limit_mw + 1e-3:
        issues.append("physical grid exchange (market - rho*R + imbalance) exceeds the substation limits")
    r_tot = plan.Rtot
    if r_tot.max() > 0 and (-(mk - r_tot)).max() > inst.grid_sell_limit_mw + 1e-3:
        issues.append("a full reserve call would exceed the export limit")
    # ---- offer / bid / reserve curves ----
    for t in range(T):
        pairs = [("DA offer", plan.Psell[:, t], scen.da_sell[:, t], +1), ("DA bid", plan.Pbuy[:, t], scen.da_buy[:, t], -1)]
        if r_tot.max() > 0 or rsv.enabled:
            pairs.append(("reserve offer", r_tot[:, t], scen.sr_up[:, t], +1))
        for nm, q, price, sign in pairs:
            if strat.bidding == "fixed":
                if np.ptp(q) > 1e-3:
                    issues.append(f"hour {t}: {nm} differs between scenarios in fixed bidding mode")
                continue
            if nm == "reserve offer" and not strat.reserve_curves:
                continue
            up, tie = monotone_chain(price, strat.tie_tol)
            if any(sign * (q[b] - q[a]) < -1e-3 for a, b in up):
                issues.append(f"hour {t}: {nm} curve is not monotone in price")
            if any(abs(q[a] - q[b]) > 1e-3 for a, b in tie):
                issues.append(f"hour {t}: {nm} violates non-anticipativity (equal prices, different quantities)")
    # ---- reserve physics ----
    if mt is not None and plan.Rmt.max() > 0:
        rr = rsv.mt_rr_mw(mt)
        if plan.Rmt.max() > rr + tol or np.any(plan.Rmt > rr * plan.u[None, :] + tol):
            issues.append("MT reserve above the 10-minute ramp bound or offered while offline")
        if np.any(plan.Pmt + plan.Rmt > mt.p_max_mw * plan.u[None, :] + tol):
            issues.append("MT output plus reserve above Pmax*u")
        if ev["mt_leftover"] > tol:
            issues.append("MT output plus reserve above the capacity of the fuel blocks")
        u_prev = np.concatenate(([1.0 if mt.initial_on else 0.0], plan.u[:-1]))
        p_prev = np.concatenate((np.full((S, 1), mt.initial_power_mw if mt.initial_on else 0.0), plan.Pmt[:, :-1]), axis=1)
        r_prev = np.concatenate((np.zeros((S, 1)), plan.Rmt[:, :-1]), axis=1)
        x = np.maximum(plan.u - u_prev, 0.0)
        y = np.maximum(u_prev - plan.u, 0.0)
        if np.any(plan.Pmt + plan.Rmt - p_prev > mt.ramp_up_mw_h * u_prev + mt.startup_ramp_mw_h * x + tol):
            issues.append("MT ramp-up limit violated at the fully called point")
        if np.any(p_prev + r_prev - plan.Pmt > mt.ramp_down_mw_h * plan.u + mt.shutdown_ramp_mw_h * y + tol):
            issues.append("MT ramp-down limit violated after a full call")
    if plan.Rdr.max() > 0 and np.any(plan.Rdr > rsv.dr_fraction * plan.Lbatch[None, :] + tol):
        issues.append("DR reserve above dr_fraction * batch load")
    if rsv.max_offer_mw is not None and r_tot.max() > rsv.max_offer_mw + tol:
        issues.append("total reserve above the market cap")
    if bess is not None:
        if np.any(plan.Rch > plan.Pch + tol):
            issues.append("BESS charging-mode reserve above the charging power")
        if bess.enforce_exclusive:
            if np.any(plan.Pdis + plan.Rdis > bess.p_max_mw + tol):
                issues.append("BESS discharge power plus reserve above the power rating")
            if np.any((plan.Rdis > tol) & (plan.Pch > tol)):
                issues.append("BESS discharge-mode reserve offered while charging")
        if np.any(ev["soc_base"] - plan.Rdis * rsv.bess_reserve_h / bess.eta_dis < bess.soc_min_mwh - 1e-3):
            issues.append("BESS cannot deliver the offered reserve for the required duration")
        sp = ev["soc_path"]
        if sp.min() < bess.soc_min_mwh - 1e-3 or sp.max() > bess.soc_max_mwh + 1e-3:
            issues.append("BESS SoC bounds violated in a real-time branch")
        if strat.terminal_soc_paths and sp[:, :, -1].min() < bess.soc_init_mwh - 1e-3:
            issues.append("BESS terminal SoC below the initial SoC in a real-time branch")
        soc_m = np.array([[[_val(model.SoCw[k, w, t]) for t in range(T)] for w in range(W)] for k in range(S)])
        if np.abs(soc_m - sp).max() > 1e-3:
            issues.append("model SoC path differs from the independent recomputation")
    # ---- objective ----
    profit = ev["profit_t"].sum(axis=2)
    joint = pi[:, None] * pw
    exp_profit = float((joint * profit).sum())
    cvar = discrete_cvar_profit(profit, joint, strat.alpha)
    risk_adj = (1 - strat.beta) * exp_profit + strat.beta * cvar
    for nm, mine, theirs in (("expected profit", exp_profit, float(pyo.value(model.exp_profit))),
                             ("risk-adjusted profit", risk_adj, risk_adj_model)):
        d, tl = mine - theirs, max(1e-2, 1e-5 * abs(mine))
        if d < -tl or ((not tail_only) and d > tl):
            issues.append(f"{nm} mismatch: recomputed {mine:.4f} vs solver {theirs:.4f}")
    return dict(passed=not issues, issues=issues, recomputed_expected_profit_eur=exp_profit,
                recomputed_cvar_eur=cvar, recomputed_risk_adjusted_eur=risk_adj)


# --------------------------------------------------------------------------- #
# Result packaging
# --------------------------------------------------------------------------- #
@dataclass
class OfferingResult:
    status: str
    risk_adjusted_profit_eur: float
    expected_profit_eur: float
    cvar_profit_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    jobs: pd.DataFrame
    offer_curves: pd.DataFrame           # hour x price scenario: the bid package in table form
    da_position: pd.DataFrame            # per hour: expectations of the offer and of Stage 2 / 3
    scenarios: pd.DataFrame              # per price scenario (conditional expectation over RT)
    joint: pd.DataFrame                  # per (s, w)
    hourly: pd.DataFrame                 # long table s x w x hour
    kpis: Dict[str, float]
    verification: Dict[str, object]
    scenario_set: OfferingScenarioSet
    rt_set: RealTimeSet
    dep_set: DeploymentSet
    strategy: StrategyConfig
    comparison: Optional[pd.DataFrame] = None
    frontier: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        k = self.kpis
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        s = self.strategy
        lines = [
            f"status={self.status}  risk-adjusted profit={self.risk_adjusted_profit_eur:,.2f} EUR  gap={gap}  "
            f"time={self.solve_time_s:.1f}s  tree={int(k['n_scenarios'])} price x {int(k['n_rt'])} RT  "
            f"bidding={s.bidding}  beta={s.beta:g}  alpha={s.alpha:g}",
            f"E[profit]={self.expected_profit_eur:,.2f}  E[net cost]={-self.expected_profit_eur:,.2f} EUR  "
            f"CVaR{int(100 * s.alpha)}={self.cvar_profit_eur:,.2f}  "
            f"VaR={k['var_profit_eur']:,.2f}  std={k['profit_std_eur']:,.2f}  min={k['profit_min_eur']:,.2f}  "
            f"max={k['profit_max_eur']:,.2f} EUR",
            f"E[revenue]: DA={k['exp_rev_da_eur']:,.2f}  ID={k['exp_rev_id_eur']:,.2f}  reserve capacity={k['exp_rev_sr_eur']:,.2f}  "
            f"activation={k['exp_rev_act_eur']:,.2f}",
            f"E[cost]: imbalance={k['exp_cost_bal_eur']:,.2f}  MT={k['exp_cost_mt_eur']:,.2f}  "
            f"BESS={k['exp_cost_bess_eur']:,.2f}  DR={k['exp_cost_dr_eur']:,.2f} EUR",
            f"reserve: avg offer={k['avg_reserve_offer_mw']:.1f} MW over {int(k['reserve_hours'])} h "
            f"(MT {k['avg_r_mt_mw']:.1f} / BESS-dis {k['avg_r_bess_dis_mw']:.1f} / BESS-ch {k['avg_r_bess_ch_mw']:.1f} / "
            f"DR {k['avg_r_dr_mw']:.1f} MW)  expected activation={k['exp_activated_mwh']:.1f} MWh",
            f"Stage 1: MT on {int(k['mt_on_hours'])} h, {int(k['mt_starts'])} start(s)  batches={int(k['n_batches'])}  "
            f"units={k['units_produced']:.0f}",
            f"verification={'PASS' if self.verification['passed'] else 'FAIL'}"]
        if self.comparison is not None:
            lines.append("strategy comparison:\n" + self.comparison.round(2).to_string(index=False))
        if self.frontier is not None:
            lines.append("risk frontier:\n" + self.frontier.round(2).to_string(index=False))
        return "\n".join(lines)

    def bid_package(self) -> dict:
        """What would be submitted to the market operator: per hour the DA offer curve (sell), the DA bid curve
        (buy) and the up-reserve offer curve as (price, MW) step points with equal prices merged."""
        oc, pkg = self.offer_curves, {}

        def steps(g: pd.DataFrame, pcol: str, qcol: str) -> List[dict]:
            pts = g.groupby(pcol, sort=True)[qcol].max().reset_index()
            return [dict(price_eur_mwh=round(float(r[pcol]), 4), quantity_mw=round(float(r[qcol]), 4)) for _, r in pts.iterrows()]

        for t, g in oc.groupby("hour"):
            pkg[int(t)] = dict(energy_offer_sell=steps(g, "price_da_sell_eur_mwh", "p_da_sell_mw"),
                               energy_bid_buy=steps(g, "price_da_buy_eur_mwh", "p_da_buy_mw"),
                               reserve_offer_up=steps(g, "price_sr_up_eur_mw_h", "r_total_mw"),
                               mt_online=int(g.mt_on.iloc[0]))
        return dict(version=__version__, currency="EUR", bidding=self.strategy.bidding, beta=self.strategy.beta,
                    alpha=self.strategy.alpha, hours=pkg)

    def save(self, outdir) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        for nm in ("jobs", "offer_curves", "da_position", "scenarios"):
            getattr(self, nm).to_csv(out / f"{nm}.csv", index=False)
        self.joint.to_csv(out / "joint_scenarios.csv", index=False)
        self.hourly.to_csv(out / "hourly_offering.csv", index=False)
        (out / "bid_package.json").write_text(json.dumps(self.bid_package(), indent=2))
        self.scenario_set.save(out / "scenario_set.json")
        self.dep_set.save(out / "deployment_set.json")
        self.rt_set.save(out / "rt_set.json")
        if self.comparison is not None:
            self.comparison.to_csv(out / "comparison.csv", index=False)
        if self.frontier is not None:
            self.frontier.to_csv(out / "frontier.csv", index=False)
        meta = dict(version=__version__, status=self.status, risk_adjusted_profit_eur=self.risk_adjusted_profit_eur,
                expected_profit_eur=self.expected_profit_eur, expected_net_cost_eur=-self.expected_profit_eur,
                cvar_profit_eur=self.cvar_profit_eur,
                    mip_gap=self.mip_gap, solve_time_s=self.solve_time_s, model_stats=self.model_stats,
                    strategy=self.strategy.__dict__, kpis=self.kpis, verification=self.verification)
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
        return out


def extract_offering(inst: Instance, mkt: IntradayMarket, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                     rsv: ReserveMarket, bal: BalancingMarket, strat: StrategyConfig, cfg: SchedulerConfig,
                     model: pyo.ConcreteModel, cands: List[da.Candidate], info: Dict[str, object]) -> OfferingResult:
    S, W, T = scen.n, rt.n_w, inst.horizon_h
    pi, pw = scen.prob, rt.prob
    joint = pi[:, None] * pw
    jobs = da.jobs_from_starts(inst, cands, model)
    plan = extract_plan(inst, jobs, model, S)
    ev = evaluate_plan(inst, scen, rt, dep, rsv, plan)
    profit = ev["profit_t"].sum(axis=2)                                      # [S, W]
    exp_profit = float((joint * profit).sum())
    cvar = discrete_cvar_profit(profit, joint, strat.alpha)
    risk_adj_model = -float(pyo.value(model.obj))
    ver = verify_offering(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, jobs, plan, ev, model, risk_adj_model)
    if not ver["passed"]:
        msg = "; ".join(ver["issues"])
        if cfg.strict_verification:
            raise VerificationError(f"solution failed independent verification: {msg}")
        log.error("verification failed: %s", msg)
    if joint.size * (1 - strat.alpha) < 1.5 and strat.beta > 0:
        log.warning("only %d joint scenarios: the %.0f%% tail is %.2f scenarios wide; use more scenarios for a "
                    "meaningful CVaR", joint.size, 100 * (1 - strat.alpha), joint.size * (1 - strat.alpha))

    hrs = np.arange(T, dtype=float)
    jobs["energy_cost_eur"] = [float((np.clip(np.minimum(r.end_h, hrs + 1) - np.maximum(r.start_h, hrs), 0, 1)
                                      * inst.price_buy_eur_mwh).sum() * r.power_mw) for r in jobs.itertuples()]
    # ---- offer curves (the bid package) ----
    rows = []
    for t in range(T):
        for rank, k in enumerate(np.argsort(scen.da_sell[:, t], kind="stable")):
            rows.append(dict(hour=t, price_rank=rank, scenario=int(k), prob=float(pi[k]),
                             price_da_sell_eur_mwh=scen.da_sell[k, t], price_da_buy_eur_mwh=scen.da_buy[k, t],
                             price_sr_up_eur_mw_h=scen.sr_up[k, t], p_da_sell_mw=plan.Psell[k, t], p_da_buy_mw=plan.Pbuy[k, t],
                             p_da_net_import_mw=plan.Pbuy[k, t] - plan.Psell[k, t], r_total_mw=plan.Rtot[k, t],
                             r_mt_mw=plan.Rmt[k, t], r_bess_dis_mw=plan.Rdis[k, t], r_bess_ch_mw=plan.Rch[k, t],
                             r_dr_mw=plan.Rdr[k, t], mt_on=int(plan.u[t])))
    offer = pd.DataFrame(rows)
    # ---- long hourly table ----
    ss, ww, tt = np.meshgrid(np.arange(S), np.arange(W), np.arange(T), indexing="ij")
    b3 = lambda a: np.broadcast_to(np.asarray(a)[:, None, :], (S, W, T))
    cols = dict(scenario=ss, rt=ww, hour=tt, prob_joint=joint[:, :, None] * np.ones((1, 1, T)),
                price_da_buy_eur_mwh=b3(scen.da_buy), price_da_sell_eur_mwh=b3(scen.da_sell),
                price_sr_up_eur_mw_h=b3(scen.sr_up), price_id_buy_eur_mwh=b3(scen.id_buy),
                price_id_sell_eur_mwh=b3(scen.id_sell), price_bal_deficit_eur_mwh=ev["lam_p"],
                price_bal_surplus_eur_mwh=ev["lam_m"], rho=dep.rho, rt_load_rel_dev=rt.load_rel_dev,
                p_da_buy_mw=b3(plan.Pbuy), p_da_sell_mw=b3(plan.Psell), p_id_buy_mw=b3(plan.Ibuy),
                p_id_sell_mw=b3(plan.Isell), p_mt_mw=b3(plan.Pmt), p_mt_rt_mw=ev["p_mt_rt"],
                p_bess_ch_mw=b3(plan.Pch), p_bess_dis_mw=b3(plan.Pdis), soc_path_mwh=ev["soc_path"],
                r_mt_mw=b3(plan.Rmt), r_bess_dis_mw=b3(plan.Rdis), r_bess_ch_mw=b3(plan.Rch), r_dr_mw=b3(plan.Rdr),
                r_total_mw=b3(plan.Rtot), load_rt_mw=ev["load_rt"], imb_deficit_mw=ev["d_plus"],
                imb_surplus_mw=ev["d_minus"], p_phys_import_mw=ev["phys"], rev_da_eur=ev["rev_da"],
                rev_id_eur=ev["rev_id"], rev_sr_eur=ev["rev_sr"], rev_act_eur=ev["rev_act"],
                cost_bal_eur=ev["cost_bal"], cost_mt_eur=ev["cost_mt"], cost_bess_eur=ev["cost_bess"],
                cost_dr_eur=ev["cost_dr"], profit_hour_eur=ev["profit_t"])
    hourly = pd.DataFrame({k: np.asarray(v).ravel() for k, v in cols.items()})

    E = lambda a: np.einsum("sw,swt->t", joint, np.asarray(a))               # expectation over the tree -> [T]
    Es = lambda a: np.einsum("s,st->t", pi, np.asarray(a))
    act_mwh = E(dep.rho * b3(plan.Rtot))
    pos = pd.DataFrame(dict(hour=np.arange(T), exp_price_da_buy_eur_mwh=Es(scen.da_buy), exp_price_da_sell_eur_mwh=Es(scen.da_sell),
                            exp_price_sr_up_eur_mw_h=Es(scen.sr_up), exp_p_da_buy_mw=Es(plan.Pbuy), exp_p_da_sell_mw=Es(plan.Psell),
                            mt_on=plan.u.astype(int), exp_p_id_buy_mw=Es(plan.Ibuy), exp_p_id_sell_mw=Es(plan.Isell),
                            exp_p_mt_mw=Es(plan.Pmt), exp_bess_ch_mw=Es(plan.Pch), exp_bess_dis_mw=Es(plan.Pdis),
                            exp_r_total_mw=Es(plan.Rtot), exp_r_mt_mw=Es(plan.Rmt), exp_r_bess_dis_mw=Es(plan.Rdis),
                            exp_r_bess_ch_mw=Es(plan.Rch), exp_r_dr_mw=Es(plan.Rdr), exp_activated_mwh=act_mwh,
                            exp_soc_mwh=E(ev["soc_path"]), exp_imb_deficit_mw=E(ev["d_plus"]),
                            exp_imb_surplus_mw=E(ev["d_minus"]), exp_profit_eur=E(ev["profit_t"])))
    comp = ("rev_da", "rev_id", "rev_sr", "rev_act", "cost_bal", "cost_mt", "cost_bess", "cost_dr")
    sc = pd.DataFrame([dict(scenario=k, prob=float(pi[k]), profit_eur=float(pw[k] @ profit[k]),
                            profit_worst_rt_eur=float(profit[k].min()),
                            **{f"{c}_eur": float(pw[k] @ ev[c][k].sum(axis=1)) for c in comp},
                            reserve_mwh=float(plan.Rtot[k].sum()), id_net_buy_mwh=float((plan.Ibuy[k] - plan.Isell[k]).sum()))
                       for k in range(S)])
    jt = pd.DataFrame([dict(scenario=k, rt=w, prob=float(joint[k, w]), profit_eur=float(profit[k, w]),
                            **{f"{c}_eur": float(ev[c][k, w].sum()) for c in comp},
                            deficit_mwh=float(ev["d_plus"][k, w].sum()), surplus_mwh=float(ev["d_minus"][k, w].sum()),
                            activated_mwh=float((dep.rho[k, w] * plan.Rtot[k]).sum())) for k in range(S) for w in range(W)])
    tot = lambda c: float((joint * ev[c].sum(axis=2)).sum())
    mean = exp_profit
    fl = profit.ravel()
    r_hours = int((plan.Rtot.max(axis=0) > 1e-6).sum())
    kpis = dict(
        n_scenarios=float(S), n_rt=float(W), n_joint=float(S * W), n_batches=float(len(jobs)),
        units_produced=float(jobs.units_out.sum()), mt_on_hours=float(plan.u.sum()),
        mt_starts=float(np.maximum(np.diff(np.concatenate(([1.0 if (inst.mt and inst.mt.initial_on) else 0.0], plan.u))), 0).sum()),
        exp_profit_eur=exp_profit, cvar_profit_eur=cvar, var_profit_eur=discrete_var_profit(profit, joint, strat.alpha),
        profit_std_eur=float(math.sqrt(max(0.0, (joint.ravel() * (fl - mean) ** 2).sum()))),
        profit_min_eur=float(fl.min()), profit_max_eur=float(fl.max()),
        exp_rev_da_eur=tot("rev_da"), exp_rev_id_eur=tot("rev_id"), exp_rev_sr_eur=tot("rev_sr"),
        exp_rev_act_eur=tot("rev_act"), exp_cost_bal_eur=tot("cost_bal"), exp_cost_mt_eur=tot("cost_mt"),
        exp_cost_bess_eur=tot("cost_bess"), exp_cost_dr_eur=tot("cost_dr"),
        bess_degradation_eur_mwh=float(inst.bess.degradation_eur_mwh) if inst.bess is not None else 0.0,
        bess_degradation_basis=inst.bess.degradation_basis if inst.bess is not None else "none",
        reserve_hours=float(r_hours), avg_reserve_offer_mw=float(pi @ plan.Rtot.mean(axis=1)),
        avg_r_mt_mw=float(pi @ plan.Rmt.mean(axis=1)), avg_r_bess_dis_mw=float(pi @ plan.Rdis.mean(axis=1)),
        avg_r_bess_ch_mw=float(pi @ plan.Rch.mean(axis=1)), avg_r_dr_mw=float(pi @ plan.Rdr.mean(axis=1)),
        exp_activated_mwh=float(act_mwh.sum()), exp_deficit_mwh=float(E(ev["d_plus"]).sum()),
        exp_surplus_mwh=float(E(ev["d_minus"]).sum()))
    return OfferingResult(info["status"], risk_adj_model, exp_profit, cvar, info["gap"], info["time"], info["stats"], jobs,
                          offer, pos, sc, jt, hourly, kpis, ver, scen, rt, dep, strat)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def optimize_offering(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                      mkt: Optional[IntradayMarket] = None, rsv: Optional[ReserveMarket] = None,
                      bal: Optional[BalancingMarket] = None, strat: Optional[StrategyConfig] = None,
                      cfg: Optional[SchedulerConfig] = None) -> OfferingResult:
    """Validate -> candidates -> extensive-form MILP -> solve -> independent verification -> package results."""
    cfg, mkt = cfg or SchedulerConfig(), (mkt or IntradayMarket()).validate()
    rsv, bal, strat = (rsv or ReserveMarket()).validate(), (bal or BalancingMarket()).validate(), (strat or StrategyConfig()).validate()
    inst.validate()
    cfg.check(inst)
    scen.validate(inst)
    validate_tree(inst, scen, rt, dep)
    cands = da.build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts, %d price x %d RT scenarios, bidding=%s, reserve=%s, beta=%g", len(cands), scen.n,
             rt.n_w, strat.bidding, "on" if rsv.enabled else "off", strat.beta)
    model = build_offering_model(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, cands)
    info = da.solve_model(model, cfg.solver)
    return extract_offering(inst, mkt, scen, rt, dep, rsv, bal, strat, cfg, model, cands, info)


def compare_strategies(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                       mkt: IntradayMarket, rsv: ReserveMarket, bal: BalancingMarket, strat: StrategyConfig,
                       cfg: SchedulerConfig, main: Optional[OfferingResult] = None) -> pd.DataFrame:
    """Same tree, same risk preference, three bidding rules.  Each is a restriction of the next, so
    self-schedule <= offering curves <= offering curves + reserve (up to the MIP gap)."""
    no_res = replace(rsv, enabled=False)
    runs = [("self-schedule (one quantity for all prices, no reserve)", replace(strat, bidding="fixed"), no_res, None),
            ("offer/bid curves, energy only", replace(strat, bidding="curve"), no_res, None),
            ("offer/bid curves + spinning reserve", strat, rsv, main)]
    rows = []
    for name, st, rs, done in runs:
        r = done or optimize_offering(inst, scen, rt, dep, mkt, rs, bal, st, cfg)
        rows.append(dict(strategy=name, risk_adjusted_eur=r.risk_adjusted_profit_eur, exp_profit_eur=r.expected_profit_eur,
                         cvar_eur=r.cvar_profit_eur, std_eur=r.kpis["profit_std_eur"],
                         reserve_revenue_eur=r.kpis["exp_rev_sr_eur"] + r.kpis["exp_rev_act_eur"], gap_pct=100 * (r.mip_gap or 0.0)))
    df = pd.DataFrame(rows)
    df["gain_vs_self_schedule_eur"] = df.risk_adjusted_eur - df.risk_adjusted_eur.iloc[0]
    return df


def compare_market_stages(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet,
                          mkt: IntradayMarket, rsv: ReserveMarket, bal: BalancingMarket,
                          strat: StrategyConfig, cfg: SchedulerConfig,
                          main: Optional[OfferingResult] = None) -> pd.DataFrame:
    """Compare DA-only, DA+ID, and DA+ID+up-reserve on the same scenario tree.

    RT imbalance settlement remains active in every case; it is not treated as an
    opt-in market because deviations are settled whether or not reserve is offered.
    """
    no_id = IntradayMarket(0.0, 0.0).validate()
    no_reserve = replace(rsv, enabled=False)
    runs = [
        ("DA only + RT imbalance settlement", no_id, no_reserve, None),
        ("DA + ID + RT imbalance settlement", mkt, no_reserve, None),
        ("DA + ID + RT settlement + up-reserve", mkt, rsv, main),
    ]
    rows = []
    for name, market, reserve, done in runs:
        result = done or optimize_offering(inst, scen, rt, dep, market, reserve, bal, strat, cfg)
        rows.append(dict(
            market_stage=name,
            expected_net_cost_eur=-result.expected_profit_eur,
            expected_profit_eur=result.expected_profit_eur,
            cvar_profit_eur=result.cvar_profit_eur,
            reserve_revenue_eur=result.kpis["exp_rev_sr_eur"] + result.kpis["exp_rev_act_eur"],
            mip_gap=result.mip_gap,
        ))
    frame = pd.DataFrame(rows)
    frame["savings_vs_da_only_eur"] = frame.expected_net_cost_eur.iloc[0] - frame.expected_net_cost_eur
    return frame


def risk_frontier(inst: Instance, scen: OfferingScenarioSet, rt: RealTimeSet, dep: DeploymentSet, mkt: IntradayMarket,
                  rsv: ReserveMarket, bal: BalancingMarket, strat: StrategyConfig, cfg: SchedulerConfig,
                  betas: Sequence[float]) -> pd.DataFrame:
    """E[profit] / CVaR pairs for a grid of risk-aversion weights (efficient frontier)."""
    rows = []
    for b in betas:
        r = optimize_offering(inst, scen, rt, dep, mkt, rsv, bal, replace(strat, beta=float(b)), cfg)
        rows.append(dict(beta=float(b), exp_profit_eur=r.expected_profit_eur, cvar_eur=r.cvar_profit_eur,
                         std_eur=r.kpis["profit_std_eur"], worst_eur=r.kpis["profit_min_eur"],
                         avg_reserve_mw=r.kpis["avg_reserve_offer_mw"]))
    return pd.DataFrame(rows)


def plot_offering(inst: Instance, res: OfferingResult, path, hours: Optional[Sequence[int]] = None) -> bool:
    """Gantt, DA offer curves at the highest-priced hours, reserve stack with SR price, profit distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    T = inst.horizon_h
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    a1, a2, a3, a4 = axes.ravel()
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=7)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_xlim(0, T)
    a1.set_title(f"Stage-1 batch schedule | expected net cost {-res.expected_profit_eur:,.0f} EUR")
    oc = res.offer_curves
    hrs = list(hours) if hours is not None else list(np.argsort(-inst.price_buy_eur_mwh)[:1]) + \
        list(np.argsort(inst.price_buy_eur_mwh)[:1]) + [h for h in (8, 14) if h < T]
    for h in dict.fromkeys(int(x) for x in hrs):
        g = oc[oc.hour == h].sort_values("price_da_sell_eur_mwh")
        a2.step(g.price_da_sell_eur_mwh, g.p_da_sell_mw - g.p_da_buy_mw, where="post", marker="o", ms=3, label=f"hour {h}")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xlabel("scenario DA price (EUR/MWh)")
    a2.set_ylabel("net DA export offer (MW)")
    a2.set_title("DA price-quantity curves (monotone in price)")
    a2.legend()
    p = res.da_position
    a3.bar(p.hour + 0.5, p.exp_r_bess_dis_mw, 0.9, label="BESS discharge", color="tab:purple")
    a3.bar(p.hour + 0.5, p.exp_r_bess_ch_mw, 0.9, bottom=p.exp_r_bess_dis_mw, label="BESS charge cut", color="tab:cyan")
    a3.bar(p.hour + 0.5, p.exp_r_dr_mw, 0.9, bottom=p.exp_r_bess_dis_mw + p.exp_r_bess_ch_mw, label="DR", color="tab:orange")
    a3.bar(p.hour + 0.5, p.exp_r_mt_mw, 0.9, bottom=p.exp_r_bess_dis_mw + p.exp_r_bess_ch_mw + p.exp_r_dr_mw, label="MT", color="tab:green")
    a3.set_ylabel("E[reserve offer] (MW)")
    a3.set_xlabel("hour of day")
    a3.set_xlim(0, T)
    ax = a3.twinx()
    ax.plot(p.hour + 0.5, p.exp_price_sr_up_eur_mw_h, "k.-", label="E[SR price]")
    ax.set_ylabel("EUR/MW/h")
    a3.legend(loc="upper left", ncol=2, fontsize=8)
    a3.set_title("Up-spinning-reserve offer by asset")
    j = res.joint
    a4.hist(j.profit_eur, bins=min(20, max(5, len(j))), weights=j.prob, color="tab:blue", alpha=0.75)
    a4.axvline(res.expected_profit_eur, color="g", label=f"E = {res.expected_profit_eur:,.0f}")
    a4.axvline(res.cvar_profit_eur, color="r", label=f"CVaR = {res.cvar_profit_eur:,.0f}")
    a4.set_xlabel("profit per joint scenario (EUR)")
    a4.set_title("Profit distribution")
    a4.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    from factory_mt_offering_plotting import plot_offering_figures

    plot_offering_figures(inst, res, Path(path).parent)
    return True


# --------------------------------------------------------------------------- #
# Self-check (needs a solver)
# --------------------------------------------------------------------------- #
def selftest(solver: str = "appsi_highs") -> int:
    """Consistency checks that follow from theory (tolerance covers the MIP gap):
       T1  self-schedule <= offering curves        (curves relax the one-quantity restriction)
       T2  offering curves <= curves + reserve     (R = 0 is feasible)
       T3  beta > 0: E[profit] falls, CVaR rises   (efficient frontier is monotone)
       T4  non-anticipativity: two scenarios with identical DA prices get identical DA quantities
       T5  every solution passes the independent verification (strict mode raises otherwise)"""
    inst = shrink_instance(da.make_benchmark_instance(2024, n_tasks=30, max_batches=10), 10)
    cfg = SchedulerConfig(start_step_h=1.0, solver=SolverSettings(solver, 1e-3, 240.0))
    S, W, T = 4, 2, inst.horizon_h
    scen = OfferingScenarioSet.generate(inst, S, 7)
    rt = RealTimeSet.generate(inst, S, W, 11)
    dep = DeploymentSet.generate(S, W, T, 13)
    mkt, rsv, bal = IntradayMarket(), ReserveMarket(), BalancingMarket()
    ok = True
    results: List[OfferingResult] = []

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    def run(sc=scen, strat=None, rs=rsv) -> OfferingResult:
        results.append(optimize_offering(inst, sc, rt, dep, mkt, rs, bal, strat or StrategyConfig(), cfg))
        return results[-1]

    tol = lambda x: max(1.0, 3e-3 * abs(x))
    fixed = run(strat=StrategyConfig(bidding="fixed"), rs=replace(rsv, enabled=False))
    curve = run(rs=replace(rsv, enabled=False))
    full = run()
    check("T1 curves >= self-schedule", curve.expected_profit_eur >= fixed.expected_profit_eur - tol(fixed.expected_profit_eur),
          f"curves={curve.expected_profit_eur:,.2f}  self-schedule={fixed.expected_profit_eur:,.2f} EUR")
    check("T2 reserve >= energy only", full.expected_profit_eur >= curve.expected_profit_eur - tol(curve.expected_profit_eur),
          f"with reserve={full.expected_profit_eur:,.2f}  energy only={curve.expected_profit_eur:,.2f} EUR")
    risk = run(strat=StrategyConfig(beta=0.7))
    check("T3 risk aversion trades E for CVaR",
          risk.expected_profit_eur <= full.expected_profit_eur + tol(full.expected_profit_eur)
          and risk.cvar_profit_eur >= full.cvar_profit_eur - tol(full.cvar_profit_eur),
          f"E: {full.expected_profit_eur:,.2f} -> {risk.expected_profit_eur:,.2f}  CVaR: {full.cvar_profit_eur:,.2f} -> {risk.cvar_profit_eur:,.2f} EUR")
    tied = replace(scen, da_buy=scen.da_buy.copy(), da_sell=scen.da_sell.copy())
    tied.da_buy[1], tied.da_sell[1] = tied.da_buy[0], tied.da_sell[0]       # same DA price, different SR / ID prices
    r_t = run(sc=tied)
    pos = r_t.offer_curves
    gap = max(abs(g.p_da_sell_mw.iloc[0] - g.p_da_sell_mw.iloc[1]) + abs(g.p_da_buy_mw.iloc[0] - g.p_da_buy_mw.iloc[1])
              for _, g in pos[pos.scenario.isin([0, 1])].groupby("hour"))
    check("T4 non-anticipativity for equal prices", gap < 1e-3, f"max quantity difference between tied scenarios = {gap:.2e} MW")
    check("T5 independent verification", all(r.verification["passed"] for r in results), f"{len(results)} solves verified")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    from factory_mt_offering_cli import main as cli_main

    return cli_main(argv, sys.modules[__name__])


if __name__ == "__main__":
    sys.exit(main())
