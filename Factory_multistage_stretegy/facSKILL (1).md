---
name: industrial-large-consumer-multistage-optimization
description: Three-stage stochastic MILP (Day-Ahead, Intraday, Real-Time balancing) for a large industrial prosumer with batch load, BESS and microturbine. All money in EUR. Version 2 - corrects the sign, price, scenario-tree, parameter and batch-model errors of version 1.
---

# SKILL: Industrial Large-Consumer Multi-Stage Energy & Market Optimization (v2)

Framework for a **large industrial consumer** with **batch production**, a **BESS** and an on-site **Microturbine (MT)** that trades on **Day-Ahead (DA)**, **Intraday (ID)** and **Real-Time (RT) balancing** markets. It matches `factory_multistage_formulation-v3.md` and the programs `factory_mt_da_scheduler.py`, `factory_mt_id_scheduler.py`, `factory_mt_rt_scheduler.py` and `factory_mt_offering_strategy.py`.

## 0. Corrections relative to v1 of this skill

| # | v1 problem | Correction in v2 |
|---|---|---|
| 1 | Balancing cost `lam*(r-*D- - r+*D+)` with r- >= 1, r+ <= 1: buying earned money, selling cost money. Balance `supply = load + D+ - D-` bought a deficit while in surplus. | `supply + D+ - D- = (1+eta)*load`. D+ = deficit bought at r+ >= 1. D- = surplus sold at r- <= 1. Cost `lam+ D+ - lam- D-`. |
| 2 | One price lam^DA for both directions (sell = 55 % of buy). | `lam+ = r+ * lam^DA,buy`, `lam- = r- * lam^DA,sell`; hence lam+ >= lam-. |
| 3 | Non-anticipativity `P^DA(w1) = P^DA(w2)` for all scenarios, which contradicts the monotone bid curve and collapses it to a self-schedule. | Equality only for scenarios with equal prices. With deterministic DA prices P^DA has no scenario index (Appendix A). |
| 4 | MT blocks 5.3/7.2/7.2/5.3 = 25 MW but Pmax - Pmin = 15 MW; SoC_max = 200 (100 %) vs a 10-90 % window. | Blocks 5.3/4.7/5.0 MW at 48.41/48.78/51.84 EUR/MWh. SoC window 20-180 MWh. |
| 5 | Batch variables carried s; imbalance carried s only (free trading channel). | Batch variables deterministic. Tree: ID scenario s, then RT scenario w. Stage 2 has index s, Stage 3 has (s, w). |
| 6 | No RT uncertainty. | eta, r+, r- random in (s, w) (section 2.1). |
| 7 | Symbol clashes (m, N, s, Delta). | Machine m, MT block b, tasks p, slot j, start step theta, start binary sigma, scenario s, RT scenario w, imbalance Delta, inventory I. |
| 8 | Batch model incomplete (sequence only, no hourly link, no discretisation). | Time-indexed start grid with non-overlap + buffer and exact hourly overlap `ov` (section 2.3). |
| 9 | MUT/MDT only for t >= MUT/MDT; no initial state. | Windows start at tau = max(1, t-MUT+1); u_0 and P_0 given; initial must-run/must-off time (section 2.4). |
| 10 | Mixed scales (60-110 MW loads vs 2.5-4.5 MW base load) and an undefined BESS "medium/large" choice. | One consistent parameter set and an explicit selection rule (section 3). |
| 11 | No terminal conditions, grid limits, ID depth, risk term or verification. | Added (sections 2.5, 2.6, 4). |
| 12 | Perfect foresight of the 24 h price path inside a scenario not mentioned. | Documented as an open limit (section 5). |

## 1. Decision architecture

```
Stage 1  here-and-now (one decision for all scenarios)
         P^{DA,buy}_t, P^{DA,sell}_t      day-ahead position (or bid curve, Appendix A)
         u_t, x_t, y_t                    MT commitment
         sigma_{m,p,j}                    batch starts (master production schedule)
              |
Stage 2  first wait-and-see, index s
         P^{ID,buy}_{t,s}, P^{ID,sell}_{t,s}      ID re-trading
         P_{MT,t,s}, P_{MT,b,t,s}                  MT re-dispatch inside the Stage-1 commitment
         P^{ch}_{t,s}, P^{dis}_{t,s}, SoC_{t,s}    BESS dispatch
              |
Stage 3  second wait-and-see, index (s, w)
         revealed: eta_{t,s,w}, r+_{t,s,w}, r-_{t,s,w}
         settled:  D+_{t,s,w} (deficit bought), D-_{t,s,w} (surplus sold)
```
The factory (batch load, product output, inventory) is fixed by Stage 1 and has no scenario index. Load shifting after Stage 1 would need scenario-indexed start binaries and is outside this model.

## 2. Mathematical formulation

### 2.1 Sets and uncertainty
- t in T = {1..24}, Delta_t = 1 h. s in S, probability pi_s, sum = 1. w in W, conditional probability pi_{w|s}, sum_w = 1 for each s.
- m in M machines, p in P tasks, j in J start slots a_j = j*theta (theta = 0.5 h), b in B MT blocks.
- Intraday: prices lam^{ID,buy}_{t,s} >= lam^{ID,sell}_{t,s}; depth Cap^{ID}_{buy}, Cap^{ID}_{sell}; base-load deviation dl_{t,s} (default 0).
- Real time: relative load deviation eta_{t,s,w} > -1 with E[eta] = 0; ratios r+ >= 1 and 0 <= r- <= 1. Prices lam+ = r+ * lam^{DA,buy}_t, lam- = r- * lam^{DA,sell}_t. Bounds D_buy, D_sell (default Q_buy + Q_sell).
- Day-ahead prices lam^{DA,buy}_t >= lam^{DA,sell}_t are deterministic here. If they are scenario dependent, use Appendix A.

### 2.2 Objective (minimise expected cost, EUR)

min  sum_t ( C^DA_t + C^MT1_t ) + sum_s pi_s sum_t ( C^ID_{t,s} + C^MT2_{t,s} + C^BESS_{t,s} ) + sum_s pi_s sum_w pi_{w|s} sum_t C^BAL_{t,s,w}

1. DA: C^DA_t = lam^{DA,buy}_t P^{DA,buy}_t - lam^{DA,sell}_t P^{DA,sell}_t
2. ID: C^ID_{t,s} = lam^{ID,buy}_{t,s} P^{ID,buy}_{t,s} - lam^{ID,sell}_{t,s} P^{ID,sell}_{t,s}
3. Balancing: C^BAL_{t,s,w} = lam+_{t,s,w} D+_{t,s,w} - lam-_{t,s,w} D-_{t,s,w}
4. MT Stage 1 (written once, not inside the scenario sum): C^MT1_t = C0 * Pmin * u_t + SUC x_t + SDC y_t, with C0 in EUR/MWh.
5. MT Stage 2 (fuel above minimum): C^MT2_{t,s} = sum_b C_b P_{MT,b,t,s}
6. BESS: C^BESS_{t,s} = C_TP (P^ch_{t,s} + P^dis_{t,s})

Optional risk term (offering strategy): max (1-beta) E[profit] + beta CVaR_alpha(profit) over the joint (s, w) scenarios.

### 2.3 Power balance, imbalance, grid limits
With the planned factory load L_{t,s} = l_{base,t} + dl_{t,s} + L_{batch,t}, for all t, s, w:

(P^{DA,buy}_t - P^{DA,sell}_t) + (P^{ID,buy}_{t,s} - P^{ID,sell}_{t,s}) + P_{MT,t,s} + P^dis_{t,s} - P^ch_{t,s} + D+_{t,s,w} - D-_{t,s,w} = (1 + eta_{t,s,w}) L_{t,s}

Imbalance D = D+ - D- = physical net import - market net import. Positive D means the plant uses more than contracted and buys the deficit at lam+. Negative D means it sells the surplus at lam-.

Bounds: 0 <= D+ <= D_buy, 0 <= D- <= D_sell. The binary z (D+ <= D_buy z, D- <= D_sell (1-z)) is optional because lam+ >= lam- makes simultaneous D+ and D- never useful.

Imbalance modes: **strategic** (above; Stage 2 may plan an unbalanced position) and **passive** (add `P^mkt + P_MT + P^dis - P^ch = (1 + sum_w pi_{w|s} eta) L` for every t, s, so D carries only the realised deviation).

Grid limits (DA, ID, market net, physical):
0 <= P^{DA,buy} <= Q_buy, 0 <= P^{DA,sell} <= Q_sell, 0 <= P^{ID,buy} <= Cap^{ID}_buy, 0 <= P^{ID,sell} <= Cap^{ID}_sell,
-Q_sell <= P^mkt_{t,s} <= Q_buy, -Q_sell <= P^mkt_{t,s} + D+ - D- <= Q_buy.

### 2.4 Microturbine
Commitment (Stage 1), u_0 and P_0 given:
- x_t - y_t = u_t - u_{t-1}, x_t + y_t <= 1.
- sum_{tau = max(1, t-MUT+1)}^{t} x_tau <= u_t, sum_{tau = max(1, t-MDT+1)}^{t} y_tau <= 1 - u_t (valid for all t including the first hours).
- If the unit starts online with remaining up-time, u_t = 1 for t <= MUT - (hours already online); likewise u_t = 0 for the remaining down-time.

Dispatch (Stage 2):
- P_{MT,t,s} = Pmin u_t + sum_b P_{MT,b,t,s}, 0 <= P_{MT,b,t,s} <= w_b u_t, sum_b w_b = Pmax - Pmin (so P_MT <= Pmax u holds automatically).
- P_{MT,t,s} - P_{MT,t-1,s} <= RU u_{t-1} + SRU x_t, P_{MT,t-1,s} - P_{MT,t,s} <= RD u_t + SRD y_t, P_{MT,0,s} = P_0.
- Block costs C_b are non-decreasing in b, so the LP fills the blocks in order.

### 2.5 BESS (Stage 2, per ID scenario)
- SoC_{t,s} = SoC_{t-1,s} + eta_ch P^ch_{t,s} Delta_t - P^dis_{t,s} Delta_t / eta_dis, SoC_{0,s} = SoC_0, **SoC_{T,s} >= SoC_0** (terminal condition).
- SoC_min <= SoC_{t,s} <= SoC_max, 0 <= P^ch <= Pbar v^ch, 0 <= P^dis <= Pbar v^dis, v^ch + v^dis <= 1.
- No separate DA baseline: the DA position already fixes the purchase, and the battery is a recourse resource.

### 2.6 Factory and warehouse (Stage 1, deterministic)
Candidate start grid: sigma_{m,p,j} in {0,1} means task p starts on machine m at a_j = j*theta.
- Each task at most once: sum_m sum_j sigma_{m,p,j} <= 1 for all p.
- Non-overlap with buffer: for every machine m and slot j, sum_p sum_{j' : a_j' <= a_j < a_j' + td_{m,p} + t_buf} sigma_{m,p,j'} <= 1.
- Exact hourly overlap: ov_{m,p,j,t} = max(0, min(a_j + td_{m,p}, t) - max(a_j, t-1)) in [0,1].
- Load and output: L_{batch,t} = sum_{m,p,j} d_{m,p} ov_{m,p,j,t} sigma_{m,p,j}; nprod_t = sum_{m,p,j} (Y_p / td_{m,p}) ov_{m,p,j,t} sigma_{m,p,j}.
- Inventory: I_t = I_{t-1} + nprod_t - delta_t, I_0 given, 0 <= I_t <= I_max, **I_T >= I_0** (terminal condition).
- The RT deviation eta scales the metered load only. It does not change production or inventory.
- Non-interruptible durations td_{m,p} (1.3-4.9 h) need not be multiples of theta. The overlap factor ov handles partial hours exactly.

## 3. Reference parameter database (EUR)

**One consistent plant scale** (matches `make_benchmark_instance`): five parallel furnaces with loads 60-110 MW (m1 70-95, m2 60-80, m3 70-90, m4 90-110, m5 65-85), base load 2.5-4.5 MW, grid limit Q_buy = Q_sell = 400 MW, MT 10-25 MW, BESS 40 MW / 200 MWh. At peak all five furnaces together can reach about 400 MW, so the grid limit can bind. MT and BESS are small relative to that peak (they cover about 6 % and 10 %), so their value comes from price arbitrage and imbalance hedging rather than from load coverage. Base load is therefore a minor term.

### 3.1 Microturbine
| Parameter | Value |
| :-- | :-- |
| P_min / P_max | 10 / 25 MW |
| RU = RD = SRU = SRD | 20 MW/h |
| SUC / SDC | 87.40 / 8.74 EUR |
| MUT / MDT | 4 / 2 h |
| C0 (no-load) | 48.41 EUR/MWh (cost C0 * Pmin * u is EUR/h) |
| Blocks (width MW, EUR/MWh) | b1: 5.3 at 48.41; b2: 4.7 at 48.78; b3: 5.0 at 51.84 (sum 15.0 = Pmax - Pmin) |

### 3.2 BESS (selection rule)
| Parameter | Utility scale (default, used by the programs) | Small scale (tests, shrunken plants) |
| :-- | :-- | :-- |
| Power | 40 MW | 2 MW |
| Energy | 200 MWh | 4 MWh |
| SoC window (10-90 %) | 20-180 MWh | 0.4-3.6 MWh |
| SoC_0 (= terminal SoC) | 20 MWh | 0.63 MWh |
Common: eta_ch 0.80, eta_dis 0.95 (round trip 0.76), C_TP 20-50 EUR/MWh. Rule: use the utility column unless the plant peak load is below about 20 MW; then use the small column.

### 3.3 Factory, grid and tariffs
Parallel machines 5, task durations 1.3-4.9 h, buffer t_buf 1.0 h, start step theta 0.5 h, demand 8 units/h, I_0 = 50, I_max = 500. ID depth Cap^{ID} and the number of scenarios, eta and ratio statistics are assumptions (set in the program docstrings).

TOU tariffs (buy / sell, EUR/MWh): off-peak (02-04, 12-14, 18-20, 22-24) 66.42 / 36.53; flat-peak (04-06) 88.56 / 48.71; mid-peak (00-02, 06-08, 14-16, 20-22) 118.08 / 64.94; on-peak (08-12, 16-18) 177.12 / 97.42.

## 4. Verification (required)
After every solve, recompute independently: factory feasibility (overlap, buffer, inventory), MT commitment and ramps, BESS SoC paths, the balance for every (s, w), grid limits, and the objective (expected cost and, with a risk term, CVaR). Fail the run on any mismatch above tolerance. With beta close to 1 the solver may waste money outside the CVaR tail, so the objective check is then one-sided (recomputed value can only be better).

## 5. Open modelling limits
- Within an ID scenario, Stage-2 decisions see the whole 24 h price path (perfect foresight). This overstates the value of re-dispatch. Removing it needs a multi-hour scenario tree.
- Objective is risk-neutral unless the CVaR term is switched on. A 5 % tail needs at least 20 joint (s, w) scenarios to be wider than one scenario.
- eta, r+, r- are drawn independently. Real imbalance prices correlate with system direction and ID prices. Use a calibrated joint scenario file when one exists.
- Virtual arbitrage between DA, ID and imbalance prices is possible whenever expected prices cross the buy/sell band. It is limited only by Q, Cap^{ID} and D bounds.

## Appendix A. DA bid-curve conditions
Only if DA prices are scenario dependent, lam^DA_t(omega), and P^DA_t(omega) may vary with omega. With scenarios sorted by price (adjacent pairs are enough by transitivity):
- P^{DA,sell}_t(w1) <= P^{DA,sell}_t(w2) if lam^{DA,sell}_t(w1) <= lam^{DA,sell}_t(w2) (offer curve non-decreasing).
- P^{DA,buy}_t(w1) >= P^{DA,buy}_t(w2) if lam^{DA,buy}_t(w1) <= lam^{DA,buy}_t(w2) (bid curve non-increasing in price, i.e. the plant buys less when price is higher).
- P^{DA}_t(w1) = P^{DA}_t(w2) if the prices are equal (non-anticipativity applies to ties only).
With deterministic DA prices these hold by construction and P^DA carries no scenario index.

## 6. Implementation guidelines
Pyomo (Python), HiGHS / CBC / Gurobi / CPLEX, relative MIP gap <= 0.01 % for exact scheduling (0.1-0.2 % is enough for tests). All money in EUR.
