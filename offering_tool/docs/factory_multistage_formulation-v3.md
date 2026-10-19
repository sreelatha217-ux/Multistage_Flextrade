# Three-Stage Stochastic MILP for an Industrial Factory with Microturbine and BESS (Day-Ahead, Intraday, Real-Time Balancing), v3

Version 3 replaces `factory_multistage_intraday_formulation-v2.md`. It is the formulation implemented by `factory_mt_da_scheduler.py`, `factory_mt_id_scheduler.py` and `factory_mt_rt_scheduler.py`. All money is in EUR.

## 0. Corrections relative to v2

| # | Where in v2 | Problem | Correction in v3 |
|---|---|---|---|
| 1 | §4 | Balance read `supply = load + (Δ⁺ − Δ⁻)`, but §5 defines Δ⁺ as a deficit that is bought. A deficit would be bought while the plant already has a surplus. | §4: `supply + Δ⁺ − Δ⁻ = load`. |
| 2 | skill vs v2 | The skill calls the shortage Δ⁻ (ratio r⁻ ≥ 1); v2 calls it Δ⁺ (ratio r⁺ ≥ 1). | v2 naming is kept: Δ⁺ is the deficit bought at r⁺ ≥ 1, Δ⁻ is the surplus sold at r⁻ ≤ 1. |
| 3 | skill §2.1 | One price λ^DA for both directions. With sell = 55 % of buy, a surplus sold at r·λ^DA,buy would pay more than any DA or ID sale. | λ⁺ = r⁺·λ^DA,buy and λ⁻ = r⁻·λ^DA,sell. |
| 4 | §2.3, §10 | Δ carries the ID scenario index s only, so Stage 2 already knows what Stage 3 reveals and Δ becomes a free trading channel. This contradicts "passive settlement". | Scenario tree: ID scenario s, then real-time scenario w. Stage-2 variables carry s only, Stage-3 variables carry (s, w). |
| 5 | §2.1 | No real-time uncertainty is defined, so Stage 3 has nothing to settle. | Relative load deviation η and price ratios r⁺, r⁻ are random in (s, w). |
| 6 | §5.2 | z_BAL binaries and the bounds Δ̄ have no values. The binaries are redundant when r⁺ ≥ 1 ≥ r⁻ and λ^DA,buy ≥ λ^DA,sell. | Binaries optional. Δ̄ defaults to Q_buy + Q_sell. |
| 7 | §9.1 | Q_buy, Q_sell are written as day-ahead limits only. | They bound the day-ahead position, the market net position and the physical exchange. |
| 8 | §2.2 | C₀ is declared in EUR/h but used as C₀·P_MT,min·u. | C₀ is in EUR/MWh (48.41), so C₀·P_MT,min·u is EUR/h. |
| 9 | §2.3, §8 | Batch variables, load, product output and inventory carry the scenario index although the batch schedule is a Stage-1 decision. | They are deterministic (no s). Real load shifting needs scenario-indexed start binaries and is outside this model. |
| 10 | §9.2 | Monotonicity and non-anticipativity of DA bids are imposed although DA prices are deterministic and P^DA has no scenario index. | Moved to Appendix A. They apply only if DA prices are scenario-dependent. |
| 11 | §2.1, §3 | Symbol clashes: s for scenario and for the batch-start binary, k for slot, Δ for the start step and for imbalance. | Scenario s, RT scenario w, start binary σ, slot j, start step θ, imbalance Δ. |
| 12 | §2.1 | Mentions renewable generation, which no equation uses. | Removed. |
| 13 | §6.1 | Commitment costs sit inside the scenario sum. | Written once in Stage 1 (equal under Σπ = 1). |
| 14 | skill §3 | MT blocks 5.3/7.2/7.2/5.3 MW sum to 25 MW, not P_max − P_min = 15 MW. With P_MT = P_min·u + Σ blocks the unit could reach 35 MW. SoC_max = 200 MWh (100 %) conflicts with the 10–90 % window. | Blocks 5.3/4.7/5.0 MW at 48.41/48.78/51.84 EUR/MWh. SoC window 20–180 MWh. |

## 1. Decision architecture

```
Stage 1  here-and-now (one decision for all scenarios)
         P^{DA,buy}_t, P^{DA,sell}_t      day-ahead position
         u_t, x_t, y_t                    microturbine commitment
         σ_{m,p,j}                        batch start times (master production schedule)
              |
Stage 2  first wait-and-see, indexed by ID scenario s
         P^{ID,buy}_{t,s}, P^{ID,sell}_{t,s}     intraday re-trading
         P_{MT,t,s}, P_{MT,b,t,s}                MT re-dispatch inside the Stage-1 commitment
         P^{ch}_{t,s}, P^{dis}_{t,s}, SoC_{t,s}  BESS dispatch
              |
Stage 3  second wait-and-see, indexed by (s, w)
         revealed: η_{t,s,w}, r^+_{t,s,w}, r^-_{t,s,w}
         settled:  Δ^+_{t,s,w} (deficit bought), Δ^-_{t,s,w} (surplus sold)
```

The factory (batch loads, product output, inventory) is fixed by Stage 1 and has no scenario index.

## 2. Sets, parameters, variables

### 2.1 Sets
- t ∈ T = {1..24}, hourly periods, Δt = 1 h.
- s ∈ S, intraday scenarios, probability π_s, Σπ_s = 1.
- w ∈ W, real-time scenarios, conditional probability π_{w|s}, Σ_w π_{w|s} = 1 for every s.
- m ∈ M machines, p ∈ P tasks, j ∈ J start slots a_j = jθ with start step θ = 0.5 h, b ∈ B microturbine blocks.

### 2.2 Parameters
- Day-ahead: λ^{DA,buy}_t, λ^{DA,sell}_t with λ^{DA,sell}_t ≤ λ^{DA,buy}_t (EUR/MWh).
- Intraday: λ^{ID,buy}_{t,s}, λ^{ID,sell}_{t,s} with sell ≤ buy; market depth Cap^{ID}_{buy}, Cap^{ID}_{sell} (MW); base-load deviation δl_{t,s} (MW, default 0).
- Real time: relative factory-load deviation η_{t,s,w} > −1 with E[η] = 0; ratios r⁺_{t,s,w} ≥ 1 and 0 ≤ r⁻_{t,s,w} ≤ 1. Prices λ⁺_{t,s,w} = r⁺·λ^{DA,buy}_t and λ⁻_{t,s,w} = r⁻·λ^{DA,sell}_t, hence λ⁺ ≥ λ⁻. Bounds Δ̄_buy, Δ̄_sell (MW), default Q_buy + Q_sell.
- Factory: l_{base,t}, d_{m,p}, td_{m,p}, t_buf, Y_p, δ_t (demand), I_0, I_max.
- Microturbine: P_min, P_max, w_b with Σ_b w_b = P_max − P_min, C_b (EUR/MWh), C₀ (EUR/MWh), SUC, SDC, RU, RD, SRU, SRD, MUT, MDT, initial state u_0 and P_0.
- BESS: P̄, SoC_min, SoC_max, SoC_0, η_ch, η_dis, C_TP.
- Grid: Q_buy, Q_sell (substation import and export limits, MW).

### 2.3 Variables
- Stage 1: P^{DA,buy}_t, P^{DA,sell}_t ≥ 0; u_t, x_t, y_t ∈ {0,1}; σ_{m,p,j} ∈ {0,1}.
- Stage 2 (index s): P^{ID,buy}, P^{ID,sell}, P_MT, P_{MT,b}, P^{ch}, P^{dis}, SoC ≥ 0; v^{ch}, v^{dis} ∈ {0,1}.
- Stage 3 (index s, w): Δ⁺_{t,s,w}, Δ⁻_{t,s,w} ≥ 0; optional z_{t,s,w} ∈ {0,1}.
- Factory (no scenario index): L_{batch,t}, nprod_t, I_t.

## 3. Objective

$$\min\; \sum_t \Big(\mathcal C^{DA}_t + \mathcal C^{MT,1}_t\Big) + \sum_s \pi_s \sum_t \Big(\mathcal C^{ID}_{t,s} + \mathcal C^{MT,2}_{t,s} + \mathcal C^{BESS}_{t,s}\Big) + \sum_s \pi_s \sum_w \pi_{w|s} \sum_t \mathcal C^{BAL}_{t,s,w}$$

1. Day-ahead: C^{DA}_t = λ^{DA,buy}_t P^{DA,buy}_t − λ^{DA,sell}_t P^{DA,sell}_t
2. Intraday: C^{ID}_{t,s} = λ^{ID,buy}_{t,s} P^{ID,buy}_{t,s} − λ^{ID,sell}_{t,s} P^{ID,sell}_{t,s}
3. Balancing: C^{BAL}_{t,s,w} = λ⁺_{t,s,w} Δ⁺_{t,s,w} − λ⁻_{t,s,w} Δ⁻_{t,s,w}
4. Microturbine, Stage 1 (commitment): C^{MT,1}_t = C₀ P_min u_t + SUC x_t + SDC y_t
5. Microturbine, Stage 2 (fuel above minimum): C^{MT,2}_{t,s} = Σ_b C_b P_{MT,b,t,s}
6. BESS: C^{BESS}_{t,s} = C_TP (P^{ch}_{t,s} + P^{dis}_{t,s})

## 4. Power balance and imbalance

With L_t = l_{base,t} + δl_{t,s} + L_{batch,t} the planned factory load, for all t, s, w:

$$\big(P^{DA,buy}_t - P^{DA,sell}_t\big) + \big(P^{ID,buy}_{t,s} - P^{ID,sell}_{t,s}\big) + P_{MT,t,s} + P^{dis}_{t,s} - P^{ch}_{t,s} + \Delta^+_{t,s,w} - \Delta^-_{t,s,w} = (1+\eta_{t,s,w})\,L_t$$

Define the market net import P^{mkt}_{t,s} = (P^{DA,buy} − P^{DA,sell}) + (P^{ID,buy} − P^{ID,sell}) and the physical net import P^{phys}_{t,s,w} = (1+η)L_t + P^{ch} − P^{dis} − P_MT. Then

$$\Delta_{t,s,w} = P^{phys}_{t,s,w} - P^{mkt}_{t,s} = \Delta^+_{t,s,w} - \Delta^-_{t,s,w}.$$

Positive Δ means the plant uses more than it has contracted and buys the deficit at λ⁺. Negative Δ means it has contracted more than it uses and sells the surplus at λ⁻.

### 4.1 Decomposition
$$0 \le \Delta^+_{t,s,w} \le \bar\Delta_{buy}\, z_{t,s,w}, \qquad 0 \le \Delta^-_{t,s,w} \le \bar\Delta_{sell}\,(1 - z_{t,s,w})$$
The binary z is optional. Because λ⁺ ≥ λ⁻, buying and selling imbalance in the same hour never lowers the cost, so the model does not need it. Without z the bounds are 0 ≤ Δ⁺ ≤ Δ̄_buy and 0 ≤ Δ⁻ ≤ Δ̄_sell.

### 4.2 Imbalance modes
- **Strategic** (the formulation above): Stage 2 may deliberately plan an unbalanced position. This hedges against η and also lets the model trade on price differences between the ID market and the imbalance price.
- **Passive**: add `supply_{t,s} = (1 + Σ_w π_{w|s} η_{t,s,w}) L_t` for every t and s. Δ then carries only the realised deviation, which is the "passive physical settlement" of v2 §10.

## 5. Grid limits

$$0 \le P^{DA,buy}_t \le Q_{buy},\quad 0 \le P^{DA,sell}_t \le Q_{sell},\quad 0 \le P^{ID,buy}_{t,s} \le Cap^{ID}_{buy},\quad 0 \le P^{ID,sell}_{t,s} \le Cap^{ID}_{sell}$$
$$-Q_{sell} \le P^{mkt}_{t,s} \le Q_{buy}, \qquad -Q_{sell} \le P^{mkt}_{t,s} + \Delta^+_{t,s,w} - \Delta^-_{t,s,w} \le Q_{buy}$$

## 6. Microturbine

Commitment (Stage 1), with u_0 given:
$$x_t - y_t = u_t - u_{t-1},\quad x_t + y_t \le 1,\quad \sum_{\tau=\max(1,t-MUT+1)}^{t} x_\tau \le u_t,\quad \sum_{\tau=\max(1,t-MDT+1)}^{t} y_\tau \le 1 - u_t$$

Dispatch (Stage 2):
$$P_{MT,t,s} = P_{min}u_t + \sum_b P_{MT,b,t,s},\qquad 0 \le P_{MT,b,t,s} \le w_b u_t$$
$$P_{MT,t,s} - P_{MT,t-1,s} \le RU\,u_{t-1} + SRU\,x_t,\qquad P_{MT,t-1,s} - P_{MT,t,s} \le RD\,u_t + SRD\,y_t,\qquad P_{MT,0,s} = P_0$$

Block costs must be non-decreasing in b so that the LP fills blocks in order. Since Σ_b w_b = P_max − P_min, P_MT ≤ P_max·u holds without a separate constraint.

## 7. BESS (Stage 2, per ID scenario)

$$SoC_{t,s} = SoC_{t-1,s} + \eta_{ch}P^{ch}_{t,s}\Delta t - P^{dis}_{t,s}\Delta t/\eta_{dis},\quad SoC_{0,s} = SoC_0,\quad SoC_{T,s} \ge SoC_0$$
$$SoC_{min} \le SoC_{t,s} \le SoC_{max},\quad 0 \le P^{ch}_{t,s} \le \bar P v^{ch}_{t,s},\quad 0 \le P^{dis}_{t,s} \le \bar P v^{dis}_{t,s},\quad v^{ch}_{t,s} + v^{dis}_{t,s} \le 1$$

The BESS has no separate day-ahead baseline. The net day-ahead position already fixes what the plant buys, and the battery is a recourse resource.

## 8. Factory and warehouse (Stage 1, deterministic)

Task execution and non-overlap with buffer:
$$\sum_m\sum_j \sigma_{m,p,j} \le 1\ \ \forall p,\qquad \sum_p\ \sum_{j':\,a_{j'} \le a_j < a_{j'}+td_{m,p}+t_{buf}} \sigma_{m,p,j'} \le 1\ \ \forall m, j$$

With the exact hourly overlap ov_{m,p,j,t} = max(0, min(a_j + td_{m,p}, t) − max(a_j, t−1)) ∈ [0,1]:
$$L_{batch,t} = \sum_{m,p,j} d_{m,p}\,ov_{m,p,j,t}\,\sigma_{m,p,j},\qquad nprod_t = \sum_{m,p,j} \frac{Y_p}{td_{m,p}}\,ov_{m,p,j,t}\,\sigma_{m,p,j}$$
$$I_t = I_{t-1} + nprod_t - \delta_t,\quad I_0 \text{ given},\quad 0 \le I_t \le I_{max},\quad I_T \ge I_0$$

The real-time deviation η scales the metered load. It does not change production or inventory.

## 9. Hierarchy

| Decision | Stage | Index |
|---|---|---|
| DA bids, MT commitment, batch starts | 1 | t (and m, p, j) |
| ID trades, MT output, BESS dispatch and SoC | 2 | t, s |
| Imbalance Δ⁺, Δ⁻ | 3 | t, s, w |

## 10. Open modelling limits

These are not errors in v2, and they are not fixed here.
- Within an ID scenario, Stage-2 decisions see the whole 24 h price path. This overstates the value of re-dispatch. Removing it needs a multi-hour scenario tree.
- The objective is risk-neutral. A CVaR term would penalise high-variance Stage-1 plans.
- η, r⁺ and r⁻ are drawn independently. Real imbalance prices depend on the system direction and correlate with ID prices. Use a calibrated joint scenario file when one exists.
- Virtual arbitrage between DA, ID and imbalance prices is possible whenever expected prices cross the buy/sell band. It is limited only by Q, the ID caps and Δ̄.
- Default values for Cap^{ID}, the number of scenarios, η and the ratios are assumptions (see the module docstrings).

## Appendix A. Day-ahead bid-curve conditions

Only if day-ahead prices are scenario-dependent, λ^{DA}_t(ω), and P^{DA}_t(ω) is allowed to vary with ω:
$$P^{DA,buy}_t(\omega_1) \le P^{DA,buy}_t(\omega_2)\ \text{ if } \lambda^{DA}_t(\omega_1) \le \lambda^{DA}_t(\omega_2),\qquad P^{DA,buy}_t(\omega_1) = P^{DA,buy}_t(\omega_2)\ \text{ if } \lambda^{DA}_t(\omega_1) = \lambda^{DA}_t(\omega_2)$$
The first condition makes the bid curve a valid market offer. The second ties scenarios with equal prices together. With deterministic DA prices, as in this model, both hold by construction.
