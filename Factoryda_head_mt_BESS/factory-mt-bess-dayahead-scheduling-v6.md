---
name: factory-mt-bess-dayahead-scheduling-v6
description: Corrected MILP model for Industrial Factory Batch Scheduling, Microturbine and BESS in the Day-Ahead market with bidirectional grid trading. Time-indexed batch formulation, resolved parameter ambiguities, explicit initial conditions and BESS degradation convention.
---

# SKILL: Factory, Microturbine & BESS Day-Ahead Energy Scheduling (v6 - Corrected)

Mixed-Integer Linear Programming (MILP) model that co-optimizes an **industrial batch manufacturing schedule**, an on-site **Microturbine (MT)** and a **Battery Energy Storage System (BESS)** in the **Day-Ahead (DA) market** with **bidirectional grid trading** and **BESS throughput degradation**. All money is in **Euros (€)**. Reference implementation: `factory_mt_bess_da_scheduler.py` / `factory-mt-bess-da` (v4.0.0, modular Pyomo + HiGHS package).

---

## 1. Changes from v5 (all 13 review findings)

| # | v5 problem | v6 resolution |
| :-- | :--- | :--- |
| 1 | BESS size given as "2.00 / 40.00 MW" and "4.00 / 200.00 MWh" | **Single pair: 40 MW / 200 MWh.** (A 4 MWh battery cannot hold $SoC_0 = 20$ MWh.) The 2 / 4 values are removed. A smaller battery is a different scenario, set through `--bess-power` / `--bess-energy`. |
| 2 | SoC range "10%-90%" labelled MWh | $SoC_{min} = 0.10 \cdot E_{max} = 20$ MWh, $SoC_{max} = 0.90 \cdot E_{max} = 180$ MWh. Fractions are parameters, MWh are derived. |
| 3 | $C_{TP}$ only a range (20-50) | **Default 35 €/MWh** (middle of the range). It is a scenario parameter (`--bess-degradation`); results must be reported together with the value used. |
| 4 | Throughput convention unclear | **Explicit:** $C_{TP}$ is charged on AC-side throughput $P_{ch}+P_{dis}$ (`degradation_basis = "throughput"`, default). The alternative "discharged energy only" ($C_{TP}\cdot P_{dis}$) is available as `degradation_basis = "discharge"` / `--bess-degradation-basis discharge` and costs half as much for the same cycling. |
| 5 | $N_{max}$ used for batch slots **and** warehouse limit | Renamed: $N_{slot}$ = max batches per machine, $I_{max}$ = warehouse capacity, $I_t$ = stock, $I_0$ = initial stock. |
| 6 | MT initial state not given | $u_0 = 0$, $P_{MT,0} = 0$ (MT offline before $t=1$), history assumed compliant with MUT/MDT. Configurable (`initial_on`, `--mt-initial-on`). BESS starts at $SoC_0$. |
| 7 | $\sum_p X = 1$ forced every batch slot full (about 50 batches) | Slots are optional: at most $N_{slot}$ batches per machine, no filler batches. |
| 8 | No task uniqueness | Each task runs **at most once** over all machines and start times (eq. 4.1). |
| 9 | $ts$ not linked to the hourly execution $o$ (batches could be split across hours) | Replaced by a **time-indexed formulation**: a batch is one binary start decision; its hourly occupancy is an exact constant, so batches are contiguous and the load is tied to the real start time. |
| 10 | No machine-hour capacity, no horizon bound | Non-overlap + buffer is enforced per machine (eq. 4.2); starts that would end after $T$ are never generated. |
| 11 | MUT/MDT only for $t \ge MUT$ / $MDT$ | Windows are truncated at $t=1$ and apply to **every** hour (eq. 5.3). |
| 12 | Ramp limits never bind while online ($RU = RD = 20 > 15$ MW range) | Kept as specified and documented: they only bind at start-up / shut-down unless $RU$, $RD$ < 15 MW/h. |
| 13 | Charge/discharge binaries presented as mandatory | Kept by default (`enforce_exclusive`), but documented as **redundant** under the conditions in 6.4; may be relaxed to an LP. |

---

## 2. Parameter Database

| Category | Symbol | Value | Unit | Note |
| :--- | :--- | :--- | :--- | :--- |
| Horizon | $T$ | 24 | h | $t \in \{1,\dots,T\}$, hour $t$ covers $[t-1, t]$; $\Delta t = 1$ h |
| Machines | $M$ | 5 | - | furnaces $m1\dots m5$ |
| Tasks | $P$ | 30 | - | tasks $p1\dots p30$ |
| Max batches per machine | $N_{slot}$ | 10 | - | cap, not a requirement |
| Start-time grid step | $\Delta$ | 0.5 | h | candidate starts $a_k = k\Delta$ |
| Task power | $d_{m,p}$ | 60 - 110 | MW | m1 70-95, m2 60-80, m3 70-90, m4 90-110, m5 65-85 |
| Task duration | $td_{m,p}$ | 1.30 - 4.90 | h | continuous, no pre-emption |
| Yield per task | $Y_p$ | 5 - 15 | units | produced pro rata during execution |
| Buffer between batches | $t_{buf}$ | 1.00 | h | per machine |
| Base load | $l_{base,t}$ | 2.50 - 4.50 | MW | |
| Import / export limit | $Q^{buy}, Q^{sell}$ | 400 / 400 | MW | |
| Warehouse capacity | $I_{max}$ | 500 | units | |
| Initial stock | $I_0$ | 50 | units | |
| Demand | $\delta_t$ | 8 | units/h | 192 per day |
| MT power range | $P^{MT}_{min}, P^{MT}_{max}$ | 10 / 25 | MW | |
| MT blocks | $w_b, C_b$ | (5.30, 48.41), (4.70, 48.78), (5.00, 51.84) | MW, €/MWh | $\sum w_b = P^{MT}_{max}-P^{MT}_{min} = 15$ MW; costs non-decreasing |
| MT base fuel rate | $C_0$ | 48.41 | €/MWh | on $P^{MT}_{min}$: 484.10 €/h while online |
| MT ramps | $RU, RD, SRU, SRD$ | 20 / 20 / 20 / 20 | MW/h | |
| MT start-up / shut-down cost | $SUC, SDC$ | 87.40 / 8.74 | € | |
| MT min up / down | $MUT, MDT$ | 4 / 2 | h | |
| MT initial state | $u_0, P^{MT}_0$ | 0 / 0 | - | offline |
| BESS power | $\bar P$ | 40 | MW | charge and discharge, AC side |
| BESS energy | $E_{max}$ | 200 | MWh | |
| BESS SoC window | $\underline{\sigma}, \bar{\sigma}$ | 10 % / 90 % | of $E_{max}$ | $SoC_{min}=20$, $SoC_{max}=180$ MWh |
| BESS initial / terminal SoC | $SoC_0$ | 20 | MWh | $SoC_T \ge SoC_0$ |
| BESS efficiency | $\eta_{ch}, \eta_{dis}$ | 0.80 / 0.95 | - | round trip $0.76$ |
| BESS degradation | $C_{TP}$ | 35 (range 20-50) | €/MWh | basis: see 6.3 |

### Day-Ahead tariffs (€/MWh)
| Period | Hours | Buy | Sell |
| :--- | :--- | :--- | :--- |
| Off-peak | 02-04, 12-14, 18-20, 22-24 | 66.42 | 36.53 |
| Flat | 04-06 | 88.56 | 48.71 |
| Mid-peak | 00-02, 06-08, 14-16, 20-22 | 118.08 | 64.94 |
| On-peak | 08-12, 16-18 | 177.12 | 97.42 |

**Requirement:** $\lambda^{sell}_t \le \lambda^{buy}_t$ for all $t$ (otherwise simultaneous buying and selling is an arbitrage loop).

---

## 3. Decision Variables

* $P^{buy}_t \in [0, Q^{buy}]$, $P^{sell}_t \in [0, Q^{sell}]$: DA purchase / sale (MW).
* $P^{MT}_t \ge 0$, $P^{MT}_{b,t} \in [0, w_b]$: MT total output and block outputs (MW).
* $u_t, x_t, y_t \in \{0,1\}$: MT online / start-up / shut-down.
* $P^{ch}_t, P^{dis}_t \in [0, \bar P]$: BESS charge / discharge (MW); $SoC_t \in [SoC_{min}, SoC_{max}]$ (MWh).
* $v^{ch}_t, v^{dis}_t \in \{0,1\}$: BESS mode flags (optional, see 6.4).
* $s_{m,p,k} \in \{0,1\}$: 1 if task $p$ starts on machine $m$ at time $a_k = k\Delta$, for every candidate $(m,p,k)$ with $a_k + td_{m,p} \le T$.
* $I_t \in [0, I_{max}]$: end-of-hour warehouse stock.

**Derived constants.** Occupancy of hour $t$ by a batch started at $a_k$:
$$ov_{m,p,k,t} = \max\!\big(0,\ \min(a_k + td_{m,p},\ t) - \max(a_k,\ t-1)\big)\in[0,1]$$

---

## 4. Factory Batch Scheduling (time-indexed, replaces v5 section 3.5)

1. **Each task at most once** (fixes #8):
$$\sum_{m}\sum_{k} s_{m,p,k} \le 1 \qquad \forall p$$

2. **Non-overlap with buffer on every machine** (fixes #9, #10). A batch started at $a_k$ blocks its machine for start times $a_j \in [a_k,\ a_k + td_{m,p} + t_{buf})$:
$$\sum_{p}\ \sum_{k:\ a_k \le a_j < a_k + td_{m,p} + t_{buf}} s_{m,p,k} \le 1 \qquad \forall m,\ \forall j$$

3. **Optional slots, bounded batch count** (fixes #7):
$$\sum_{p}\sum_{k} s_{m,p,k} \le N_{slot} \qquad \forall m$$

4. **Batch load and production** (exact, contiguous execution):
$$L^{batch}_t = \sum_{m,p,k} d_{m,p}\, ov_{m,p,k,t}\, s_{m,p,k}, \qquad
n^{prod}_t = \sum_{m,p,k} \frac{Y_p}{td_{m,p}}\, ov_{m,p,k,t}\, s_{m,p,k}$$

5. **Warehouse:**
$$I_t = I_{t-1} + n^{prod}_t - \delta_t,\qquad 0 \le I_t \le I_{max},\qquad I_T \ge I_0$$

*Batch position $n$ and start time $ts_{m,n}$ are recovered after the solve by sorting each machine's chosen starts.*

---

## 5. Microturbine

1. **Piecewise output** (exact 25 MW cap because $\sum_b w_b = 15$ MW):
$$P^{MT}_t = P^{MT}_{min}\,u_t + \sum_b P^{MT}_{b,t},\qquad 0 \le P^{MT}_{b,t} \le w_b\, u_t$$
2. **Ramps:**
$$P^{MT}_t - P^{MT}_{t-1} \le RU\, u_{t-1} + SRU\, x_t,\qquad P^{MT}_{t-1} - P^{MT}_t \le RD\, u_t + SRD\, y_t$$
3. **Minimum up / down time, every hour, windows truncated at $t=1$** (fixes #11):
$$\sum_{\tau=\max(1,\,t-MUT+1)}^{t} x_\tau \le u_t,\qquad \sum_{\tau=\max(1,\,t-MDT+1)}^{t} y_\tau \le 1-u_t \qquad \forall t$$
4. **Commitment logic:** $x_t - y_t = u_t - u_{t-1}$, $\ x_t + y_t \le 1$, with $u_0 = 0$, $P^{MT}_0 = 0$.

*Remark (#12): while online the MT range is 15 MW, so $RU = RD = 20$ MW/h cannot bind; ramps matter only at start-up / shut-down unless $RU, RD < 15$ MW/h.*

---

## 6. BESS

1. **Energy balance:**
$$SoC_t = SoC_{t-1} + \eta_{ch}\, P^{ch}_t\,\Delta t - \frac{P^{dis}_t\,\Delta t}{\eta_{dis}},\qquad SoC_{0}\ \text{given}$$
2. **Bounds and terminal energy:** $SoC_{min} \le SoC_t \le SoC_{max}$, $\ SoC_T \ge SoC_0$.
3. **Degradation cost** (fixes #3, #4):
$$C^{deg}_t = C_{TP}\,\big(P^{ch}_t + P^{dis}_t\big)\quad(\texttt{throughput, default});\qquad
C^{deg}_t = C_{TP}\,P^{dis}_t\quad(\texttt{discharge basis})$$
4. **Power limits and exclusivity** (optional, #13): $P^{ch}_t \le \bar P v^{ch}_t$, $P^{dis}_t \le \bar P v^{dis}_t$, $v^{ch}_t + v^{dis}_t \le 1$. When $\eta_{ch}\eta_{dis} < 1$, $C_{TP} \ge 0$, $\lambda^{sell} \le \lambda^{buy}$ and the export limit does not bind, simultaneous charging and discharging only wastes energy, so the binaries can be dropped (LP relaxation, same optimum, faster). Keep them if prices can be negative or the export limit can bind.

---

## 7. Objective and Balance

$$\min \sum_{t=1}^{T}\Big(\lambda^{buy}_t P^{buy}_t - \lambda^{sell}_t P^{sell}_t + C_0 P^{MT}_{min} u_t + \sum_b C_b P^{MT}_{b,t} + SUC\,x_t + SDC\,y_t + C^{deg}_t\Big)$$

$$P^{buy}_t - P^{sell}_t + P^{MT}_t + P^{dis}_t - P^{ch}_t = l_{base,t} + L^{batch}_t \qquad \forall t$$

---

## 8. Reference Results (benchmark, seed 2024, HiGHS, gap <= 0.01 %)

| Scenario | Total cost | BESS use |
| :--- | :--- | :--- |
| $C_{TP} = 35$ (default) | 139,377 € | idle (arbitrage margin is negative) |
| $C_{TP} = 5$, throughput basis | 139,350 € | 90.7 MWh charged |
| $C_{TP} = 5$, discharge basis | 138,262 € | 254.9 MWh charged |
| $C_{TP} = 0$ | 137,293 € | 255.0 MWh charged |

Break-even: charging off-peak (66.42) and avoiding on-peak purchases (177.12) pays only while $C_{TP} \lesssim 38.7$ €/MWh (throughput basis); selling at on-peak (97.42) pays only while $C_{TP} \lesssim 4.3$ €/MWh. In this benchmark the factory already moves batches to cheap hours and on-peak load is about 4 MW, so there is little for the battery to displace.

---

## 9. Solver Guidelines

* **Framework:** Pyomo (Python) or GAMS. **Solvers:** HiGHS, Gurobi, CPLEX, CBC.
* **MIP gap:** <= 0.01 %. After solving, re-check every constraint from the extracted schedule (power balance, SoC, MT logic, machine buffers, inventory) before reporting.
