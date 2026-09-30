---
name: industrial-large-consumer-multistage-optimization
description: Multi-stage stochastic MILP optimization model for large industrial prosumers (batch load, BESS, microturbine) trading across Day-Ahead, Intraday, and Real-Time electricity markets in Euros (€).
---

# SKILL: Industrial Large-Consumer Multi-Stage Energy & Market Optimization

This skill provides a comprehensive, multi-stage stochastic mixed-integer linear programming (MILP) framework for optimizing electricity market trading and operational dispatch of a **large industrial consumer** equipped with **industrial batch production processes**, a **Battery Energy Storage System (BESS)**, and an on-site **Microturbine (MT)**.

The formulation strictly handles **Day-Ahead (DA)**, **Intraday (ID)**, and **Real-Time Balancing** electricity markets, using all monetary parameters in **Euros (€)**.

---

## 1. Multi-Stage Decision Architecture

The decision-making timeline is structured into three sequential stages to account for market gate closures and the progressive realization of uncertainties (market prices, batch execution delays, and load fluctuations):

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-Market Commitments & Baseline (Day-Ahead Stage: "Here-and-Now")             │
│ • Submit Day-Ahead (DA) market energy purchase/sell schedules: P_{t}^{DA}                │
│ • Microturbine Unit Commitment: Binary online/offline status u_{MT,t} ∈ {0,1}           │
│ • Master Production Schedule: Allocate batch start times ts_{m,n} and machine assignments  │
│ • BESS Day-Ahead Charging/Discharging Baseline: P_{ch,t}^{DA}, P_{dis,t}^{DA}            │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Intraday Re-Trading & Operational Recourse (Intraday Stage: "1st Wait-and-See") │
│ • Submit Intraday buy/sell adjustments: P_{buy,t,s}^{ID}, P_{sell,t,s}^{ID}               │
│ • Microturbine Output Adjustment: Re-dispatch MT generation P_{MT,t,s} against ID prices  │
│ • Industrial Load Shifting: Shift flexible batch tasks in response to price spikes/delays  │
│ • BESS Recourse Dispatch: Re-adjust charge/discharge to capture intraday price spreads    │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Real-Time Balancing Settlement (Real-Time Stage: "2nd Wait-and-See")            │
│ • Measure physical deviation between net market commitment (P_t^{DA} + P_t^{ID}) and       │
│   actual net physical consumption                                                          │
│ • Settle positive and negative imbalance volumes (Δ_{t,s}^+, Δ_{t,s}^-) under balancing  │
│   penalty pricing ratios (r_{t,s}^+, r_{t,s}^-)                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Objective Function: Total Expected System Cost Minimization

The primary objective is to minimize the expected total electricity procurement costs, microturbine fuel and operational expenses, and battery aging costs across all time periods $t \in T$ and stochastic scenarios $s \in S$:

$$\min \mathbb{E} \left[ \sum_{t=1}^T \left( \mathcal{C}_t^{DA} + \mathcal{C}_{t,s}^{ID} + \mathcal{C}_{t,s}^{BAL} + \mathcal{C}_{t,s}^{MT} + \mathcal{C}_{t,s}^{BESS} \right) \right]$$

#### Detailed Cost Components:

1. **Day-Ahead Energy Trading Cost (€):**
   $$\mathcal{C}_t^{DA} = \lambda_t^{DA} \cdot P_t^{DA}$$
   *(where $P_t^{DA} > 0$ denotes power purchase from the grid, and $P_t^{DA} < 0$ denotes power sale back to the grid).*

2. **Intraday Market Re-Trading Net Cost (€):**
   $$\mathcal{C}_{t,s}^{ID} = \lambda_{t,s}^{ID} \cdot \left( P_{buy,t,s}^{ID} - P_{sell,t,s}^{ID} \right)$$

3. **Real-Time Imbalance Settlement Cost (€):**
   $$\mathcal{C}_{t,s}^{BAL} = \lambda_{t,s}^{DA} \cdot \left( r_{t,s}^- \cdot \Delta_{t,s}^- - r_{t,s}^+ \cdot \Delta_{t,s}^+ \right)$$
   *(where $r_{t,s}^- \ge 1.0$ is the under-generation/over-consumption penalty ratio, and $r_{t,s}^+ \le 1.0$ is the over-generation/under-consumption discount selling ratio).*

4. **Microturbine Fuel, Start-Up, and Shut-Down Costs (€):**
   $$\mathcal{C}_{t,s}^{MT} = \sum_{m=1}^{N_m} C_{m}^{MT} \cdot P_{MT,m,t,s} + SUC \cdot x_{MT,t} + SDC \cdot y_{MT,t}$$
   *(where $C_{m}^{MT}$ is the marginal cost in €/MWh for block $m$ of the piecewise linear cost curve, $SUC$ is the start-up cost, and $SDC$ is the shut-down cost).*

5. **Battery Degradation Throughput Aging Cost (€):**
   $$\mathcal{C}_{t,s}^{BESS} = C_{TP} \cdot \left( P_{ch,t,s}^{BESS} + P_{dis,t,s}^{BESS} \right)$$
   *(where $C_{TP}$ is the throughput degradation cost in €/MWh).*

---

### 2.2 System Power Balance Constraints

At every hour $t$ and scenario $s$, net electricity imported from market contracts and on-site generation must match factory consumption and storage charging:

$$P_t^{DA} + P_{buy,t,s}^{ID} - P_{sell,t,s}^{ID} + P_{MT,t,s} + P_{dis,t,s}^{BESS} - P_{ch,t,s}^{BESS} = P_{factory,t,s} + \left( \Delta_{t,s}^+ - \Delta_{t,s}^- \right)$$

where total factory consumption $P_{factory,t,s}$ is composed of non-shiftable base load and flexible batch load:

$$P_{factory,t,s} = l_{base,t} + \sum_{m \in M} \sum_{n \in N} l_{m,n,t,s}$$

---

### 2.3 Industrial Factory & Batch Scheduling Constraints

Industrial load is modeled as parallel processing machines $m \in M$ executing batch production tasks $n \in N$.

1. **Task Duration and Machine Power Demand:**
   $$l_{m,n,t,s} = d_{m,p} \cdot o_{m,n,t,p,s}$$
   where $d_{m,p}$ is the power demand (MW) of task $p$ on machine $m$, and $o_{m,n,t,p,s}$ is the continuous processing indicator during time step $t$.

2. **Single Task Assignment per Machine Sequence:**
   $$\sum_{p \in P} X_{m,n,p} = 1 \quad \forall m \in M, n \in N$$

3. **Batch Non-Overlap & Maintenance/Cooling Buffer Constraint:**
   $$ts_{m,n} \ge ts_{m,n-1} + td_{m,p} \cdot X_{m,n-1,p} + t_{buffer}$$
   where $ts_{m,n}$ is the task start time, $td_{m,p}$ is task processing duration, and $t_{buffer}$ is the mandatory inter-batch setup/cooling buffer.

4. **Product Warehouse Inventory Balance:**
   $$N_{t,s} = N_{t-1,s} + n_{prod,t,s} - n_{prod,dem,t}$$
   $$0 \le N_{t,s} \le N_{max}$$
   where $N_{t,s}$ is stored product inventory, $n_{prod,t,s}$ is newly completed units, and $n_{prod,dem,t}$ is the required delivery schedule.

---

### 2.4 Microturbine (MT) Operational Constraints

1. **Piecewise Linear Power Block Limits:**
   $$P_{MT,t,s} = P_{MT,min} \cdot u_{MT,t} + \sum_{m=1}^{N_m} P_{MT,m,t,s}$$
   $$0 \le P_{MT,m,t,s} \le P_{MT,m,max} \cdot u_{MT,t} \quad \forall m \in \{1, \dots, N_m\}$$

2. **Ramp-Up and Ramp-Down Rate Limits:**
   $$P_{MT,t,s} - P_{MT,t-1,s} \le RU_{MT} \cdot u_{MT,t-1} + SRU_{MT} \cdot x_{MT,t}$$
   $$P_{MT,t-1,s} - P_{MT,t,s} \le RD_{MT} \cdot u_{MT,t} + SRD_{MT} \cdot y_{MT,t}$$

3. **Minimum Up Time (MUT) and Minimum Down Time (MDT):**
   $$\sum_{\tau = t - MUT + 1}^{t} x_{MT,\tau} \le u_{MT,t} \quad \forall t \ge MUT$$
   $$\sum_{\tau = t - MDT + 1}^{t} y_{MT,\tau} \le 1 - u_{MT,t} \quad \forall t \ge MDT$$

4. **Unit Commitment Logic:**
   $$x_{MT,t} - y_{MT,t} = u_{MT,t} - u_{MT,t-1}$$
   $$x_{MT,t} + y_{MT,t} \le 1 \quad (x_{MT}, y_{MT}, u_{MT} \in \{0, 1\})$$

---

### 2.5 Battery Energy Storage System (BESS) Constraints

1. **State of Charge (SoC) Dynamics:**
   $$SoC_{t,s} = SoC_{t-1,s} + \eta_{ch} \cdot P_{ch,t,s}^{BESS} \cdot \Delta t - \frac{1}{\eta_{dis}} \cdot P_{dis,t,s}^{BESS} \cdot \Delta t$$

2. **SoC and Power Limits:**
   $$SoC_{min} \le SoC_{t,s} \le SoC_{max}$$
   $$0 \le P_{ch,t,s}^{BESS} \le P_{ch,max} \cdot v_{ch,t,s}$$
   $$0 \le P_{dis,t,s}^{BESS} \le P_{dis,max} \cdot v_{dis,t,s}$$

3. **Non-Simultaneous Charging and Discharging:**
   $$v_{ch,t,s} + v_{dis,t,s} \le 1 \quad (v_{ch,t,s}, v_{dis,t,s} \in \{0, 1\})$$

---

### 2.6 Multi-Stage Offering & Bidding Curve Constraints

To ensure valid Day-Ahead bidding packages submitted before real-time price realizations:

1. **Non-Decreasing Bidding Curve Condition:**
   $$P_{t,\omega_1}^{DA} \le P_{t,\omega_2}^{DA} \quad \text{for scenarios where } \lambda_{t,\omega_1}^{DA} \le \lambda_{t,\omega_2}^{DA}$$

2. **Non-Anticipativity Condition:**
   $$P_{t,\omega_1}^{DA} = P_{t,\omega_2}^{DA} \quad \forall \omega_1, \omega_2 \in \Omega_{DA}$$

---

## 3. Reference Parameter Database (All Values in EUR / €)

### 3.1 Microturbine (MT) Technical & Economic Parameters

| Parameter | Symbol | Value | Unit | Description / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Minimum Generation** | $P_{MT,min}$ | **10.00** | MW | Minimum stable power output |
| **Maximum Capacity** | $P_{MT,max}$ | **25.00** | MW | Maximum rated capacity |
| **Ramp-Up Rate** | $RU_{MT}$ | **20.00** | MW/h | Operational upward ramp limit |
| **Ramp-Down Rate** | $RD_{MT}$ | **20.00** | MW/h | Operational downward ramp limit |
| **Start-Up Ramp Limit** | $SRU_{MT}$ | **20.00** | MW/h | Ramp limit during start-up interval |
| **Shut-Down Ramp Limit** | $SRD_{MT}$ | **20.00** | MW/h | Ramp limit during shut-down interval |
| **Start-Up Cost** | $SUC$ | **87.40** | € | Fixed cost per start-up event |
| **Shut-Down Cost** | $SDC$ | **8.74** | € | Fixed cost per shut-down event |
| **Minimum Up Time** | $MUT$ | **4** | Hours | Mandatory online duration |
| **Minimum Down Time** | $MDT$ | **2** | Hours | Mandatory offline duration |

#### MT Piecewise Linear Fuel Cost Curve Blocks (€/MWh):
* **Block 1 ($m=1$):** Width = **5.30 MW**, Marginal Cost = **48.41 €/MWh**
* **Block 2 ($m=2$):** Width = **7.20 MW**, Marginal Cost = **48.78 €/MWh**
* **Block 3 ($m=3$):** Width = **7.20 MW**, Marginal Cost = **51.84 €/MWh**
* **Block 4 ($m=4$):** Width = **5.30 MW**, Marginal Cost = **55.40 €/MWh**

---

### 3.2 Battery Energy Storage System (BESS) Parameters

| Parameter | Symbol | Medium Scale | Large Utility Scale | Unit | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Power Capacity** | $P_{max}^{BESS}$ | **2.00** | **40.00** | MW | Max charging/discharging rate |
| **Energy Capacity** | $E_{max}^{BESS}$ | **4.00** | **200.00** | MWh | Total nominal battery capacity |
| **Min State of Charge** | $SoC_{min}$ | **0.40** (10%) | **20.00** (10%) | MWh | Lower safety depth of discharge limit |
| **Max State of Charge** | $SoC_{max}$ | **3.60** (90%) | **200.00** (100%) | MWh | Upper state of charge limit |
| **Initial State of Charge** | $SoC_0$ | **0.63** | **20.00** | MWh | Initial stored energy at $t=0$ |
| **Charging Efficiency** | $\eta_{ch}$ | **80.0%** | **80.0%** | % | Charge conversion efficiency |
| **Discharging Efficiency** | $\eta_{dis}$ | **95.0%** | **95.0%** | % | Discharge conversion efficiency |
| **Round-Trip Efficiency** | $\eta_{BESS}$ | **76.0%** | **76.0%** | % | Overall efficiency ($\eta_{ch} \cdot \eta_{dis}$) |
| **Throughput Cost** | $C_{TP}$ | **20.00 – 50.00** | **20.00 – 50.00** | €/MWh | Cell degradation throughput cost |

---

### 3.3 Industrial Factory & Time-of-Use (TOU) Tariff Parameters

| Parameter | Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Grid Substation Line Cap** | $Q_{md}$ | **400.00** | MW | Maximum power line import capacity |
| **Parallel Machines** | $M$ | **5** | Units | E.g., parallel smelting furnaces |
| **Plant Base Load** | $l_{base,t}$ | **2.50 – 4.50** | MW | Non-shiftable continuous base consumption |
| **Batch Task Power Load** | $d_{m,p}$ | **60.00 – 110.00** | MW | Power required per task across machines |
| **Batch Task Duration** | $td_{m,p}$ | **1.50 – 4.90** | Hours | Non-interruptible task processing time |
| **Inter-Batch Buffer** | $t_{buffer}$ | **1.00** | Hour | Setup/cooling buffer between batches |

#### Time-of-Use (TOU) Electricity Tariffs (€/MWh):
* **Off-Peak (02:00–04:00, 12:00–14:00, 18:00–20:00, 22:00–24:00):** Buy Price = **66.42 €/MWh**, Sell Price = **36.53 €/MWh**
* **Flat-Peak (04:00–06:00):** Buy Price = **88.56 €/MWh**, Sell Price = **48.71 €/MWh**
* **Mid-Peak (00:00–02:00, 06:00–08:00, 14:00–16:00, 20:00–22:00):** Buy Price = **118.08 €/MWh**, Sell Price = **64.94 €/MWh**
* **On-Peak (08:00–12:00, 16:00–18:00):** Buy Price = **177.12 €/MWh**, Sell Price = **97.42 €/MWh**

---

## 4. Implementation Guidelines & Solvers

1. **Modeling Framework:** Formulate in **Pyomo (Python)** or **GAMS** as a Mixed-Integer Linear Program (MILP).
2. **Solvers:** Compatible with commercial solvers (**CPLEX**, **Gurobi**) or open-source solvers (**CBC**, **HiGHS**).
3. **Optimality Gap:** Recommended relative MIP gap tolerance set to $\le 0.01\%$ for exact global scheduling.
