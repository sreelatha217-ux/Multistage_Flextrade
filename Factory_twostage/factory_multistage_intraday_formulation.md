# Multi-Stage Stochastic MILP Formulation for Industrial Factory, Microturbine, and BESS (Up to Intraday Market Stage)

## 1. Executive Overview & Multi-Stage Decision Architecture

This document presents the complete mathematical formulation for co-optimizing an **industrial batch manufacturing plant**, an on-site **Microturbine (MT)**, and a **Battery Energy Storage System (BESS)** across sequential electricity market stages. Building upon the baseline Day-Ahead model defined in `factory-mt-bess-dayahead-scheduling-v6md.pdf` [14], this formulation extends the architecture into a **multi-stage stochastic mixed-integer linear programming (MILP)** framework up to the **Intraday Market Stage**, strictly adhering to `industrial-large-consumer-multistage-optimizationmd.pdf` [15].

In accordance with market gate closures and progressive uncertainty realization, the decision timeline up to Intraday operations is structured into two main stages:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-Market Commitments & Baseline (Day-Ahead Stage: "Here-and-Now")              │
│ • Submit Day-Ahead (DA) market purchase/sale quantities: P_t^{DA,buy}, P_t^{DA,sell}      │
│ • Microturbine Unit Commitment: Binary online/offline/start/stop status u_t, x_t, y_t ∈ {0,1} │
│ • Master Production Schedule: Time-indexed batch start decisions s_{m,p,k} ∈ {0,1}         │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Intraday Re-Trading & Operational Recourse ("1st Wait-and-See", Scenarios s ∈ S)│
│ • Submit Intraday buy/sell adjustment quantities: P_{buy,t,s}^{ID}, P_{sell,t,s}^{ID}      │
│ • Microturbine Power Generation Re-Dispatch: P_{MT,t,s} within Stage 1 commitment bounds   │
│ • BESS Recourse Dispatch: Re-optimize charging/discharging P_{ch,t,s}^{BESS}, P_{dis,t,s}^{BESS}│
│ • Flexible Batch Load Shifting & Execution: L_{batch,t,s} and Warehouse Stock I_{t,s}       │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sets, Parameters, and Decision Variables

### 2.1 Sets and Indices
* $t \in T = \{1, 2, \dots, T\}$: Hourly time periods over the scheduling horizon ($T = 24\text{ h}$).
* $s \in S$: Stochastic scenarios representing realizations of intraday prices and operational fluctuations.
* $m \in M$: Parallel industrial processing machines.
* $p \in P$: Production batch tasks to be executed.
* $k \in K$: Candidate batch start time slots $a_k = k \Delta$ (where $\Delta = 0.5\text{ h}$).
* $b \in B$: Piecewise power output blocks for the Microturbine.

### 2.2 Parameters
* $\pi_s$: Probability of scenario $s$ ($\sum_{s \in S} \pi_s = 1$).
* $\lambda_t^{DA,buy}, \lambda_t^{DA,sell}$: Day-Ahead electricity purchase and sale prices (€/MWh).
* $\lambda_{buy,t,s}^{ID}, \lambda_{sell,t,s}^{ID}$: Intraday electricity adjustment purchase and sale prices in scenario $s$ (€/MWh).
* $l_{base,t}$: Base factory non-shiftable electrical load (MW).
* $d_{m,p}$: Power demand of task $p$ on machine $m$ (MW).
* $td_{m,p}$: Processing duration of task $p$ on machine $m$ (hours).
* $t_{buf}$: Mandatory inter-batch setup/cooling buffer time on each machine (hours).
* $Y_p$: Product yield resulting from completing task $p$ (units).
* $\delta_t$: Hourly product delivery demand required from the warehouse (units/h).
* $I_{max}, I_0$: Warehouse storage capacity and initial product stock (units).
* $P_{MT,min}, P_{MT,max}$: Minimum and maximum Microturbine electrical output (MW).
* $w_b$: Upper capacity bound for piecewise block $b$ of the MT (MW).
* $C_0$: MT no-load operating cost coefficient (€/h).
* $C_b$: Marginal fuel/operating cost for MT output block $b$ (€/MWh).
* $SUC, SDC$: MT start-up cost and shut-down cost (€/event).
* $RU, RD$: MT ramp-up and ramp-down limits during continuous online operation (MW/h).
* $SRU, SRD$: MT start-up and shut-down ramping limits (MW/h).
* $MUT, MDT$: Minimum up-time and minimum down-time for MT (hours).
* $\bar{P}$: Maximum BESS charge/discharge power capacity (MW).
* $SoC_{min}, SoC_{max}, SoC_0$: Minimum, maximum, and initial State of Charge of the BESS (MWh).
* $\eta_{ch}, \eta_{dis}$: BESS charging and discharging efficiency factors ($0 < \eta < 1$).
* $C_{TP}$: BESS throughput degradation cost coefficient (€/MWh).
* $Q_{buy}, Q_{sell}$: Maximum Day-Ahead grid import/export capacity limits (MW).
* $Cap_{buy}^{ID}, Cap_{sell}^{ID}$: Maximum Intraday re-trading buy/sell capacity limits (MW).

### 2.3 Decision Variables

#### Stage 1: Pre-Market Commitments ("Here-and-Now")
* $P_t^{DA,buy} \ge 0$: Day-Ahead power purchased from the grid in hour $t$ (MW).
* $P_t^{DA,sell} \ge 0$: Day-Ahead power sold to the grid in hour $t$ (MW).
* $u_t \in \{0,1\}$: Binary status indicating if the Microturbine is online ($1$) or offline ($0$) in hour $t$.
* $x_t \in \{0,1\}$: Binary start-up trigger for the Microturbine in hour $t$.
* $y_t \in \{0,1\}$: Binary shut-down trigger for the Microturbine in hour $t$.
* $s_{m,p,k} \in \{0,1\}$: Binary decision to start task $p$ on machine $m$ at time candidate $a_k$.

#### Stage 2: Intraday Operational Recourse ("1st Wait-and-See", Scenario-Dependent)
* $P_{buy,t,s}^{ID} \ge 0$: Additional power purchased in the Intraday market in hour $t$, scenario $s$ (MW).
* $P_{sell,t,s}^{ID} \ge 0$: Additional power sold in the Intraday market in hour $t$, scenario $s$ (MW).
* $P_{MT,t,s} \ge 0$: Total active electrical generation from the MT in hour $t$, scenario $s$ (MW).
* $P_{MT,b,t,s} \ge 0$: Power output generated in block $b$ of the MT in hour $t$, scenario $s$ (MW).
* $P_{ch,t,s}^{BESS} \ge 0$: BESS active charging power in hour $t$, scenario $s$ (MW).
* $P_{dis,t,s}^{BESS} \ge 0$: BESS active discharging power in hour $t$, scenario $s$ (MW).
* $SoC_{t,s} \ge 0$: BESS State of Charge at the end of hour $t$, scenario $s$ (MWh).
* $v_{ch,t,s}, v_{dis,t,s} \in \{0,1\}$: Binary BESS operational mode indicators (charging vs. discharging).
* $L_{batch,t,s} \ge 0$: Total factory industrial batch power consumption in hour $t$, scenario $s$ (MW).
* $nprod_{t,s} \ge 0$: Quantity of finished product units completed in hour $t$, scenario $s$ (units).
* $I_{t,s} \ge 0$: End-of-hour warehouse product inventory level in hour $t$, scenario $s$ (units).

---

## 3. Multi-Stage Objective Function

The objective function minimizes total deterministic Stage 1 Day-Ahead commitment costs plus the expected Stage 2 Intraday trading adjustments, Microturbine operation and commitment costs, and BESS degradation expenses across all scenarios $s \in S$:

$$\min \sum_{t=1}^T \mathcal{C}_t^{DA} + \sum_{s \in S} \pi_s \sum_{t=1}^T \left( \mathcal{C}_{t,s}^{ID} + \mathcal{C}_{t,s}^{MT} + \mathcal{C}_{t,s}^{BESS} \right)$$

### Cost Component Formulations

1. **Day-Ahead Energy Procurement Cost (€)**:
   $$\mathcal{C}_t^{DA} = \lambda_t^{DA,buy} P_t^{DA,buy} - \lambda_t^{DA,sell} P_t^{DA,sell} \quad \forall t \in T$$

2. **Intraday Re-Trading Adjustment Cost (€)**:
   $$\mathcal{C}_{t,s}^{ID} = \lambda_{buy,t,s}^{ID} P_{buy,t,s}^{ID} - \lambda_{sell,t,s}^{ID} P_{sell,t,s}^{ID} \quad \forall t \in T, \forall s \in S$$

3. **Microturbine Fuel, Operating & Unit Commitment Cost (€)**:
   $$\mathcal{C}_{t,s}^{MT} = C_0 P_{MT,min} u_t + \sum_{b \in B} C_b P_{MT,b,t,s} + SUC \cdot x_t + SDC \cdot y_t \quad \forall t \in T, \forall s \in S$$

4. **BESS Throughput Degradation Cost (€)**:
   $$\mathcal{C}_{t,s}^{BESS} = C_{TP} \left( P_{ch,t,s}^{BESS} + P_{dis,t,s}^{BESS} \right) \quad \forall t \in T, \forall s \in S$$

---

## 4. System Power Balance Constraint

For every hour $t \in T$ and scenario $s \in S$, total net power imported from the grid (Day-Ahead and Intraday) plus on-site generation and BESS discharge must exactly balance factory consumption (base load plus active batch load) and BESS charging:

$$\left( P_t^{DA,buy} - P_t^{DA,sell} \right) + \left( P_{buy,t,s}^{ID} - P_{sell,t,s}^{ID} \right) + P_{MT,t,s} + P_{dis,t,s}^{BESS} - P_{ch,t,s}^{BESS} = l_{base,t} + L_{batch,t,s} \quad \forall t \in T, \forall s \in S$$

---

## 5. Microturbine (MT) Constraints

Microturbine binary unit commitment decisions ($u_t, x_t, y_t$) are established in Stage 1 ("Here-and-Now"), while active power generation ($P_{MT,t,s}$) is adjusted dynamically in Stage 2 across scenarios.

### 5.1 Output Capacity & Piecewise Linearization
$$P_{MT,t,s} = P_{MT,min} u_t + \sum_{b \in B} P_{MT,b,t,s} \quad \forall t \in T, \forall s \in S$$
$$0 \le P_{MT,b,t,s} \le w_b u_t \quad \forall b \in B, \forall t \in T, \forall s \in S$$

### 5.2 Unit Commitment Logic & Status Consistency (Stage 1)
$$x_t - y_t = u_t - u_{t-1} \quad \forall t \in T \quad (u_0 = 0)$$
$$x_t + y_t \le 1 \quad \forall t \in T$$
$$u_t, x_t, y_t \in \{0, 1\} \quad \forall t \in T$$

### 5.3 Minimum Up-Time and Minimum Down-Time Limits (Stage 1)
To ensure physical thermal stability, minimum up/down time windows are enforced across all hours $t \in T$ (truncated at $t = 1$):
$$\sum_{\tau=\max(1, t-MUT+1)}^t x_\tau \le u_t \quad \forall t \in T$$
$$\sum_{\tau=\max(1, t-MDT+1)}^t y_\tau \le 1 - u_t \quad \forall t \in T$$

### 5.4 Scenario-Dependent Inter-temporal Ramp Rate Limits (Stage 2)
Power output adjustments between consecutive hours are bounded by operational ramp rates and start-up/shut-down limits:
$$P_{MT,t,s} - P_{MT,t-1,s} \le RU \cdot u_{t-1} + SRU \cdot x_t \quad \forall t \in T, \forall s \in S \quad (P_{MT,0,s} = 0)$$
$$P_{MT,t-1,s} - P_{MT,t,s} \le RD \cdot u_t + SRD \cdot y_t \quad \forall t \in T, \forall s \in S$$

---

## 6. Battery Energy Storage System (BESS) Recourse Constraints

BESS charging, discharging, and State of Charge (SoC) profiles are re-optimized in Stage 2 to capture intraday market price spreads and compensate for batch load variations.

### 6.1 State of Charge Dynamics
$$SoC_{t,s} = SoC_{t-1,s} + \eta_{ch} P_{ch,t,s}^{BESS} \Delta t - \frac{1}{\eta_{dis}} P_{dis,t,s}^{BESS} \Delta t \quad \forall t \in T, \forall s \in S \quad (SoC_{0,s} = SoC_0)$$

### 6.2 Capacity Bounds & Terminal Energy Requirement
$$SoC_{min} \le SoC_{t,s} \le SoC_{max} \quad \forall t \in T, \forall s \in S$$
$$SoC_{T,s} \ge SoC_0 \quad \forall s \in S$$

### 6.3 Power Limits & Non-Simultaneous Operation
$$0 \le P_{ch,t,s}^{BESS} \le \bar{P} v_{ch,t,s} \quad \forall t \in T, \forall s \in S$$
$$0 \le P_{dis,t,s}^{BESS} \le \bar{P} v_{dis,t,s} \quad \forall t \in T, \forall s \in S$$
$$v_{ch,t,s} + v_{dis,t,s} \le 1 \quad (v_{ch,t,s}, v_{dis,t,s} \in \{0,1\}) \quad \forall t \in T, \forall s \in S$$

---

## 7. Industrial Batch Production & Warehouse Constraints

Adopting the precise **time-indexed batch scheduling formulation** from `factory-mt-bess-dayahead-scheduling-v6md.pdf` [14]:

### 7.1 Single Task Assignment & Non-Overlap with Buffer
1. **Task Execution Bounds**: Each manufacturing task runs at most once across all available machines and start candidates:
   $$\sum_{m \in M} \sum_{k \in K} s_{m,p,k} \le 1 \quad \forall p \in P$$

2. **Machine Non-Overlap & Cooling Buffer**: A batch started at time $a_k$ blocks machine $m$ for all candidate start times $a_j \in [a_k, a_k + td_{m,p} + t_{buf})$:
   $$\sum_{p \in P} \sum_{k: a_k \le a_j < a_k + td_{m,p} + t_{buf}} s_{m,p,k} \le 1 \quad \forall m \in M, \forall j \in K$$

### 7.2 Hourly Batch Power Load & Product Yield
Let $ov_{m,p,k,t}$ denote the exact continuous fraction of hour $t$ occupied by task $p$ started at candidate time $a_k = k \Delta$ on machine $m$:
$$ov_{m,p,k,t} = \max\left(0, \min(a_k + td_{m,p}, t) - \max(a_k, t - 1)\right) \in [0, 1]$$

The resulting active batch power load $L_{batch,t,s}$ and hourly product yield $nprod_{t,s}$ are:
$$L_{batch,t,s} = \sum_{m \in M} \sum_{p \in P} \sum_{k \in K} d_{m,p} \cdot ov_{m,p,k,t} \cdot s_{m,p,k} \quad \forall t \in T, \forall s \in S$$
$$nprod_{t,s} = \sum_{m \in M} \sum_{p \in P} \sum_{k \in K} \frac{Y_p}{td_{m,p}} \cdot ov_{m,p,k,t} \cdot s_{m,p,k} \quad \forall t \in T, \forall s \in S$$

### 7.3 Warehouse Product Inventory Balance
Finished product inventory accumulates from hourly manufacturing output and is drawn down by fixed hourly customer deliveries $\delta_t$:
$$I_{t,s} = I_{t-1,s} + nprod_{t,s} - \delta_t \quad \forall t \in T, \forall s \in S \quad (I_{0,s} = I_0)$$
$$0 \le I_{t,s} \le I_{max} \quad \forall t \in T, \forall s \in S$$
$$I_{T,s} \ge I_0 \quad \forall s \in S$$

---

## 8. Trading Limits & Non-Anticipativity Constraints

### 8.1 Grid Trading Capacity Bounds
$$0 \le P_t^{DA,buy} \le Q_{buy}, \quad 0 \le P_t^{DA,sell} \le Q_{sell} \quad \forall t \in T$$
$$0 \le P_{buy,t,s}^{ID} \le Cap_{buy}^{ID}, \quad 0 \le P_{sell,t,s}^{ID} \le Cap_{sell}^{ID} \quad \forall t \in T, \forall s \in S$$

### 8.2 Day-Ahead Bidding Curve Monotonicity & Non-Anticipativity
To guarantee valid market bidding curves submitted prior to price realization:
1. **Monotonicity**:
   $$P_t^{DA,buy}(\omega_1) \le P_t^{DA,buy}(\omega_2) \quad \text{for scenarios where } \lambda_t^{DA}(\omega_1) \le \lambda_t^{DA}(\omega_2)$$
2. **Non-Anticipativity**:
   $$P_t^{DA,buy}(\omega_1) = P_t^{DA,buy}(\omega_2) \quad \text{for scenarios where } \lambda_t^{DA}(\omega_1) = \lambda_t^{DA}(\omega_2)$$

---

## 9. Mathematical Comparison & Variable Mapping

The table below summarizes how the baseline Day-Ahead model (`factory-mt-bess-dayahead-scheduling-v6md.pdf` [14]) is transformed into the multi-stage stochastic formulation up to Intraday (`industrial-large-consumer-multistage-optimizationmd.pdf` [15]):

| Model Component | Baseline Day-Ahead Model (`v6md`) | Multi-Stage Intraday Model (`multistage`) | Decision Stage |
| :--- | :--- | :--- | :--- |
| **Market Power Import/Export** | $P_t^{buy}, P_t^{sell}$ | $P_t^{DA,buy}, P_t^{DA,sell}$ (DA) / $P_{buy,t,s}^{ID}, P_{sell,t,s}^{ID}$ (ID) | Stage 1 (DA) / Stage 2 (ID) |
| **MT Commitment Status** | $u_t, x_t, y_t \in \{0,1\}$ | $u_t, x_t, y_t \in \{0,1\}$ (Fixed across scenarios) | Stage 1 (Here-and-Now) |
| **MT Output Power** | $P_{MT,t}, P_{MT,b,t}$ | $P_{MT,t,s}, P_{MT,b,t,s}$ (Re-dispatched per scenario) | Stage 2 (Operational Recourse) |
| **BESS Charge/Discharge** | $P_t^{ch}, P_t^{dis}, SoC_t$ | $P_{ch,t,s}^{BESS}, P_{dis,t,s}^{BESS}, SoC_{t,s}$ (Recourse dispatch) | Stage 2 (Operational Recourse) |
| **Batch Schedule Decisions** | $s_{m,p,k} \in \{0,1\}$ | $s_{m,p,k} \in \{0,1\}$ (Fixed baseline master schedule) | Stage 1 (Here-and-Now) |
| **Factory Batch Load** | $L_{batch,t}$ | $L_{batch,t,s}$ | Stage 2 (Operational Recourse) |
| **Warehouse Inventory** | $I_t$ | $I_{t,s}$ | Stage 2 (Operational Recourse) |

---

## References

1. **factory-mt-bess-dayahead-scheduling-v6md.pdf** [14]: Mixed-Integer Linear Programming (MILP) model for co-optimizing industrial batch scheduling, Microturbine, and BESS in the Day-Ahead market.
2. **industrial-large-consumer-multistage-optimizationmd.pdf** [15]: Multi-stage stochastic optimization framework for large industrial consumers in Day-Ahead, Intraday, and Balancing markets.
