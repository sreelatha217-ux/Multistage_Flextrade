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
│ • (No BESS baseline: the net DA position fixes purchases, the BESS is Stage-2 recourse)   │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Intraday Re-Trading & Operational Recourse (Intraday Stage: "1st Wait-and-See") │
│ • Submit Intraday buy/sell adjustments: P_{buy,t,s}^{ID}, P_{sell,t,s}^{ID}               │
│ • Microturbine Output Adjustment: Re-dispatch MT generation P_{MT,t,s} against ID prices  │
│ • (Batch schedule stays fixed from Stage 1; load shifting would need scenario-indexed     │
│   start binaries and is not part of this model)                                            │
│ • BESS Recourse Dispatch: Re-adjust charge/discharge to capture intraday price spreads    │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Real-Time Balancing Settlement (Real-Time Stage: "2nd Wait-and-See")            │
│ • Real-time load deviation η_{t,s,w} and price ratios r^+, r^- are revealed (scenario w)  │
│ • Measure deviation between net market commitment (P^{DA} + P^{ID}) and physical use      │
│ • Settle deficit Δ^+ (bought at r^+·λ^{DA,buy}, r^+ ≥ 1) and surplus Δ^- (sold at         │
│   r^-·λ^{DA,sell}, r^- ≤ 1). Stage-2 variables carry s only, Stage-3 variables (s, w)     │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Objective Function: Total Expected System Cost Minimization

The primary objective is to minimize the expected total electricity procurement costs, microturbine fuel and operational expenses, and battery aging costs across all time periods $t \in T$ over the scenario tree: intraday scenarios $s$ (probability $\pi_s$) and, for each $s$, real-time scenarios $w$ (conditional probability $\pi_{w|s}$):

$$\min \sum_t \left(\mathcal{C}_t^{DA} + \mathcal{C}_t^{MT,1}\right) + \sum_s \pi_s \sum_t \left(\mathcal{C}_{t,s}^{ID} + \mathcal{C}_{t,s}^{MT,2} + \mathcal{C}_{t,s}^{BESS}\right) + \sum_s \pi_s \sum_w \pi_{w|s} \sum_t \mathcal{C}_{t,s,w}^{BAL}$$

#### Detailed Cost Components:

1. **Day-Ahead Energy Trading Cost (€):**
   $$\mathcal{C}_t^{DA} = \lambda_t^{DA} \cdot P_t^{DA}$$
   *(where $P_t^{DA} > 0$ denotes power purchase from the grid, and $P_t^{DA} < 0$ denotes power sale back to the grid).*

2. **Intraday Market Re-Trading Net Cost (€):**
   $$\mathcal{C}_{t,s}^{ID} = \lambda_{t,s}^{ID} \cdot \left( P_{buy,t,s}^{ID} - P_{sell,t,s}^{ID} \right)$$

3. **Real-Time Imbalance Settlement Cost (€):**
   $$\mathcal{C}_{t,s,w}^{BAL} = r_{t,s,w}^+ \lambda_t^{DA,buy} \Delta_{t,s,w}^+ - r_{t,s,w}^- \lambda_t^{DA,sell} \Delta_{t,s,w}^-$$
   *($\Delta^+ \ge 0$ is the deficit bought from the system, with penalty ratio $r^+ \ge 1$; $\Delta^- \ge 0$ is the surplus sold, with discount ratio $0 \le r^- \le 1$. Separate buy and sell base prices are required: a single $\lambda^{DA}$ would let a surplus earn $r\lambda^{DA,buy}$, more than any DA or ID sale.)*

4. **Microturbine Fuel, Start-Up, and Shut-Down Costs (€):**
   $$\mathcal{C}_t^{MT,1} = C_0 P_{MT,min} u_{MT,t} + SUC \cdot x_{MT,t} + SDC \cdot y_{MT,t}, \qquad \mathcal{C}_{t,s}^{MT,2} = \sum_{b=1}^{N_b} C_{b}^{MT} \cdot P_{MT,b,t,s}$$
   *(where $C_{b}^{MT}$ is the marginal cost in €/MWh of block $b$, $C_0$ = 48.41 €/MWh is the fuel rate on the minimum output so $C_0 P_{MT,min}$ = 484.10 €/h while online, $SUC$ is the start-up cost and $SDC$ the shut-down cost. Block index $b$ is used so it does not clash with machine index $m$.)*

5. **Battery Degradation Throughput Aging Cost (€):**
   $$\mathcal{C}_{t,s}^{BESS} = C_{TP} \cdot \left( P_{ch,t,s}^{BESS} + P_{dis,t,s}^{BESS} \right)$$
   *(where $C_{TP}$ is the throughput degradation cost in €/MWh).*

---

### 2.2 System Power Balance Constraints

At every hour $t$ and scenario pair $(s,w)$, market positions, on-site generation and the settled imbalance must match factory consumption and storage charging:

$$P_t^{DA} + P_{buy,t,s}^{ID} - P_{sell,t,s}^{ID} + P_{MT,t,s} + P_{dis,t,s}^{BESS} - P_{ch,t,s}^{BESS} + \Delta_{t,s,w}^+ - \Delta_{t,s,w}^- = (1+\eta_{t,s,w})\, P_{factory,t}$$

where $\eta$ is the zero-mean relative real-time load deviation and total factory consumption is the base load plus the batch load, which is fixed in Stage 1 and has no scenario index:

$$P_{factory,t} = l_{base,t} + \sum_{m \in M} \sum_{n \in N} l_{m,n,t}$$

Grid limits apply to the day-ahead position, the market net position and the physical exchange:
$$-Q_{sell} \le P^{DA}_t + P^{ID}_{buy,t,s} - P^{ID}_{sell,t,s} \le Q_{buy}, \qquad -Q_{sell} \le P^{DA}_t + P^{ID}_{buy,t,s} - P^{ID}_{sell,t,s} + \Delta^+_{t,s,w} - \Delta^-_{t,s,w} \le Q_{buy}$$
$\Delta^+ \le \bar\Delta_{buy}\, z$, $\Delta^- \le \bar\Delta_{sell}(1-z)$ with $z \in \{0,1\}$ is optional: with $r^+ \ge 1 \ge r^-$ and $\lambda^{buy} \ge \lambda^{sell}$ simultaneous buying and selling never pays. Default $\bar\Delta = Q_{buy} + Q_{sell}$.

Two modes: *strategic* (Stage 2 may plan an unbalanced position) and *passive* (add supply $= (1 + E_w[\eta])\, P_{factory}$ so $\Delta$ carries only the realised deviation).

---

### 2.3 Industrial Factory & Batch Scheduling Constraints

Industrial load is modeled as parallel processing machines $m \in M$ executing batch production tasks $n \in N$.

1. **Task Duration and Machine Power Demand:**
   $$l_{m,n,t} = d_{m,p} \cdot o_{m,n,t,p}$$
   where $d_{m,p}$ is the power demand (MW) of task $p$ on machine $m$, and $o_{m,n,t,p}$ is the processing fraction of hour $t$ (in [0,1], exact overlap with the batch interval). The solver uses an equivalent time-indexed form with start binaries on a 0.5 h grid.

2. **Single Task Assignment per Machine Sequence:**
   $$\sum_{p \in P} X_{m,n,p} = 1 \quad \forall m \in M, n \in N$$

3. **Batch Non-Overlap & Maintenance/Cooling Buffer Constraint:**
   $$ts_{m,n} \ge ts_{m,n-1} + td_{m,p} \cdot X_{m,n-1,p} + t_{buffer}$$
   where $ts_{m,n}$ is the task start time, $td_{m,p}$ is task processing duration, and $t_{buffer}$ is the mandatory inter-batch setup/cooling buffer.

4. **Product Warehouse Inventory Balance:**
   $$I_t = I_{t-1} + n_{prod,t} - n_{prod,dem,t}, \qquad 0 \le I_t \le I_{max}, \qquad I_T \ge I_0$$
   where $I_t$ is stored product inventory (named $I$ so it does not clash with the batch-position set $N$), $n_{prod,t}$ is newly completed units, and $n_{prod,dem,t}$ is the required delivery schedule. The real-time deviation $\eta$ scales the metered load only.

---

### 2.4 Microturbine (MT) Operational Constraints

1. **Piecewise Linear Power Block Limits:**
   $$P_{MT,t,s} = P_{MT,min} \cdot u_{MT,t} + \sum_{b=1}^{N_b} P_{MT,b,t,s}$$
   $$0 \le P_{MT,b,t,s} \le P_{MT,b,max} \cdot u_{MT,t} \quad \forall b \in \{1, \dots, N_b\}$$
   The block widths must sum to $P_{MT,max} - P_{MT,min}$ (15 MW), and block costs must be non-decreasing.

2. **Ramp-Up and Ramp-Down Rate Limits:**
   $$P_{MT,t,s} - P_{MT,t-1,s} \le RU_{MT} \cdot u_{MT,t-1} + SRU_{MT} \cdot x_{MT,t}$$
   $$P_{MT,t-1,s} - P_{MT,t,s} \le RD_{MT} \cdot u_{MT,t} + SRD_{MT} \cdot y_{MT,t}$$

3. **Minimum Up Time (MUT) and Minimum Down Time (MDT):**
   $$\sum_{\tau = \max(1,\,t - MUT + 1)}^{t} x_{MT,\tau} \le u_{MT,t} \quad \forall t$$
   $$\sum_{\tau = \max(1,\,t - MDT + 1)}^{t} y_{MT,\tau} \le 1 - u_{MT,t} \quad \forall t$$
   (Windows are truncated at $t=1$. Restricting to $t \ge MUT$ would leave starts in the first hours unconstrained.)

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

These apply only when day-ahead prices are scenario-dependent. With deterministic $\lambda^{DA}$ and one $P^{DA}_t$ for all scenarios (the model here), non-anticipativity holds by construction and monotonicity has nothing to act on.

1. **Non-Decreasing Bidding Curve Condition** (scenario-dependent DA prices $\lambda^{DA}_t(\omega)$):
   $$P_{t}^{DA}(\omega_1) \le P_{t}^{DA}(\omega_2) \quad \text{for scenarios where } \lambda_{t}^{DA}(\omega_1) \le \lambda_{t}^{DA}(\omega_2)$$

2. **Equal prices, equal quantities:**
   $$P_{t}^{DA}(\omega_1) = P_{t}^{DA}(\omega_2) \quad \text{for scenarios where } \lambda_{t}^{DA}(\omega_1) = \lambda_{t}^{DA}(\omega_2)$$
   (The earlier version required equality for all scenarios, which contradicts condition 1.)

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

#### MT Piecewise Linear Fuel Cost Curve Blocks (€/MWh), widths sum to $P_{max} - P_{min}$ = 15 MW:
* **Block 1 ($b=1$):** Width = **5.30 MW**, Marginal Cost = **48.41 €/MWh**
* **Block 2 ($b=2$):** Width = **4.70 MW**, Marginal Cost = **48.78 €/MWh**
* **Block 3 ($b=3$):** Width = **5.00 MW**, Marginal Cost = **51.84 €/MWh**
* **No-load fuel rate** $C_0$ = **48.41 €/MWh** on $P_{min}$ (484.10 €/h while online)

*(Earlier versions listed four blocks of 5.3/7.2/7.2/5.3 MW = 25 MW. Added to $P_{min}$ = 10 MW that allows 35 MW, above the 25 MW rating.)*

---

### 3.2 Battery Energy Storage System (BESS) Parameters

| Parameter | Symbol | Medium Scale | Large Utility Scale | Unit | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Power Capacity** | $P_{max}^{BESS}$ | **2.00** | **40.00** | MW | Max charging/discharging rate |
| **Energy Capacity** | $E_{max}^{BESS}$ | **4.00** | **200.00** | MWh | Total nominal battery capacity |
| **Min State of Charge** | $SoC_{min}$ | **0.40** (10%) | **20.00** (10%) | MWh | Lower safety depth of discharge limit |
| **Max State of Charge** | $SoC_{max}$ | **3.60** (90%) | **180.00** (90%) | MWh | Upper state of charge limit |
| **Initial State of Charge** | $SoC_0$ | **0.63** | **20.00** | MWh | Initial stored energy; also the required terminal SoC |
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

### 3.4 Real-Time Balancing Parameters (assumed defaults, no source value exists)

| Parameter | Symbol | Default | Description |
| :--- | :--- | :--- | :--- |
| Load deviation | $\eta_{t,s,w}$ | std 3 %, AR(1) $\rho$ = 0.6, mean 0 | Relative deviation of factory load |
| Deficit ratio | $r^+$ | $1 + 0.25\,g$, $g$ lognormal (mean 1, $\sigma$ 0.3) | Penalty on bought deficit, $\ge 1$ |
| Surplus ratio | $r^-$ | $1 - 0.25\,g$ clipped to [0,1] | Discount on sold surplus |
| RT scenarios per ID scenario | $\lvert W \rvert$ | 5 | One zero path and two antithetic pairs |
| Imbalance bound | $\bar\Delta$ | $Q_{buy}+Q_{sell}$ | Non-binding default |
| ID market depth | $Cap^{ID}$ | 100 MW per direction | Each direction |

Real imbalance prices depend on the system direction. Replace the independent draw with calibrated data when available.

## 4. Implementation Guidelines & Solvers

1. **Modeling Framework:** Formulate in **Pyomo (Python)** or **GAMS** as a Mixed-Integer Linear Program (MILP).
2. **Solvers:** Compatible with commercial solvers (**CPLEX**, **Gurobi**) or open-source solvers (**CBC**, **HiGHS**).
3. **Optimality Gap:** Recommended relative MIP gap tolerance set to $\le 0.01\%$ for exact global scheduling.

## 5. Reference implementation

The package implementation is organized under `src/factory_multistage/`, with separate data, model, solver, result, and CLI modules for each market stage. From this directory, install the package with `uv sync --extra test`; run `uv run factory-mt-da`, `uv run factory-mt-id`, or `uv run factory-mt-rt`. The original `factory_mt_*_scheduler.py` files remain compatibility launchers.
