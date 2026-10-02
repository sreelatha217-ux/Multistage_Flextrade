---
name: factory-mt-bess-dayahead-scheduling-v5
description: Complete MILP optimization model for Industrial Factory Batch Scheduling, Microturbine, and BESS in Day-Ahead Electricity Market with bidirectional grid trading and battery degradation.
---

# SKILL: Factory, Microturbine & BESS Day-Ahead Energy Scheduling Optimization (v5)

This skill provides a complete Mixed-Integer Linear Programming (MILP) mathematical formulation for co-optimizing an **industrial factory batch manufacturing schedule**, an on-site **Microturbine (MT)**, and a **Battery Energy Storage System (BESS)** participating in **Day-Ahead (DA) electricity markets** with **bidirectional grid power trading (import & export)** and **battery throughput degradation modeling**.

All monetary parameters are strictly expressed in **Euros (€)**.

---

## 1. Parameter & Benchmark Database

| Parameter Category | Symbol | Value / Benchmark Range | Unit | Description / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Scheduling Horizon** | $T$ | **24** | Hours | Daily scheduling horizon ($t \in \{1, \dots, 24\}$) |
| **Parallel Machines** | $M$ | **5** | Units | Parallel industrial smelting furnaces ($m \in \{m1, \dots, m5\}$) |
| **Production Task Set** | $P$ | **30** | Tasks | Total manufacturing tasks ($p \in \{p1, \dots, p30\}$) |
| **Batch Sequence Order**| $N$ | **1 – 10** | Positions | Sequential batch slots per machine ($n \in \{1, \dots, N_{max}\}$) |
| **Task Power Demand** | $d_{m,p}$ | **60.00 – 110.00** | MW | Power drawn by task $p$ on machine $m$ |
| **Task Processing Duration**| $td_{m,p}$ | **1.30 – 4.90** | Hours | Continuous processing time required for task $p$ |
| **Product Yield per Task** | $Y_p$ | **5 – 15** | Units/Task | Finished goods produced upon completing task $p$ |
| **Inter-Batch Buffer Time** | $t_{buffer}$ | **1.00** | Hour | Mandatory setup/cooling buffer between consecutive batches |
| **Non-Shiftable Base Load** | $l_{base,t}$ | **2.50 – 4.50** | MW | Facility auxiliary continuous baseline consumption |
| **Substation Import Limit** | $Q_{md}^{buy}$ | **400.00** | MW | Maximum substation power import capacity |
| **Substation Export Limit** | $Q_{md}^{sell}$| **400.00** | MW | Maximum substation power export capacity |
| **Warehouse Storage Limit**| $N_{max}$ | **500.00** | Units | Maximum warehouse product inventory limit |
| **Initial Product Stock** | $N_0$ | **50.00** | Units | Warehouse inventory at $t=0$ |
| **Hourly Customer Demand** | $n_{prod,dem,t}$| **8.00** | Units/Hour | Required product shipment schedule (192 units/day) |
| **MT Minimum Power** | $P_{MT,min}$ | **10.00** | MW | Minimum stable power output |
| **MT Maximum Power** | $P_{MT,max}$ | **25.00** | MW | Maximum power capacity limit |
| **MT Incremental Capacity** | $\Delta P_{MT,max}$| **15.00** | MW | Total incremental block capacity ($25 - 10 = 15\text{ MW}$) |
| **MT Base Minimum Fuel Rate**| $C_0^{MT}$ | **48.41** | €/MWh | Applied to $P_{MT,min} = 10\text{ MW}$ ($\rightarrow 484.10\ \text{€/h}$) |
| **MT Fuel Block 1** | $w_1, C_1^{MT}$ | **5.30 MW, 48.41** | €/MWh | Width = 5.30 MW, Marginal Cost = 48.41 €/MWh |
| **MT Fuel Block 2** | $w_2, C_2^{MT}$ | **4.70 MW, 48.78** | €/MWh | Width = 4.70 MW, Marginal Cost = 48.78 €/MWh |
| **MT Fuel Block 3** | $w_3, C_3^{MT}$ | **5.00 MW, 51.84** | €/MWh | Width = 5.00 MW, Marginal Cost = 51.84 €/MWh |
| **MT Ramp-Up Rate** | $RU_{MT}$ | **20.00** | MW/h | Operational upward ramp limit |
| **MT Ramp-Down Rate** | $RD_{MT}$ | **20.00** | MW/h | Operational downward ramp limit |
| **MT Start-Up / Shut-Down Cost**| $SUC, SDC$| **87.40 / 8.74** | € | Fixed start-up / shut-down cost |
| **MT Min Up / Down Time** | $MUT, MDT$| **4 / 2** | Hours | Mandatory online / offline duration |
| **BESS Power Capacity** | $P_{max}^{BESS}$ | **2.00 / 40.00** | MW | Max charging / discharging rate |
| **BESS Energy Capacity** | $E_{max}^{BESS}$ | **4.00 / 200.00**| MWh | Total nominal battery capacity |
| **BESS SoC Range** | $SoC_{min}, SoC_{max}$| **10% – 90%** | MWh | Lower depth of discharge and upper state-of-charge bounds |
| **BESS Initial SoC** | $SoC_0$ | **20.00** | MWh | Starting stored energy at $t=0$ |
| **BESS Charging Efficiency**| $\eta_{ch}$ | **80.0%** | % | Charge conversion efficiency |
| **BESS Discharging Efficiency**| $\eta_{dis}$ | **95.0%** | % | Discharge conversion efficiency |
| **BESS Round-Trip Efficiency**| $\eta_{BESS}$ | **76.0%** | % | Overall efficiency ($\eta_{ch} \cdot \eta_{dis}$) |
| **BESS Degradation Cost** | $C_{TP}$ | **20.00 – 50.00** | €/MWh | Cell throughput aging degradation cost |

### Day-Ahead Time-of-Use (TOU) Electricity Tariffs (€/MWh):
* **Off-Peak (02:00–04:00, 12:00–14:00, 18:00–20:00, 22:00–24:00):** Buy = **66.42 €/MWh**, Sell = **36.53 €/MWh**
* **Flat-Peak (04:00–06:00):** Buy = **88.56 €/MWh**, Sell = **48.71 €/MWh**
* **Mid-Peak (00:00–02:00, 06:00–08:00, 14:00–16:00, 20:00–22:00):** Buy = **118.08 €/MWh**, Sell = **64.94 €/MWh**
* **On-Peak (08:00–12:00, 16:00–18:00):** Buy = **177.12 €/MWh**, Sell = **97.42 €/MWh**

---

## 2. Decision Variables

* $P_{buy,t}^{DA} \ge 0$: Power purchased from grid in Day-Ahead market at hour $t$ (MW).
* $P_{sell,t}^{DA} \ge 0$: Power sold to grid in Day-Ahead market at hour $t$ (MW).
* $P_{MT,t} \ge 0$: Total power output generated by Microturbine at hour $t$ (MW).
* $P_{MT,m,t} \ge 0$: Power generated in incremental block $m$ of Microturbine at hour $t$ (MW).
* $u_{MT,t} \in \{0, 1\}$: Binary variable; $1$ if Microturbine is online at hour $t$, $0$ otherwise.
* $x_{MT,t}, y_{MT,t} \in \{0, 1\}$: Binary start-up and shut-down indicators for Microturbine at hour $t$.
* $P_{ch,t}^{BESS} \ge 0$: Power drawn to charge BESS at hour $t$ (MW).
* $P_{dis,t}^{BESS} \ge 0$: Power discharged from BESS at hour $t$ (MW).
* $SoC_t \ge 0$: State-of-charge stored energy in BESS at the end of hour $t$ (MWh).
* $v_{ch,t}, v_{dis,t} \in \{0, 1\}$: Binary charging and discharging status flags for BESS.
* $X_{m,n,p} \in \{0, 1\}$: Binary variable; $1$ if task $p$ is assigned to $n$-th batch position on machine $m$, $0$ otherwise.
* $ts_{m,n} \ge 0$: Scheduled start time of batch $n$ on machine $m$ (Hours).
* $o_{m,n,t,p} \in [0, 1]$: Execution time proportion indicator for task $p$ of $n$-th batch on machine $m$ during hour $t$.
* $n_{prod,t} \ge 0$: Finished goods completed during hour $t$ (Units).
* $N_t \ge 0$: Stored product inventory in warehouse at the end of hour $t$ (Units).

---

## 3. Complete Mathematical Model Formulation

### 3.1 Objective Function: Total Net Operating Cost Minimization
Minimize net grid power costs, microturbine fuel/commitment costs, and BESS throughput degradation costs:

$$\min \quad \mathcal{F}_{cost} = \sum_{t=1}^{T} \left( \underbrace{\lambda_{buy,t}^{DA} \cdot P_{buy,t}^{DA} - \lambda_{sell,t}^{DA} \cdot P_{sell,t}^{DA}}_{\text{Net Grid Electricity Cost}} + \underbrace{C_0^{MT} \cdot P_{MT,min} \cdot u_{MT,t} + \sum_{m=1}^{N_m} C_m^{MT} \cdot P_{MT,m,t} + SUC \cdot x_{MT,t} + SDC \cdot y_{MT,t}}_{\text{Microturbine Fuel, Start-Up \& Shut-Down Costs}} + \underbrace{C_{TP} \cdot \left( P_{ch,t}^{BESS} + P_{dis,t}^{BESS} \right)}_{\text{BESS Degradation Throughput Cost}} \right)$$

---

### 3.2 Substation Power Balance Constraint
At every hour $t$, net grid imports plus microturbine generation plus battery discharge must balance factory consumption plus battery charging plus grid power sales:

$$P_{buy,t}^{DA} - P_{sell,t}^{DA} + P_{MT,t} + P_{dis,t}^{BESS} - P_{ch,t}^{BESS} = l_{base,t} + \sum_{m \in M} \sum_{n \in N} \sum_{p \in P} \left( d_{m,p} \cdot o_{m,n,t,p} \right) \quad \forall t \in T$$

---

### 3.3 Battery Energy Storage System (BESS) Constraints

1. **State-of-Charge (SoC) Dynamic Balance:**
   $$SoC_t = SoC_{t-1} + \eta_{ch} \cdot P_{ch,t}^{BESS} \cdot \Delta t - \frac{1}{\eta_{dis}} \cdot P_{dis,t}^{BESS} \cdot \Delta t \quad \forall t \in T$$

2. **SoC Bounds & Terminal Energy Sustainability:**
   $$SoC_{min} \le SoC_t \le SoC_{max} \quad \forall t \in T$$
   $$SoC_T \ge SoC_0$$

3. **Charging and Discharging Power Limits:**
   $$0 \le P_{ch,t}^{BESS} \le P_{max}^{BESS} \cdot v_{ch,t} \quad \forall t \in T$$
   $$0 \le P_{dis,t}^{BESS} \le P_{max}^{BESS} \cdot v_{dis,t} \quad \forall t \in T$$

4. **Non-Simultaneous Charge/Discharge Lock:**
   $$v_{ch,t} + v_{dis,t} \le 1 \quad \forall t \in T \quad (v_{ch,t}, v_{dis,t} \in \{0, 1\})$$

---

### 3.4 Microturbine (MT) Operational Constraints

1. **Piecewise Power Generation & Exact 25 MW Capacity Cap:**
   $$P_{MT,t} = P_{MT,min} \cdot u_{MT,t} + \sum_{m=1}^{N_m} P_{MT,m,t} \quad \forall t \in T$$
   $$0 \le P_{MT,m,t} \le w_m \cdot u_{MT,t} \quad \forall m \in \{1, \dots, N_m\}, \forall t \in T$$
   *(where $w_1 = 5.30\text{ MW}, w_2 = 4.70\text{ MW}, w_3 = 5.00\text{ MW}$, ensuring $P_{MT,t} \le 10 + 15 = 25\text{ MW}$).*

2. **Ramp-Up and Ramp-Down Rate Limits:**
   $$P_{MT,t} - P_{MT,t-1} \le RU_{MT} \cdot u_{MT,t-1} + SRU_{MT} \cdot x_{MT,t} \quad \forall t \in T$$
   $$P_{MT,t-1} - P_{MT,t} \le RD_{MT} \cdot u_{MT,t} + SRD_{MT} \cdot y_{MT,t} \quad \forall t \in T$$

3. **Minimum Up Time (MUT) and Minimum Down Time (MDT):**
   $$\sum_{\tau = t - MUT + 1}^{t} x_{MT,\tau} \le u_{MT,t} \quad \forall t \ge MUT$$
   $$\sum_{\tau = t - MDT + 1}^{t} y_{MT,\tau} \le 1 - u_{MT,t} \quad \forall t \ge MDT$$

4. **Unit Commitment Logic:**
   $$x_{MT,t} - y_{MT,t} = u_{MT,t} - u_{MT,t-1} \quad \forall t \in T$$
   $$x_{MT,t} + y_{MT,t} \le 1 \quad \forall t \in T$$

---

### 3.5 Industrial Batch Scheduling Constraints

1. **Single Task Assignment per Batch Slot:**
   $$\sum_{p \in P} X_{m,n,p} = 1 \quad \forall m \in M, \forall n \in N$$

2. **Batch Non-Overlap & Cooling Buffer Constraint:**
   $$ts_{m,n} \ge ts_{m,n-1} + \sum_{p \in P} \left( td_{m,p} \cdot X_{m,n-1,p} \right) + t_{buffer} \quad \forall m \in M, \forall n > 1$$

3. **Task Execution Allocation Logic:**
   $$o_{m,n,t,p} \le X_{m,n,p} \quad \forall m \in M, \forall n \in N, \forall p \in P, \forall t \in T$$
   $$\sum_{t=1}^{T} o_{m,n,t,p} = td_{m,p} \cdot X_{m,n,p} \quad \forall m \in M, \forall n \in N, \forall p \in P$$

---

### 3.6 Warehouse Inventory & Production Coupling Constraints

1. **Hourly Production Yield Coupling:**
   $$n_{prod,t} = \sum_{m \in M} \sum_{n \in N} \sum_{p \in P} \left( Y_p \cdot \frac{o_{m,n,t,p}}{td_{m,p}} \right) \quad \forall t \in T$$

2. **Dynamic Inventory Balance:**
   $$N_t = N_{t-1} + n_{prod,t} - n_{prod,dem,t} \quad \forall t \in T$$

3. **Storage Capacity Bounds:**
   $$0 \le N_t \le N_{max} \quad \forall t \in T$$
   $$N_T \ge N_0$$

---

### 3.7 Substation Grid Capacity Limits
$$0 \le P_{buy,t}^{DA} \le Q_{md}^{buy} \quad \forall t \in T$$
$$0 \le P_{sell,t}^{DA} \le Q_{md}^{sell} \quad \forall t \in T$$

---

## 4. Solver Implementation Guidelines

* **Modeling Framework:** Pyomo (Python) or GAMS.
* **Solvers:** CPLEX, Gurobi, CBC, or HiGHS.
* **Optimality Tolerance:** MIP gap $\le 0.01\%$.
