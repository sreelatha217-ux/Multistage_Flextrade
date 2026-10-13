# Offering owner-report figure guide

This guide explains every PNG in [`offering_owner_report_results/`](offering_owner_report_results/).
The images are outputs from the saved 24-hour owner run, not live market charts. The run
uses three day-ahead/intraday price scenarios and three real-time scenarios (nine joint
paths), with equal probabilities. Hour values are hourly intervals; plotted points at
`h + 0.5` are centered within hour `h`.

## BESS degradation setting verified

The saved [`summary.json`](offering_owner_report_results/summary.json) records
`bess_degradation_eur_mwh: 5.0` and `bess_degradation_basis: "throughput"`. The
reproduction command in [`README.md`](README.md) also passes
`--bess-degradation 5 --bess-degradation-basis throughput`.

The offering model charges €5 per MWh of expected battery throughput: scheduled
charging energy after reserve activation curtailment, plus scheduled discharge energy
and reserve-activated discharge energy. The saved expected degradation cost is
**€747.45**; at €5/MWh this corresponds to about **149.49 MWh** of expected
throughput-equivalent energy. The BESS plot title displays the configured rate and
expected charge. Degradation is one operating-cost component, not the total BESS
electricity cost or the total factory operating cost.

## Overview and operating schedules

### `factory_owner_dashboard.png`

Four owner-level summaries:

1. **Value of market participation:** modeled expected net cost for DA-only (with
   real-time imbalance settlement), DA+ID, and DA+ID+up-reserve. The savings labels
   compare each case with DA-only. Lower cost is better.
2. **Expected cost bridge:** shows how DA and ID settlement, reserve-capacity and
   activation revenue, imbalance, microturbine, BESS degradation, and demand-response
   costs contribute to expected net cost. Revenue offsets costs.
3. **Value of production rescheduling:** per-ID-scenario savings from rescheduling
   jobs rather than retaining the Stage 1 jobs; the dashed line is the probability-
   weighted saving.
4. **Production maintained across reschedules:** unit and job counts for the Stage 1
   plan and each Stage 2 ID-scenario plan.

The market-stage comparison and production-rescheduling study are separate
calculations. The dashboard footnote calls this out: rescheduling savings are measured
against the fixed Stage 1 job plan and are not added into market-stage savings.

### `offering.png`

The compact, four-panel overview: Stage 1 production schedule, selected DA
price-quantity curves, the expected upward-reserve offer split by asset with reserve
price on a second axis, and the probability-weighted profit distribution with expected
profit and CVaR markers. Its expected net cost title is the negative of modeled
expected profit; it is not product-sales-inclusive factory profit.

### `schedule_rt.png`

Stage 1 production Gantt chart. Each horizontal bar is a job on a machine; its
horizontal position and length show the scheduled start and duration. Job labels
identify the task. The title reports modeled expected net cost.

### `stage1_here_and_now.png`

Three Stage 1 decisions shared before price uncertainty is known: production schedule,
day-ahead microturbine on/off commitment, and selected day-ahead offer curves. The
offer panel plots net export (sell minus buy), so values below zero mean net import,
not a negative bid quantity.

## Market and asset operation

### `market_positions_rt.png`

Expected net positions by hour for day-ahead, intraday, and real-time imbalance.
Positive values mean net import/deficit; negative values mean net export/surplus.
These are expected net quantities, rather than individual scenario traces.

### `market_prices_rt.png`

Expected day-ahead buy/sell, intraday buy/sell, and real-time deficit/surplus
settlement prices by hour. The shaded bands show the minimum-to-maximum scenario
range for day-ahead buy and intraday buy prices. The line families are scenario-tree
expectations.

### `bess_dispatch_rt.png`

Expected BESS charging (above zero) and discharge (below zero) by hour, together
with the state of charge (SoC) for joint scenarios and its expectation on the second
y-axis. Dotted horizontal lines mark SoC limits. The title states the €5/MWh
throughput degradation rate and the run's expected degradation cost (**€747.45**).
Power (MW) and energy/SoC (MWh) use different y-axes.

### `microturbine_dispatch_rt.png`

Expected real-time microturbine generation and the minimum-to-maximum range across
joint scenarios. A step line on the right axis shows the shared day-ahead commitment
(Off/On). Commitment is not the same as the scenario-dependent dispatch level.

### `realtime_imbalance.png`

Real-time imbalance for each joint scenario and the expected imbalance. The plotted
quantity is deficit minus surplus: positive means deficit; negative means surplus.
The zero line separates the two.

### `da_offer_curves.png`

Day-ahead net export offer curves for selected hours. The horizontal axis is the
scenario day-ahead sell price and the vertical axis is sell quantity minus buy
quantity. A negative y-value means the optimized position is a net buy/import for
that scenario; it does not mean that the buy or sell quantity itself is negative.
Curves are price-ordered and step-shaped because they are price-quantity schedules.

### `reserve_offers.png`

Expected upward-reserve offers stacked by source: BESS discharge, BESS charge
curtailment, demand response, and microturbine. The overlaid line on a second axis is
the expected reserve price. Bars show offered reserve capacity, not necessarily
activated energy; activation is scenario-dependent.

## Offering economics and scenario recourse

### `profit_distribution.png`

Probability-weighted histogram of modeled profit across joint price/real-time
scenarios. Vertical markers show expected profit and CVaR profit. In this run the
values are negative (net modeled cost); the scenario count is small, so the tail-risk
marker is sensitive to the sampled scenarios. This model's objective excludes product
sales revenue and other non-energy factory economics.

### `stage2_wait_and_see.png`

Four hourly series panels for each revealed price scenario: net intraday import
(buy minus sell), microturbine dispatch, BESS net dispatch (discharge minus charge),
and total reserve offer. This is Stage 2 recourse after price information is revealed,
not one common expected schedule. Consequently, negative net-trade or BESS values
are possible and indicate export or net charging, respectively.

### `stage3_wait_and_see.png`

Three scenario-specific real-time panels for each joint path `(s, w)`: imbalance
(deficit minus surplus), activated reserve (`deployment fraction × offered reserve`),
and BESS SoC after reserve activation. Color identifies the price scenario and line
style the real-time scenario. SoC dotted lines are the battery limits.

## Separate intraday production-rescheduling study

These three figures describe a standalone production rescheduling analysis. They do
not show an additional stage integrated into the offering-profit optimization, and
their savings should not be added to the market participation comparison.

### `stage2_production_reschedules.png`

The first panel is the Stage 1 baseline schedule; subsequent panels show one
production schedule per ID price scenario. Job bars show machine, task, and scheduled
time. Each scenario title reports that schedule's operating cost and number of
changed jobs. The configured rescheduling cutoff is hour 0 for this run, so no jobs
had started and none were frozen.

### `intraday_prices_and_load.png`

Top panel: ID buy-price paths by scenario, with dashed lines for ID sell prices.
Bottom panel: implied production load for each scenario's Stage 2 schedule, compared
with the Stage 1 baseline load. It helps show how production timing responds to each
scenario's prices; price and load use separate y-axes/panels.

### `intraday_reschedule_metrics.png`

Top panel: matched operating-cost comparison for each ID scenario, with Stage 1 jobs
fixed versus jobs rescheduled under that same scenario's ID prices and the same
microturbine commitment. Bottom panel: cost saved (fixed-plan cost minus rescheduled
cost); positive means rescheduling reduced cost, negative would mean it increased
cost. The x-axis includes scenario probability.

## Reading the charts carefully

- These are results from modeled scenarios and assumptions, not observations or
  forecasts from a live electricity market.
- Net quantities are signed: check the axis label and definition before interpreting
  negative values. For example, negative net DA export means net import.
- Expected curves and bars aggregate the modeled scenario probabilities; they need
  not match any one scenario's actual path.
- EUR labels refer to modeled energy-market and operating settlements. They are not
  total factory profit, because product-sales revenue and other non-energy business
  costs are not included.
