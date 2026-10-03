# Factory Multistage Offering Strategy

This project contains the day-ahead, intraday, and real-time schedulers plus an optional strategic offering extension. The base three-stage formulation is documented in [factory_multistage_formulation-v3.md](factory_multistage_formulation-v3.md). The offering model adds scenario-dependent DA bid/offer curves, up-spinning-reserve offers, deployment, and CVaR risk control.

The current full-run owner report is [OWNER_REPORT.md](OWNER_REPORT.md), with plots and data in [offering_owner_report_results/](offering_owner_report_results/).

The existing scheduler modules remain colocated so their shared model objects and helper functions are reused directly. The offering CLI is separated into `factory_mt_offering_cli.py`; `factory_mt_offering_strategy.py` remains the public API and backward-compatible script entry point.

See [OWNER_REPORT.md](OWNER_REPORT.md) for the full seed-2024 owner results, financial interpretation, plots, and limitations.

## Install

Install [uv](https://docs.astral.sh/uv/) and run these commands from this directory:

```sh
uv sync --extra test
```

The project supports Python 3.10 and later. `uv.lock` pins the runtime and test dependencies. HiGHS is provided by `highspy`.

## Run

```sh
uv run factory-mt-da --help
uv run factory-mt-id --help
uv run factory-mt-rt --help
uv run factory-mt-offering --help
```

Run the offering model's built-in solver-backed checks with:

```sh
uv run factory-mt-offering --selftest
```

Set the battery degradation cost explicitly (throughput basis, charge plus discharge) with `--bess-degradation 5 --bess-degradation-basis throughput`. The configured EUR/MWh rate and expected degradation cost are included in the result summary and BESS plot.

The offering command accepts the shared instance, solver, ID, RT, and output options from the RT scheduler, plus offering-specific options such as `--bidding`, `--beta`, `--alpha`, reserve settings, `--compare`, and `--frontier`. Add `--plot` to write the baseline-style `schedule_rt.png`, `market_positions_rt.png`, `market_prices_rt.png`, `bess_dispatch_rt.png`, `microturbine_dispatch_rt.png`, and `realtime_imbalance.png` figures; offering-specific `da_offer_curves.png`, `reserve_offers.png`, and `profit_distribution.png`; and explicit `stage1_here_and_now.png`, `stage2_wait_and_see.png`, and `stage3_wait_and_see.png` views. `offering.png` remains the combined overview. Stage 1 shows shared production, commitment, and DA offers; Stage 2 shows decisions by price scenario; Stage 3 shows decisions by joint price/RT scenario. Use `--help` for the complete list. The script form remains available for existing workflows:

The CLI also creates a separate production-only Stage-2 reschedule for each ID price scenario, using that scenario's ID prices and the Stage-1 MT commitment. It writes `intraday_rescheduled_jobs.csv` and `intraday_reschedule_summary.csv`; the summary compares each movable schedule against the same Stage-1 jobs fixed under the same ID prices and MT commitment, so `production_reschedule_savings_eur` is a matched rescheduling value. `--plot` adds `stage2_production_reschedules.png`, `intraday_prices_and_load.png`, and `intraday_reschedule_metrics.png`. `--id-reschedule-hour` controls when the ID plan is applied. Its default is hour 0, matching a next-day schedule made before production starts; jobs already started before a nonzero cutoff are fixed. These standalone reschedule costs are not included in the offering-profit objective.

The rescheduler uses the project’s hourly market horizon and does not enforce Denmark’s continuous intraday execution clock or 0.1 MW trade increments. The scenario schedules are operational studies, not market-ready DK1/DK2 bids.

Run `--compare --plot` to add `market_stage_comparison.csv` and `factory_owner_dashboard.png`. The market ladder compares DA-only, DA+ID, and DA+ID+modeled up-reserve on the same scenario tree; RT imbalance settlement remains active in all cases. The dashboard summarizes market-stage net costs, cost drivers, matched ID rescheduling value, and production volumes. Product sales revenue is not modeled, so these are operating/settlement cost benefits, not total factory profit.

```sh
uv run python factory_mt_offering_strategy.py --help
```

Run the package checks with:

```sh
uv run pytest
```

## Reproduce the owner run

Run from this project directory. This command uses the built-in 30-task factory with seed 2024, the full 24-hour horizon, three ID price scenarios, three RT branches per ID scenario, and the €5/MWh BESS throughput degradation used in the report. It runs both market-stage comparisons and all plots.

```sh
uv run factory-mt-offering \
	--seed 2024 --max-batches 10 --start-step 0.5 \
	--scenario-seed 7 --scenarios 3 \
	--price-sigma 0.15 --level-sigma 0.05 --id-spread-sigma 0.08 \
	--sr-ratio 0.12 --sr-sigma 0.20 --load-sigma 0.0 \
	--rt-seed 11 --rt-scenarios 3 --rt-load-sigma 0.03 --rt-load-rho 0.6 \
	--r-premium 0.25 --r-discount 0.25 --r-sigma 0.3 \
	--rho-seed 13 --rho-mean 0.08 --rho-sigma 0.8 \
	--bess-degradation 5 --bess-degradation-basis throughput \
	--id-reschedule-hour 0 --beta 0 --alpha 0.95 --bidding curve \
	--time-limit 300 --mip-gap 0.01 \
	--compare --plot --out results/owner_report
```

The built-in run needs no input JSON. The seeds make the generated scenario tree reproducible with the current code and dependency lock. To rerun against custom plant or market assumptions, provide the JSON files described below. To replay a saved scenario tree exactly, use:

```sh
uv run factory-mt-offering \
	--seed 2024 --max-batches 10 \
	--offering-scenario-file offering_owner_report_results/scenario_set.json \
	--rt-file offering_owner_report_results/rt_set.json \
	--deploy-file offering_owner_report_results/deployment_set.json \
	--bess-degradation 5 --bess-degradation-basis throughput \
	--id-reschedule-hour 0 --time-limit 300 --mip-gap 0.01 \
	--compare --plot --out results/replay
```

## Required inputs

For a plant-specific run, pass `--instance plant.json`. Required arrays use `T = horizon_h`, `M = machines`, and `P = tasks`:

| Field | Shape / meaning |
|---|---|
| `machines`, `tasks` | Unique machine/task names |
| `power_mw` | Machine-task power matrix `[M,P]` |
| `duration_h` | Machine-task processing duration `[M,P]` |
| `yield_units` | Production yield by task `[P]` |
| `base_load_mw` | Fixed hourly load `[T]` |
| `price_buy_eur_mwh`, `price_sell_eur_mwh` | Day-ahead tariff arrays `[T]`; sell must not exceed buy |
| `demand_units` | Hourly production demand `[T]` |

Optional fields include `horizon_h`, `buffer_h`, grid import/export limits, batch limits, and inventory limits. If `mt` is omitted, the built-in MT defaults apply; set `"mt": null` to disable the MT. BESS is absent if omitted or null. Small valid example with no MT or BESS:

```json
{
	"machines": ["m1"],
	"tasks": ["batch1"],
	"power_mw": [[2.0]],
	"duration_h": [[1.0]],
	"yield_units": [10.0],
	"base_load_mw": [1.0, 1.0],
	"price_buy_eur_mwh": [100.0, 80.0],
	"price_sell_eur_mwh": [55.0, 44.0],
	"demand_units": [5.0, 5.0],
	"horizon_h": 2,
	"buffer_h": 0.0,
	"grid_limit_mw": 50.0,
	"grid_sell_limit_mw": 50.0,
	"max_batches_per_machine": 1,
	"inventory_init": 0.0,
	"inventory_max": 10.0,
	"mt": null,
	"bess": null
}
```

### Offering price scenarios

Pass `--offering-scenario-file offering.json` to supply price scenarios rather than generate them. `prob` is `[S]`; each price/load field below is `[S,T]`. Probabilities must sum to one; sell prices must not exceed buy prices.

```json
{
	"prob": [0.5, 0.5],
	"da_buy": [[100.0, 80.0], [120.0, 60.0]],
	"da_sell": [[55.0, 44.0], [66.0, 33.0]],
	"sr_up": [[10.0, 8.0], [14.0, 7.0]],
	"id_buy": [[95.0, 70.0], [130.0, 65.0]],
	"id_sell": [[52.25, 38.5], [71.5, 35.75]],
	"load_dev_mw": [[0.0, 0.0], [0.0, 0.0]]
}
```

`da_buy`, `da_sell`, `id_buy`, and `id_sell` are EUR/MWh; `sr_up` is EUR/MW/h; `load_dev_mw` is MW.

### RT and reserve deployment scenarios

Pass `--rt-file rt.json` for RT scenarios. `prob` is `[S,W]`, with every row summing to one; `load_rel_dev`, `r_plus`, and `r_minus` are `[S,W,T]`.

```json
{
	"prob": [[0.5, 0.5], [0.5, 0.5]],
	"load_rel_dev": [[[0.0, 0.0], [0.02, -0.02]], [[0.0, 0.0], [-0.02, 0.02]]],
	"r_plus": [[[1.1, 1.1], [1.2, 1.2]], [[1.1, 1.1], [1.2, 1.2]]],
	"r_minus": [[[0.9, 0.9], [0.8, 0.8]], [[0.9, 0.9], [0.8, 0.8]]]
}
```

Require `load_rel_dev > -1`, `r_plus >= 1`, and `0 <= r_minus <= 1`. For reserve deployment, pass `--deploy-file deployment.json`; the root object contains `rho` with shape `[S,W,T]` and values in `[0,1]`:

```json
{
	"rho": [[[0.0, 0.1], [0.2, 0.0]], [[0.0, 0.1], [0.2, 0.0]]]
}
```

The scenario counts and horizon in all files must match. The model's ID schedule rescheduler uses hourly intervals; it is not a continuous intraday order execution simulator.

## Expected outputs

Each output directory contains:

- `summary.json`: solve status, expected net cost/profit, CVaR, asset cost breakdown, scenario count, and independent verification.
- `jobs.csv`: shared Stage 1 production schedule.
- `intraday_rescheduled_jobs.csv` and `intraday_reschedule_summary.csv`: one Stage 2 production schedule per ID scenario and its matched fixed-plan cost/savings.
- `hourly_offering.csv`, `da_position.csv`, `offer_curves.csv`, `scenarios.csv`, and `joint_scenarios.csv`: operating/market details.
- `scenario_set.json`, `rt_set.json`, and `deployment_set.json`: inputs required to replay the scenario tree.
- With `--compare`: `comparison.csv` and `market_stage_comparison.csv`.
- With `--plot`: schedule, market, asset, imbalance, reserve, profit, Stage 1/2/3, ID rescheduling, and owner-dashboard PNGs. `factory_owner_dashboard.png` is only written with both `--compare` and `--plot`.

`expected_net_cost_eur` is not total factory profit: this model does not include product-sales revenue. Generated-price runs are scenario studies, not guaranteed market savings or DK1/DK2-compliant bid recommendations.

## Reproduce the owner run

From this project directory, the following command reproduces the full built-in benchmark run used in the owner report. It does not use `--shrink`.

```sh
uv run factory-mt-offering \
	--seed 2024 --max-batches 10 --start-step 0.5 \
	--scenario-seed 7 --scenarios 3 \
	--rt-seed 11 --rt-scenarios 3 --rho-seed 13 \
	--bess-degradation 5 --bess-degradation-basis throughput \
	--id-reschedule-hour 0 \
	--time-limit 300 --mip-gap 0.01 \
	--compare --plot --out results/owner_report
```

This uses the deterministic benchmark factory/task data for seed 2024 and generates synthetic DA/ID/reserve and RT scenarios using the listed seeds. It is a scenario study, not a run on live Nord Pool or Energinet data. `--id-reschedule-hour 0` treats all delivery-day jobs as future decisions; a nonzero hour freezes baseline jobs that have already started.

To replay the exact saved scenario tree from the committed owner run, supply all three saved scenario files instead of generating new ones:

```sh
uv run factory-mt-offering \
	--seed 2024 --max-batches 10 \
	--offering-scenario-file offering_owner_report_results/scenario_set.json \
	--rt-file offering_owner_report_results/rt_set.json \
	--deploy-file offering_owner_report_results/deployment_set.json \
	--bess-degradation 5 --bess-degradation-basis throughput \
	--id-reschedule-hour 0 --time-limit 300 --mip-gap 0.01 \
	--compare --plot --out results/replay
```

The committed run used a 1% relative MIP-gap target. For higher-confidence financial studies, use more price/RT scenarios, calibrated market data, and a tighter MIP gap, and allow for the longer solve time.

## Required inputs

No external data files are required for a benchmark run. For a plant-specific run, provide `--instance plant.json`. The required fields are:

| Field | Shape / meaning |
|---|---|
| `machines`, `tasks` | Machine and task names |
| `power_mw` | Machine by task matrix `[M,P]` |
| `duration_h` | Machine by task duration matrix `[M,P]` |
| `yield_units` | Task yields `[P]` |
| `base_load_mw` | Hourly fixed load `[T]` |
| `price_buy_eur_mwh`, `price_sell_eur_mwh` | Day-ahead buy/sell tariffs `[T]`; sell must not exceed buy |
| `demand_units` | Hourly factory demand `[T]` |

Optional instance fields include `horizon_h`, `buffer_h`, `grid_limit_mw`, `grid_sell_limit_mw`, `max_batches_per_machine`, `inventory_init`, `inventory_max`, `mt`, and `bess`. If `mt` is omitted the built-in MT defaults are used; set `"mt": null` to disable it. BESS is absent when omitted or null. The following small two-hour example has no MT or BESS:

```json
{
	"machines": ["m1"],
	"tasks": ["batch1"],
	"power_mw": [[2.0]],
	"duration_h": [[1.0]],
	"yield_units": [10.0],
	"base_load_mw": [1.0, 1.0],
	"price_buy_eur_mwh": [100.0, 80.0],
	"price_sell_eur_mwh": [55.0, 44.0],
	"demand_units": [5.0, 5.0],
	"horizon_h": 2,
	"buffer_h": 0.0,
	"grid_limit_mw": 50.0,
	"grid_sell_limit_mw": 50.0,
	"max_batches_per_machine": 1,
	"inventory_init": 0.0,
	"inventory_max": 10.0,
	"mt": null,
	"bess": null
}
```

### Scenario files

All scenario arrays must match the plant horizon `T`. If custom scenario files are used, set compatible scenario counts and probability distributions:

- `--offering-scenario-file`: `prob` has shape `[S]`; `da_buy`, `da_sell`, `sr_up`, `id_buy`, `id_sell`, and `load_dev_mw` have shape `[S,T]`. Prices are EUR/MWh except `sr_up` (EUR/MW/h); load deviation is MW.
- `--rt-file`: `prob` has shape `[S,W]`, and each row sums to 1. `load_rel_dev`, `r_plus`, and `r_minus` have shape `[S,W,T]`. Require `load_rel_dev > -1`, `r_plus >= 1`, and `0 <= r_minus <= 1`.
- `--deploy-file`: root object contains `rho` with shape `[S,W,T]`; every value must be between 0 and 1.

Example offering scenarios for the two-hour plant above (`S=2`, `T=2`):

```json
{
	"prob": [0.5, 0.5],
	"da_buy": [[100.0, 80.0], [120.0, 60.0]],
	"da_sell": [[55.0, 44.0], [66.0, 33.0]],
	"sr_up": [[10.0, 8.0], [14.0, 7.0]],
	"id_buy": [[95.0, 70.0], [130.0, 65.0]],
	"id_sell": [[52.25, 38.5], [71.5, 35.75]],
	"load_dev_mw": [[0.0, 0.0], [0.0, 0.0]]
}
```

Matching RT and deployment examples (`W=2`):

```json
{
	"prob": [[0.5, 0.5], [0.5, 0.5]],
	"load_rel_dev": [[[0.0, 0.0], [0.02, -0.02]], [[0.0, 0.0], [-0.02, 0.02]]],
	"r_plus": [[[1.1, 1.1], [1.2, 1.2]], [[1.1, 1.1], [1.2, 1.2]]],
	"r_minus": [[[0.9, 0.9], [0.8, 0.8]], [[0.9, 0.9], [0.8, 0.8]]]
}
```

```json
{
	"rho": [[[0.0, 0.1], [0.2, 0.0]], [[0.0, 0.1], [0.2, 0.0]]]
}
```

Save each JSON object in its own file and pass all four files to the command, for example `--instance plant.json --offering-scenario-file offering.json --rt-file rt.json --deploy-file deployment.json`. Use the generated `scenario_set.json`, `rt_set.json`, and `deployment_set.json` from a result folder to reproduce that exact uncertainty tree.

## Output files

Every run writes `summary.json`, `jobs.csv`, `offer_curves.csv`, `da_position.csv`, `scenarios.csv`, `joint_scenarios.csv`, `hourly_offering.csv`, `bid_package.json`, and the scenario JSON files. `--compare` additionally writes `comparison.csv` and `market_stage_comparison.csv`. Production rescheduling writes `intraday_rescheduled_jobs.csv` and `intraday_reschedule_summary.csv`. `--plot` writes the schedule, price, market-position, asset, imbalance, reserve, profit, and Stage 1/2/3 figures, plus the owner dashboard when used with `--compare`.