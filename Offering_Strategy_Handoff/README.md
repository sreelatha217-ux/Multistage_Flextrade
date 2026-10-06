# Factory Offering Strategy - Share Package

This folder is a self-contained handoff of the factory energy-offering
optimizer. It contains the scheduler implementation, install metadata, the
`uv` lockfile, a runnable set of example JSON inputs, a sample output, and
focused tests.

The model makes a day-ahead factory production schedule and energy bid curves,
then models intraday and real-time recourse, optional up-spinning-reserve
offers, reserve deployment, and profit-risk trade-offs using CVaR. It is a
command-line/Python package; **it does not expose an HTTP/UI service**.

## Package contents

- `factory_mt_offering_strategy.py`: offering model and Python API.
- `factory_mt_offering_cli.py`: offering command-line interface.
- `factory_mt_da_scheduler.py`, `factory_mt_id_scheduler.py`,
  `factory_mt_rt_scheduler.py`: shared underlying stages and data models.
- `factory_mt_production_rescheduler.py`: intraday production rescheduling.
- `factory_mt_offering_plotting.py`: optional plots.
- `pyproject.toml`, `uv.lock`: project metadata, entry points, and pinned
  dependency resolution.
- `examples/input/`: a small plant plus explicit DA/ID/SR, RT, and reserve
  deployment JSON scenario files.
- `examples/output/`: output generated from those example inputs.
- `tests/`: focused tests for the offering integration.

## Requirements and install

Install Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/). From this
folder run:

```sh
uv sync --extra test
```

`uv.lock` pins the resolved dependency versions. HiGHS is installed through
`highspy`; no separate solver installation is needed for the default
`appsi_highs` solver.

Run the focused tests:

```sh
uv run pytest
```

## Run the example

From this folder:

```sh
uv run factory-mt-offering \
  --instance examples/input/plant.json \
  --offering-scenario-file examples/input/offering_scenarios.json \
  --rt-file examples/input/rt_scenarios.json \
  --deploy-file examples/input/reserve_deployment.json \
  --out examples/output
```

To regenerate plots as well:

```sh
uv run factory-mt-offering \
  --instance examples/input/plant.json \
  --offering-scenario-file examples/input/offering_scenarios.json \
  --rt-file examples/input/rt_scenarios.json \
  --deploy-file examples/input/reserve_deployment.json \
  --plot --out examples/output
```

The example is intentionally small and illustrative. The scenario arrays are
synthetic and must not be treated as live market data or market-ready bids.
The CLI's `--help` lists additional inputs, solver controls, risk settings,
reserve controls, and analysis options.

## Inputs the UI/backend team should provide

The CLI accepts JSON files. In a service integration, the UI can collect the
same fields and the backend can validate/serialize them to these files or call
the Python API directly.

### Plant: `--instance plant.json`

| Field | Shape | Meaning |
|---|---|---|
| `machines`, `tasks` | lists | Machine and task identifiers |
| `power_mw` | `[machine][task]` | Task power draw in MW |
| `duration_h` | `[machine][task]` | Processing duration in hours |
| `yield_units` | `[task]` | Output quantity for each task |
| `base_load_mw` | `[hour]` | Non-task load in MW |
| `price_buy_eur_mwh`, `price_sell_eur_mwh` | `[hour]` | DA tariff input; sell must not exceed buy |
| `demand_units` | `[hour]` | Factory product delivery requirement |
| `horizon_h` | scalar | Number of hourly delivery periods |

Additional optional fields include `buffer_h`, `grid_limit_mw`,
`grid_sell_limit_mw`, `max_batches_per_machine`, `inventory_init`,
`inventory_max`, `mt`, and `bess`. In this model, the task definitions are a
machine-by-task matrix rather than an arbitrary list of independent work
orders. `mt: null` or `bess: null` disables that resource. If MT is omitted,
its built-in defaults are used; BESS is disabled if omitted.

### Price uncertainty: `--offering-scenario-file offering_scenarios.json`

Let `S` be the number of price scenarios and `T` be the plant horizon.
`prob` is `[S]`; the price/deviation arrays are `[S,T]`.

| Field | Unit | Meaning |
|---|---|---|
| `da_buy`, `da_sell` | EUR/MWh | DA buy/sell scenarios |
| `id_buy`, `id_sell` | EUR/MWh | ID buy/sell scenarios |
| `sr_up` | EUR/MW/h | Up-reserve capacity price |
| `load_dev_mw` | MW | Additive deviation from the plant base-load forecast |

Scenario probabilities must sum to one. Sell prices must not exceed buy prices.

### Real-time and reserve: `--rt-file` and `--deploy-file`

Let `W` be the number of real-time branches per price scenario.
`rt_scenarios.json` has `prob` shaped `[S,W]` (each row sums to one) and
`load_rel_dev`, `r_plus`, and `r_minus` shaped `[S,W,T]`. Load deviation is
relative (for example, `0.02` means +2%); `r_plus >= 1`, and `r_minus` is in
`[0,1]`.

`reserve_deployment.json` has `rho` shaped `[S,W,T]`, with values in `[0,1]`;
it represents the fraction of offered reserve called by the system operator.
Scenario counts and horizon lengths must match across all files.

## Outputs the UI can consume

Every run writes an output folder. Most useful files for integration are:

- `bid_package.json`: market-facing per-hour energy bid and reserve offer price
  / quantity curves, plus the strategy parameters.
- `jobs.csv`: selected shared production plan with machine, task, start/end,
  power, and output.
- `da_position.csv`: hourly DA position, MT/BESS schedule, reserve quantities,
  and expected operating values.
- `hourly_offering.csv`: detailed per-hour offering and expected dispatch.
- `offer_curves.csv`: normalized/long-form offer curve rows.
- `summary.json`: solver status, expected and risk-adjusted objective, CVaR,
  KPI breakdown, solve statistics, and independent verification.
- `scenarios.csv`, `joint_scenarios.csv`: scenario-level details.
- `scenario_set.json`, `rt_set.json`, `deployment_set.json`: saved scenario
  inputs used by the run.
- `intraday_rescheduled_jobs.csv` and `intraday_reschedule_summary.csv`:
  scenario-specific production-only intraday schedules and matched comparisons
  with the fixed day-ahead task schedule.
- With `--plot`, PNG charts; with `--compare`, strategy and market-stage
  comparison CSVs.

`expected_net_cost_eur` is not total factory profit: product-sales revenue is
not included. Intraday production-rescheduling savings are reported separately
and are not included in the offering objective.

## Backend/UI integration notes

1. Keep each plant configuration, operating date, and day-ahead result
   associated with the same run.
2. Return `bid_package.json`, `jobs.csv`, and selected KPI fields to the UI
   after a successful solve; surface validation and solver failures instead of
   returning an empty or stale schedule.
3. For repeatable scenario studies, send explicit scenario files. If those
   files are omitted, the CLI generates synthetic scenarios.
4. The current CLI runs the complete configured study. It is not an
   asynchronous job service, live data connector, or rolling-horizon endpoint.
   Those pieces must be supplied by the integrating backend.
5. These results are optimization outputs, not a guarantee of market
   acceptance, dispatch, revenue, or compliance with a particular exchange's
   bid increments and gate-closure rules.
