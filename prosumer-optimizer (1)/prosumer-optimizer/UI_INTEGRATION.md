# UI / backend integration handoff

This package schedules an industrial site across the day-ahead (DA), intraday
(ID), and real-time balancing stages. It accepts configuration files in JSON or
YAML and exposes a Python API. **It does not currently expose an HTTP API.**
The JSON below is an `AppConfig` file, not an HTTP request envelope.

## What the UI can configure

- **Factory machines and tasks:** there is no separate machine-count setting in
  an explicit factory configuration. Each `factory.jobs[]` entry is one
  non-interruptible batch task. `machine` identifies the machine; the number of
  machines is the number of distinct machine names. `sequence` orders tasks on
  the same machine. `power_mw` and `duration_h` define the task's load.
- **Other factory load and production:** `base_load_mw` is the non-shiftable
  load, while `demand_units` is the delivery requirement per time step. Each
  can be one scalar for the entire horizon or an array with one value per step.
- **BESS:** the current model supports one aggregate battery. Choose `large`
  (40 MW / 200 MWh) or `medium` (2 MW / 4 MWh), or override its parameters.
  Separate multiple battery units are not modeled individually.
- **Microturbine (MT):** the current model supports one aggregate MT with
  configurable output, ramp, commitment, and fuel-cost parameters. Separate
  multiple MT units are not modeled individually.
- **Grid connection:** import/export limits at the point of common coupling.
- **Intraday flexibility:** the maximum permitted shift of each task start
  relative to the DA plan, plus the step at which re-optimization starts.

Units: power in MW, energy in MWh, time in hours, production/inventory in units,
and money in EUR (prices in EUR/MWh). `time.dt_h` is the duration of one step;
the horizon must be an integer multiple of it. At the default 24-hour, 1-hour
step, arrays such as `base_load_mw` have 24 values indexed from hour 0.

## Example JSON configuration

This file can be passed directly to `AppConfig.from_file()` or to the CLI with
`--config configs/ui_factory.example.json`. Fields not specified are filled
from the built-in defaults.

```json
{
  "time": {
    "horizon_h": 24.0,
    "dt_h": 1.0
  },
  "grid": {
    "import_limit_mw": 400.0,
    "export_limit_mw": 400.0
  },
  "microturbine": {
    "p_min_mw": 10.0,
    "p_max_mw": 25.0
  },
  "bess": {
    "profile": "large"
  },
  "factory": {
    "base_load_mw": 3.5,
    "demand_units": 0.0,
    "buffer_h": 1.0,
    "inventory_init": 0.0,
    "inventory_max": 1000.0,
    "intraday_shift_window_h": 2.0,
    "unmet_penalty_eur_per_unit": 5000.0,
    "jobs": [
      {
        "job_id": "LINE-1-BATCH-1",
        "machine": "LINE-1",
        "sequence": 0,
        "duration_h": 3.0,
        "power_mw": 90.0,
        "units_out": 100.0,
        "earliest_start_h": 0.0,
        "latest_finish_h": 24.0
      },
      {
        "job_id": "LINE-1-BATCH-2",
        "machine": "LINE-1",
        "sequence": 1,
        "duration_h": 2.5,
        "power_mw": 80.0,
        "units_out": 100.0
      },
      {
        "job_id": "LINE-2-BATCH-1",
        "machine": "LINE-2",
        "sequence": 0,
        "duration_h": 4.0,
        "power_mw": 70.0,
        "units_out": 120.0
      }
    ]
  },
  "intraday": {
    "step": 10
  },
  "output": {
    "directory": "results",
    "save": true,
    "save_config": true
  }
}
```

For the default grid above, the factory has two machines and three tasks. A
task must have unique `job_id`; `(machine, sequence)` pairs must also be
unique. Tasks on a machine run in sequence with the configured `buffer_h`
between them. Omitted job values default to `units_out: 100`,
`earliest_start_h: 0`, and `latest_finish_h: horizon_h`.

To use the generated demo factory instead, omit `factory` (or set it to
`null`). That produces five randomly generated machines with two tasks per
machine by default; it is intended for demonstration, not live plant data.

## Built-in defaults

These are defaults from the current implementation. Any setting may be
overridden in JSON; omitted setting groups keep their defaults.

| Setting | Default |
|---|---:|
| Horizon / step | 24 h / 1 h |
| Grid import / export limit | 400 / 400 MW |
| MT min / max output | 10 / 25 MW |
| MT ramp up / down | 20 / 20 MW/h |
| MT startup / shutdown cost | EUR 87.40 / EUR 8.74 |
| MT minimum up / down time | 4 / 2 h |
| BESS profile | `large` |
| Large BESS power / energy | 40 MW / 200 MWh |
| Large BESS initial / minimum / maximum SoC | 20 / 20 / 200 MWh |
| BESS charge / discharge efficiency | 0.80 / 0.95 |
| BESS throughput cost | EUR 30/MWh |
| Factory buffer / inventory maximum | 1 h / 1000 units |
| Intraday batch shift window | 2 h |
| Unmet-demand penalty | EUR 5000/unit |
| Scenario counts (ID / RT) | 5 / 3 |
| Intraday demo re-optimization step | 10 (hour 10 at 1-hour steps) |
| Solver / time limit / MIP gap | `appsi_highs` / 300 s / 0.0001 |

The `medium` BESS profile is 2 MW / 4 MWh, with SoC minimum 0.4 MWh,
maximum 3.6 MWh, and initial SoC 0.63 MWh. For either profile, the default
terminal target is the initial SoC.

## Current data-source limitation and live intraday operation

The `run_pipeline()` convenience workflow generates synthetic market scenarios
and, if intraday is enabled, a synthetic price spike and scenario. The
`scenarios` and `intraday` sections configure those generators; they are **not**
arrays of real market prices. The current JSON config does not accept actual
day-ahead/intraday price curves.

For a live intraday solve, the backend must retain the day-ahead `StageResult`
and call `MultiStageProsumerOptimizer.solve_intraday()` with:

- `t0`: the first step to re-optimize; earlier steps remain history.
- measured `InitialState`: battery `soc_mwh`, MT `mt_online`,
  `mt_output_mw`, `mt_hours_in_state`, and factory `inventory_units`.
- an updated `IDScenario` with arrays for `da_buy`, `da_sell`, `id_buy`, and
  `id_sell`, plus one or more real-time branches. Each price/load/settlement
  array must have `horizon_h / dt_h` entries.
- optionally, a refreshed `base_load_mw` scalar or full-horizon array.

The intraday stage keeps DA market commitments and MT on/off commitments
frozen, while re-dispatching the remaining MT output, BESS, and eligible batch
starts (within the configured shift window). Steps before `t0` are not
re-optimized. The package exposes these objects through its Python API but does
not currently parse them from a live-operation JSON request. An application
service/API adapter is needed to connect UI payloads and live measurements to
that API.

## Results returned by the optimizer

`run_pipeline(cfg)` returns a `PipelineResult` with `day_ahead` and, if enabled,
`intraday`, each a `StageResult`. A stage result contains:

- `schedule`: one row per time step. DA columns include `hour`, `da_buy_mw`,
  `da_sell_mw`, `da_net_mw`, `mt_on`, `mt_startup`, `mt_shutdown`,
  `exp_mt_mw`, `exp_id_net_mw`, `exp_bess_ch_mw`, `exp_bess_dis_mw`,
  `exp_soc_mwh`, `exp_batch_load_mw`, `exp_inventory`,
  `exp_imb_surplus_mw`, `exp_imb_shortfall_mw`, and `exp_da_price`.
  `exp_` values are scenario-probability-weighted expectations. An intraday
  result preserves the full-day index; past/unoptimized rows may be empty.
- `batch_plan`: one row per job with `job_id`, `machine`, `sequence`,
  `power_mw`, `units_out`, DA `start_step` / `start_hour`, `end_hour`,
  `duration_steps`, and scenario-specific start-hour columns.
- `scenario_detail`: per-scenario, per-step dispatch and market values.
- `scenario_costs`, `cost_breakdown`, and `risk` (mean, standard deviation,
  worst, best, and CVaR95 cost).
- Solve metadata: `stage`, `t0`, `status`, `objective_eur`, `mip_gap`,
  `solve_time_s`, and `model_stats`.

With saving enabled, each stage currently writes its schedule, batch plan,
scenario detail, and scenario costs as CSV files, plus a summary JSON file.
`config_used.json` records the resolved configuration. There is not currently
one combined JSON response containing schedule rows; the UI/API adapter should
serialize the `StageResult` fields above into its response contract.

## Suggested UI-to-service flow

1. Submit plant configuration and next-day jobs; the backend validates it,
   builds `AppConfig`, and solves day-ahead.
2. Return the DA schedule and batch plan to the UI.
3. On an intraday update, submit `t0`, measured state, refreshed price/forecast
   arrays, and optional base load. The service reuses the saved DA result and
   returns the intraday schedule.
4. Keep each day-ahead plan and its subsequent reschedules associated with the
   same plant and operating day.

The UI can use the config example and result field names above as the initial
contract, but the service/API wrapper and live-price/state JSON schema still
need to be implemented.
