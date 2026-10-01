# Factory Day-Ahead Scheduler

An installable Python package for minimizing the day-ahead electricity cost of
industrial batch production. It uses a time-indexed Pyomo MILP to schedule
non-preemptive jobs while enforcing machine sequencing, inventory, demand, and
grid limits.

## Install

```bash
uv sync --extra test --extra plot
```

Or install with pip:

```bash
python -m pip install -e ".[test,plot]"
```

The default solver is HiGHS, provided by `highspy`. Plotting support is optional.

## Run

```bash
uv run factory-da --help
uv run factory-da --out da_results --plot
```

The same CLI is available as `python -m factory_da` or through the legacy
`python factory_da_scheduler.py` script. The Python API is exported from
`factory_da`:

```python
from factory_da import SchedulerConfig, make_benchmark_instance, optimize_day_ahead

instance = make_benchmark_instance()
result = optimize_day_ahead(instance, SchedulerConfig())
print(result.summary())
```

## Input Data

To use your own factory data, create a JSON file such as `instance.json` in the
project directory and pass it with `--instance`. The following fields are
required:

- `machines`: machine names, for example `["F1"]`.
- `tasks`: task names, for example `["batch_A"]`.
- `power_mw`: task power in MW, with one row per machine and one column per task.
- `duration_h`: task duration in hours, with the same dimensions as `power_mw`.
- `yield_units`: units produced by each completed task, one value per task.
- `base_load_mw`: non-shiftable load in MW, one value per hour.
- `price_eur_mwh`: electricity price in EUR/MWh, one value per hour.
- `demand_units`: product demand in units, one value per hour.

Optional fields are `horizon_h` (hours, default `24`), `buffer_h` (hours between
batches, default `1`), `grid_limit_mw` (default `400`),
`max_batches_per_machine` (default `10`), `inventory_init` (default `50`), and
`inventory_max` (default `500`). The three hourly arrays must each contain
`horizon_h` values. Matrix rows follow the order in `machines`; columns follow
the order in `tasks`.

Example for one machine, one task, and a four-hour horizon:

```json
{
	"machines": ["F1"],
	"tasks": ["batch_A"],
	"power_mw": [[5]],
	"duration_h": [[1]],
	"yield_units": [10],
	"base_load_mw": [2, 2, 2, 2],
	"price_eur_mwh": [50, 20, 30, 40],
	"demand_units": [0, 0, 0, 10],
	"horizon_h": 4,
	"buffer_h": 0,
	"grid_limit_mw": 20,
	"max_batches_per_machine": 1,
	"inventory_init": 0,
	"inventory_max": 20
}
```

Replace the example values with your factory's data, then run:

```bash
uv run factory-da --instance instance.json --start-step 1 --out results --plot
```

## Package layout

- `data` and `exceptions`: input, settings, result types, and validation errors
- `benchmark` and `candidates`: sample data and feasible time-indexed starts
- `model` and `solver`: MILP construction and solver integration
- `results` and `plotting`: result extraction, verification, KPIs, and charts
- `optimizer` and `cli`: public workflow and command-line interface