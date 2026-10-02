# Factory MT Day-Ahead Scheduler

An installable Python package for day-ahead scheduling of industrial batch
production with optional on-site microturbine commitment and dispatch. The
time-indexed Pyomo MILP minimizes grid, fuel, and start/stop costs while
enforcing machine sequencing, inventory, demand, grid limits, and MT operating
constraints.

## Install and Run

```bash
uv sync --extra test
uv run factory-mt-da --help
uv run factory-mt-da --out da_mt_results --plot
```

The command is also available as `uv run python -m factory_mt_da` or through
the compatibility script `uv run python factory_mt_da_scheduler.py`.

## Use the Python API

```python
from factory_mt_da import SchedulerConfig, make_benchmark_instance, optimize_day_ahead

instance = make_benchmark_instance()
result = optimize_day_ahead(instance, SchedulerConfig())
print(result.summary())
```

## Configure an Instance

The benchmark and solver defaults are centralized in `src/factory_mt_da/parameters.py`.
For production data, export a benchmark JSON with `--export-instance instance.json`,
edit its machine/task matrices, load profile, tariff, demand, inventory, and `mt`
settings, then run:

```bash
uv run factory-mt-da --instance instance.json --out results --plot
```

Set `"mt": null` in the JSON to disable the microturbine. MT parameters include
minimum/maximum power, ramp rates, start/stop costs, minimum up/down hours, fuel
block widths and marginal prices, and the initial commitment state. CLI options
override solver/search settings for one run.

## Results

Each run writes `jobs.csv`, `hourly.csv`, and `summary.json`. The hourly table
reports grid import, MT output and commitment, separate grid/fuel/start-stop
costs, total cost, production, demand, and inventory. `--plot` also writes
`schedule.png` with the batch Gantt schedule, stacked grid/MT dispatch, grid
limit, and day-ahead price.

## Package Layout

- `parameters`, `data`, and `exceptions`: defaults, validated inputs, and domain errors
- `benchmark` and `candidates`: sample instance and feasible batch starts
- `model` and `solver`: MILP construction and solver integration
- `optimizer`: end-to-end scheduling workflow
- `results` and `plotting`: independent checks, KPIs, tables, and schedule figure
- `cli`: command-line interface