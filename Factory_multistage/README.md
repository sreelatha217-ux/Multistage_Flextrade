# Factory Multistage Scheduler

Three-stage stochastic MILP scheduling for industrial production and electricity-market participation across day-ahead (DA), intraday (ID), and real-time (RT) balancing.

## Setup

Install the package and test dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra test
```

## Run

```sh
uv run factory-mt-da --help
uv run factory-mt-id --help
uv run factory-mt-rt --help
uv run factory-mt-rt --selftest
```

The original `factory_mt_da_scheduler.py`, `factory_mt_id_scheduler.py`, and `factory_mt_rt_scheduler.py` filenames remain as compatibility launchers. Running the RT command without a custom scenario file generates the ID and conditional RT scenarios from the configured random seeds.

Pass `--plot` to the RT command to write six separate PNGs: `schedule_rt.png`, `market_positions_rt.png`, `market_prices_rt.png`, `bess_dispatch_rt.png`, `microturbine_dispatch_rt.png`, and `realtime_imbalance.png`. The MT and BESS operating results are also included in `summary.json` and `hourly_rt.csv`.

## Package Layout

- `src/factory_multistage/data.py`, `benchmark.py`, `candidates.py`: validated plant data and production scheduling inputs
- `model.py`, `solver.py`, `results.py`, `optimizer.py`, `plotting.py`: reusable day-ahead model and solution workflow
- `scenarios.py`, `intraday_model.py`, `intraday_results.py`, `intraday_optimizer.py`: intraday uncertainty and Stage 2 recourse
- `realtime_scenarios.py`, `realtime_model.py`, `realtime_results.py`, `realtime_optimizer.py`: conditional RT scenario tree and Stage 3 settlement
- `cli.py`, `intraday_cli.py`, `realtime_cli.py`: stage-specific command-line interfaces
- `configs/default.json`: bundled benchmark, solver, ID, and RT defaults

Use `uv run pytest` to run the package tests. The built-in scenario file interfaces remain JSON; results are written as CSV and JSON files under the selected output directory.