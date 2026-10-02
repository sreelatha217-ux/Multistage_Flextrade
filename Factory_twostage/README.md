# Factory MT + BESS Two-Stage Scheduler

A modular Pyomo MILP for day-ahead production and market positions with scenario-based intraday trading, microturbine redispatch, and BESS recourse.

## Setup

```sh
uv sync --extra test
```

## Run

Run the two-stage stochastic scheduler with the bundled configuration:

```sh
uv run factory-mt-id --plot --out da_id_results
```

Run the deterministic day-ahead model separately:

```sh
uv run factory-mt-da --plot --out da_mt_results
```

Use `--config configs/default.json` to load a run configuration, `--instance path.json` for a custom factory instance, or `--scenario-file path.json` to supply intraday scenarios. CLI options override configuration values. `--vss` also computes the value of the stochastic solution.

```sh
uv run factory-mt-id --help
uv run pytest
```

## Package Layout

- `src/factory_twostage/data.py`, `config.py`, and `scenarios.py`: validated instance, run settings, and stochastic scenario data
- `src/factory_twostage/candidates.py`, `model_blocks.py`, and `model.py`: batch candidates and reusable deterministic DA constraints
- `src/factory_twostage/intraday_model.py`: extensive-form two-stage DA + ID MILP
- `src/factory_twostage/solver.py`, `optimizer.py`, `intraday_optimizer.py`: solver and orchestration
- `src/factory_twostage/results.py` and `intraday_results.py`: independent checks, KPIs, and CSV/JSON output
- `src/factory_twostage/cli.py` and `intraday_cli.py`: deterministic and stochastic CLI entry points
- `configs/default.json`: editable benchmark, asset, solver, and scenario settings

Intraday outputs include `jobs.csv`, `da_position.csv`, `scenarios.csv`, `hourly_scenarios.csv`, `scenario_set.json`, and `summary.json`.