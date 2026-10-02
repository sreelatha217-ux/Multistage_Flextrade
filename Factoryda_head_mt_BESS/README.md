# Factory MT + BESS Day-Ahead Scheduler

A modular Pyomo MILP scheduler for industrial batch production, an on-site microturbine, a battery energy storage system, and bidirectional day-ahead grid trading.

## Setup

```sh
uv sync --extra test
```

## Run

```sh
uv run factory-mt-bess-da --config configs/default.json --plot --out da_mt_bess_results
```

The bundled JSON run configuration separates benchmark, scheduler, solver, microturbine, and BESS settings. Use `--config path.json` to load another configuration or `--instance path.json` for custom input data; command-line flags override the corresponding run settings.

```sh
uv run factory-mt-bess-da --help
uv run pytest
```

## Package layout

- `src/factory_mt_bess_da/data.py`: validated instances, MT/BESS assets, solver settings, and results
- `src/factory_mt_bess_da/parameters.py`: benchmark and default parameter values
- `src/factory_mt_bess_da/config.py`: JSON configuration loading and construction
- `configs/default.json`: editable default benchmark and asset settings
- `src/factory_mt_bess_da/candidates.py` and `model.py`: time-indexed batch candidates and MILP
- `src/factory_mt_bess_da/solver.py`, `results.py`, and `optimizer.py`: solving, verification, and orchestration
- `src/factory_mt_bess_da/plotting.py` and `cli.py`: figures and command-line interface
