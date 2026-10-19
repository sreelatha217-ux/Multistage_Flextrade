# Offering Strategy Tool: code for the UI developer

An industrial consumer enters production, microturbine (MT), battery (BESS) and market data. The tool returns the
day-ahead offers, reserve offers, production schedule and expected result.

The UI talks to **one input file and one output file**:

```
UI form -> input.json -> offering_service -> model (HiGHS) -> output.json -> UI
```

The field-by-field definition of both files is in the specification document shared with this zip
("Offering Strategy Tool: Input and Output Specification"). This README covers the code.

## What is in the zip

| Path | What it is |
|---|---|
| `offering_service/` | The service layer written for the UI: input checks, mapping, database loader, output builder, runner |
| `factory_mt_*.py` | The optimisation model (unchanged copy of `Factory_multistage_stretegy`) |
| `sql/da_prices.sql`, `sql/mfrr_up_capacity.sql` | **Placeholders.** Replace with your own queries (see "Database") |
| `examples/` | `input_generate.json` (generated scenarios) and `input_database.json` (real DA and mFRR from PostgreSQL) |
| `sample_results/` | Result files of one saved model run, used by the tests |
| `tests/test_service.py` | Tests of the service layer (no solver needed) |
| `tests/test_model_package.py` | The model's own tests (need pyomo and highspy) |
| `docs/` | Model formulation (v3) and the model README (all model command-line options) |

## Install

Python 3.10 or newer. From this folder:

```sh
uv sync --extra test --extra db
```

If `uv` complains that `uv.lock` is out of date (the `db` extra is new), run `uv lock` once and repeat.

## Run one job

```sh
uv run python -m offering_service run --input examples/input_generate.json --output output.json
```

- `output.json` is always written. On failure it contains `meta.status = "error"` and `meta.error` with `code`,
  `message` and, where known, the input `field`. Codes: `INVALID_INPUT`, `DATA_UNAVAILABLE`, `INFEASIBLE`,
  `SOLVER_ERROR`, `TIMEOUT`, `INTERNAL`.
- Add `--workdir DIR` to keep the working files. `DIR/status.json` shows `state` (`running`, `done`, `error`) and the
  current `step`. Poll it from the UI.
- The model runs as a separate process. The overall time limit is `solver.time_limit_s x 8 + 120` seconds unless
  `run_settings.solver.overall_timeout_s` is set. The model runs several solves (main run, ID re-plan, comparison).
- Run jobs in the background (queue or worker). A full run can take minutes.

## The owner-run numbers as a ready input and output

`examples/input_owner_run.json` is the full owner run as one input file: the built-in 30-task, 5-machine plant (seed
2024), MT, BESS (5 EUR/MWh wear cost), reserve settings, and the saved 3 price scenarios x 3 real-time branches
(`provided` mode, so the run is exactly repeatable). `examples/output_owner_run.json` is the matching output,
built from the solver results saved with that run (expected result -129,533.6 EUR, CVaR -142,356.0 EUR, 1% gap).

Use the pair as the golden test: the UI sends the input, the service must return an output with the same structure,
and the numbers must agree within the solver gap.

```sh
uv run python -m offering_service run --input examples/input_owner_run.json --output output.json
```

The same run without the service layer (model files only, in `examples/owner_run/`):

```sh
uv run factory-mt-offering --instance examples/owner_run/plant.json --offering-scenario-file examples/owner_run/scenario_set.json --rt-file examples/owner_run/rt_set.json --deploy-file examples/owner_run/deployment_set.json --id-reschedule-hour 0 --time-limit 300 --mip-gap 0.01 --compare --out results/replay
```

## Database (real DA and mFRR prices)

Used when `market_scenarios.mode = "database"`.

1. Replace the two files in `sql/` with your own queries. The comments in each file list the exact output columns,
   units and parameters. One row per hour, Europe/Copenhagen, EUR. Average 15-minute rows to hours in the SQL.
2. Set the connection in the **server environment**, never in `input.json` or the browser:
   `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGSSLMODE=require`.
3. Use a read-only database user. The code also opens the session read-only and sets a statement timeout
   (`OFFERING_DB_STATEMENT_TIMEOUT_MS`, default 30000) and a connect timeout (`OFFERING_DB_CONNECT_TIMEOUT_S`, default 10).
4. Check the connection with `uv run python -m offering_service test-db` (use the same call behind a "Test connection" button).

What the loader does:

- Each requested day with 24 complete hours in both queries becomes one scenario with equal probability. Days with
  missing hours, empty values, or a clock change (23 or 25 hours) are skipped and listed in `scenarios_used.database`.
- `da_buy = spot + buy_adder_eur_mwh` and `da_sell = spot - sell_deduct_eur_mwh`.
- **Negative prices are set to 0.** The model rejects negative prices. The number of changed values is returned in
  `meta.warnings`.
- Intraday prices are not read from the database. They are the DA price times a random factor
  (`id_spread_sigma`, `seed`). Balancing prices and reserve calls come from `realtime_scenarios`.
- If `grid.reference_tariff` is missing, the average of the scenarios is used as the reference tariff.

## Differences from the specification document

These were found while writing the code. The code is correct; update the document where it differs.

1. Reserve asset names are `mt`, `bess_dis`, `bess_ch`, `dr` (the model's `ASSETS`).
2. The model does not support negative prices (plant tariff, scenario tables). Section 9.2 of the spec said not to
   clip them; the loader sets them to 0 instead. In `provided` mode negative prices are rejected with the field name.
3. New optional fields `grid.id_cap_buy_mw` and `grid.id_cap_sell_mw` (intraday trade depth, MW per hour, default 100).
4. `grid.reference_tariff` is optional in `database` mode.
5. Additional model checks reported with the field name: daily demand must not exceed starting inventory plus the
   yield of all tasks, and the base load must not exceed the grid import limit plus the MT capacity.
6. `microturbine`: leave the key out for the model's default MT, set it to `null` for no MT. `bess`: missing or `null`
   means no battery.

## Tests

```sh
uv run python -m unittest tests.test_service -v
```

The service tests replace the model run with the saved results in `sample_results/`. They check, among other things,
that every command-line flag the service passes exists in the model's parsers, that the reserve asset names and the
MT and BESS field names match the model source, and that the output file is strict JSON.

## Not verified

- I could not run the optimisation itself when preparing this zip (no pyomo or HiGHS in my environment).
  `output_owner_run.json` is the owner run's saved solver output reshaped by the service code, not a new solve.
  Run `examples/input_owner_run.json` once and compare with it. `input_generate.json` and `input_database.json`
  use a made-up 3-machine plant: if the model reports `INFEASIBLE` for them, change the plant data, not the code.
- The database loader is tested with a fake connection. It has not been run against your PostgreSQL.
- The model solves hourly steps and is a scenario study, not a market-ready DK1/DK2 bid (see `docs/MODEL_README.md`).
