# Factory Multistage Scheduler

A Pyomo mixed-integer optimization model for factory production, a microturbine (MT), a battery energy storage system (BESS), and grid positions across day-ahead (DA), intraday (ID), and real-time (RT) balancing.

This is a **price-taking scheduling model**, not a market-clearing or bid-price optimizer. It chooses quantities at supplied or generated prices. The DA and ID schedules are optimized; RT load deviations are settled as imbalance. The model does not submit offer-price curves or re-dispatch the MT/BESS in RT.

## 1. Install

Run commands from this directory (`Factory_multistage`):

```sh
uv sync --extra test
```

The project requires Python 3.10 or later. `uv.lock` pins the resolved dependencies. The default open-source solver is HiGHS through `highspy`.

## 2. Model stages

1. **DA:** choose the shared production batch schedule, MT on/off commitment, and DA buy/sell quantities.
2. **ID:** for each ID price scenario, adjust ID buy/sell, MT dispatch, and BESS charge/discharge. The DA production schedule remains fixed.
3. **RT:** for each conditional RT scenario, settle load mismatch as deficit or surplus imbalance. Production, MT dispatch, and BESS dispatch are not changed at this stage.

## 3. Inputs

### Use the built-in benchmark

No input files are required. By default the program generates the factory, TOU tariffs, ID price scenarios, and RT scenarios from the configured seeds. The starting settings are in `src/factory_multistage/configs/default.json`.

### Supply a custom factory instance

Use `--instance path/to/instance.json`. The required plant fields are:

| Field | Shape / meaning |
|---|---|
| `machines` | Machine names, length M |
| `tasks` | Task names, length P |
| `power_mw` | M by P task power matrix |
| `duration_h` | M by P task duration matrix |
| `yield_units` | Yield per task, length P |
| `base_load_mw` | Non-shiftable load, length T |
| `price_buy_eur_mwh` | DA purchase price, length T |
| `price_sell_eur_mwh` | DA sale price, length T; must not exceed the buy price |
| `demand_units` | Product demand, length T |

T is `horizon_h` (24 by default). Optional instance fields include `buffer_h`, `grid_limit_mw`, `grid_sell_limit_mw`, `max_batches_per_machine`, `inventory_init`, and `inventory_max`. If `mt` is omitted, the default MT is used. A custom instance has no BESS unless a `bess` object is provided.

Small valid example (`inputs/plant.json`):

```json
{
	"machines": ["furnace-1"],
	"tasks": ["batch-1"],
	"power_mw": [[2.0]],
	"duration_h": [[1.0]],
	"yield_units": [10.0],
	"base_load_mw": [1.0, 1.0],
	"price_buy_eur_mwh": [100.0, 120.0],
	"price_sell_eur_mwh": [50.0, 60.0],
	"demand_units": [0.0, 5.0],
	"horizon_h": 2,
	"buffer_h": 0.0,
	"grid_limit_mw": 50.0,
	"grid_sell_limit_mw": 50.0,
	"max_batches_per_machine": 1,
	"inventory_init": 0.0,
	"inventory_max": 20.0,
	"bess": {
		"p_max_mw": 2.0,
		"e_max_mwh": 10.0,
		"soc_min_frac": 0.1,
		"soc_max_frac": 0.9,
		"soc_init_mwh": 2.0,
		"degradation_eur_mwh": 5.0
	}
}
```

### MT and BESS omission rules

`mt` and `bess` behave differently when omitted from a custom instance:

- If the `mt` key is **not present**, the instance creates an MT with the default parameters from `data.py`.
- To disable the MT, set `"mt": null`.
- If the `bess` key is **not present** or is `null`, the instance has no BESS.
- To include a BESS, provide a `bess` object. You can specify only the values you want to change; missing values inside the object use the BESS defaults from `data.py`.

For example, this disables the MT and creates a BESS using all default BESS parameters:

```json
{
	"mt": null,
	"bess": {}
}
```

Powers are MW, energy is MWh, durations are hours, prices are EUR/MWh, and inventory/yield/demand are product units.

### Supply scenarios instead of generating them

For ID scenarios, use `--scenario-file path.json`. The JSON root has `prob`, `id_buy_eur_mwh`, `id_sell_eur_mwh`, and `load_dev_mw`. Price and load arrays have shape **[S, T]**; probabilities have length S and sum to one.

```json
{
	"prob": [0.5, 0.5],
	"id_buy_eur_mwh": [[95.0, 110.0], [105.0, 125.0]],
	"id_sell_eur_mwh": [[52.0, 60.0], [57.0, 68.0]],
	"load_dev_mw": [[0.0, 0.0], [0.1, -0.1]]
}
```

For RT scenarios, use `--rt-file path.json`. The root has `prob`, `load_rel_dev`, `r_plus`, and `r_minus`. `prob` has shape **[S, W]**; each ID-scenario row sums to one. The other arrays have shape **[S, W, T]**. `r_plus` must be at least 1; `r_minus` must be between 0 and 1. Scenario files saved in a previous result directory (`scenario_set.json`, `rt_set.json`) can be reused to reproduce a run.

When scenario files are not provided, ID prices are generated from DA prices and RT imbalance prices are generated as ratios of DA buy/sell prices. RT load deviation is also generated. These are modelling assumptions; use calibrated files when available.

## 4. Change parameters

Edit `src/factory_multistage/configs/default.json` for common run settings. Command-line options override config values.

| Config section | Change these values |
|---|---|
| `benchmark` | `seed`, `max_batches`, `with_mt`, `with_bess` |
| `scheduler` | `start_step_h` (batch start-time resolution), `unique_tasks` |
| `solver` | solver `name`, `mip_gap`, `time_limit_s`, `threads`, `verbose` |
| `intraday` | buy/sell market caps, scenario count/seed, price and load volatility |
| `realtime` | RT scenarios/seed, load deviation volatility and persistence, imbalance price ratios, `imbalance_mode`, imbalance limit |

Change common asset parameters from CLI flags when available:

```sh
--no-mt                       # disable the MT
--no-bess                     # disable the BESS
--mt-ramp-up 15 --mt-ramp-down 15
--bess-degradation 5          # EUR/MWh
--bess-degradation-basis throughput
--bess-power 20               # MW
--bess-energy 100              # MWh
```

For other MT/BESS engineering values (efficiencies, SoC limits, fuel blocks, startup costs, etc.), add or edit the nested `mt` and `bess` objects in a custom instance JSON. The dataclass defaults and validation rules are in `src/factory_multistage/data.py`. `--bess-degradation` changes the current run only; edit the custom instance JSON to save that value with the asset data.

Scenario and solver options are also available as CLI flags. Check the complete list with:

```sh
uv run factory-mt-da --help
uv run factory-mt-id --help
uv run factory-mt-rt --help
```

## 5. Run the schedulers

Run each stage from this directory. `--out` selects the output folder; `--plot` enables plots.

**DA only:**

```sh
uv run factory-mt-da --config src/factory_multistage/configs/default.json \
	--out results/day_ahead --plot
```

**DA + ID:**

```sh
uv run factory-mt-id --config src/factory_multistage/configs/default.json \
	--out results/intraday --plot
```

**Full DA + ID + RT:**

```sh
uv run factory-mt-rt --config src/factory_multistage/configs/default.json \
	--out results/multistage --plot
```

For a quicker test run, reduce the optimization size and loosen the solver tolerance:

```sh
uv run factory-mt-rt --config src/factory_multistage/configs/default.json \
	--max-batches 5 --scenarios 3 --rt-scenarios 3 \
	--time-limit 120 --mip-gap 0.01 --out results/quick --plot
```

Use custom plant and scenario inputs like this:

```sh
uv run factory-mt-rt --config src/factory_multistage/configs/default.json \
	--instance inputs/plant.json \
	--scenario-file inputs/id_scenarios.json \
	--rt-file inputs/rt_scenarios.json \
	--out results/custom --plot
```

To change the BESS degradation cost for one run, add `--bess-degradation 5`. The solver's `--mip-gap` is a stopping tolerance, not a guarantee that two separate runs will have an exact cost ranking.

## 6. Read the outputs

Each run prints a cost summary and writes files under the selected `--out` directory.

| Output | Written by | Purpose |
|---|---|---|
| `summary.json` | All stages | Objective/expected cost, cost KPIs, solver status/gap/time, model size, and independent verification result |
| `jobs.csv` | All stages | Selected machine, task, batch position, start/end, power, yield, and energy cost |
| `hourly.csv` | DA | Hourly load, DA buy/sell, MT, BESS, inventory, and cost details |
| `da_position.csv` | ID and RT | DA positions plus expected ID trades, MT/BESS dispatch, SoC, and (for RT) expected imbalance |
| `scenarios.csv` | ID and RT | One aggregated row per ID scenario: probability, cost, trading volumes, and asset use |
| `hourly_scenarios.csv` | ID | Hour-by-hour records for each ID scenario |
| `hourly_rt.csv` | RT | Hourly records for every ID/RT scenario pair, including MT/BESS dispatch, market prices, imbalance, and cost components |
| `joint_scenarios.csv` | RT | One aggregate row for each ID/RT scenario pair |
| `scenario_set.json` | ID and RT | Exact ID scenario data used by the run; reusable with `--scenario-file` |
| `rt_set.json` | RT | Exact conditional RT scenario data used by the run; reusable with `--rt-file` |

With `--plot`, DA writes `schedule.png`; ID writes `schedule.png`; RT writes separate figures so the charts stay readable:

- `schedule_rt.png`: production batch schedule
- `market_positions_rt.png`: DA, expected ID, and expected imbalance quantities
- `market_prices_rt.png`: DA, expected ID, and expected RT settlement prices and scenario ranges
- `bess_dispatch_rt.png`: expected BESS charge/discharge and scenario/expected SoC
- `microturbine_dispatch_rt.png`: MT dispatch range, expected output, and DA commitment
- `realtime_imbalance.png`: RT imbalance paths and expected imbalance

For MT/BESS cost and operating values, start with `summary.json`; inspect `hourly_rt.csv` for time-series details. Important columns include `p_mt_mw`, `mt_on`, `mt_startup`, `mt_shutdown`, `p_bess_ch_mw`, `p_bess_dis_mw`, `soc_mwh`, `imb_deficit_mw`, and `imb_surplus_mw`.

## 7. Tests and compatibility

```sh
uv run --extra test pytest
uv run factory-mt-rt --selftest
```

The selftest runs solver-backed consistency checks. The original `factory_mt_da_scheduler.py`, `factory_mt_id_scheduler.py`, and `factory_mt_rt_scheduler.py` filenames remain compatibility launchers.