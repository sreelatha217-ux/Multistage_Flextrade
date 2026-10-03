# Factory Multistage Offering Strategy

This project contains the day-ahead, intraday, and real-time schedulers plus an optional strategic offering extension. The base three-stage formulation is documented in [factory_multistage_formulation-v3.md](factory_multistage_formulation-v3.md). The offering model adds scenario-dependent DA bid/offer curves, up-spinning-reserve offers, deployment, and CVaR risk control.

The existing scheduler modules remain colocated so their shared model objects and helper functions are reused directly. The offering CLI is separated into `factory_mt_offering_cli.py`; `factory_mt_offering_strategy.py` remains the public API and backward-compatible script entry point.

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