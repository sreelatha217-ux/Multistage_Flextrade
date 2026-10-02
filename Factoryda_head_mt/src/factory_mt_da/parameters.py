"""Single source of truth for scheduler and benchmark defaults."""

VERSION = "2.0.0"
EPS = 1e-9

DEFAULT_HORIZON_H = 24
DEFAULT_BUFFER_H = 1.0
DEFAULT_GRID_LIMIT_MW = 400.0
DEFAULT_MAX_BATCHES_PER_MACHINE = 10
DEFAULT_INVENTORY_INIT = 50.0
DEFAULT_INVENTORY_MAX = 500.0
DEFAULT_START_STEP_H = 0.5
DEFAULT_UNIQUE_TASKS = True
DEFAULT_CHARGE_MIN_POWER_FUEL = False
DEFAULT_STRICT_VERIFICATION = True

DEFAULT_SOLVER = "appsi_highs"
DEFAULT_MIP_GAP = 1e-4
DEFAULT_TIME_LIMIT_S = 180.0
DEFAULT_SOLVER_THREADS = None
DEFAULT_OUTPUT_DIR = "da_mt_results"

DEFAULT_MT = {
    "p_min_mw": 10.0,
    "p_max_mw": 25.0,
    "ramp_up_mw_h": 20.0,
    "ramp_down_mw_h": 20.0,
    "startup_ramp_mw_h": 20.0,
    "shutdown_ramp_mw_h": 20.0,
    "startup_cost_eur": 87.40,
    "shutdown_cost_eur": 8.74,
    "min_up_h": 4,
    "min_down_h": 2,
    "block_width_mw": (5.3, 7.2, 7.2, 5.3),
    "block_cost_eur_mwh": (48.41, 48.78, 51.84, 55.40),
    "initial_on": False,
    "initial_power_mw": 0.0,
}

BENCHMARK_MACHINE_POWER_RANGES_MW = {
    "m1": (70, 95),
    "m2": (60, 80),
    "m3": (70, 90),
    "m4": (90, 110),
    "m5": (65, 85),
}
BENCHMARK_TARIFF_BLOCKS = (
    (66.42, ((2, 4), (12, 14), (18, 20), (22, 24))),
    (88.56, ((4, 6),)),
    (118.08, ((0, 2), (6, 8), (14, 16), (20, 22))),
    (177.12, ((8, 12), (16, 18))),
)
BENCHMARK_TASK_COUNT = 30
BENCHMARK_SEED = 2024
BENCHMARK_DEMAND_UNITS_PER_H = 8.0
BENCHMARK_BASE_LOAD_CENTER_MW = 3.5
BENCHMARK_BASE_LOAD_AMPLITUDE_MW = 1.0
BENCHMARK_TASK_DURATION_RANGE_H = (1.3, 4.9)
BENCHMARK_TASK_YIELD_RANGE_UNITS = (5, 16)