"""
Reference parameter database (all money in EUR, power in MW, energy in MWh).

These are the *default values* used by :mod:`prosumer_opt.parameters` and
:mod:`prosumer_opt.config`. Values follow the reference database of the
industrial large-consumer multi-stage optimisation skill (sections 3.1 - 3.3).
Nothing in this module has behaviour; change values in a YAML/JSON config file
rather than editing this file where possible.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

# --------------------------------------------------------------------------- #
# Numerical
# --------------------------------------------------------------------------- #
TOL: float = 1e-9

# --------------------------------------------------------------------------- #
# Time grid
# --------------------------------------------------------------------------- #
DEFAULT_HORIZON_H: float = 24.0
DEFAULT_DT_H: float = 1.0

# --------------------------------------------------------------------------- #
# Grid connection (substation line cap Q_md)
# --------------------------------------------------------------------------- #
GRID_IMPORT_LIMIT_MW: float = 400.0
GRID_EXPORT_LIMIT_MW: float = 400.0

# --------------------------------------------------------------------------- #
# Microturbine (skill section 3.1)
# --------------------------------------------------------------------------- #
MT_P_MIN_MW: float = 10.0
MT_P_MAX_MW: float = 25.0
MT_RAMP_UP_MW_H: float = 20.0
MT_RAMP_DOWN_MW_H: float = 20.0
MT_STARTUP_RAMP_MW_H: float = 20.0
MT_SHUTDOWN_RAMP_MW_H: float = 20.0
MT_STARTUP_COST_EUR: float = 87.40
MT_SHUTDOWN_COST_EUR: float = 8.74
MT_MIN_UP_H: float = 4.0
MT_MIN_DOWN_H: float = 2.0
#: Piecewise-linear fuel-cost blocks from 0 MW upwards: (width MW, marginal EUR/MWh).
MT_COST_BLOCKS: Tuple[Tuple[float, float], ...] = (
    (5.30, 48.41), (7.20, 48.78), (7.20, 51.84), (5.30, 55.40))

# --------------------------------------------------------------------------- #
# Battery (skill section 3.2)
# --------------------------------------------------------------------------- #
BESS_ETA_CH: float = 0.80
BESS_ETA_DIS: float = 0.95
BESS_THROUGHPUT_COST_EUR_MWH: float = 30.0   # skill range: 20 - 50

#: Named BESS size profiles. Keys map 1:1 onto ``BESSParams`` fields.
BESS_PROFILES: Dict[str, Dict[str, float]] = {
    "large": dict(p_max_mw=40.0, e_max_mwh=200.0, soc_min_mwh=20.0,
                  soc_max_mwh=200.0, soc_init_mwh=20.0),
    "medium": dict(p_max_mw=2.0, e_max_mwh=4.0, soc_min_mwh=0.4,
                   soc_max_mwh=3.6, soc_init_mwh=0.63),
}

# --------------------------------------------------------------------------- #
# Factory (skill section 3.3)
# --------------------------------------------------------------------------- #
FACTORY_BASE_LOAD_MW: float = 3.5
FACTORY_BUFFER_H: float = 1.0
FACTORY_INVENTORY_MAX: float = 1000.0
FACTORY_SHIFT_WINDOW_H: float = 2.0
FACTORY_UNMET_PENALTY_EUR_PER_UNIT: float = 5000.0
BATCH_DURATION_RANGE_H: Tuple[float, float] = (1.5, 4.9)
BATCH_POWER_RANGE_MW: Tuple[float, float] = (60.0, 110.0)

# --------------------------------------------------------------------------- #
# Time-of-use tariff (skill section 3.3), EUR/MWh
# --------------------------------------------------------------------------- #
class TariffTier(NamedTuple):
    buy_eur_mwh: float
    sell_eur_mwh: float
    blocks: Tuple[Tuple[int, int], ...]   # hour-of-day blocks [start, end)


TOU_TIERS: Dict[str, TariffTier] = {
    "off":  TariffTier(66.42, 36.53, ((2, 4), (12, 14), (18, 20), (22, 24))),
    "flat": TariffTier(88.56, 48.71, ((4, 6),)),
    "mid":  TariffTier(118.08, 64.94, ((0, 2), (6, 8), (14, 16), (20, 22))),
    "on":   TariffTier(177.12, 97.42, ((8, 12), (16, 18))),
}

# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
SOLVER_NAME: str = "appsi_highs"
SOLVER_MIP_GAP: float = 1e-4        # skill recommendation: <= 0.01 %
SOLVER_TIME_LIMIT_S: float = 300.0

# --------------------------------------------------------------------------- #
# Synthetic scenario / demo-data generators
# --------------------------------------------------------------------------- #
SCENARIO_SELL_RATIO: float = 0.55
SCENARIO_DA_AR_COEFF: float = 0.70    # AR(1) persistence of day-ahead price shocks
SCENARIO_DA_SIGMA: float = 0.10
SCENARIO_ID_SIGMA: float = 0.06       # intraday basis noise vs day-ahead
SCENARIO_LOAD_SIGMA: float = 0.03     # real-time load deviation
SCENARIO_R_MINUS_BASE: float = 1.10
SCENARIO_R_MINUS_SPREAD: float = 0.25
SCENARIO_R_PLUS_BASE: float = 0.90
SCENARIO_R_PLUS_SPREAD: float = 0.25

DEMO_UNITS_OUT_PER_BATCH: float = 100.0
DEMO_DEMAND_UNITS_PER_H: float = 25.0
DEMO_DEMAND_WINDOW_H: Tuple[float, float] = (8.0, 24.0)
DEMO_BASE_LOAD_OFF_SHIFT_MW: float = 3.0
DEMO_BASE_LOAD_ON_SHIFT_MW: float = 4.5
DEMO_ON_SHIFT_WINDOW_H: Tuple[float, float] = (8.0, 18.0)

INTRADAY_PRICE_NOISE_SIGMA: float = 0.05
INTRADAY_LOAD_DEV_SIGMA: float = 0.02
#: Real-time branches of the refreshed intraday scenario:
#: (probability, mean load deviation, r_minus, r_plus)
INTRADAY_RT_BRANCHES: Tuple[Tuple[float, float, float, float], ...] = (
    (0.5, 0.02, 1.2, 0.8),
    (0.5, -0.02, 1.3, 0.7),
)
