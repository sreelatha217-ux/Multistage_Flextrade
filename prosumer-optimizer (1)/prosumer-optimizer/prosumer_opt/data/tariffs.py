"""Time-of-use electricity tariff (skill section 3.3)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import TOU_TIERS
from ..parameters import TimeGrid


def tou_prices(tg: TimeGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Reference TOU buy and sell prices (EUR/MWh) on the optimisation grid."""
    buy, sell = np.zeros(tg.n_steps), np.zeros(tg.n_steps)
    for t in range(tg.n_steps):
        h = (t * tg.dt_h) % 24.0
        for tier in TOU_TIERS.values():
            if any(a <= h < e for a, e in tier.blocks):
                buy[t], sell[t] = tier.buy_eur_mwh, tier.sell_eur_mwh
    return buy, sell
