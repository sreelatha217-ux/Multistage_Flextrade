"""Pre-solve sanity checks, so that impossible inputs fail with a clear message instead of an
opaque 'infeasible' from the solver."""
from __future__ import annotations

import math
from typing import Dict

from .exceptions import DataValidationError
from .parameters import BESSParams, FactoryParams, MicroturbineParams, TimeGrid
from .state import InitialState
from .utils import require


def check_static_feasibility(tg: TimeGrid, factory: FactoryParams) -> None:
    """Every job fits its time window and every machine's job sequence fits the horizon."""
    T = tg.n_steps
    factory.base_series(T)
    factory.demand_series(T)
    buf = tg.steps(factory.buffer_h)
    by_machine: Dict[str, float] = {}
    for j in factory.jobs:
        d = tg.steps(j.duration_h)
        e = tg.steps(j.earliest_start_h)
        lf = T if j.latest_finish_h is None else min(T, int(math.floor(j.latest_finish_h / tg.dt_h + 1e-9)))
        require(lf - d >= e, f"job {j.job_id}: cannot fit inside its time window")
        by_machine[j.machine] = by_machine.get(j.machine, 0.0) + d + buf
    for machine, occupied in by_machine.items():
        require(occupied - buf <= T, f"machine {machine}: sequence needs {occupied} steps, horizon has {T}")


def check_state(st: InitialState, bess: BESSParams, mt: MicroturbineParams, factory: FactoryParams) -> None:
    """The measured/initial state lies inside the physical operating ranges."""
    require(bess.soc_min_mwh - 1e-6 <= st.soc_mwh <= bess.soc_max_mwh + 1e-6,
            f"initial SoC {st.soc_mwh:.3f} outside [{bess.soc_min_mwh}, {bess.soc_max_mwh}]",
            DataValidationError)
    require(0 <= st.mt_output_mw <= mt.p_max_mw + 1e-6, "initial MT output out of range", DataValidationError)
    require(bool(st.mt_online) or st.mt_output_mw < 1e-6, "MT offline but output > 0", DataValidationError)
    require(0 <= st.inventory_units <= factory.inventory_max + 1e-6,
            "initial inventory outside [0, inventory_max]", DataValidationError)
