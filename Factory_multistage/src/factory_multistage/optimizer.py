from __future__ import annotations

from typing import Optional

from ._internal import log
from .candidates import build_candidates
from .data import Instance, SchedulerConfig, SchedulingResult
from .model import build_model
from .results import extract_result
from .solver import solve_model


def optimize_day_ahead(inst: Instance, cfg: Optional[SchedulerConfig] = None) -> SchedulingResult:
    """Validate -> build candidates -> build MILP -> solve -> verify -> package results."""
    cfg = cfg or SchedulerConfig()
    inst.validate()
    cfg.check(inst)
    cands = build_candidates(inst, cfg.start_step_h)
    log.info("%d candidate batch starts (machines=%d, tasks=%d, step=%.2f h, MT=%s)", len(cands),
             len(inst.machines), len(inst.tasks), cfg.start_step_h, "yes" if inst.mt else "no")
    model = build_model(inst, cfg, cands)
    info = solve_model(model, cfg.solver)
    return extract_result(inst, cfg, model, cands, info)
