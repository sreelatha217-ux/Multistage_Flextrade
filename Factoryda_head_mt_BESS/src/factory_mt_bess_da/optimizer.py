"""High-level day-ahead optimization orchestration."""

import logging

from .candidates import build_candidates
from .data import Instance, SchedulerConfig, SchedulingResult
from .model import build_model
from .results import extract_result
from .solver import solve_model

log = logging.getLogger("factory_mt_bess_da")


def optimize_day_ahead(inst: Instance, cfg: SchedulerConfig | None = None) -> SchedulingResult:
    """Validate inputs, build and solve the MILP, then independently verify results."""
    cfg = cfg or SchedulerConfig()
    inst.validate()
    cfg.check(inst)
    candidates = build_candidates(inst, cfg.start_step_h)
    log.info(
        "%d candidate starts (machines=%d, tasks=%d, step=%.2f h, MT=%s, BESS=%s)",
        len(candidates), len(inst.machines), len(inst.tasks), cfg.start_step_h,
        "yes" if inst.mt else "no", "yes" if inst.bess else "no",
    )
    model = build_model(inst, cfg, candidates)
    solve_info = solve_model(model, cfg.solver)
    return extract_result(inst, cfg, model, candidates, solve_info)
