"""Result container returned by every solve, with summary printing and CSV/JSON export."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ._version import __version__
from .state import DAPlan


@dataclass
class StageResult:
    """
    Outcome of one optimisation stage.

    schedule         per-step expected values (DA position, MT, BESS, ID, load, imbalance, SoC, inventory)
    batch_plan       per job: machine, DA start/end, power, per-scenario start
    scenario_detail  long table with every scenario-level decision
    cost_breakdown   expected EUR by component (DA, ID, BAL, MT, BESS, UNMET, TOTAL)
    scenario_costs   cost by scenario
    risk             mean, std, worst, best, CVaR95 of total cost
    plan             frozen DA commitments (input of the intraday stage)
    """
    stage: str
    t0: int
    status: str
    objective_eur: float
    mip_gap: Optional[float]
    solve_time_s: float
    model_stats: Dict[str, int]
    schedule: pd.DataFrame
    batch_plan: pd.DataFrame
    scenario_detail: pd.DataFrame
    cost_breakdown: Dict[str, float]
    scenario_costs: pd.DataFrame
    risk: Dict[str, float]
    plan: DAPlan

    def summary(self) -> str:
        cb = "  ".join(f"{k}={v:,.1f}" for k, v in self.cost_breakdown.items())
        gap = "n/a" if self.mip_gap is None else f"{100 * self.mip_gap:.4f}%"
        return (f"[{self.stage}] status={self.status} expected_cost={self.objective_eur:,.2f} EUR "
                f"gap={gap} time={self.solve_time_s:.1f}s\n  {cb}\n  "
                f"risk: " + "  ".join(f"{k}={v:,.1f}" for k, v in self.risk.items()))

    def save(self, outdir) -> Path:
        """Write four CSV tables and a JSON summary, prefixed with the stage name."""
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        tag = self.stage.lower()
        self.schedule.to_csv(out / f"{tag}_schedule.csv", index_label="step")
        self.batch_plan.to_csv(out / f"{tag}_batch_plan.csv", index=False)
        self.scenario_detail.to_csv(out / f"{tag}_scenario_detail.csv", index=False)
        self.scenario_costs.to_csv(out / f"{tag}_scenario_costs.csv", index=False)
        meta = dict(stage=self.stage, t0=self.t0, status=self.status, objective_eur=self.objective_eur,
                    mip_gap=self.mip_gap, solve_time_s=self.solve_time_s, model_stats=self.model_stats,
                    cost_breakdown=self.cost_breakdown, risk=self.risk, version=__version__)
        (out / f"{tag}_summary.json").write_text(json.dumps(meta, indent=2))
        return out
