"""Optional schedule visualizations."""
import numpy as np

from ._internal import log
from .data import Instance, SchedulingResult


def plot_schedule(inst: Instance, res: SchedulingResult, path) -> bool:
    """Gantt chart plus load/price profile. Returns False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[3, 2]), sharex=True)
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Day-ahead batch schedule  |  cost {res.objective_eur:,.0f} EUR")
    h = res.hourly
    a2.bar(h.hour + 0.5, h.p_da_mw, width=0.9, color="tab:blue", label="P_DA (MW)")
    a2.axhline(inst.grid_limit_mw, color="r", ls="--", lw=1, label="Q_md")
    a2.set_ylabel("MW")
    a3 = a2.twinx()
    a3.step(np.append(h.hour, inst.horizon_h), np.append(h.price_eur_mwh, h.price_eur_mwh.iloc[-1]),
            where="post", color="tab:orange", label="DA price")
    a3.set_ylabel("EUR/MWh")
    a2.set_xlabel("hour of day")
    a2.set_xlim(0, inst.horizon_h)
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
