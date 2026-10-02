from __future__ import annotations

import numpy as np

from ._internal import log
from .data import Instance, SchedulingResult


def plot_schedule(inst: Instance, res: SchedulingResult, path) -> bool:
    """Gantt chart plus supply stack (grid + MT) and price. Returns False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    has_b = inst.bess is not None
    fig, axes = plt.subplots(3 if has_b else 2, 1, figsize=(12, 11 if has_b else 8.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 2.2, 1.8] if has_b else [3, 2.2]))
    a1, a2 = axes[0], axes[1]
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Day-ahead batch schedule with microturbine + BESS  |  total cost {res.objective_eur:,.0f} EUR")
    h = res.hourly
    a2.bar(h.hour + 0.5, h.p_buy_mw, width=0.9, color="tab:blue", label="P_buy (MW)")
    a2.bar(h.hour + 0.5, h.p_mt_mw, width=0.9, bottom=h.p_buy_mw, color="tab:green", label="P_MT (MW)")
    a2.bar(h.hour + 0.5, h.p_bess_dis_mw, width=0.9, bottom=h.p_buy_mw + h.p_mt_mw, color="tab:purple", label="BESS dis (MW)")
    a2.bar(h.hour + 0.5, -h.p_sell_mw, width=0.9, color="tab:red", label="P_sell (MW)")
    a2.axhline(inst.grid_limit_mw, color="r", ls="--", lw=1, label="Q_buy")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("MW")
    a3 = a2.twinx()
    xs = np.append(h.hour, inst.horizon_h)
    a3.step(xs, np.append(h.price_buy_eur_mwh, h.price_buy_eur_mwh.iloc[-1]), where="post", color="tab:orange", label="DA buy")
    a3.step(xs, np.append(h.price_sell_eur_mwh, h.price_sell_eur_mwh.iloc[-1]), where="post", color="tab:olive", ls=":", label="DA sell")
    a3.set_ylabel("EUR/MWh")
    if has_b:
        a4 = axes[2]
        a4.bar(h.hour + 0.5, h.p_bess_dis_mw, width=0.9, color="tab:purple", label="discharge (MW)")
        a4.bar(h.hour + 0.5, -h.p_bess_ch_mw, width=0.9, color="tab:cyan", label="charge (MW)")
        a4.axhline(0, color="k", lw=0.6)
        a4.set_ylabel("BESS MW")
        a5 = a4.twinx()
        a5.plot(h.hour + 0.5, h.soc_mwh, color="k", marker="o", ms=3, label="SoC (MWh)")
        a5.axhline(inst.bess.soc_min_mwh, color="gray", ls=":", lw=1)
        a5.axhline(inst.bess.soc_max_mwh, color="gray", ls=":", lw=1)
        a5.set_ylabel("SoC MWh")
        a4.legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("hour of day")
    a2.set_xlim(0, inst.horizon_h)
    a2.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
