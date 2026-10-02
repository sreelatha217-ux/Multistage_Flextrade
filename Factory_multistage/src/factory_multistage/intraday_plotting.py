from __future__ import annotations

import numpy as np

from ._internal import log
from .day_ahead import Instance
from .intraday_results import IntradayResult


def plot_intraday(inst: Instance, res: IntradayResult, path) -> bool:
    """Gantt chart, DA position vs expected ID trade with the ID price fan, and the BESS SoC fan."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False
    T = inst.horizon_h
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 2.4, 2]))
    for i, mach in enumerate(inst.machines):
        for r in res.jobs[res.jobs.machine == mach].itertuples():
            a1.barh(i, r.duration_h, left=r.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            a1.text(r.start_h + r.duration_h / 2, i, r.task, ha="center", va="center", fontsize=8)
    a1.set_yticks(range(len(inst.machines)), inst.machines)
    a1.set_title(f"Stage-1 batch schedule | expected cost {res.objective_eur:,.0f} EUR | {int(res.kpis['n_scenarios'])} scenarios")
    p = res.da_position
    a2.bar(p.hour + 0.3, p.p_da_buy_mw - p.p_da_sell_mw, width=0.4, color="tab:blue", label="DA net import (MW)")
    a2.bar(p.hour + 0.7, p.exp_p_id_buy_mw - p.exp_p_id_sell_mw, width=0.4, color="tab:orange", label="E[ID net import] (MW)")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("MW")
    pr = a2.twinx()
    for k in range(res.scenario_set.n):
        pr.step(np.append(np.arange(T), T), np.append(res.scenario_set.id_buy_eur_mwh[k], res.scenario_set.id_buy_eur_mwh[k][-1]),
                where="post", color="gray", lw=0.5, alpha=0.5)
    pr.step(np.append(p.hour, T), np.append(p.price_da_buy_eur_mwh, p.price_da_buy_eur_mwh.iloc[-1]), where="post",
            color="tab:red", lw=1.4, label="DA buy price")
    pr.set_ylabel("EUR/MWh (grey = ID buy scenarios)")
    a2.legend(loc="upper left", ncol=2)
    pr.legend(loc="upper right")
    if inst.bess is not None:
        for k in range(res.scenario_set.n):
            hk = res.hourly[res.hourly.scenario == k]
            a3.plot(hk.hour + 0.5, hk.soc_mwh, color="tab:purple", lw=0.8, alpha=0.6)
        a3.axhline(inst.bess.soc_min_mwh, color="gray", ls=":")
        a3.axhline(inst.bess.soc_max_mwh, color="gray", ls=":")
        a3.set_ylabel("BESS SoC (MWh), one line per scenario")
    if inst.mt is not None:
        a4 = a3.twinx()
        a4.step(p.hour, p.mt_on * inst.mt.p_max_mw, where="post", color="tab:green", lw=1.2, label="MT committed (x Pmax)")
        a4.plot(p.hour + 0.5, p.exp_p_mt_mw, color="tab:olive", lw=1.2, label="E[P_MT] (MW)")
        a4.set_ylabel("MT MW")
        a4.legend(loc="upper right")
    a3.set_xlabel("hour of day")
    a3.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
