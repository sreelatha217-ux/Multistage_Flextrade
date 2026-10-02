"""Schedule visualization."""

import logging
from pathlib import Path

from .data import Instance, SchedulingResult

log = logging.getLogger("factory_mt_bess_da")


def plot_schedule(inst: Instance, result: SchedulingResult, path: str | Path) -> bool:
    """Save a Gantt, grid/MT dispatch, tariff, and optional battery plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    has_bess = inst.bess is not None
    figure, axes = plt.subplots(
        3 if has_bess else 2,
        1,
        figsize=(12, 11 if has_bess else 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2.2, 1.8] if has_bess else [3, 2.2]},
    )
    gantt, dispatch = axes[0], axes[1]
    for machine_index, machine in enumerate(inst.machines):
        machine_jobs = result.jobs[result.jobs.machine == machine]
        for row in machine_jobs.itertuples():
            gantt.barh(machine_index, row.duration_h, left=row.start_h,
                       color=plt.cm.tab20(machine_index * 2), edgecolor="k")
            gantt.text(row.start_h + row.duration_h / 2, machine_index, row.task,
                       ha="center", va="center", fontsize=8)
    gantt.set_yticks(range(len(inst.machines)), inst.machines)
    gantt.set_title(
        f"Factory + microturbine + BESS schedule | total cost {result.objective_eur:,.0f} EUR"
    )
    hourly = result.hourly
    x = hourly.hour + 0.5
    dispatch.bar(x, hourly.p_buy_mw, width=0.9, color="tab:blue", label="Grid buy (MW)")
    dispatch.bar(x, hourly.p_mt_mw, width=0.9, bottom=hourly.p_buy_mw,
                 color="tab:green", label="MT (MW)")
    dispatch.bar(x, hourly.p_bess_dis_mw, width=0.9,
                 bottom=hourly.p_buy_mw + hourly.p_mt_mw,
                 color="tab:purple", label="BESS discharge (MW)")
    dispatch.bar(x, -hourly.p_sell_mw, width=0.9, color="tab:red", label="Grid sell (MW)")
    dispatch.axhline(inst.grid_limit_mw, color="r", ls="--", lw=1, label="Import limit")
    dispatch.axhline(0, color="k", lw=0.6)
    dispatch.set_ylabel("MW")
    tariff_axis = dispatch.twinx()
    step_x = hourly.hour.to_numpy()
    tariff_axis.step(
        list(step_x) + [inst.horizon_h],
        list(hourly.price_buy_eur_mwh) + [hourly.price_buy_eur_mwh.iloc[-1]],
        where="post", color="tab:orange", label="DA buy",
    )
    tariff_axis.step(
        list(step_x) + [inst.horizon_h],
        list(hourly.price_sell_eur_mwh) + [hourly.price_sell_eur_mwh.iloc[-1]],
        where="post", color="tab:olive", ls=":", label="DA sell",
    )
    tariff_axis.set_ylabel("EUR/MWh")
    if has_bess:
        battery_axis = axes[2]
        battery_axis.bar(x, hourly.p_bess_dis_mw, width=0.9,
                         color="tab:purple", label="Discharge (MW)")
        battery_axis.bar(x, -hourly.p_bess_ch_mw, width=0.9,
                         color="tab:cyan", label="Charge (MW)")
        battery_axis.axhline(0, color="k", lw=0.6)
        battery_axis.set_ylabel("BESS MW")
        soc_axis = battery_axis.twinx()
        soc_axis.plot(x, hourly.soc_mwh, color="k", marker="o", ms=3, label="SoC (MWh)")
        soc_axis.axhline(inst.bess.soc_min_mwh, color="gray", ls=":", lw=1)
        soc_axis.axhline(inst.bess.soc_max_mwh, color="gray", ls=":", lw=1)
        soc_axis.set_ylabel("SoC MWh")
        battery_axis.legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Hour of day")
    dispatch.set_xlim(0, inst.horizon_h)
    dispatch.legend(loc="upper left", ncol=5)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return True
