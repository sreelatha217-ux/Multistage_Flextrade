"""Optional schedule and microturbine dispatch visualization."""

import numpy as np

from ._internal import log
from .data import Instance, SchedulingResult


def plot_schedule(inst: Instance, result: SchedulingResult, path) -> bool:
    """Save the batch Gantt chart and grid/MT supply stack with DA prices."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    figure, (gantt_axis, dispatch_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8.5),
        gridspec_kw={"height_ratios": [3, 2.2]},
        sharex=True,
    )
    for machine_index, machine in enumerate(inst.machines):
        machine_jobs = result.jobs[result.jobs.machine == machine]
        for job in machine_jobs.itertuples():
            gantt_axis.barh(
                machine_index,
                job.duration_h,
                left=job.start_h,
                color=plt.cm.tab20(machine_index * 2),
                edgecolor="black",
            )
            gantt_axis.text(
                job.start_h + job.duration_h / 2,
                machine_index,
                job.task,
                ha="center",
                va="center",
                fontsize=8,
            )
    gantt_axis.set_yticks(range(len(inst.machines)), inst.machines)
    gantt_axis.set_title(
        f"Day-ahead schedule with microturbine | total cost {result.objective_eur:,.0f} EUR"
    )

    hourly = result.hourly
    dispatch_axis.bar(
        hourly.hour + 0.5,
        hourly.p_da_mw,
        width=0.9,
        color="tab:blue",
        label="P_DA (MW)",
    )
    dispatch_axis.bar(
        hourly.hour + 0.5,
        hourly.p_mt_mw,
        width=0.9,
        bottom=hourly.p_da_mw,
        color="tab:green",
        label="P_MT (MW)",
    )
    dispatch_axis.axhline(
        inst.grid_limit_mw,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="Grid limit",
    )
    dispatch_axis.set_ylabel("Power (MW)")
    price_axis = dispatch_axis.twinx()
    price_axis.step(
        np.append(hourly.hour, inst.horizon_h),
        np.append(hourly.price_eur_mwh, hourly.price_eur_mwh.iloc[-1]),
        where="post",
        color="tab:orange",
        label="DA price",
    )
    price_axis.set_ylabel("Price (EUR/MWh)")
    dispatch_axis.set_xlabel("Hour of day")
    dispatch_axis.set_xlim(0, inst.horizon_h)
    dispatch_axis.legend(loc="upper left", ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return True