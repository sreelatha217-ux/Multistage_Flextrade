"""Small set of market and production plots for the offering optimizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_selected_offering_figures(inst, res, outdir, schedule_path=None) -> None:
    """Write market positions, market prices, and production schedule figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    horizon = inst.horizon_h
    positions = res.da_position
    scenarios = res.scenario_set
    realtime = res.rt_set
    hours = np.arange(horizon, dtype=float) + 0.5

    figure, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(positions.hour - 0.25, positions.exp_p_da_buy_mw - positions.exp_p_da_sell_mw,
           width=0.25, color="tab:blue", label="Expected DA net position")
    ax.bar(positions.hour, positions.exp_p_id_buy_mw - positions.exp_p_id_sell_mw,
           width=0.25, color="tab:orange", label="Expected ID net position")
    ax.bar(positions.hour + 0.25, positions.exp_imb_deficit_mw - positions.exp_imb_surplus_mw,
           width=0.25, color="tab:red", label="Expected RT imbalance")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Power (MW)", title="Market positions")
    ax.legend(loc="upper left", ncol=3)
    figure.tight_layout()
    figure.savefig(output / "market_positions.png", dpi=150)
    plt.close(figure)

    joint_prob = scenarios.prob[:, None] * realtime.prob
    da_buy = scenarios.prob @ scenarios.da_buy
    da_sell = scenarios.prob @ scenarios.da_sell
    id_buy = scenarios.prob @ scenarios.id_buy
    id_sell = scenarios.prob @ scenarios.id_sell
    rt_buy = realtime.r_plus * scenarios.da_buy[:, None, :]
    rt_sell = realtime.r_minus * scenarios.da_sell[:, None, :]
    expected_rt_buy = np.einsum("sw,swt->t", joint_prob, rt_buy)
    expected_rt_sell = np.einsum("sw,swt->t", joint_prob, rt_sell)

    figure, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(hours, scenarios.da_buy.min(axis=0), scenarios.da_buy.max(axis=0),
                    color="tab:blue", alpha=0.10, label="DA buy scenario range")
    ax.fill_between(hours, scenarios.id_buy.min(axis=0), scenarios.id_buy.max(axis=0),
                    color="tab:orange", alpha=0.12, label="ID buy scenario range")
    ax.plot(hours, da_buy, color="tab:blue", lw=1.8, label="Expected DA buy")
    ax.plot(hours, da_sell, color="deepskyblue", ls="--", label="Expected DA sell")
    ax.plot(hours, id_buy, color="tab:orange", lw=1.6, label="Expected ID buy")
    ax.plot(hours, id_sell, color="goldenrod", ls="--", label="Expected ID sell")
    ax.plot(hours, expected_rt_buy, color="tab:red", lw=1.6, label="Expected RT deficit")
    ax.plot(hours, expected_rt_sell, color="tab:pink", ls="--", label="Expected RT surplus")
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Price (EUR/MWh)", title="Market prices")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "market_prices.png", dpi=150)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(13, 5.5))
    for index, machine in enumerate(inst.machines):
        for job in res.jobs[res.jobs.machine == machine].itertuples():
            ax.barh(index, job.duration_h, left=job.start_h, color=plt.cm.tab20(index * 2), edgecolor="k")
            ax.text(job.start_h + job.duration_h / 2, index, job.task, ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(inst.machines)), inst.machines)
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Machine", title="Production schedule")
    ax.grid(True, axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(schedule_path or output / "production_schedule.png", dpi=150)
    plt.close(figure)


def plot_intraday_reschedules(inst, baseline_jobs, scenario_jobs, scenario_summary, outdir,
                              reschedule_hour: float) -> None:
    """Compare the day-ahead production schedule with each intraday reschedule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    horizon = inst.horizon_h
    scenarios = sorted(scenario_summary.id_scenario.astype(int).unique())
    figure, axes = plt.subplots(len(scenarios) + 1, 1,
                                figsize=(13, max(6, 3.2 * (len(scenarios) + 1))),
                                sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]

    def draw_schedule(ax, jobs, title):
        for machine_index, machine in enumerate(inst.machines):
            machine_jobs = jobs[jobs.machine == machine]
            for job in machine_jobs.itertuples():
                ax.barh(machine_index, job.duration_h, left=job.start_h,
                        color=plt.cm.tab10(machine_index % 10), edgecolor="black")
                ax.text(job.start_h + job.duration_h / 2, machine_index, job.task,
                        ha="center", va="center", fontsize=7)
        if 0 < reschedule_hour < horizon:
            ax.axvline(reschedule_hour, color="black", linestyle="--", linewidth=1,
                       label=f"Reschedule cutoff: h={reschedule_hour:g}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set(yticks=range(len(inst.machines)), yticklabels=inst.machines,
               xlim=(0, horizon), ylabel="Machine", title=title)
        ax.grid(True, axis="x", alpha=0.2)

    draw_schedule(axes[0], baseline_jobs, "Day-ahead production schedule")
    for axis, scenario in zip(axes[1:], scenarios):
        jobs = scenario_jobs[scenario_jobs.id_scenario == scenario]
        summary = scenario_summary[scenario_summary.id_scenario == scenario].iloc[0]
        draw_schedule(axis, jobs,
                      f"Intraday scenario {scenario} | cost {summary.operating_cost_eur:,.0f} EUR | "
                      f"{int(summary.jobs_changed)} jobs changed")
    axes[-1].set_xlabel("Hour of delivery day")
    figure.suptitle(
        f"Production reschedule by ID price scenario | cutoff hour {reschedule_hour:g} | "
        "already-started jobs remain fixed", y=0.995
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output / "production_reschedule.png", dpi=150)
    plt.close(figure)
