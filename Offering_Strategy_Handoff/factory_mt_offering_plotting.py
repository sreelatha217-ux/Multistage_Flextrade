"""Separate diagnostic figures for the strategic offering results."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_offering_figures(inst, res, outdir) -> None:
    """Write RT-style scheduler plots and offering-specific diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    horizon = inst.horizon_h
    hours = np.arange(horizon, dtype=float) + 0.5
    positions = res.da_position
    scenarios = res.scenario_set
    realtime = res.rt_set
    hourly = res.hourly
    expected_net_cost = -res.expected_profit_eur

    def save(fig, name):
        if fig._suptitle is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    figure, ax = plt.subplots(figsize=(13, 5.5))
    for index, machine in enumerate(inst.machines):
        for job in res.jobs[res.jobs.machine == machine].itertuples():
            ax.barh(index, job.duration_h, left=job.start_h, color=plt.cm.tab20(index * 2), edgecolor="k")
            ax.text(job.start_h + job.duration_h / 2, index, job.task, ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(inst.machines)), inst.machines)
    ax.set(xlim=(0, horizon), xlabel="Hour of day",
           title=f"Stage-1 production schedule | expected net cost {expected_net_cost:,.0f} EUR")
    save(figure, "schedule_rt.png")

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
    save(figure, "market_positions_rt.png")

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
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Price (EUR/MWh)",
           title="DA, ID, and RT settlement prices")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    save(figure, "market_prices_rt.png")

    figure, power_ax = plt.subplots(figsize=(13, 5.5))
    if inst.bess is not None:
        power_ax.bar(positions.hour - 0.18, positions.exp_bess_ch_mw, width=0.36,
                     color="tab:green", label="Expected charge")
        power_ax.bar(positions.hour + 0.18, -positions.exp_bess_dis_mw, width=0.36,
                     color="tab:red", label="Expected discharge")
        soc_ax = power_ax.twinx()
        for (scenario, rt), path in hourly.groupby(["scenario", "rt"]):
            soc_ax.plot(path.hour + 0.5, path.soc_path_mwh, color="tab:purple", lw=0.8, alpha=0.28,
                        label="Joint-scenario SoC" if scenario == 0 and rt == 0 else None)
        soc_ax.plot(hours, positions.exp_soc_mwh, color="purple", lw=1.8, label="Expected SoC")
        soc_ax.axhline(inst.bess.soc_min_mwh, color="gray", ls=":", lw=0.9, label="SoC limits")
        soc_ax.axhline(inst.bess.soc_max_mwh, color="gray", ls=":", lw=0.9)
        power_ax.axhline(0, color="black", lw=0.6)
        power_ax.set_ylabel("Power (MW; discharge below zero)")
        soc_ax.set_ylabel("State of charge (MWh)")
        power_ax.set_title(
            f"BESS dispatch | degradation {inst.bess.degradation_eur_mwh:g} EUR/MWh | "
            f"expected degradation cost {res.kpis['exp_cost_bess_eur']:,.0f} EUR"
        )
        power_handles, power_labels = power_ax.get_legend_handles_labels()
        soc_handles, soc_labels = soc_ax.get_legend_handles_labels()
        power_ax.legend(power_handles + soc_handles, power_labels + soc_labels,
                        loc="upper left", ncol=3, fontsize=8)
    else:
        power_ax.text(0.5, 0.5, "No BESS configured", ha="center", va="center", transform=power_ax.transAxes)
        power_ax.set_ylabel("BESS")
    power_ax.set(xlim=(0, horizon), xlabel="Hour of day")
    save(figure, "bess_dispatch_rt.png")

    figure, mt_ax = plt.subplots(figsize=(13, 5.5))
    if inst.mt is not None:
        mt_paths = hourly.pivot_table(index=["scenario", "rt"], columns="hour", values="p_mt_rt_mw").sort_index()
        mt_values = mt_paths.to_numpy().reshape(scenarios.n, realtime.n_w, horizon)
        mt_mean = np.einsum("sw,swt->t", joint_prob, mt_values)
        mt_ax.fill_between(hours, mt_values.min(axis=(0, 1)), mt_values.max(axis=(0, 1)),
                           color="tab:green", alpha=0.16, label="Joint scenario range")
        mt_ax.plot(hours, mt_mean, color="tab:green", lw=1.8, label="Expected RT MT output")
        commitment_ax = mt_ax.twinx()
        commitment_ax.step(hours, positions.mt_on, where="mid", color="tab:purple", ls="--",
                           label="DA MT commitment")
        commitment_ax.set_ylim(-0.05, 1.05)
        commitment_ax.set_yticks([0, 1], ["Off", "On"])
        commitment_ax.set_ylabel("Commitment")
        mt_ax.set_ylabel("MT generation (MW)")
        mt_ax.set_title("Microturbine output and commitment")
        output_handles, output_labels = mt_ax.get_legend_handles_labels()
        commitment_handles, commitment_labels = commitment_ax.get_legend_handles_labels()
        mt_ax.legend(output_handles + commitment_handles, output_labels + commitment_labels,
                     loc="upper left", ncol=3)
    else:
        mt_ax.text(0.5, 0.5, "No microturbine configured", ha="center", va="center",
                   transform=mt_ax.transAxes)
        mt_ax.set_ylabel("Microturbine")
    mt_ax.set(xlim=(0, horizon), xlabel="Hour of day")
    save(figure, "microturbine_dispatch_rt.png")

    figure, rt_ax = plt.subplots(figsize=(13, 5.5))
    for _, path in hourly.groupby(["scenario", "rt"]):
        rt_ax.plot(path.hour + 0.5, path.imb_deficit_mw - path.imb_surplus_mw,
                   color="gray", lw=0.8, alpha=0.45)
    rt_ax.plot(hours, positions.exp_imb_deficit_mw - positions.exp_imb_surplus_mw,
               color="tab:red", lw=1.8, label="Expected imbalance")
    rt_ax.axhline(0, color="black", lw=0.7)
    rt_ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Imbalance (MW; deficit positive)",
              title="Real-time imbalance by scenario")
    rt_ax.legend(loc="upper left")
    save(figure, "realtime_imbalance.png")

    figure, ax = plt.subplots(figsize=(13, 5.5))
    selected_hours = list(np.argsort(-positions.exp_price_da_buy_eur_mwh)[:2])
    selected_hours += list(np.argsort(positions.exp_price_da_buy_eur_mwh)[:2])
    for hour in dict.fromkeys([*selected_hours, *[h for h in (8, 14) if h < horizon]]):
        curve = res.offer_curves[res.offer_curves.hour == hour].sort_values("price_da_sell_eur_mwh")
        ax.step(curve.price_da_sell_eur_mwh, curve.p_da_sell_mw - curve.p_da_buy_mw,
                where="post", marker="o", ms=3, label=f"hour {hour}")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel="Scenario DA sell price (EUR/MWh)", ylabel="Net DA export offer (MW)",
           title="DA price-quantity offer curves")
    ax.legend(loc="best", ncol=2)
    save(figure, "da_offer_curves.png")

    figure, power_ax = plt.subplots(figsize=(13, 5.5))
    power_ax.bar(positions.hour + 0.5, positions.exp_r_bess_dis_mw, 0.9,
                 label="BESS discharge", color="tab:purple")
    power_ax.bar(positions.hour + 0.5, positions.exp_r_bess_ch_mw, 0.9,
                 bottom=positions.exp_r_bess_dis_mw, label="BESS charge curtailment", color="tab:cyan")
    power_ax.bar(positions.hour + 0.5, positions.exp_r_dr_mw, 0.9,
                 bottom=positions.exp_r_bess_dis_mw + positions.exp_r_bess_ch_mw,
                 label="Demand response", color="tab:orange")
    power_ax.bar(positions.hour + 0.5, positions.exp_r_mt_mw, 0.9,
                 bottom=positions.exp_r_bess_dis_mw + positions.exp_r_bess_ch_mw + positions.exp_r_dr_mw,
                 label="Microturbine", color="tab:green")
    power_ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Expected reserve offer (MW)",
                 title="Up-spinning reserve by asset")
    reserve_price_ax = power_ax.twinx()
    reserve_price_ax.plot(positions.hour + 0.5, positions.exp_price_sr_up_eur_mw_h,
                          "k.-", label="Expected reserve price")
    reserve_price_ax.set_ylabel("Reserve price (EUR/MW/h)")
    power_handles, power_labels = power_ax.get_legend_handles_labels()
    price_handles, price_labels = reserve_price_ax.get_legend_handles_labels()
    power_ax.legend(power_handles + price_handles, power_labels + price_labels,
                    loc="upper left", ncol=3, fontsize=8)
    save(figure, "reserve_offers.png")

    figure, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(res.joint.profit_eur, bins=min(20, max(5, len(res.joint))),
            weights=res.joint.prob, color="tab:blue", alpha=0.75)
    ax.axvline(res.expected_profit_eur, color="green", label=f"E = {res.expected_profit_eur:,.0f} EUR")
    ax.axvline(res.cvar_profit_eur, color="red", label=f"CVaR = {res.cvar_profit_eur:,.0f} EUR")
    ax.set(xlabel="Profit per joint scenario (EUR)", ylabel="Probability", title="Profit distribution")
    ax.legend()
    save(figure, "profit_distribution.png")

    figure, (production_ax, commitment_ax, offers_ax) = plt.subplots(3, 1, figsize=(13, 12))
    for index, machine in enumerate(inst.machines):
        for job in res.jobs[res.jobs.machine == machine].itertuples():
            production_ax.barh(index, job.duration_h, left=job.start_h,
                               color=plt.cm.tab20(index * 2), edgecolor="k")
            production_ax.text(job.start_h + job.duration_h / 2, index, job.task,
                               ha="center", va="center", fontsize=8)
    production_ax.set_yticks(range(len(inst.machines)), inst.machines)
    production_ax.set(xlim=(0, horizon), ylabel="Machine",
                      title=f"Shared production | expected net cost {expected_net_cost:,.0f} EUR")
    commitment_ax.step(hours, positions.mt_on, where="mid", color="tab:purple", lw=1.8)
    commitment_ax.set(xlim=(0, horizon), ylim=(-0.05, 1.05), ylabel="MT commitment",
                      yticks=[0, 1], yticklabels=["Off", "On"], title="Shared day-ahead MT commitment")
    selected_hours = list(np.argsort(-positions.exp_price_da_buy_eur_mwh)[:2])
    selected_hours += list(np.argsort(positions.exp_price_da_buy_eur_mwh)[:2])
    for hour in dict.fromkeys([*selected_hours, *[h for h in (8, 14) if h < horizon]]):
        curve = res.offer_curves[res.offer_curves.hour == hour].sort_values("price_da_sell_eur_mwh")
        offers_ax.step(curve.price_da_sell_eur_mwh, curve.p_da_sell_mw - curve.p_da_buy_mw,
                       where="post", marker="o", ms=3, label=f"hour {hour}")
    offers_ax.axhline(0, color="black", lw=0.7)
    offers_ax.set(xlabel="Scenario DA sell price (EUR/MWh)", ylabel="Net export offer (MW)",
                  title="DA offer curves committed here-and-now")
    offers_ax.legend(loc="best", ncol=2)
    figure.suptitle(
        f"Stage 1 | here-and-now decisions shared before uncertainty is revealed | "
        f"expected net cost {expected_net_cost:,.0f} EUR", y=0.99
    )
    save(figure, "stage1_here_and_now.png")

    stage2 = hourly[hourly.rt == 0]
    figure, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    stage2_series = (
        ("p_id_buy_mw", "p_id_sell_mw", "ID net trade", "Net import (MW)"),
        ("p_mt_mw", None, "MT dispatch", "Generation (MW)"),
        ("p_bess_dis_mw", "p_bess_ch_mw", "BESS net dispatch", "Net injection (MW)"),
        ("r_total_mw", None, "Reserve offer", "Reserve offer (MW)"),
    )
    for ax, (positive, negative, title, ylabel) in zip(axes, stage2_series):
        for scenario, path in stage2.groupby("scenario"):
            path = path.sort_values("hour")
            values = path[positive] if negative is None else path[positive] - path[negative]
            ax.plot(path.hour + 0.5, values, color=plt.cm.tab10(int(scenario) % 10), marker=".", lw=1.3,
                    label=f"Price scenario s={scenario}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(ylabel=ylabel, title=title)
        ax.legend(loc="upper left", ncol=min(scenarios.n, 4), fontsize=8)
    axes[-1].set(xlim=(0, horizon), xlabel="Hour of day")
    figure.suptitle("Stage 2 | wait-and-see recourse by revealed price scenario", y=0.99)
    save(figure, "stage2_wait_and_see.png")

    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for (scenario, rt), path in hourly.groupby(["scenario", "rt"]):
        path = path.sort_values("hour")
        label = f"(s={scenario}, w={rt})"
        color = plt.cm.tab10(int(scenario) % 10)
        line_style = ("-", "--", ":", "-.")[int(rt) % 4]
        axes[0].plot(path.hour + 0.5, path.imb_deficit_mw - path.imb_surplus_mw,
                     color=color, linestyle=line_style, lw=1.0, alpha=0.8, label=label)
        axes[1].plot(path.hour + 0.5, path.rho * path.r_total_mw,
                     color=color, linestyle=line_style, lw=1.0, alpha=0.8, label=label)
        if inst.bess is not None:
            axes[2].plot(path.hour + 0.5, path.soc_path_mwh, color=color, linestyle=line_style,
                         lw=1.0, alpha=0.75, label=label)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("Imbalance (MW)")
    axes[0].set_title("Real-time imbalance settlement")
    axes[1].set_ylabel("Activated reserve (MW)")
    axes[1].set_title("Reserve deployment revealed in real time")
    if inst.bess is not None:
        axes[2].axhline(inst.bess.soc_min_mwh, color="gray", ls=":", lw=0.8)
        axes[2].axhline(inst.bess.soc_max_mwh, color="gray", ls=":", lw=0.8)
        axes[2].set_ylabel("BESS SoC (MWh)")
        axes[2].set_title("BESS state after scenario-specific reserve activation")
    else:
        axes[2].text(0.5, 0.5, "No BESS configured", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_ylabel("BESS SoC")
    axes[-1].set(xlim=(0, horizon), xlabel="Hour of day")
    axes[0].legend(loc="upper left", ncol=min(scenarios.n * realtime.n_w, 4), fontsize=8)
    figure.suptitle("Stage 3 | wait-and-see settlement by joint (price, real-time) scenario", y=0.99)
    save(figure, "stage3_wait_and_see.png")


def plot_intraday_reschedules(inst, baseline_jobs, scenario_jobs, scenario_summary, outdir,
                              reschedule_hour: float) -> None:
    """Compare the Stage-1 plan with one ID-price-based Stage-2 production plan per scenario."""
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

    draw_schedule(axes[0], baseline_jobs, "Stage 1 baseline production schedule")
    for axis, scenario in zip(axes[1:], scenarios):
        jobs = scenario_jobs[scenario_jobs.id_scenario == scenario]
        summary = scenario_summary[scenario_summary.id_scenario == scenario].iloc[0]
        draw_schedule(axis, jobs,
                      f"Stage 2 ID scenario {scenario} | cost {summary.operating_cost_eur:,.0f} EUR | "
                      f"{int(summary.jobs_changed)} jobs changed")
    axes[-1].set_xlabel("Hour of delivery day")
    figure.suptitle(
        f"Production rescheduling by ID price scenario | cutoff hour {reschedule_hour:g} | "
        "Stage-1 started jobs remain fixed", y=0.995
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output / "stage2_production_reschedules.png", dpi=150)
    plt.close(figure)


def plot_intraday_reschedule_analysis(inst, baseline_jobs, scenario_jobs, scenario_summary,
                                      scenario_set, outdir) -> None:
    """Plot the ID price paths, their planned production loads, and per-scenario metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    horizon = inst.horizon_h
    hours = np.arange(horizon, dtype=float) + 0.5
    scenarios = sorted(scenario_summary.id_scenario.astype(int).unique())
    colors = {scenario: plt.cm.tab10(scenario % 10) for scenario in scenarios}

    def hourly_batch_load(jobs):
        load = np.zeros(horizon)
        hour_start = np.arange(horizon, dtype=float)
        for job in jobs.itertuples():
            overlap = np.clip(np.minimum(job.end_h, hour_start + 1.0) - np.maximum(job.start_h, hour_start),
                              0.0, 1.0)
            load += job.power_mw * overlap
        return load

    figure, (price_ax, load_ax) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for scenario in scenarios:
        color = colors[scenario]
        price_ax.plot(hours, scenario_set.id_buy[scenario], color=color, lw=1.5,
                      label=f"ID buy, s={scenario}")
        price_ax.plot(hours, scenario_set.id_sell[scenario], color=color, lw=0.9, linestyle="--", alpha=0.75)
        jobs = scenario_jobs[scenario_jobs.id_scenario == scenario]
        load_ax.plot(hours, hourly_batch_load(jobs), color=color, lw=1.5, marker=".",
                     label=f"Stage 2, s={scenario}")
    load_ax.plot(hours, hourly_batch_load(baseline_jobs), color="black", lw=1.7, linestyle=":",
                 label="Stage 1 baseline")
    price_ax.set(ylabel="ID price (EUR/MWh)", title="Intraday buy prices by scenario (dashed = sell)")
    load_ax.set(xlim=(0, horizon), xlabel="Hour of delivery day", ylabel="Batch production load (MW)",
                title="Factory load implied by the production schedules")
    price_ax.legend(loc="upper left", ncol=min(len(scenarios), 4), fontsize=8)
    load_ax.legend(loc="upper left", ncol=min(len(scenarios) + 1, 4), fontsize=8)
    figure.suptitle("Intraday prices and Stage-2 production rescheduling", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "intraday_prices_and_load.png", dpi=150)
    plt.close(figure)

    summary = scenario_summary.sort_values("id_scenario")
    labels = [f"s={int(row.id_scenario)}\n{row.probability:.0%}" for row in summary.itertuples()]
    x = np.arange(len(summary))
    width = 0.36
    figure, (cost_ax, saving_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fixed_bars = cost_ax.bar(x - width / 2, summary.fixed_plan_cost_eur, width,
                             color="0.65", label="Stage-1 jobs fixed")
    recourse_bars = cost_ax.bar(x + width / 2, summary.operating_cost_eur, width,
                                color=[colors[s] for s in scenarios], label="Stage-2 jobs rescheduled")
    cost_ax.bar_label(fixed_bars, fmt="%.0f", padding=3, fontsize=8)
    cost_ax.bar_label(recourse_bars, fmt="%.0f", padding=3, fontsize=8)
    cost_ax.set_ylabel("Operating cost (EUR)")
    cost_ax.set_title("Matched ID-price operating costs")
    cost_ax.legend(loc="best", fontsize=8)
    savings = summary.production_reschedule_savings_eur
    saving_bars = saving_ax.bar(x, savings,
                                color=["tab:green" if value >= 0 else "tab:red" for value in savings])
    saving_ax.axhline(0, color="black", linewidth=0.8)
    saving_ax.bar_label(saving_bars, fmt="%+.0f", padding=3, fontsize=8)
    saving_ax.set(xticks=x, xticklabels=labels, xlabel="ID scenario (probability)",
                  ylabel="Cost saved (EUR)", title="Production rescheduling value: fixed cost - recourse cost")
    figure.suptitle("Stage-2 production reschedule value", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "intraday_reschedule_metrics.png", dpi=150)
    plt.close(figure)


def plot_owner_dashboard(inst, result, reschedule_summary, market_stages, outdir) -> None:
    """Create an owner-facing summary of participation costs, cost drivers, and flexibility value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    stage_ax, bridge_ax, recourse_ax, production_ax = axes.ravel()
    kpis = result.kpis

    stages = market_stages.reset_index(drop=True)
    stage_costs = stages.expected_net_cost_eur.to_numpy()
    stage_labels = ["DA only\n(+ RT settlement)", "DA + ID\n(+ RT settlement)", "DA + ID\n+ up-reserve"]
    stage_colors = ["0.55", "tab:blue", "tab:green"]
    bars = stage_ax.bar(np.arange(len(stages)), stage_costs, color=stage_colors)
    stage_ax.bar_label(bars, labels=[f"{value:,.0f}" for value in stage_costs], padding=3, fontsize=8)
    stage_ax.set(xticks=np.arange(len(stages)), xticklabels=stage_labels,
                 ylabel="Expected net cost (EUR)", title="Value of market participation")
    stage_ax.tick_params(axis="x", labelsize=8)
    baseline_cost = float(stage_costs[0])
    for index, value in enumerate(stage_costs):
        stage_ax.text(index, value * 0.88, f"saved {baseline_cost - value:,.0f}",
                      ha="center", va="top", fontsize=8, color="white" if index else "black")

    items = [
        ("DA net\nsettlement", -kpis["exp_rev_da_eur"]),
        ("ID net\nsettlement", -kpis["exp_rev_id_eur"]),
        ("Reserve\ncapacity", -kpis["exp_rev_sr_eur"]),
        ("Reserve\nactivation", -kpis["exp_rev_act_eur"]),
        ("Imbalance", kpis["exp_cost_bal_eur"]),
        ("MT", kpis["exp_cost_mt_eur"]),
        ("BESS\ndegradation", kpis["exp_cost_bess_eur"]),
        ("Demand\nresponse", kpis["exp_cost_dr_eur"]),
    ]
    running = 0.0
    for index, (label, delta) in enumerate(items):
        end = running + delta
        bridge_ax.bar(index, abs(delta), bottom=min(running, end),
                      color="tab:red" if delta >= 0 else "tab:green", width=0.72)
        running = end
    net_cost = -result.expected_profit_eur
    bridge_ax.bar(len(items), net_cost, color="tab:purple", width=0.72)
    bridge_ax.set_xticks(range(len(items) + 1), [label for label, _ in items] + ["Expected\nnet cost"])
    bridge_ax.set_ylabel("EUR")
    bridge_ax.set_title("Expected cost bridge (reserve revenue offsets cost)")
    bridge_ax.tick_params(axis="x", labelsize=7, rotation=25)
    bridge_ax.grid(True, axis="y", alpha=0.2)

    recourse = reschedule_summary.sort_values("id_scenario")
    scenario_ids = recourse.id_scenario.astype(int).tolist()
    savings = recourse.production_reschedule_savings_eur.to_numpy()
    recourse_ax.bar(scenario_ids, savings,
                    color=["tab:green" if value >= 0 else "tab:red" for value in savings])
    recourse_ax.axhline(0, color="black", linewidth=0.8)
    expected_saving = float(np.dot(recourse.probability, savings))
    recourse_ax.axhline(expected_saving, color="tab:blue", linestyle="--", linewidth=1,
                        label=f"Probability-weighted saving: {expected_saving:,.0f} EUR")
    recourse_ax.set(xticks=scenario_ids, xlabel="ID price scenario", ylabel="Cost saved (EUR)",
                    title="Value of production rescheduling")
    recourse_ax.legend(loc="best", fontsize=8)

    labels = ["Stage 1 baseline"] + [f"ID scenario {scenario}" for scenario in scenario_ids]
    units = [float(result.jobs.units_out.sum())] + recourse.units_produced.astype(float).tolist()
    batches = [len(result.jobs)] + recourse.batches.astype(int).tolist()
    bars = production_ax.bar(np.arange(len(labels)), units, color=["0.55"] + [plt.cm.tab10(s % 10) for s in scenario_ids])
    production_ax.bar_label(bars, labels=[f"{value:.0f} units\n{batch} jobs" for value, batch in zip(units, batches)],
                            padding=3, fontsize=8)
    production_ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
                      ylabel="Production (units)", title="Production maintained across reschedules")
    production_ax.tick_params(axis="x", labelsize=8)

    figure.suptitle(
        f"Factory owner dashboard | {int(result.kpis['n_scenarios'])} ID x {int(result.kpis['n_rt'])} RT scenarios | "
        f"BESS degradation {inst.bess.degradation_eur_mwh if inst.bess else 0:g} EUR/MWh",
        y=0.99,
    )
    figure.text(0.5, 0.015,
                "Market stages use the same scenario tree; Stage-2 reschedule savings are against the fixed Stage-1 job plan.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(output / "factory_owner_dashboard.png", dpi=160)
    plt.close(figure)