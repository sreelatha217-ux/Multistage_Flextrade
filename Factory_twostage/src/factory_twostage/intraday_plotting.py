"""Optional visualization for two-stage schedule and scenario recourse."""

import logging

import numpy as np

from .data import Instance
from .intraday_results import IntradayResult

log = logging.getLogger("factory_twostage")


def plot_intraday(inst: Instance, result: IntradayResult, path) -> bool:
    """Plot the shared production plan, DA/ID trades, prices, and scenario SoC."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    hours = inst.horizon_h
    figure, (schedule_axis, market_axis, asset_axis) = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2.4, 2]},
    )
    for machine_index, machine in enumerate(inst.machines):
        machine_jobs = result.jobs[result.jobs.machine == machine]
        for job in machine_jobs.itertuples():
            schedule_axis.barh(
                machine_index,
                job.duration_h,
                left=job.start_h,
                color=plt.cm.tab20(machine_index * 2),
                edgecolor="k",
            )
            schedule_axis.text(
                job.start_h + job.duration_h / 2,
                machine_index,
                job.task,
                ha="center",
                va="center",
                fontsize=8,
            )
    schedule_axis.set_yticks(range(len(inst.machines)), inst.machines)
    schedule_axis.set_title(
        f"Stage-1 batch schedule | expected cost {result.objective_eur:,.0f} EUR | "
        f"{int(result.kpis['n_scenarios'])} scenarios"
    )

    positions = result.da_position
    market_axis.bar(
        positions.hour + 0.3,
        positions.p_da_buy_mw - positions.p_da_sell_mw,
        width=0.4,
        color="tab:blue",
        label="DA net import (MW)",
    )
    market_axis.bar(
        positions.hour + 0.7,
        positions.exp_p_id_buy_mw - positions.exp_p_id_sell_mw,
        width=0.4,
        color="tab:orange",
        label="Expected ID net import (MW)",
    )
    market_axis.axhline(0, color="k", linewidth=0.6)
    market_axis.set_ylabel("MW")
    price_axis = market_axis.twinx()
    for scenario in range(result.scenario_set.n):
        prices = result.scenario_set.id_buy_eur_mwh[scenario]
        price_axis.step(
            np.append(np.arange(hours), hours),
            np.append(prices, prices[-1]),
            where="post",
            color="gray",
            linewidth=0.5,
            alpha=0.5,
        )
    da_prices = positions.price_da_buy_eur_mwh.to_numpy()
    price_axis.step(
        np.append(positions.hour, hours),
        np.append(da_prices, da_prices[-1]),
        where="post",
        color="tab:red",
        linewidth=1.4,
        label="DA buy price",
    )
    price_axis.set_ylabel("EUR/MWh (gray = ID scenarios)")
    market_axis.legend(loc="upper left", ncol=2)
    price_axis.legend(loc="upper right")

    if inst.bess is not None:
        for scenario in range(result.scenario_set.n):
            scenario_hours = result.hourly[result.hourly.scenario == scenario]
            asset_axis.plot(
                scenario_hours.hour + 0.5,
                scenario_hours.soc_mwh,
                color="tab:purple",
                linewidth=0.8,
                alpha=0.6,
            )
        asset_axis.axhline(inst.bess.soc_min_mwh, color="gray", linestyle=":")
        asset_axis.axhline(inst.bess.soc_max_mwh, color="gray", linestyle=":")
        asset_axis.set_ylabel("BESS SoC (MWh)")
    if inst.mt is not None:
        commitment_axis = asset_axis.twinx()
        commitment_axis.step(
            positions.hour,
            positions.mt_on * inst.mt.p_max_mw,
            where="post",
            color="tab:green",
            linewidth=1.2,
            label="MT committed",
        )
        commitment_axis.plot(
            positions.hour + 0.5,
            positions.exp_p_mt_mw,
            color="tab:olive",
            linewidth=1.2,
            label="Expected MT output",
        )
        commitment_axis.set_ylabel("MT MW")
        commitment_axis.legend(loc="upper right")

    asset_axis.set_xlabel("hour of day")
    asset_axis.set_xlim(0, hours)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return True


def plot_bess_scenarios(inst: Instance, result: IntradayResult, path) -> bool:
    """Plot scenario SoC trajectories and charge/discharge heatmaps separately."""
    if inst.bess is None:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping BESS plot")
        return False

    scenario_count = result.scenario_set.n
    hours = inst.horizon_h
    scenario_hourly = [
        result.hourly[result.hourly.scenario == scenario]
        .sort_values("hour")
        for scenario in range(scenario_count)
    ]
    soc = np.stack([frame.soc_mwh.to_numpy() for frame in scenario_hourly])
    charge = np.stack([frame.p_bess_ch_mw.to_numpy() for frame in scenario_hourly])
    discharge = np.stack([frame.p_bess_dis_mw.to_numpy() for frame in scenario_hourly])
    probabilities = result.scenario_set.prob
    hour_centers = np.arange(hours) + 0.5
    figure, (soc_axis, charge_axis, discharge_axis) = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5, 1.5]},
        constrained_layout=True,
    )

    scenario_colors = plt.get_cmap("tab10")
    for scenario in range(scenario_count):
        soc_axis.plot(
            hour_centers,
            soc[scenario],
            color=scenario_colors(scenario % 10),
            linewidth=1.4,
            alpha=0.8,
        )
    mean_soc = probabilities @ soc
    soc_axis.plot(
        hour_centers,
        mean_soc,
        color="black",
        linewidth=2.5,
        label="Probability-weighted mean",
        zorder=3,
    )
    soc_axis.axhline(
        inst.bess.soc_min_mwh,
        color="dimgray",
        linestyle=":",
        linewidth=1.4,
        label="SoC limits",
    )
    soc_axis.axhline(
        inst.bess.soc_max_mwh,
        color="dimgray",
        linestyle=":",
        linewidth=1.4,
    )
    soc_axis.set_ylim(inst.bess.soc_min_mwh - 5, inst.bess.soc_max_mwh + 5)
    soc_axis.set_ylabel("Stored energy (MWh)")
    soc_axis.set_title("BESS state of charge by intraday scenario")
    soc_axis.legend(loc="upper right")

    scenario_labels = [f"S{scenario}" for scenario in range(scenario_count)]
    for axis, values, title, color_map in (
        (charge_axis, charge, "BESS charging power by scenario", "Blues"),
        (discharge_axis, discharge, "BESS discharging power by scenario", "Oranges"),
    ):
        maximum = max(float(values.max()), 1e-9)
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(0, hours, -0.5, scenario_count - 0.5),
            cmap=color_map,
            vmin=0,
            vmax=maximum,
        )
        axis.set_yticks(np.arange(scenario_count), scenario_labels)
        axis.set_ylabel("Scenario")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, pad=0.015, label="MW")

    discharge_axis.set_xlabel("Hour of day")
    discharge_axis.set_xticks(np.arange(0, hours + 1, 2))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True