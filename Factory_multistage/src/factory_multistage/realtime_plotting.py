from __future__ import annotations

from pathlib import Path

import numpy as np

from ._internal import log
from .day_ahead import Instance
from .realtime_results import RealTimeResult


def plot_realtime(inst: Instance, res: RealTimeResult, path) -> bool:
    """Write separate production, market, BESS, MT, and RT PNGs beside ``path``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    horizon = inst.horizon_h
    hours = np.arange(horizon, dtype=float) + 0.5
    positions = res.da_position

    figure, ax = plt.subplots(figsize=(13, 5.5))
    for i, machine in enumerate(inst.machines):
        for job in res.jobs[res.jobs.machine == machine].itertuples():
            ax.barh(i, job.duration_h, left=job.start_h, color=plt.cm.tab20(i * 2), edgecolor="k")
            ax.text(job.start_h + job.duration_h / 2, i, job.task, ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(inst.machines)), inst.machines)
    ax.set(xlim=(0, horizon), xlabel="Hour of day", title="Stage-1 production schedule")
    _save_figure(figure, output_path, plt)

    figure, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(positions.hour - 0.25, positions.p_da_buy_mw - positions.p_da_sell_mw,
           width=0.25, color="tab:blue", label="DA net position")
    ax.bar(positions.hour, positions.exp_p_id_buy_mw - positions.exp_p_id_sell_mw,
           width=0.25, color="tab:orange", label="Expected ID net position")
    ax.bar(positions.hour + 0.25, positions.exp_imb_deficit_mw - positions.exp_imb_surplus_mw,
           width=0.25, color="tab:red", label="Expected RT imbalance")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Power (MW)", title="Market positions")
    ax.legend(loc="upper left", ncol=3)
    _save_figure(figure, output_path.with_name("market_positions_rt.png"), plt)

    scenario_set, rt_set = res.scenario_set, res.rt_set
    expected_id_buy = scenario_set.prob @ scenario_set.id_buy_eur_mwh
    expected_id_sell = scenario_set.prob @ scenario_set.id_sell_eur_mwh
    joint_prob = scenario_set.prob[:, None] * rt_set.prob
    rt_buy, rt_sell = rt_set.lam_plus(inst), rt_set.lam_minus(inst)
    expected_rt_buy = np.einsum("sw,swt->t", joint_prob, rt_buy)
    expected_rt_sell = np.einsum("sw,swt->t", joint_prob, rt_sell)

    figure, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(hours, scenario_set.id_buy_eur_mwh.min(axis=0),
                    scenario_set.id_buy_eur_mwh.max(axis=0), color="tab:orange", alpha=0.12,
                    label="ID buy scenario range")
    ax.fill_between(hours, rt_buy.min(axis=(0, 1)), rt_buy.max(axis=(0, 1)),
                    color="tab:red", alpha=0.10, label="RT deficit price range")
    ax.plot(hours, inst.price_buy_eur_mwh, color="tab:blue", lw=1.8, label="DA buy")
    ax.plot(hours, inst.price_sell_eur_mwh, color="deepskyblue", ls="--", label="DA sell")
    ax.plot(hours, expected_id_buy, color="tab:orange", lw=1.6, label="Expected ID buy")
    ax.plot(hours, expected_id_sell, color="goldenrod", ls="--", label="Expected ID sell")
    ax.plot(hours, expected_rt_buy, color="tab:red", lw=1.6, label="Expected RT deficit")
    ax.plot(hours, expected_rt_sell, color="tab:pink", ls="--", label="Expected RT surplus")
    ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Price (EUR/MWh)",
           title="DA, ID, and RT settlement prices")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    _save_figure(figure, output_path.with_name("market_prices_rt.png"), plt)

    figure, power_ax = plt.subplots(figsize=(13, 5.5))
    if inst.bess is not None:
        power_ax.bar(positions.hour - 0.18, positions.exp_bess_ch_mw, width=0.36,
                     color="tab:green", label="Expected charge")
        power_ax.bar(positions.hour + 0.18, -positions.exp_bess_dis_mw, width=0.36,
                     color="tab:red", label="Expected discharge")
        soc_ax = power_ax.twinx()
        for scenario in range(scenario_set.n):
            scenario_hourly = res.hourly[(res.hourly.scenario == scenario) & (res.hourly.rt == 0)]
            soc_ax.plot(scenario_hourly.hour + 0.5, scenario_hourly.soc_model_mwh,
                        color="tab:purple", lw=0.9, alpha=0.35,
                        label="ID-scenario SoC" if scenario == 0 else None)
        soc_ax.plot(hours, positions.exp_soc_mwh, color="purple", lw=1.8, label="Expected SoC")
        soc_ax.axhline(inst.bess.soc_min_mwh, color="gray", ls=":", lw=0.9, label="SoC limits")
        soc_ax.axhline(inst.bess.soc_max_mwh, color="gray", ls=":", lw=0.9)
        power_ax.axhline(0, color="black", lw=0.6)
        power_ax.set_ylabel("Power (MW; discharge below zero)")
        soc_ax.set_ylabel("State of charge (MWh)")
        power_ax.set_title(
            f"BESS dispatch | degradation {inst.bess.degradation_eur_mwh:g} EUR/MWh | "
            f"expected throughput {res.kpis['exp_bess_throughput_mwh']:.1f} MWh"
        )
        power_handles, power_labels = power_ax.get_legend_handles_labels()
        soc_handles, soc_labels = soc_ax.get_legend_handles_labels()
        power_ax.legend(power_handles + soc_handles, power_labels + soc_labels,
                        loc="upper left", ncol=3, fontsize=8)
    else:
        power_ax.text(0.5, 0.5, "No BESS configured", ha="center", va="center",
                      transform=power_ax.transAxes)
        power_ax.set_ylabel("BESS")
    power_ax.set(xlim=(0, horizon), xlabel="Hour of day")
    _save_figure(figure, output_path.with_name("bess_dispatch_rt.png"), plt)

    figure, mt_ax = plt.subplots(figsize=(13, 5.5))
    if inst.mt is not None:
        mt_frames = res.hourly[res.hourly.rt == 0]
        mt_by_scenario = np.stack([
            mt_frames[mt_frames.scenario == scenario].sort_values("hour").p_mt_mw.to_numpy()
            for scenario in range(scenario_set.n)
        ])
        mt_ax.fill_between(hours, mt_by_scenario.min(axis=0), mt_by_scenario.max(axis=0),
                           color="tab:green", alpha=0.16, label="ID scenario range")
        mt_ax.plot(hours, positions.exp_p_mt_mw, color="tab:green", lw=1.8, label="Expected MT output")
        commitment_ax = mt_ax.twinx()
        commitment_ax.step(hours, positions.mt_on, where="mid", color="tab:purple", ls="--",
                           label="DA MT commitment")
        commitment_ax.set_ylim(-0.05, 1.05)
        commitment_ax.set_yticks([0, 1], ["Off", "On"])
        commitment_ax.set_ylabel("Commitment")
        mt_ax.set_ylabel("MT generation (MW)")
        mt_ax.set_title(
            f"Microturbine | expected energy {positions.exp_p_mt_mw.sum():.1f} MWh | "
            f"expected operating cost {res.kpis['exp_mt_cost_eur']:,.0f} EUR"
        )
        output_handles, output_labels = mt_ax.get_legend_handles_labels()
        commitment_handles, commitment_labels = commitment_ax.get_legend_handles_labels()
        mt_ax.legend(output_handles + commitment_handles, output_labels + commitment_labels,
                     loc="upper left", ncol=3)
    else:
        mt_ax.text(0.5, 0.5, "No microturbine configured", ha="center", va="center",
                   transform=mt_ax.transAxes)
        mt_ax.set_ylabel("Microturbine")
    mt_ax.set(xlim=(0, horizon), xlabel="Hour of day")
    _save_figure(figure, output_path.with_name("microturbine_dispatch_rt.png"), plt)

    figure, rt_ax = plt.subplots(figsize=(13, 5.5))
    for _, scenario_hourly in res.hourly.groupby(["scenario", "rt"]):
        rt_ax.plot(scenario_hourly.hour + 0.5, scenario_hourly.imb_net_mw,
                   color="gray", lw=0.8, alpha=0.45)
    rt_ax.plot(hours, positions.exp_imb_deficit_mw - positions.exp_imb_surplus_mw,
               color="tab:red", lw=1.8, label="Expected imbalance")
    rt_ax.axhline(0, color="black", lw=0.7)
    rt_ax.set(xlim=(0, horizon), xlabel="Hour of day", ylabel="Imbalance (MW; deficit positive)",
              title="Real-time imbalance by scenario")
    rt_ax.legend(loc="upper left")
    _save_figure(figure, output_path.with_name("realtime_imbalance.png"), plt)
    return True


def _save_figure(figure, path: Path, plt) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
