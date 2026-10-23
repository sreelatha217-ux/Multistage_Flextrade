"""Real-data versions of the decision figures (paper Figs 2-12 were drawn from the old SYNTHETIC day).

All inputs come from the reference tree (history end 2026-10-11, real DK1 DA + mFRR prices, Stage-1 reserve):
  <root>/real_data_results/ladder_mfrr_stage1/2026-10-11/{jobs,da_position,offer_curves,hourly_offering,
                                                          joint_scenarios,market_stage_comparison}.csv, summary.json
  <root>/real_data_results/resched_backtest/backtest.csv
  <root>/real_data_results_gurobi/realized/oos/oos.csv
Usage (from anywhere): python real_data_figures.py [--root <Offering_Strategy_Handoff folder>] --out Figures
Writes fig_real_dashboard, fig_real_stage1, fig_real_reserve, fig_real_bess_mt, fig_real_stage2, fig_real_stage3 (.pdf/.png)
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))  # folder containing real_data_results/
ap.add_argument("--out", default="Figures")
ap.add_argument("--tree", default="2026-10-11")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = os.path.join(a.root, "real_data_results", "ladder_mfrr_stage1", a.tree)
rd = lambda f: pd.read_csv(os.path.join(T, f))
jobs, pos, oc, hr, jt, cmp_ = (rd("jobs.csv"), rd("da_position.csv"), rd("offer_curves.csv"),
                               rd("hourly_offering.csv"), rd("joint_scenarios.csv"), rd("market_stage_comparison.csv"))
summ = json.load(open(os.path.join(T, "summary.json")))
K = summ["kpis"]
bt = pd.read_csv(os.path.join(a.root, "real_data_results", "resched_backtest", "backtest.csv"))
oos = pd.read_csv(os.path.join(a.root, "real_data_results_gurobi", "realized", "oos", "oos.csv"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titleweight": "bold",
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False})
SC = ["#2E86AB", "#E07A1F", "#7B2D8E"]          # price scenarios s = 0,1,2 (cheap -> expensive)
H = np.arange(24)
prob_s = hr.groupby("scenario").prob_joint.sum().values


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"{name}.{e}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 1. dashboard
fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
c = cmp_.expected_net_cost_eur.values
ax[0, 0].bar(["L1  DA only", "L2  DA+ID", "L3  DA+ID+reserve"], c / 1e3, color=["#8C8C8C", "#4C78A8", "#F58518"], width=.6)
for i, v in enumerate(c):
    ax[0, 0].text(i, v / 1e3 + .3, f"{v:,.0f}", ha="center", fontsize=8)
ax[0, 0].set_ylim(c.min() / 1e3 - 8, c.max() / 1e3 + 3); ax[0, 0].set_ylabel("Expected net cost (kEUR)")
ax[0, 0].set_title(f"(a) Market ladder, reference tree {a.tree}")

items = [("DA net purchases", -K["exp_rev_da_eur"]), ("Intraday net", -K["exp_rev_id_eur"]), ("MT fuel + commitment", K["exp_cost_mt_eur"]),
         ("BESS degradation", K["exp_cost_bess_eur"]), ("DR activation", K["exp_cost_dr_eur"]), ("RT imbalance", K["exp_cost_bal_eur"]),
         ("Reserve capacity", -K["exp_rev_sr_eur"]), ("Reserve activation", -K["exp_rev_act_eur"])]
cum = 0
for i, (n, v) in enumerate(items):
    ax[0, 1].bar(i, v / 1e3, bottom=cum / 1e3 if v >= 0 else (cum + v) / 1e3, color="#C0504D" if v > 0 else "#4F9D69", width=.65)
    cum += v
ax[0, 1].bar(len(items), cum / 1e3, color="#333333", width=.65)
ax[0, 1].set_xticks(range(len(items) + 1)); ax[0, 1].set_xticklabels([n for n, _ in items] + ["Total"], rotation=40, ha="right", fontsize=7)
ax[0, 1].set_ylabel("kEUR"); ax[0, 1].set_title(f"(b) Cost bridge (total {cum:,.0f} EUR)")

x = np.arange(len(bt))
ax[1, 0].bar(x, bt.saving_strict_eur / 1e3, color="#2E86AB", width=.6, label="Saving, rescheduling +/-3 h")
ax[1, 0].errorbar(x, bt.saving_strict_eur / 1e3, yerr=bt.noise_eur / 1e3, fmt="none", ecolor="k", capsize=3, lw=1, label="Solver noise")
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([d[5:] for d in bt.day], rotation=0)
ax[1, 0].set_ylabel("kEUR per day"); ax[1, 0].legend(frameon=False, fontsize=7.5)
ax[1, 0].set_title(f"(c) Rescheduling after DA clearing (mean {bt.saving_strict_eur.mean():,.0f} EUR)")

w = .38
ax[1, 1].bar(x - w / 2, oos.ins_saving_da_only_eur / 1e3, w, color="#BBBBBB", label="Expected on tree")
ax[1, 1].bar(x + w / 2, oos.saving_da_only_eur / 1e3, w, color="#F58518", label="Realized on actual day")
ax[1, 1].axhline(0, color="k", lw=.6)
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([d[5:] for d in oos.day]); ax[1, 1].set_ylabel("Saving vs DA only (kEUR)")
ax[1, 1].legend(frameon=False, fontsize=7.5)
ax[1, 1].set_title(f"(d) Realized {oos.saving_da_only_eur.mean():,.0f} EUR vs expected {oos.ins_saving_da_only_eur.mean():,.0f} EUR (mean)")
fig.tight_layout(); save(fig, "fig_real_dashboard")

# ---------------------------------------------------------------- 2. Stage-1: Gantt, MT, DA curves
# ---------------------------------------------------------------- 2. Stage-1: single-column figure
# Designed for direct insertion into a two-column paper.
fig = plt.figure(figsize=(3.4, 5.8))
gs = fig.add_gridspec(
    3, 1,
    height_ratios=[1.5, 0.85, 1.65],
    hspace=0.65
)

# (a) Production schedule
ax = fig.add_subplot(gs[0])
mach = sorted(jobs.machine.unique())
cm = plt.cm.tab10

for _, r in jobs.iterrows():
    y = mach.index(r.machine)
    ax.barh(
        y, r.end_h - r.start_h,
        left=r.start_h,
        color=cm(y),
        edgecolor="white",
        height=0.62
    )
    ax.text(
        (r.start_h + r.end_h) / 2, y, r.task,
        ha="center", va="center",
        fontsize=6, color="white"
    )

ax.set_yticks(range(len(mach)))
ax.set_yticklabels(mach, fontsize=6)
ax.invert_yaxis()
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 6))
ax.tick_params(axis="both", labelsize=6, pad=2)
ax.set_xlabel("Hour of day", fontsize=6)
ax.set_title(
    f"(a) Production schedule: {len(jobs)} jobs, "
    f"{jobs.units_out.sum():.0f} units",
    fontsize=6, pad=5
)

# (b) Microturbine commitment
ax = fig.add_subplot(gs[1])
ax.step(H, pos.mt_on, where="post",
        color="#7B2D8E", label="MT online")
ax.plot(H, pos.exp_p_mt_mw / 25,
        color="#4F9D69", label="Expected output / 25 MW")

ax.set_ylim(-0.05, 1.15)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 6))
ax.tick_params(axis="both", labelsize=6, pad=2)
ax.set_xlabel("Hour of day", fontsize=6)
ax.set_title(
    f"(b) Microturbine commitment "
    f"({K['mt_on_hours']:.0f} h online, "
    f"{K['mt_starts']:.0f} start-up)",
    fontsize=6, pad=5
)
ax.legend(frameon=False, fontsize=6, loc="lower left")

# (c) Day-Ahead offer curves
ax = fig.add_subplot(gs[2])
oc["net_export"] = oc.p_da_sell_mw - oc.p_da_buy_mw

rng = oc.groupby("hour").net_export.agg(
    lambda s: s.max() - s.min()
).sort_values(ascending=False)

hours = sorted(rng.index[:7])

for h, col in zip(
    hours, plt.cm.viridis(np.linspace(0, 0.9, len(hours)))
):
    g = oc[oc.hour == h].sort_values("price_da_sell_eur_mwh")
    ax.step(
        g.price_da_sell_eur_mwh,
        g.net_export,
        where="post",
        marker="o", ms=2,
        color=col, label=f"h{h}"
    )

ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("DA sell price (EUR/MWh)", fontsize=6)
ax.set_ylabel("Net export (MW)", fontsize=6)
ax.tick_params(axis="both", labelsize=6, pad=2)
ax.set_title(
    "(c) Day-Ahead offer curves (7 hours)",
    fontsize=6, pad=5
)
ax.legend(
    frameon=False, ncol=4, fontsize=6,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),
    columnspacing=0.8, handlelength=1.2
)

save(fig, "fig_real_stage1")


# ---------------------------------------------------------------- 3. Reserve stack
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.7, 1]))
parts = [("exp_r_bess_dis_mw", "BESS discharge headroom", "#7B2D8E"), ("exp_r_bess_ch_mw", "BESS charge curtailment", "#2AB0C5"),
         ("exp_r_dr_mw", "Demand response", "#F58518"), ("exp_r_mt_mw", "MT ramp headroom", "#4F9D69")]
b = np.zeros(24)
for col, lab, cc in parts:
    ax[0].bar(H, pos[col], bottom=b, color=cc, label=lab, width=.8); b += pos[col].values
ax[0].set_xlabel("Hour of day"); ax[0].set_ylabel("Reserve offer (MW)")
a2 = ax[0].twinx(); a2.spines["right"].set_visible(True)
for s in range(3):
    p = oc[oc.scenario == s].sort_values("hour")
    a2.plot(p.hour, p.price_sr_up_eur_mw_h, color=SC[s], lw=1.2, label=f"mFRR price, day type {s}")
a2.set_ylabel("mFRR up-capacity price (EUR/MW/h)")
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, ncol=2, loc="upper left")
ax[0].set_ylim(0, b.max() * 1.55)
ax[0].set_title(f"(a) Stage-1 reserve by asset ({K['avg_reserve_offer_mw']:.1f} MW mean, {K['reserve_hours']:.0f} h offered)")
tot = [pos[c_].mean() for c_, _, _ in parts]
ax[1].barh([p[1] for p in parts][::-1], tot[::-1], color=[p[2] for p in parts][::-1])
for i, v in enumerate(tot[::-1]):
    ax[1].text(v + .1, i, f"{v:.1f} MW", va="center", fontsize=8)
ax[1].set_xlabel("Mean over 24 h (MW)"); ax[1].set_xlim(0, max(tot) * 1.3); ax[1].set_title("(b) Asset contribution")
fig.tight_layout(); save(fig, "fig_real_reserve")

# ---------------------------------------------------------------- 4. BESS and MT


# ----------------------------------------------------------------
# 4. BESS and MT — wider Elsevier single-column figure
# ----------------------------------------------------------------
fig, ax = plt.subplots(
    2, 1,
    figsize=(3.5, 5.3),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.4, 1.0],
        "hspace": 0.10
    }
)

FS_TITLE = 6
FS_LABEL = 7
FS_TICK = 6
FS_LEGEND = 5.5

# (a) BESS dispatch and state of charge
ax[0].bar(
    H, pos.exp_bess_ch_mw,
    color="#4F9D69", width=0.75,
    label="Expected charge"
)
ax[0].bar(
    H, -pos.exp_bess_dis_mw,
    color="#C0504D", width=0.75,
    label="Expected discharge"
)

ax[0].set_ylabel("BESS power (MW)", fontsize=FS_LABEL, labelpad=2)
ax[0].tick_params(axis="both", labelsize=FS_TICK, pad=1.5, length=2)

# Secondary axis for SoC
a3 = ax[0].twinx()

for (s, w_), g in hr.groupby(["scenario", "rt"]):
    a3.plot(
        g.hour, g.soc_path_mwh,
        color=SC[s], linewidth=0.6, alpha=0.35,
        zorder=1
    )

a3.plot(
    pos.hour, pos.exp_soc_mwh,
    color="#5B2C83", linewidth=1.8,
    label="Expected SoC", zorder=4
)

a3.axhline(20, color="grey", linestyle="--", linewidth=0.65)
a3.axhline(180, color="grey", linestyle="--", linewidth=0.65)
a3.set_ylabel("SoC (MWh)", fontsize=FS_LABEL, labelpad=2)
a3.tick_params(axis="y", labelsize=FS_TICK, pad=1.5, length=2)

# Combined legend placed below the two-line heading
h1, l1 = ax[0].get_legend_handles_labels()
h2, l2 = a3.get_legend_handles_labels()

ax[0].legend(
    h1 + h2, l1 + l2,
    frameon=False,
    ncol=2,
    fontsize=FS_LEGEND,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.99),
    handlelength=1.2,
    handletextpad=0.35,
    columnspacing=0.65,
    labelspacing=0.2,
    borderaxespad=0.15
)

ax[0].set_title(
    f"(a) BESS dispatch and SoC\n"
    f"(9 joint paths; bounds 20–180 MWh; "
    f"throughput cost {K['exp_cost_bess_eur']:,.0f} EUR)",
    fontsize=FS_TITLE,
    pad=4,
    linespacing=1.15
)

# (b) Real-time microturbine dispatch
env = hr.groupby("hour").p_mt_rt_mw.agg(["min", "max"])

ax[1].fill_between(
    env.index, env["min"], env["max"],
    color="#4F9D69", alpha=0.25,
    label="Scenario envelope"
)

ax[1].plot(
    pos.hour, pos.exp_p_mt_mw,
    color="#2E7D4F", linewidth=1.5,
    label="Expected output"
)

ax[1].axhline(
    10, color="#7B2D8E", linestyle="--",
    linewidth=0.8, label="Minimum 10 MW"
)
ax[1].axhline(
    25, color="grey", linestyle=":",
    linewidth=0.8, label="Maximum 25 MW"
)

ax[1].set_xlabel("Hour of day", fontsize=FS_LABEL, labelpad=2)
ax[1].set_ylabel("MT power (MW)", fontsize=FS_LABEL, labelpad=2)
ax[1].tick_params(axis="both", labelsize=FS_TICK, pad=1.5, length=2)

ax[1].legend(
    frameon=False,
    ncol=2,
    fontsize=FS_LEGEND,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    handlelength=1.2,
    handletextpad=0.35,
    columnspacing=0.65,
    labelspacing=0.2,
    borderaxespad=0.1
)

ax[1].set_title(
    "(b) Real-time microturbine dispatch",
    fontsize=FS_TITLE,
    pad=3
)

# Shared x-axis
ax[1].set_xlim(0, 24)
ax[1].set_xticks(range(0, 25, 6))

# Recover horizontal plotting space
fig.subplots_adjust(
    left=0.17,
    right=0.85,
    top=0.96,
    bottom=0.15,
    hspace=0.10
)

save(fig, "fig_real_bess_mt")



# ----------------------------------------------------------------
# Font sizes
FS_TITLE = 9
FS_LABEL = 8
FS_TICK = 6
FS_LEGEND = 5.5

# ---------------------------------------------------------------- 5. Stage 2 recourse
fig, ax = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True)

s2 = hr[hr.rt == 0]

for s in range(3):
    g = s2[s2.scenario == s].sort_values("hour")
    lab = f"day type {s} (p={prob_s[s]:.2f})"

    ax[0, 0].step(
        g.hour, g.p_id_buy_mw - g.p_id_sell_mw,
        where="mid", color=SC[s], label=lab
    )
    ax[0, 1].plot(g.hour, g.p_mt_mw, color=SC[s])
    ax[1, 0].plot(
        g.hour, g.p_bess_dis_mw - g.p_bess_ch_mw,
        color=SC[s]
    )
    ax[1, 1].step(
        g.hour, g.r_total_mw, where="mid", color=SC[s]
    )

titles_s2 = [
    "(a) Net intraday import (+) / export (-)",
    "(b) Microturbine dispatch",
    "(c) BESS net discharge (+) / charge (-)",
    "(d) Total up-reserve offer\n(identical across day types: Stage-1 commitment)"
]

for a_, t_ in zip(ax.ravel(), titles_s2):
    a_.set_title(t_, fontsize=FS_TITLE, pad=4, linespacing=1.1)
    a_.set_ylabel("MW", fontsize=FS_LABEL)
    a_.axhline(0, color="k", lw=0.4)
    a_.tick_params(axis="both", labelsize=FS_TICK)

for a_ in ax[1]:
    a_.set_xlabel("Hour of day", fontsize=FS_LABEL)

ax[0, 0].legend(frameon=False, fontsize=FS_LEGEND)

fig.tight_layout()
save(fig, "fig_real_stage2")


# ---------------------------------------------------------------- 6. Stage 3 + profit distribution
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))

paths = [(s, w_) for s in range(3) for w_ in range(3)]

for (s, w_), g in hr.groupby(["scenario", "rt"]):
    g = g.sort_values("hour")
    ls = ["-", "--", ":"][w_]

    ax[0, 0].plot(
        g.hour, g.imb_deficit_mw - g.imb_surplus_mw,
        color=SC[s], ls=ls, lw=1
    )
    ax[0, 1].plot(
        g.hour, g.rho * g.r_total_mw,
        color=SC[s], ls=ls, lw=1
    )

ax[0, 0].axhline(0, color="k", lw=0.4)
ax[0, 0].set_title(
    "(a) Real-time imbalance,\ndeficit (+) / surplus (-)",
    fontsize=FS_TITLE, pad=4, linespacing=1.1
)
ax[0, 0].set_ylabel("MW", fontsize=FS_LABEL)
ax[0, 0].set_xlabel("Hour of day", fontsize=FS_LABEL)

ax[0, 1].set_title(
    "(b) Activated up-reserve (rho × offer)",
    fontsize=FS_TITLE, pad=4
)
ax[0, 1].set_ylabel("MW", fontsize=FS_LABEL)
ax[0, 1].set_xlabel("Hour of day", fontsize=FS_LABEL)

from matplotlib.lines import Line2D

ax[0, 0].legend(
    [Line2D([0], [0], color=c_) for c_ in SC]
    + [Line2D([0], [0], color="k", ls=l_) for l_ in ["-", "--", ":"]],
    [f"day type {s}" for s in range(3)]
    + [f"RT branch {w_}" for w_ in range(3)],
    frameon=False,
    fontsize=FS_LEGEND,
    ncol=2
)

lab = [f"s{int(r.scenario)}-w{int(r.rt)}" for _, r in jt.iterrows()]

ax[1, 0].bar(
    range(len(jt)), jt.profit_eur / 1e3,
    color=[SC[int(s)] for s in jt.scenario], width=0.7
)

ep = K["exp_profit_eur"] / 1e3
cv = K["cvar_profit_eur"] / 1e3

ax[1, 0].axhline(
    ep, color="#4F9D69", lw=1.6,
    label=f"Expected {ep:,.1f} k"
)
ax[1, 0].axhline(
    cv, color="#C0504D", lw=1.6, ls="--",
    label=f"CVaR95 {cv:,.1f} k"
)

ax[1, 0].set_xticks(range(len(jt)))
ax[1, 0].set_xticklabels(lab, rotation=45, fontsize=FS_TICK)
ax[1, 0].set_ylabel("Profit (kEUR; negative = cost)", fontsize=FS_LABEL)
ax[1, 0].tick_params(axis="y", labelsize=FS_TICK)
ax[1, 0].legend(
    frameon=False, fontsize=FS_LEGEND, loc="lower left"
)
ax[1, 0].set_title(
    "(c) Profit on the 9 joint paths",
    fontsize=FS_TITLE, pad=4
)

cols = ["rev_da_eur", "rev_id_eur", "rev_sr_eur", "rev_act_eur"]
sc = pd.read_csv(os.path.join(T, "scenarios.csv"))

ax[1, 1].bar(sc.scenario, sc.prob, color=SC, width=0.5)

for i, r in sc.iterrows():
    ax[1, 1].text(
        i, r.prob + 0.01,
        f"p={r.prob:.2f}\n"
        f"mean DA {hr[hr.scenario == i].price_da_buy_eur_mwh.mean():.0f} EUR/MWh\n"
        f"mFRR {hr[hr.scenario == i].price_sr_up_eur_mw_h.mean():.1f}",
        ha="center", fontsize=FS_TICK
    )

ax[1, 1].set_ylim(0, sc.prob.max() * 1.55)
ax[1, 1].set_xticks(sc.scenario)
ax[1, 1].set_xticklabels(
    [f"day type {i}" for i in sc.scenario],
    fontsize=FS_TICK
)
ax[1, 1].set_ylabel("Probability", fontsize=FS_LABEL)
ax[1, 1].set_title(
    "(d) Day types from the 90 previous real days",
    fontsize=FS_TITLE, pad=4
)

# Apply consistent tick-label sizes to all Stage 3 axes
for a_ in ax.ravel():
    a_.tick_params(axis="both", labelsize=FS_TICK)

fig.tight_layout()
save(fig, "fig_real_stage3")


print("done")
