"""Seasonal analysis of the DK1 real-data study (heat maps + seasonal statistics).

Inputs  (all from Offering_Strategy_Handoff/):
  dk1_hourly_da.csv, mfrr_capacity_dk1.csv,
  real_data_results_gurobi/ladder_mfrr_stage1/<date>/{hourly_offering.csv,market_stage_comparison.csv}
  real_data_results/ladder_mfrr_stage1/2026-10-11/...   (HiGHS reference tree)
Outputs: Figures/seasonal_heatmaps.{pdf,png}, seasonal_stats.csv, seasonal_ladder.csv
Usage:   python seasonal_analysis.py --root <path to Offering_Strategy_Handoff> --out <dir>
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="Offering_Strategy_Handoff")
ap.add_argument("--da", nargs="*", default=None, help="extra/other Day-Ahead CSVs (wide h00..h23); default: <root>/dk1_hourly_da.csv")
ap.add_argument("--mfrr", nargs="*", default=None, help="extra/other mFRR CSVs; default: <root>/mfrr_capacity_dk1.csv")
ap.add_argument("--extra-trees", default=None, help="dir with ladder_mfrr_stage1/<history_end>/ for additional (e.g. winter) trees")
ap.add_argument("--out", default=".")
a = ap.parse_args()
R, O = a.root, a.out
os.makedirs(os.path.join(O, "Figures"), exist_ok=True)

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
ORDER = ["Winter", "Spring", "Summer", "Autumn"]

# ---------------- Day-Ahead (raw, negatives kept for the statistics) ----------------
da_files = a.da or [os.path.join(R, "dk1_hourly_da.csv")]
da = (pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in da_files])
        .drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True))
hc = [f"h{h:02d}" for h in range(24)]
long = da.melt(id_vars="date", value_vars=hc, var_name="hh", value_name="da")
long["hour"] = long.hh.str[1:].astype(int)
long["ym"] = long.date.dt.to_period("M")

# ---------------- mFRR up-capacity (Danish local time) ----------------
mf_files = a.mfrr or [os.path.join(R, "mfrr_capacity_dk1.csv")]
mf = (pd.concat([pd.read_csv(f, parse_dates=["TimeDK"]) for f in mf_files])
        .drop_duplicates("TimeDK", keep="last").sort_values("TimeDK"))
mf = mf.rename(columns={"UpPriceEUR": "mfrr"})
mf["hour"] = mf.TimeDK.dt.hour
mf["ym"] = mf.TimeDK.dt.to_period("M")
mf = mf[(mf.TimeDK >= "2025-10-01") & (mf.TimeDK < "2026-10-13")]

months = pd.period_range("2025-10", "2026-10", freq="M")
lab = [m.strftime("%b %y") for m in months]
lab[-1] += "*"                       # partial month (1-12 Oct)

hm_da = long.pivot_table(index="ym", columns="hour", values="da", aggfunc="mean").reindex(months)
hm_mf = mf.pivot_table(index="ym", columns="hour", values="mfrr", aggfunc="mean").reindex(months)

# ---------------- Optimizer output by seasonal tree ----------------
trees = ["2026-03-31", "2026-04-28", "2026-05-26", "2026-06-23",
         "2026-07-21", "2026-08-18", "2026-09-15", "2026-10-11"]
EXTRA = {}
if a.extra_trees:
    root = os.path.join(a.extra_trees, "ladder_mfrr_stage1")
    for d in sorted(os.listdir(root)):
        EXTRA[d] = os.path.join(root, d)
    trees = sorted(set(trees) | set(EXTRA))
def tpath(d, f):
    if d in EXTRA:
        return os.path.join(EXTRA[d], f)
    base = "real_data_results" if d == "2026-10-11" else "real_data_results_gurobi"
    return os.path.join(R, base, "ladder_mfrr_stage1", d, f)

off, lad = [], []
for d in trees:
    h = pd.read_csv(tpath(d, "hourly_offering.csv"))
    off.append(h.groupby("hour").r_total_mw.mean().reindex(range(24)).values)
    c = pd.read_csv(tpath(d, "market_stage_comparison.csv"))
    l1, l2, l3 = c.expected_net_cost_eur.values[:3]
    s = pd.read_csv(tpath(d, "scenarios.csv"))
    hist_end = pd.Timestamp(d)
    hw = h.copy(); hw["w"] = hw.prob_joint / hw.prob_joint.sum() * 1.0
    sr = (hw.price_sr_up_eur_mw_h * hw.prob_joint).sum() / hw.prob_joint.sum()   # probability-weighted tree mean
    lad.append(dict(history_end=d, L1=l1, L2=l2, L3=l3, id_saving=l1 - l2, res_saving=l2 - l3,
                    total_saving=l1 - l3, total_pct=100 * (l1 - l3) / l1,
                    offer_mw=float(np.mean(off[-1])), mean_sr_price=sr))
off = np.array(off)
lad = pd.DataFrame(lad)
lad["season"] = [SEASON[pd.Timestamp(d).month] for d in lad.history_end]
lad.to_csv(os.path.join(O, "seasonal_ladder.csv"), index=False)

# ---------------- Seasonal statistics ----------------
day = long.assign(season=long.date.dt.month.map(SEASON))
rows = []
for s in ORDER:
    x = day[day.season == s]
    prof = x.groupby("hour").da.mean()
    m = mf[mf.TimeDK.dt.month.map(SEASON) == s]
    rows.append(dict(season=s, days=x.date.nunique(), da_mean=x.da.mean(), da_std=x.da.std(),
                     evening_peak=prof.loc[17:20].mean(), midday=prof.loc[11:14].mean(),
                     spread=prof.loc[17:20].mean() - prof.loc[11:14].mean(),
                     neg_hours=int((x.da < 0).sum()),
                     mfrr_mean=m.mfrr.mean(), mfrr_median=m.mfrr.median(),
                     mfrr_p95=m.mfrr.quantile(.95)))
st = pd.DataFrame(rows); st.to_csv(os.path.join(O, "seasonal_stats.csv"), index=False)
print(st.round(2).to_string(index=False)); print(lad.round(2).to_string(index=False))
dayavg = long.groupby("date").da.mean()
print("coef var of monthly means:", long.groupby("ym").da.mean().round(1).to_dict())

SCOL = {"Winter": "#3B6FB6", "Spring": "#3C9A3A", "Summer": "#E0911A", "Autumn": "#B5552B"}
def colour_ticks(ax, seasons):
    for t, se in zip(ax.get_yticklabels(), seasons):
        t.set_color(SCOL[se]); t.set_fontweight("bold")

# ---------------- Figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titleweight": "bold",
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5})
fig = plt.figure(figsize=(12.6, 4.4 + 0.23 * max(len(months), len(trees)) * 2), dpi=200)
gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26, left=0.075, right=0.955, top=0.93, bottom=0.075)
hrs = np.arange(24)

def heat(ax, M, cmap, norm, ylabels, title, cbar_label, annotate_fmt=None, fs=6.2):
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(0, 24, 2)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True); ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Hour of day (CET/CEST)")
    cb = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label(cbar_label); cb.outline.set_visible(False); cb.ax.tick_params(length=0)
    if annotate_fmt:
        rgba = im.cmap(im.norm(np.ma.masked_invalid(M)))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    r, g, b, _ = rgba[i, j]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, annotate_fmt.format(v), ha="center", va="center", fontsize=fs,
                            color="white" if lum < 0.5 else "#1a1a1a")
    return im

# season separators on month heat maps
def season_lines(ax):
    prev = None
    for i, m in enumerate(months):
        s = SEASON[m.month]
        if prev is not None and s != prev:
            ax.axhline(i - .5, color="#222", lw=1.4)
        prev = s

# (a) DA
axA = fig.add_subplot(gs[0, 0])
heat(axA, hm_da.values, "magma_r", Normalize(0, np.nanpercentile(hm_da.values, 99)), lab,
     "(a) Day-Ahead price, mean by month and hour", "EUR/MWh", "{:.0f}")
season_lines(axA); colour_ticks(axA, [SEASON[m.month] for m in months])
# (b) mFRR
axB = fig.add_subplot(gs[0, 1])
heat(axB, hm_mf.values, "viridis", LogNorm(vmin=max(1, np.nanmin(hm_mf.values)), vmax=np.nanmax(hm_mf.values)),
     lab, "(b) mFRR up-capacity price, mean by month and hour (log scale)", "EUR/MW/h", "{:.0f}")
season_lines(axB); colour_ticks(axB, [SEASON[m.month] for m in months])
# (c) reserve offer
axC = fig.add_subplot(gs[1, 0])
tl = [pd.Timestamp(d).strftime("%d %b %Y") + f"  ({SEASON[pd.Timestamp(d).month][:3]})" for d in trees]
tl = [t.replace("(Win)", "(Win)") for t in tl]
heat(axC, off, "cividis", Normalize(0, np.nanmax(off)), tl,
     "(c) Stage-1 reserve offer by hour, one tree per 4 weeks", "MW", "{:.0f}")
axC.set_ylabel("Tree history end"); colour_ticks(axC, [SEASON[pd.Timestamp(d).month] for d in trees])

# (d) savings decomposition
axD = fig.add_subplot(gs[1, 1])
x = np.arange(len(trees)); w = 0.62
c_id, c_res = "#4C78A8", "#F58518"
axD.bar(x, lad.id_saving / 1e3, w, color=c_id, label="Intraday saving", zorder=3)
axD.bar(x, lad.res_saving / 1e3, w, bottom=lad.id_saving / 1e3, color=c_res, label="Up-reserve saving", zorder=3)
seas_t = [SEASON[pd.Timestamp(d).month] for d in trees]
i0 = 0
for i in range(1, len(trees) + 1):
    if i == len(trees) or seas_t[i] != seas_t[i0]:
        axD.axvspan(i0 - .5, i - .5, color=SCOL[seas_t[i0]], alpha=.08, zorder=0)
        axD.text((i0 + i - 1) / 2, 0.985, seas_t[i0], transform=axD.get_xaxis_transform(), ha="center", va="top",
                 fontsize=7.5, color=SCOL[seas_t[i0]], fontweight="bold")
        i0 = i
axD.set_xticks(x); axD.set_xticklabels([pd.Timestamp(d).strftime("%b\n%d") for d in trees])
axD.set_ylabel("Expected saving vs Day-Ahead only (kEUR)")
axD.set_ylim(0, (lad.total_saving.max() / 1e3) * 1.28)
axD.grid(axis="y", color="#e5e5e5", zorder=0); axD.spines[["top", "right"]].set_visible(False)
ax2 = axD.twinx()
ax2.plot(x, lad.mean_sr_price, "-o", color="#2F2F2F", lw=1.6, ms=4.5, label="90-day mean mFRR price", zorder=4)
ax2.set_ylabel("Tree mean mFRR price (EUR/MW/h)"); ax2.spines[["top"]].set_visible(False)
ax2.set_ylim(0, lad.mean_sr_price.max() * 1.18)
ytop = axD.get_ylim()[1]; y2 = ax2.get_ylim()[1]
for i, r in lad.iterrows():
    pos = max(r.total_saving / 1e3, r.mean_sr_price / y2 * ytop)
    axD.text(i, pos + 0.45, f"{r.total_pct:.1f}%", ha="center", fontsize=7.5, fontweight="bold", zorder=6)
h1, l1 = axD.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axD.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=7.5)
axD.set_title("(d) Expected saving by tree (label: total, % of Level 1)", loc="left", pad=7)

fig.text(0.075, 0.004, "Rows in (a),(b): calendar months; black lines separate the seasons; * = partial month. "
         "Raw prices incl. negative hours (floored at zero only inside the optimizer).", fontsize=7, color="#555")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(O, "Figures", f"seasonal_heatmaps.{ext}"), bbox_inches="tight")
print("saved")
