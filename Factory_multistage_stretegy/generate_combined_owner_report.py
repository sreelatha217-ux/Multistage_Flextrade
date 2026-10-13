"""Build a combined PDF from the saved offering and ID-rescheduling run."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "offering_owner_report_results"

FIGURES = [
    ("factory_owner_dashboard.png", "Owner dashboard",
     "Four-part overview of market participation costs, the operating-cost bridge, "
     "standalone rescheduling value, and production volumes."),
    ("offering.png", "Offering overview",
     "Compact view of the Stage 1 production plan, DA price-quantity curves, reserve "
     "offers by asset, and scenario profit distribution."),
    ("schedule_rt.png", "Stage 1 production schedule",
     "Shared production schedule before uncertainty is revealed. Bars show jobs by "
     "machine, start time, and duration."),
    ("stage1_here_and_now.png", "Stage 1: here-and-now decisions",
     "Shared production, day-ahead microturbine commitment, and selected DA offers."),
    ("market_positions_rt.png", "Expected market positions",
     "Expected net DA, ID, and real-time positions. Positive values indicate net "
     "import/deficit; negative values indicate net export/surplus."),
    ("market_prices_rt.png", "Market settlement prices",
     "Expected DA, ID, and RT prices by hour, with scenario ranges for DA-buy and "
     "ID-buy prices."),
    ("bess_dispatch_rt.png", "BESS dispatch and state of charge",
     "Expected charge/discharge and joint-scenario state of charge. The run charges "
     "€5/MWh on throughput and reports €747.45 expected degradation cost."),
    ("microturbine_dispatch_rt.png", "Microturbine dispatch",
     "Expected RT output and scenario range, alongside the shared DA on/off "
     "commitment."),
    ("realtime_imbalance.png", "Real-time imbalance",
     "Scenario and expected deficit-minus-surplus. Values below zero represent "
     "surplus, not a negative physical quantity."),
    ("da_offer_curves.png", "Day-ahead offer curves",
     "Net DA export is sell quantity minus buy quantity. Negative values therefore "
     "mean net import; individual buy/sell quantities remain nonnegative."),
    ("reserve_offers.png", "Upward-reserve offers",
     "Expected reserve capacity by asset, with expected reserve price on a second "
     "axis. Offers are capacity and need not equal activated energy."),
    ("profit_distribution.png", "Scenario profit distribution",
     "Probability-weighted modeled profit with expected-profit and CVaR markers. "
     "The objective excludes product-sales revenue and other non-energy business costs."),
    ("stage2_wait_and_see.png", "Stage 2: price-scenario recourse",
     "Intraday net trade, MT dispatch, BESS net dispatch, and reserve offer for each "
     "price scenario after that scenario is revealed."),
    ("stage3_wait_and_see.png", "Stage 3: real-time recourse",
     "Joint-path imbalance, reserve activation, and BESS state of charge after "
     "real-time uncertainty is revealed."),
    ("stage2_production_reschedules.png", "Standalone production reschedules",
     "Stage 1 schedule followed by a separate Stage 2 production plan for each ID "
     "scenario. This is not integrated into the offering-profit optimization."),
    ("intraday_prices_and_load.png", "ID prices and implied production load",
     "Intraday price scenarios and resulting Stage 2 production load, compared with "
     "the Stage 1 baseline."),
    ("intraday_reschedule_metrics.png", "Matched rescheduling cost comparison",
     "Scenario-by-scenario fixed-plan and rescheduled costs and their difference. "
     "Scenario 1 produces one fewer unit, so its cost reduction is not a strict "
     "same-output comparison."),
]


def eur(value: float) -> str:
    return f"€{value:,.2f}"


def add_footer(fig, page: int, total: int) -> None:
    fig.text(0.5, 0.025, f"Seed-2024 owner run | {page} / {total}",
             ha="center", fontsize=8, color="0.4")


def text_page(pdf: PdfPages, page: int, total: int, title: str,
              paragraphs: list[str], table: tuple[list[str], list[list[str]]] | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.075, 0.94, title, fontsize=19, weight="bold", va="top")
    y = 0.885
    for paragraph in paragraphs:
        if paragraph.startswith("## "):
            fig.text(0.075, y, paragraph[3:], fontsize=12, weight="bold", va="top")
            y -= 0.034
        else:
            lines = textwrap.wrap(paragraph, width=104, break_long_words=False)
            block = "\n".join(lines)
            fig.text(0.075, y, block, fontsize=9.5, va="top", linespacing=1.35)
            y -= 0.024 * len(lines) + 0.018
    if table:
        headers, rows = table
        table_height = min(0.32, 0.055 + 0.035 * len(rows))
        ax = fig.add_axes([0.075, max(0.08, y - table_height - 0.015), 0.85, table_height])
        ax.axis("off")
        rendered = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8.1)
        rendered.scale(1, 1.35)
        for (row, _), cell in rendered.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#24476b")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")
    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf: PdfPages, page: int, total: int, image_path: Path,
               title: str, caption: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold", va="top")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        image_ax = fig.add_axes([0.04, 0.12, 0.92, 0.78])
        image_ax.imshow(rgb)
        image_ax.set_axis_off()
        image_ax.set_aspect("auto")
    fig.text(0.05, 0.075, "\n".join(textwrap.wrap(caption, width=155)),
             fontsize=8.5, va="top")
    add_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(output: Path) -> Path:
    summary = json.loads((RESULTS / "summary.json").read_text())
    market = pd.read_csv(RESULTS / "market_stage_comparison.csv")
    reschedule = pd.read_csv(RESULTS / "intraday_reschedule_summary.csv")
    weights = reschedule.probability.to_numpy(dtype=float)
    savings = reschedule.production_reschedule_savings_eur.to_numpy(dtype=float)
    weighted_reschedule_saving = float(np.dot(weights, savings))
    weighted_fixed_cost = float(np.dot(weights, reschedule.fixed_plan_cost_eur))
    weighted_recourse_cost = float(np.dot(weights, reschedule.operating_cost_eur))
    weighted_recourse_units = float(np.dot(weights, reschedule.units_produced))
    market_saving = float(market.iloc[0].expected_net_cost_eur - market.iloc[-1].expected_net_cost_eur)
    market_da_id = float(market.iloc[0].expected_net_cost_eur - market.iloc[1].expected_net_cost_eur)
    market_id_reserve = float(market.iloc[1].expected_net_cost_eur - market.iloc[-1].expected_net_cost_eur)
    if not np.isclose(weighted_reschedule_saving, weighted_fixed_cost - weighted_recourse_cost):
        raise ValueError("Reschedule totals do not reconcile with scenario-weighted savings")
    for filename, _, _ in FIGURES:
        if not (RESULTS / filename).is_file():
            raise FileNotFoundError(f"Required owner-run figure is missing: {RESULTS / filename}")

    text_pages: list[tuple[str, list[str], tuple[list[str], list[list[str]]] | None]] = [
        (
            "Executive summary",
            [
                "This report brings together two related but distinct analyses from the "
                "saved seed-2024 owner run: (1) the multistage market-offering optimization "
                "and (2) a standalone production rescheduling comparison by ID-price scenario.",
                f"The offering model's DA + ID + up-reserve case has expected modeled net "
                f"cost {eur(float(market.iloc[-1].expected_net_cost_eur))}, compared with "
                f"{eur(float(market.iloc[0].expected_net_cost_eur))} for DA-only plus RT "
                f"imbalance settlement. That is a modeled market-participation difference "
                f"of {eur(market_saving)} ({market_saving / float(market.iloc[0].expected_net_cost_eur):.1%}).",
                f"Separately, the production rescheduler estimates probability-weighted "
                f"cost reduction of {eur(weighted_reschedule_saving)} versus fixing the "
                f"Stage 1 jobs, under scenario-matched ID prices and the same MT commitment. "
                f"This is about {weighted_reschedule_saving / weighted_fixed_cost:.2%} of "
                f"the standalone fixed-plan comparison cost.",
                "Do not add these two savings figures. The rescheduling result is a separate "
                "counterfactual, not a benefit already optimized inside the offering model. "
                "The combined report is an analysis of both, not a combined integrated-solver result.",
                "All amounts are modeled energy-market/operating costs, not factory profit "
                "or ROI: product-sale revenue and other non-energy business costs are excluded.",
            ],
            (
                ["Indicator", "Saved run result"],
                [
                    ["Market case cost (DA + ID + up-reserve)", eur(float(market.iloc[-1].expected_net_cost_eur))],
                    ["DA-only market-case cost", eur(float(market.iloc[0].expected_net_cost_eur))],
                    ["Market-case difference", eur(market_saving)],
                    ["Separate weighted reschedule difference", eur(weighted_reschedule_saving)],
                    ["BESS degradation rate / expected cost", "€5/MWh throughput / " + eur(float(summary["kpis"]["exp_cost_bess_eur"]))],
                ],
            ),
        ),
        (
            "How to interpret the two savings results",
            [
                "The offering optimization chooses production, DA bids/offers, ID positions, "
                "reserve offers, dispatch, and settlements within its model. It reports "
                "expected market/operating cost under the modeled scenario tree.",
                "The separate rescheduler takes the Stage 1 job schedule and MT commitment, "
                "then re-solves production and operating dispatch for each ID scenario. It "
                "compares the resulting cost with a fixed-job-plan solve using that same "
                "scenario's ID prices and commitment. The rescheduler does not feed those "
                "schedules back into DA offers, the offering objective, or RT offering results.",
                f"The market comparison moves from DA-only to DA+ID by {eur(market_da_id)}, "
                f"then from DA+ID to DA+ID+reserve by {eur(market_id_reserve)}. The complete "
                f"market-ladder difference is {eur(market_saving)}. The separate reschedule "
                f"comparison is {eur(weighted_reschedule_saving)}. They use different "
                "counterfactuals and scopes, so their arithmetic sum is not a validated "
                "combined saving.",
                "An integrated value would require a single re-optimization in which "
                "scenario-contingent production schedules affect market positions and all "
                "settlements, with consistent output/service constraints. No such integrated "
                "result is claimed in this report.",
            ],
            (
                ["Comparison", "Baseline", "Alternative", "Difference"],
                [
                    ["Market ladder: DA-only to full offering",
                     eur(float(market.iloc[0].expected_net_cost_eur)),
                     eur(float(market.iloc[-1].expected_net_cost_eur)), eur(market_saving)],
                    ["Standalone ID rescheduler: weighted fixed jobs to recourse",
                     eur(weighted_fixed_cost), eur(weighted_recourse_cost), eur(weighted_reschedule_saving)],
                ],
            ),
        ),
        (
            "Market offering results",
            [
                "The same 3 ID-price × 3 RT scenario tree is used in the three market-stage "
                "cases; RT imbalance settlement remains active in every case. Adding ID "
                "trading reduces the baseline modeled cost modestly. Adding the modeled "
                "up-reserve option accounts for the larger incremental reduction in this run.",
                "The full case records €16,602.20 combined reserve capacity and activation "
                "revenue, while its incremental cost reduction versus DA+ID is smaller. "
                "Gross reserve revenue is not equal to net value: the optimization also "
                "changes energy positions and operating decisions.",
                "Full-case expected cost drivers are DA net settlement €83,067.40, ID net "
                "settlement €29,439.00, reserve capacity revenue −€12,071.82, reserve "
                "activation revenue −€4,530.38, imbalance €3,860.31, MT cost €28,408.53, "
                "BESS degradation €747.45, and demand-response cost €613.16.",
                "The model terminated optimally with a reported MIP gap of 0.986% and "
                "independent verification passed. The scenario set is small, so these results "
                "are scenario-study outputs, not a robust statistical forecast.",
            ],
            (
                ["Market participation case", "Expected net cost", "Saving vs. DA-only", "MIP gap"],
                [[str(row.market_stage), eur(float(row.expected_net_cost_eur)),
                  eur(float(row.savings_vs_da_only_eur)), f"{float(row.mip_gap):.3%}"]
                 for row in market.itertuples()],
            ),
        ),
        (
            "Standalone ID production rescheduling results",
            [
                "All three ID scenarios have probability 1/3. Scenario 0 reduces the "
                "matched fixed-plan cost by €6,069.81; scenario 1 reduces it by €932.30; "
                "scenario 2 retains the fixed plan and has zero saving. Their probability-"
                "weighted result is €2,334.04.",
                "The scenario-0 and scenario-1 schedules each change 13 jobs. Scenario 2 "
                "changes none because the fixed Stage 1 plan is retained as the least-cost "
                "incumbent. All reported reschedule solves passed their recorded verification.",
                "Important output caveat: Stage 1 and scenarios 0 and 2 produce 193 units, "
                "but scenario 1 produces 192. Its €932.30 lower cost therefore is not a "
                "strict same-output comparison; the one-unit output difference may contribute "
                "to the cost reduction. Treat the weighted €2,334 as a scheduling-study "
                "estimate, not a pure timing-only saving. Re-run with equal production "
                "requirements before attributing all cost difference to flexibility.",
                "At cutoff hour 0, no jobs have started and none are frozen. At a later "
                "cutoff, already-started jobs are fixed. This run's reschedule MIP gaps are "
                "about 0.97%, 0.93%, and 0% for scenarios 0, 1, and 2.",
            ],
            (
                ["ID scenario", "Probability", "Fixed-plan cost", "Rescheduled cost",
                 "Cost saved", "Jobs changed", "Units"],
                [[str(int(row.id_scenario)), f"{row.probability:.1%}",
                  eur(float(row.fixed_plan_cost_eur)), eur(float(row.operating_cost_eur)),
                  eur(float(row.production_reschedule_savings_eur)), str(int(row.jobs_changed)),
                  f"{row.units_produced:.0f}"]
                 for row in reschedule.itertuples()]
                + [["Weighted / expected", "100.0%", eur(weighted_fixed_cost),
                    eur(weighted_recourse_cost), eur(weighted_reschedule_saving), "—",
                    f"{weighted_recourse_units:.1f} (expected)"]],
            ),
        ),
        (
            "BESS, risk, and decision-use caveats",
            [
                "BESS degradation was explicitly set to €5/MWh on a throughput basis. "
                "The model charges throughput for scheduled charging after activation "
                "curtailment and for scheduled plus reserve-activated discharge. The saved "
                f"expected degradation cost is {eur(float(summary['kpis']['exp_cost_bess_eur']))}; "
                "this is not the full cost of buying energy to charge the battery.",
                "The run contains nine joint paths. beta=0 means CVaR is reported but does "
                "not influence this run's objective; with only nine paths, the 95% tail "
                "estimate is especially sample-sensitive.",
                "The modeled reserve is generic upward reserve, not proof of a qualified "
                "market product or a compliant Danish bid. Validate current market rules, "
                "technical qualification, gate closure, and settlement details before "
                "operational use.",
                "For a stronger rescheduling business case: enforce equal required output "
                "and service levels in fixed and recourse cases; use more realistic ID "
                "execution constraints; increase scenario coverage; then integrate production "
                "recourse with the offering and settlement model before reporting a combined "
                "value.",
            ],
            (
                ["Run setting", "Value"],
                [
                    ["BESS degradation", "€5/MWh, throughput basis"],
                    ["Expected BESS degradation cost", eur(float(summary["kpis"]["exp_cost_bess_eur"]))],
                    ["Expected reserve offer / active hours",
                     f"{summary['kpis']['avg_reserve_offer_mw']:.2f} MW / "
                     f"{int(summary['kpis']['reserve_hours'])} hours"],
                    ["Expected activated reserve", f"{summary['kpis']['exp_activated_mwh']:.2f} MWh"],
                    ["Scenario tree / CVaR weight", "3 ID × 3 RT = 9 joint paths / beta = 0"],
                    ["Offering verification", "Passed"],
                ],
            ),
        ),
    ]

    total_pages = len(text_pages) + len(FIGURES)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        pdf.infodict()["Title"] = "Combined Offering and Intraday Rescheduling Analysis"
        pdf.infodict()["Author"] = "Factory Multistage Offering Strategy"
        for page, (title, paragraphs, table) in enumerate(text_pages, start=1):
            text_page(pdf, page, total_pages, title, paragraphs, table)
        for index, (filename, title, caption) in enumerate(FIGURES, start=len(text_pages) + 1):
            image_page(pdf, index, total_pages, RESULTS / filename, title, caption)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "combined_offering_reschedule_analysis.pdf",
        help="PDF destination (default: offering_owner_report_results/combined_offering_reschedule_analysis.pdf)",
    )
    args = parser.parse_args()
    result = build_pdf(args.output.resolve())
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
