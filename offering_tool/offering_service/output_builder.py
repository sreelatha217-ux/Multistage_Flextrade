"""Assemble output.json from the result folder written by the model (summary.json, bid_package.json, CSV files)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

NAN_TEXT = {"", "nan", "NaN", "None"}


def _conv(s: str):
    if s in NAN_TEXT:
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        f = float(s)
    except ValueError:
        return s
    if not math.isfinite(f):
        return None
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [{k: _conv(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pick(row: dict, keys: list[str]) -> dict:
    return {k: row.get(k) for k in keys}


JOB_KEYS = ["machine", "position_n", "task", "start_h", "end_h", "start_time", "end_time", "duration_h",
            "power_mw", "units_out", "energy_mwh", "energy_cost_eur"]
POSITION_KEYS = ["hour", "exp_price_da_buy_eur_mwh", "exp_price_da_sell_eur_mwh", "exp_price_sr_up_eur_mw_h",
                 "exp_p_da_buy_mw", "exp_p_da_sell_mw", "exp_p_mt_mw", "exp_bess_ch_mw", "exp_bess_dis_mw",
                 "exp_r_total_mw", "exp_r_mt_mw", "exp_r_bess_dis_mw", "exp_r_bess_ch_mw", "exp_r_dr_mw"]
SUMMARY_MAP = {                                  # output name -> summary.json kpis name
    "expected_profit_eur": "exp_profit_eur", "cvar_profit_eur": "cvar_profit_eur",
    "profit_std_eur": "profit_std_eur", "profit_min_eur": "profit_min_eur", "profit_max_eur": "profit_max_eur",
    "revenue_da_eur": "exp_rev_da_eur", "revenue_id_eur": "exp_rev_id_eur", "revenue_reserve_eur": "exp_rev_sr_eur",
    "revenue_activation_eur": "exp_rev_act_eur", "cost_imbalance_eur": "exp_cost_bal_eur",
    "cost_mt_eur": "exp_cost_mt_eur", "cost_bess_eur": "exp_cost_bess_eur", "cost_dr_eur": "exp_cost_dr_eur",
    "n_batches": "n_batches", "units_produced": "units_produced", "mt_on_hours": "mt_on_hours",
    "mt_starts": "mt_starts", "reserve_hours": "reserve_hours", "avg_reserve_offer_mw": "avg_reserve_offer_mw",
    "exp_activated_mwh": "exp_activated_mwh", "exp_deficit_mwh": "exp_deficit_mwh",
    "exp_surplus_mwh": "exp_surplus_mwh"}


def _hourly_offers(bid: dict) -> list[dict]:
    out = []
    for h in sorted(bid["hours"], key=int):
        e = bid["hours"][h]
        out.append({
            "hour": int(h), "mt_online": e.get("mt_online"),
            "energy_offer_sell": e.get("energy_offer_sell", []),
            "energy_bid_buy": e.get("energy_bid_buy", []),
            # the model labels the reserve price price_eur_mwh; the unit is EUR/MW/h
            "reserve_offer_up": [{"price_eur_mw_h": s["price_eur_mwh"], "quantity_mw": s["quantity_mw"]}
                                 for s in e.get("reserve_offer_up", [])]})
    return out


def build_output(result_dir: Path, include_scenario_detail: bool = False, requested_mip_gap: float | None = None,
                 extra_warnings: list[str] | None = None, database_info: dict | None = None) -> dict:
    result_dir = Path(result_dir)
    summ = read_json(result_dir / "summary.json")
    bid = read_json(result_dir / "bid_package.json")
    if summ is None or bid is None:
        raise FileNotFoundError(f"summary.json or bid_package.json missing in {result_dir}")
    kp = summ.get("kpis", {})
    warnings = list(extra_warnings or [])
    gap = summ.get("mip_gap")
    if requested_mip_gap is not None and gap is not None and gap > requested_mip_gap + 1e-12:
        warnings.append(f"achieved optimality gap {gap:.4f} is above the requested {requested_mip_gap}")
    ver = summ.get("verification", {})
    if ver.get("passed") is False:
        warnings.append("the model's own solution check failed; do not use this result")

    position = read_csv(result_dir / "da_position.csv")
    id_summary = read_csv(result_dir / "intraday_reschedule_summary.csv")
    id_jobs = read_csv(result_dir / "intraday_rescheduled_jobs.csv")
    reschedules = []
    for s in id_summary:
        sid = s["id_scenario"]
        reschedules.append({
            "id_scenario": sid, "probability": s.get("probability"),
            "operating_cost_eur": s.get("operating_cost_eur"), "fixed_plan_cost_eur": s.get("fixed_plan_cost_eur"),
            "production_reschedule_savings_eur": s.get("production_reschedule_savings_eur"),
            "jobs_changed": s.get("jobs_changed"), "frozen_jobs": s.get("frozen_jobs"),
            "batches": s.get("batches"), "units_produced": s.get("units_produced"),
            "mip_gap": s.get("mip_gap"), "solve_time_s": s.get("solve_time_s"),
            "verification_passed": s.get("verification_passed"),
            "jobs": [_pick(j, JOB_KEYS) for j in id_jobs if j["id_scenario"] == sid]})

    real_time = {"hourly_expected": [
        {"hour": r["hour"], "exp_imb_deficit_mw": r.get("exp_imb_deficit_mw"),
         "exp_imb_surplus_mw": r.get("exp_imb_surplus_mw"), "exp_activated_mwh": r.get("exp_activated_mwh"),
         "exp_soc_mwh": r.get("exp_soc_mwh"), "exp_profit_eur": r.get("exp_profit_eur")} for r in position]}
    if include_scenario_detail:
        real_time["scenario_detail"] = read_csv(result_dir / "hourly_offering.csv")

    out = {
        "meta": {"status": summ.get("status"), "solve_time_s": summ.get("solve_time_s"), "mip_gap": gap,
                 "model_stats": summ.get("model_stats"),
                 "verification": {"passed": ver.get("passed"), "issues": ver.get("issues", [])},
                 "warnings": warnings},
        "summary": {"risk_adjusted_profit_eur": summ.get("risk_adjusted_profit_eur"),
                    **{k: kp.get(v) for k, v in SUMMARY_MAP.items()}},
        "day_ahead": {
            "production_schedule": [_pick(j, JOB_KEYS) for j in read_csv(result_dir / "jobs.csv")],
            "hourly_offers": _hourly_offers(bid),
            "expected_position": [_pick(r, POSITION_KEYS) for r in position]},
        "intraday": {"reschedules": reschedules,
                     "expected_id_position": [_pick(r, ["hour", "exp_p_id_buy_mw", "exp_p_id_sell_mw"])
                                              for r in position]},
        "real_time": real_time,
        "strategy_comparison": {"strategies": read_csv(result_dir / "comparison.csv"),
                                "market_stages": read_csv(result_dir / "market_stage_comparison.csv")},
        "scenarios_used": {"price": read_json(result_dir / "scenario_set.json"),
                           "real_time": read_json(result_dir / "rt_set.json"),
                           "deployment": read_json(result_dir / "deployment_set.json")},
        "strategy_settings": {k: bid.get(k) for k in ("bidding", "beta", "alpha", "currency")},
    }
    if database_info:
        out["scenarios_used"]["database"] = database_info
        out["meta"]["warnings"] = warnings
    return out
