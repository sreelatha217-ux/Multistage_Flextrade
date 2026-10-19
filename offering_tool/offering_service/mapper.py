"""Turn input.json into the plant file, the scenario files and the command-line options of the model."""
from __future__ import annotations

import json
from pathlib import Path

from . import db_market
from .errors import InvalidInput
from .validate import ASSETS, horizon

# Fields of the model's Microturbine / BESS dataclasses (factory_mt_da_scheduler.py)
MT_KEYS = {"p_min_mw", "p_max_mw", "ramp_up_mw_h", "ramp_down_mw_h", "startup_ramp_mw_h", "shutdown_ramp_mw_h",
           "startup_cost_eur", "shutdown_cost_eur", "min_up_h", "min_down_h", "block_width_mw", "block_cost_eur_mwh",
           "base_cost_eur_mwh", "initial_on", "initial_power_mw", "initial_hours_in_state"}
BESS_KEYS = {"p_max_mw", "e_max_mwh", "soc_min_frac", "soc_max_frac", "soc_init_mwh", "eta_ch", "eta_dis",
             "degradation_eur_mwh", "degradation_basis", "enforce_exclusive"}


def _only(block: dict, allowed: set[str], field: str) -> dict:
    extra = sorted(set(block) - allowed)
    if extra:
        raise InvalidInput(f"unknown field(s): {', '.join(extra)}", f"{field}.{extra[0]}")
    return dict(block)


def build_instance(d: dict, tariff: dict | None) -> dict:
    p, g = d["plant"], d.get("grid") or {}
    tar = tariff or g["reference_tariff"]
    inst = {
        "machines": p["machines"], "tasks": p["tasks"], "power_mw": p["power_mw"], "duration_h": p["duration_h"],
        "yield_units": p["yield_units"], "base_load_mw": p["base_load_mw"], "demand_units": p["demand_units"],
        "price_buy_eur_mwh": tar["buy_eur_mwh"], "price_sell_eur_mwh": tar["sell_eur_mwh"],
        "horizon_h": horizon(d), "buffer_h": p.get("buffer_h", 1.0),
        "grid_limit_mw": g.get("import_limit_mw", 400.0), "grid_sell_limit_mw": g.get("export_limit_mw", 400.0),
        "max_batches_per_machine": p.get("max_batches_per_machine", 10),
        "inventory_init": p.get("inventory_init", 50.0), "inventory_max": p.get("inventory_max", 500.0),
        "bess": None,
    }
    if "microturbine" in d:                      # missing key -> the model's default MT; null -> no MT
        inst["mt"] = None if d["microturbine"] is None else _only(d["microturbine"], MT_KEYS, "microturbine")
    if d.get("bess") is not None:
        inst["bess"] = _only(d["bess"], BESS_KEYS, "bess")
    return inst


def build_files(d: dict, workdir: Path) -> tuple[list[str], dict]:
    """Write the files into workdir and return (argv for factory_mt_offering_strategy.py, info)."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "results"
    rs, ms = d.get("run_settings") or {}, d.get("market_scenarios") or {}
    rt, rf, g = d.get("realtime_scenarios") or {}, d.get("reserve_flex") or {}, d.get("grid") or {}
    sv = rs.get("solver") or {}
    info: dict = {"warnings": [], "out_dir": out_dir}
    mode = ms.get("mode", "generate")

    tariff = None
    scen_file = None
    if mode == "database":
        scen, tariff, dbinfo = db_market.build_scenarios(ms, horizon(d))
        info["database"] = dbinfo
        info["warnings"] += dbinfo["warnings"]
        if g.get("reference_tariff"):
            tariff = g["reference_tariff"]       # the user's own tariff wins over the average of the scenarios
        scen_file = workdir / "scenario_set.json"
        scen_file.write_text(json.dumps(scen))
    elif mode == "provided":
        scen = {k: ms[k] for k in ("prob", "da_buy", "da_sell", "sr_up", "id_buy", "id_sell", "load_dev_mw")}
        scen_file = workdir / "scenario_set.json"
        scen_file.write_text(json.dumps(scen))

    inst_file = workdir / "plant.json"
    inst_file.write_text(json.dumps(build_instance(d, tariff)))

    a = ["--instance", str(inst_file), "--out", str(out_dir),
         "--start-step", str(rs.get("start_step_h", 0.5)),
         "--solver", str(sv.get("name", "appsi_highs")), "--mip-gap", str(sv.get("mip_gap", 0.01)),
         "--time-limit", str(sv.get("time_limit_s", 300)),
         "--beta", str(rs.get("risk_aversion_beta", 0.0)), "--alpha", str(rs.get("cvar_alpha", 0.95)),
         "--bidding", str(rs.get("bidding", "curve")),
         "--id-reschedule-hour", str(rs.get("id_reschedule_hour", 0.0)),
         "--id-cap-buy", str(g.get("id_cap_buy_mw", 100.0)), "--id-cap-sell", str(g.get("id_cap_sell_mw", 100.0))]
    if rs.get("compare_strategies", True):
        a.append("--compare")

    # reserve
    if not rs.get("include_reserve", True):
        a.append("--no-reserve")
    a += ["--reserve-assets", ",".join(rf.get("assets", ASSETS)),
          "--mt-rr-minutes", str(rf.get("mt_response_minutes", 10.0)),
          "--dr-fraction", str(rf.get("dr_fraction", 0.10)), "--dr-cost", str(rf.get("dr_cost_eur_mwh", 150.0)),
          "--act-ratio", str(rf.get("activation_price_ratio", 1.0)),
          "--bess-reserve-hours", str(rf.get("bess_reserve_hours", 1.0))]
    if rf.get("max_reserve_mw") is not None:
        a += ["--max-reserve", str(rf["max_reserve_mw"])]

    # price scenarios
    if scen_file is not None:
        a += ["--offering-scenario-file", str(scen_file)]
    else:
        a += ["--scenarios", str(ms.get("n_scenarios", 6)), "--scenario-seed", str(ms.get("seed", 7)),
              "--price-sigma", str(ms.get("price_sigma", 0.15)), "--level-sigma", str(ms.get("level_sigma", 0.05)),
              "--id-spread-sigma", str(ms.get("id_spread_sigma", 0.08)), "--sr-ratio", str(ms.get("sr_ratio", 0.12)),
              "--sr-sigma", str(ms.get("sr_sigma", 0.20)), "--load-sigma", str(ms.get("load_sigma", 0.0))]

    # real-time branches and reserve calls
    if rt.get("mode", "generate") == "provided":
        rt_file, dep_file = workdir / "rt_set.json", workdir / "deployment_set.json"
        rt_file.write_text(json.dumps({k: rt[k] for k in ("prob", "load_rel_dev", "r_plus", "r_minus")}))
        dep_file.write_text(json.dumps({"rho": rt["rho"]}))
        a += ["--rt-file", str(rt_file), "--deploy-file", str(dep_file)]
    else:
        a += ["--rt-scenarios", str(rt.get("n_branches", 3)), "--rt-seed", str(rt.get("rt_seed", 11)),
              "--rt-load-sigma", str(rt.get("rt_load_sigma", 0.03)), "--rt-load-rho", str(rt.get("rt_load_rho", 0.6)),
              "--r-premium", str(rt.get("r_premium", 0.25)), "--r-discount", str(rt.get("r_discount", 0.25)),
              "--r-sigma", str(rt.get("r_sigma", 0.3)), "--rho-seed", str(rt.get("rho_seed", 13)),
              "--rho-mean", str(rt.get("rho_mean", 0.08)), "--rho-sigma", str(rt.get("rho_sigma", 0.8))]
    return a, info
