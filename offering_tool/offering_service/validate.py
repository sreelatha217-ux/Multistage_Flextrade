"""Input checks that run before the model, so the UI gets the name of the bad field.

The model repeats most of these checks itself (it raises InstanceValidationError). Checking here first gives
`meta.error.field` and avoids a solver start-up for obviously wrong input.
Returns a list of warning texts; raises InvalidInput on the first error.
"""
from __future__ import annotations

import math
from typing import Any

from .errors import InvalidInput

MODES_MARKET = ("generate", "provided", "database")
MODES_RT = ("generate", "provided")
ASSETS = ("mt", "bess_dis", "bess_ch", "dr")      # same as factory_mt_offering_strategy.ASSETS
EPS = 1e-9


def _bad(field: str, msg: str):
    raise InvalidInput(msg, field)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec(d: dict, key: str, n: int, field: str, lo: float | None = 0.0) -> list[float]:
    v = d.get(key)
    if not isinstance(v, list) or len(v) != n or not all(_is_num(x) for x in v):
        _bad(f"{field}.{key}", f"expected a list of {n} numbers")
    if lo is not None and min(v) < lo - EPS:
        _bad(f"{field}.{key}", f"values must be >= {lo}")
    return v


def _shape(a: Any, dims: tuple[int, ...], field: str) -> None:
    """Check a nested list has exactly the given shape and only finite numbers."""
    def rec(x, level):
        if level == len(dims):
            if not _is_num(x):
                _bad(field, "contains a value that is not a finite number")
            return
        if not isinstance(x, list) or len(x) != dims[level]:
            _bad(field, f"expected shape {list(dims)}")
        for y in x:
            rec(y, level + 1)
    rec(a, 0)


def _num(d: dict, key: str, field: str, lo=None, hi=None, lo_open=False, hi_open=False, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    if not _is_num(v):
        _bad(f"{field}.{key}", "must be a number")
    if lo is not None and (v < lo or (lo_open and v <= lo)):
        _bad(f"{field}.{key}", f"must be {'>' if lo_open else '>='} {lo}")
    if hi is not None and (v > hi or (hi_open and v >= hi)):
        _bad(f"{field}.{key}", f"must be {'<' if hi_open else '<='} {hi}")
    return v


def horizon(d: dict) -> int:
    return int((d.get("run_settings") or {}).get("horizon_h", 24))


def validate_input(d: dict) -> list[str]:
    warnings: list[str] = []
    if not isinstance(d, dict):
        _bad("input", "input.json must contain a JSON object")
    for sec in ("plant",):
        if not isinstance(d.get(sec), dict):
            _bad(sec, "section is required")

    rs = d.get("run_settings") or {}
    T = horizon(d)
    if T <= 0:
        _bad("run_settings.horizon_h", "must be positive")
    step = _num(rs, "start_step_h", "run_settings", 0, None, lo_open=True, default=0.5)
    r = T / step
    if abs(r - round(r)) > 1e-9:
        _bad("run_settings.horizon_h", "horizon_h must be a whole multiple of start_step_h")
    if rs.get("bidding", "curve") not in ("curve", "fixed"):
        _bad("run_settings.bidding", "must be 'curve' or 'fixed'")
    _num(rs, "risk_aversion_beta", "run_settings", 0, 1, default=0.0)
    _num(rs, "cvar_alpha", "run_settings", 0, 1, lo_open=True, hi_open=True, default=0.95)
    _num(rs, "id_reschedule_hour", "run_settings", 0, T, default=0.0)
    sv = rs.get("solver") or {}
    _num(sv, "mip_gap", "run_settings.solver", 0, 1, hi_open=True, default=0.01)
    _num(sv, "time_limit_s", "run_settings.solver", 0, None, lo_open=True, default=300)

    # ---- plant
    p = d["plant"]
    M, P = len(p.get("machines") or []), len(p.get("tasks") or [])
    if M == 0 or P == 0:
        _bad("plant", "machines and tasks must not be empty")
    if len(set(p["machines"])) != M:
        _bad("plant.machines", "machine names must be unique")
    if len(set(p["tasks"])) != P:
        _bad("plant.tasks", "task names must be unique")
    _shape(p.get("power_mw"), (M, P), "plant.power_mw")
    _shape(p.get("duration_h"), (M, P), "plant.duration_h")
    if min(min(row) for row in p["power_mw"]) < 0:
        _bad("plant.power_mw", "values must be >= 0")
    dur = p["duration_h"]
    if min(min(row) for row in dur) <= 0:
        _bad("plant.duration_h", "values must be > 0")
    if max(max(row) for row in dur) > T:
        _bad("plant.duration_h", "a task duration exceeds the horizon")
    y = _vec(p, "yield_units", P, "plant")
    base = _vec(p, "base_load_mw", T, "plant")
    dem = _vec(p, "demand_units", T, "plant")
    inv0 = _num(p, "inventory_init", "plant", 0, default=50.0)
    invmax = _num(p, "inventory_max", "plant", 0, default=500.0)
    if inv0 > invmax:
        _bad("plant.inventory_init", "must not exceed inventory_max")
    _num(p, "buffer_h", "plant", 0, default=1.0)
    nb = p.get("max_batches_per_machine", 10)
    if not isinstance(nb, int) or isinstance(nb, bool) or nb < 1:
        _bad("plant.max_batches_per_machine", "must be an integer >= 1")
    if sum(dem) > sum(y) + inv0 + EPS:
        _bad("plant.demand_units", f"daily demand {sum(dem):.0f} exceeds starting inventory plus the yield of all "
                                   f"tasks ({inv0 + sum(y):.0f}); no schedule can meet it")

    # ---- grid
    g = d.get("grid") or {}
    imp = _num(g, "import_limit_mw", "grid", 0, lo_open=True, default=400.0)
    _num(g, "export_limit_mw", "grid", 0, lo_open=True, default=400.0)
    _num(g, "id_cap_buy_mw", "grid", 0, default=100.0)
    _num(g, "id_cap_sell_mw", "grid", 0, default=100.0)

    mode = (d.get("market_scenarios") or {}).get("mode", "generate")
    if mode not in MODES_MARKET:
        _bad("market_scenarios.mode", f"must be one of {MODES_MARKET}")
    tar = g.get("reference_tariff")
    if tar is None:
        if mode != "database":
            _bad("grid.reference_tariff", "required unless market_scenarios.mode is 'database'")
    else:
        buy = _vec(tar, "buy_eur_mwh", T, "grid.reference_tariff")
        sell = _vec(tar, "sell_eur_mwh", T, "grid.reference_tariff")
        if any(s > b + EPS for s, b in zip(sell, buy)):
            _bad("grid.reference_tariff.sell_eur_mwh", "sell price must not exceed the buy price in any hour")

    # ---- microturbine
    mt = d.get("microturbine", {})
    mt_cap = 0.0
    if mt is not None:
        lo = _num(mt, "p_min_mw", "microturbine", 0, default=10.0)
        hi = _num(mt, "p_max_mw", "microturbine", 0, lo_open=True, default=25.0)
        if lo > hi:
            _bad("microturbine.p_min_mw", "must not exceed p_max_mw")
        mt_cap = hi
        w = mt.get("block_width_mw", [5.3, 4.7, 5.0])
        c = mt.get("block_cost_eur_mwh", [48.41, 48.78, 51.84])
        if not (isinstance(w, list) and isinstance(c, list) and w and len(w) == len(c)
                and all(_is_num(x) for x in w + c)):
            _bad("microturbine.block_width_mw", "block_width_mw and block_cost_eur_mwh must be equal-length lists")
        if min(w) <= 0 or min(c) < 0:
            _bad("microturbine.block_width_mw", "widths must be > 0 and costs >= 0")
        if any(c[i + 1] < c[i] - EPS for i in range(len(c) - 1)):
            _bad("microturbine.block_cost_eur_mwh", "marginal costs must not decrease from one block to the next")
        if abs(sum(w) - (hi - lo)) > 1e-6:
            warnings.append("microturbine.block_width_mw does not add up to p_max_mw - p_min_mw")
    if max(base) > imp + mt_cap + EPS:
        _bad("plant.base_load_mw", "base load alone exceeds the grid import limit plus the MT capacity")

    # ---- bess
    b = d.get("bess")
    if b is not None:
        pm = _num(b, "p_max_mw", "bess", 0, lo_open=True, default=40.0)
        em = _num(b, "e_max_mwh", "bess", 0, lo_open=True, default=200.0)
        lo = _num(b, "soc_min_frac", "bess", 0, 1, default=0.10)
        hi = _num(b, "soc_max_frac", "bess", 0, 1, default=0.90)
        if not lo < hi:
            _bad("bess.soc_min_frac", "must be below soc_max_frac")
        s0 = _num(b, "soc_init_mwh", "bess", 0, default=20.0)
        if not (lo * em - 1e-9 <= s0 <= hi * em + 1e-9):
            _bad("bess.soc_init_mwh", f"must lie between {lo * em:.1f} and {hi * em:.1f} MWh")
        for k in ("eta_ch", "eta_dis"):
            _num(b, k, "bess", 0, 1, lo_open=True, default=0.9)
        _num(b, "degradation_eur_mwh", "bess", 0, default=35.0)
        if b.get("degradation_basis", "throughput") not in ("throughput", "discharge"):
            _bad("bess.degradation_basis", "must be 'throughput' or 'discharge'")
        del pm

    # ---- reserve
    rf = d.get("reserve_flex") or {}
    for a in rf.get("assets", ASSETS):
        if a not in ASSETS:
            _bad("reserve_flex.assets", f"unknown asset '{a}', allowed: {list(ASSETS)}")
    for k, lo_ in (("mt_response_minutes", 0), ("dr_fraction", 0), ("dr_cost_eur_mwh", 0),
                   ("activation_price_ratio", 0), ("bess_reserve_hours", 0)):
        _num(rf, k, "reserve_flex", lo_)
    _num(rf, "dr_fraction", "reserve_flex", 0, 1)

    # ---- market scenarios
    ms = d.get("market_scenarios") or {}
    S = None
    if mode == "provided":
        S = len(ms.get("prob") or [])
        if S == 0:
            _bad("market_scenarios.prob", "at least one scenario is required")
        _shape(ms["prob"], (S,), "market_scenarios.prob")
        if min(ms["prob"]) < 0 or abs(sum(ms["prob"]) - 1.0) > 1e-9:
            _bad("market_scenarios.prob", "probabilities must be >= 0 and sum to 1")
        for k in ("da_buy", "da_sell", "sr_up", "id_buy", "id_sell", "load_dev_mw"):
            _shape(ms.get(k), (S, T), f"market_scenarios.{k}")
        for k in ("da_buy", "da_sell", "sr_up", "id_buy", "id_sell"):
            if min(min(r_) for r_ in ms[k]) < 0:
                _bad(f"market_scenarios.{k}", "negative prices are not supported by the model; set negative "
                                              "prices to 0 before sending")
        for a_, b_ in (("da_sell", "da_buy"), ("id_sell", "id_buy")):
            for rs_, rb_ in zip(ms[a_], ms[b_]):
                if any(s > bb + EPS for s, bb in zip(rs_, rb_)):
                    _bad(f"market_scenarios.{a_}", f"{a_} must not exceed {b_}")
    elif mode == "generate":
        S = int(ms.get("n_scenarios", 6))
        if S < 1:
            _bad("market_scenarios.n_scenarios", "must be >= 1")
        for k in ("price_sigma", "level_sigma", "id_spread_sigma", "sr_ratio", "sr_sigma", "load_sigma"):
            _num(ms, k, "market_scenarios", 0)
    else:  # database
        if not isinstance(ms.get("area"), str) or not ms["area"]:
            _bad("market_scenarios.area", "required, for example 'DK1'")
        days = ms.get("days")
        if not (isinstance(days, list) and days) and not (ms.get("from") and ms.get("to")):
            _bad("market_scenarios.days", "give a list of dates, or 'from' and 'to'")
        for k in ("buy_adder_eur_mwh", "sell_deduct_eur_mwh"):
            if k not in ms:
                _bad(f"market_scenarios.{k}", "required in database mode (0 or more)")
            _num(ms, k, "market_scenarios", 0)
        _num(ms, "id_spread_sigma", "market_scenarios", 0)

    # ---- real time
    rt = d.get("realtime_scenarios") or {}
    rmode = rt.get("mode", "generate")
    if rmode not in MODES_RT:
        _bad("realtime_scenarios.mode", f"must be one of {MODES_RT}")
    if rmode == "provided":
        if S is None:
            _bad("realtime_scenarios.mode", "'provided' needs market_scenarios.mode 'provided' (the scenario count "
                                            "must be known to check the shapes)")
        W = len((rt.get("prob") or [[]])[0])
        if W == 0:
            _bad("realtime_scenarios.prob", "expected shape [S][W]")
        _shape(rt["prob"], (S, W), "realtime_scenarios.prob")
        for row in rt["prob"]:
            if min(row) < 0 or abs(sum(row) - 1.0) > 1e-9:
                _bad("realtime_scenarios.prob", "each row must be >= 0 and sum to 1")
        for k in ("load_rel_dev", "r_plus", "r_minus", "rho"):
            _shape(rt.get(k), (S, W, T), f"realtime_scenarios.{k}")
        flat = lambda a: [x for s in a for w in s for x in w]
        if min(flat(rt["load_rel_dev"])) <= -1:
            _bad("realtime_scenarios.load_rel_dev", "values must be > -1")
        if min(flat(rt["r_plus"])) < 1 - EPS:
            _bad("realtime_scenarios.r_plus", "values must be >= 1")
        if min(flat(rt["r_minus"])) < -EPS or max(flat(rt["r_minus"])) > 1 + EPS:
            _bad("realtime_scenarios.r_minus", "values must lie in [0, 1]")
        if min(flat(rt["rho"])) < -EPS or max(flat(rt["rho"])) > 1 + EPS:
            _bad("realtime_scenarios.rho", "values must lie in [0, 1]")
    else:
        if int(rt.get("n_branches", 3)) < 1:
            _bad("realtime_scenarios.n_branches", "must be >= 1")
        for k in ("rt_load_sigma", "rt_load_rho", "r_premium", "r_discount", "r_sigma", "rho_mean", "rho_sigma"):
            _num(rt, k, "realtime_scenarios", 0)
    return warnings
