"""Read real DA and mFRR prices from PostgreSQL and build the price scenario table.

Connection: standard libpq environment variables, never the input file:
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSSLMODE (use 'require')
Use a read-only database user. Only this server-side module talks to the database.

The two SQL files in ./sql are the developer's own queries. They must return the columns named in
sql/da_prices.sql and sql/mfrr_up_capacity.sql (one row per hour, Europe/Copenhagen time, EUR units).
All user choices (area, dates) are passed as query parameters, never pasted into the SQL text.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .errors import DataUnavailable, InvalidInput

LOCAL_TZ = ZoneInfo("Europe/Copenhagen")
SQL_DIR = Path(os.environ.get("OFFERING_SQL_DIR", Path(__file__).resolve().parent.parent / "sql"))
CONNECT_TIMEOUT_S = int(os.environ.get("OFFERING_DB_CONNECT_TIMEOUT_S", "10"))
STATEMENT_TIMEOUT_MS = int(os.environ.get("OFFERING_DB_STATEMENT_TIMEOUT_MS", "30000"))


# --------------------------------------------------------------------------- connection
def connect():
    """Open a read-only connection from the PG* environment variables (psycopg 3, else psycopg2)."""
    missing = [k for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD") if not os.environ.get(k)]
    if missing:
        raise DataUnavailable("database settings missing in the server environment: " + ", ".join(missing))
    opts = f"-c statement_timeout={STATEMENT_TIMEOUT_MS} -c default_transaction_read_only=on"
    try:
        try:
            import psycopg                                  # type: ignore
            conn = psycopg.connect(connect_timeout=CONNECT_TIMEOUT_S, options=opts,
                                   sslmode=os.environ.get("PGSSLMODE", "require"))
        except ImportError:
            import psycopg2                                 # type: ignore
            conn = psycopg2.connect(connect_timeout=CONNECT_TIMEOUT_S, options=opts,
                                    sslmode=os.environ.get("PGSSLMODE", "require"))
    except ImportError as err:
        raise DataUnavailable("install a PostgreSQL driver: pip install 'psycopg[binary]'") from err
    except Exception as err:                                # driver-specific error classes
        raise DataUnavailable(f"cannot connect to the database: {err}") from err
    return conn


def test_connection() -> str:
    """Run one tiny query. Used by the 'Test connection' button."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("select 1")
        cur.fetchone()
        return "ok"
    finally:
        conn.close()


def _run(conn, sql_file: str, params: dict) -> list[dict]:
    text = (SQL_DIR / sql_file).read_text()
    try:
        cur = conn.cursor()
        cur.execute(text, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except DataUnavailable:
        raise
    except Exception as err:
        raise DataUnavailable(f"query {sql_file} failed: {err}") from err


# --------------------------------------------------------------------------- rows -> days
def _local(ts) -> dt.datetime:
    if isinstance(ts, str):
        ts = dt.datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        raise DataUnavailable("delivery_hour_start must carry a time zone (timestamp with time zone)")
    return ts.astimezone(LOCAL_TZ)


def _group_days(rows: list[dict], col: str, what: str) -> tuple[dict, dict]:
    """-> ({date: [24 values]}, {date: reason}). A day is kept only with exactly 24 distinct local hours."""
    for need in ("delivery_hour_start", col):
        if rows and need not in rows[0]:
            raise DataUnavailable(f"{what} query must return the column '{need}'")
    by_day: dict[dt.date, dict[int, float | None]] = defaultdict(dict)
    dup: set[dt.date] = set()
    for r in rows:
        t = _local(r["delivery_hour_start"])
        d, h = t.date(), t.hour
        if h in by_day[d]:
            dup.add(d)
        v = r[col]
        by_day[d][h] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    ok, dropped = {}, {}
    for d, hours in by_day.items():
        if d in dup:
            dropped[d] = f"{what}: repeated hour (25-hour day or duplicate rows)"
        elif set(hours) != set(range(24)):
            dropped[d] = f"{what}: {len(hours)} hours present, 24 needed"
        elif any(v is None for v in hours.values()):
            dropped[d] = f"{what}: empty price value"
        else:
            ok[d] = [hours[h] for h in range(24)]
    return ok, dropped


def _parse_date(x) -> dt.date:
    try:
        return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))
    except ValueError as err:
        raise InvalidInput(f"not a date (YYYY-MM-DD): {x}", "market_scenarios.days") from err


# --------------------------------------------------------------------------- scenario builder
def build_scenarios(ms: dict, horizon_h: int = 24, conn=None) -> tuple[dict, dict, dict]:
    """Return (scenario_set_dict, reference_tariff, info).

    scenario_set_dict has the format of OfferingScenarioSet.save(); reference_tariff is
    {'buy_eur_mwh': [T], 'sell_eur_mwh': [T]}; info lists days used, dropped days and warnings.
    """
    if horizon_h != 24:
        raise InvalidInput("database mode supports a 24-hour horizon only", "run_settings.horizon_h")
    if ms.get("days"):
        wanted = sorted({_parse_date(x) for x in ms["days"]})
    else:
        a, b = _parse_date(ms["from"]), _parse_date(ms["to"])
        if b < a:
            raise InvalidInput("'to' is before 'from'", "market_scenarios.to")
        wanted = [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]
    params = {"area": ms["area"], "first_day": wanted[0], "last_day": wanted[-1]}

    own = conn is None
    conn = conn or connect()
    try:
        da_rows = _run(conn, "da_prices.sql", params)
        sr_rows = _run(conn, "mfrr_up_capacity.sql", params)
    finally:
        if own:
            conn.close()

    da_days, drop_da = _group_days(da_rows, "da_price_eur_mwh", "DA")
    sr_days, drop_sr = _group_days(sr_rows, "mfrr_up_capacity_price_eur_mw_h", "mFRR")
    dropped = {}
    valid = []
    for d in wanted:
        if d in da_days and d in sr_days:
            valid.append(d)
        else:
            dropped[d.isoformat()] = (drop_da.get(d) or drop_sr.get(d) or
                                      ("DA: no rows" if d not in da_days else "mFRR: no rows"))
    if not valid:
        raise DataUnavailable("no complete day found in the database for the requested dates and area "
                              f"{ms['area']}: {dropped}")

    n_req = ms.get("n_scenarios")
    if n_req is not None and int(n_req) > len(valid):
        raise DataUnavailable(f"{n_req} scenarios requested but only {len(valid)} complete days exist")
    if n_req is not None and int(n_req) < len(valid):
        idx = np.unique(np.round(np.linspace(0, len(valid) - 1, int(n_req))).astype(int))
        valid = [valid[i] for i in idx]

    add, ded = float(ms["buy_adder_eur_mwh"]), float(ms["sell_deduct_eur_mwh"])
    spot = np.array([da_days[d] for d in valid])
    sr = np.array([sr_days[d] for d in valid])
    raw_buy, raw_sell = spot + add, spot - ded
    # The model does not support negative prices: floor at 0 and report how many hours were changed.
    n_floor = int(np.sum(raw_buy < 0) + np.sum(raw_sell < 0) + np.sum(sr < 0))
    da_buy, da_sell, sr_up = (np.maximum(raw_buy, 0.0), np.maximum(raw_sell, 0.0), np.maximum(sr, 0.0))

    S = len(valid)
    sigma = float(ms.get("id_spread_sigma", 0.08))
    rng = np.random.default_rng(int(ms.get("seed", 7)))
    g = np.exp(sigma * rng.standard_normal((S, horizon_h)) - sigma ** 2 / 2.0) if sigma > 0 else np.ones((S, horizon_h))
    scen = {"prob": [1.0 / S] * S, "da_buy": da_buy.tolist(), "da_sell": da_sell.tolist(), "sr_up": sr_up.tolist(),
            "id_buy": (da_buy * g).tolist(), "id_sell": (da_sell * g).tolist(),
            "load_dev_mw": np.zeros((S, horizon_h)).tolist()}
    tariff = {"buy_eur_mwh": da_buy.mean(axis=0).tolist(), "sell_eur_mwh": da_sell.mean(axis=0).tolist()}
    warnings = []
    if n_floor:
        warnings.append(f"{n_floor} price values were below 0 and were set to 0 (the model does not support negative prices)")
    if dropped:
        warnings.append(f"{len(dropped)} requested day(s) were skipped (incomplete or clock-change days)")
    info = {"days_used": [d.isoformat() for d in valid], "days_dropped": dropped,
            "negative_values_floored": n_floor, "warnings": warnings}
    return scen, tariff, info
