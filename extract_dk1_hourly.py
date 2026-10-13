"""Build a clean day x 24 hourly DK1 day-ahead price table from the Aiven DB (read-only).

Only source='dayahead_real' rows are used (real 15-min MTU data, 2025-10-01 onward).
The interpolated_from_elspotprices rows are never touched.

Usage:
  export DATABASE_URL='postgres://...'
  uv run --with psycopg2-binary --with pandas python extract_dk1_hourly.py
Writes dk1_hourly_da.csv (one row per complete local day, columns h00..h23, EUR/MWh, negatives kept)
and prints a data-quality report. Paste the report back.
"""
import os
import sys

import pandas as pd

TZ = "Europe/Copenhagen"


def build_hourly(df: pd.DataFrame):
    """df: columns time_utc, price_eur (15-min). Returns (days, report_dict).

    A local day is kept only if it has exactly 24 local hours with exactly 4 quarter-hour prices each.
    This drops the two DST days (23 h / 25 h) and any partial day (e.g. the current day) automatically.
    """
    d = df.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
    d = d.drop_duplicates("time_utc").sort_values("time_utc")
    loc = d["time_utc"].dt.tz_convert(TZ)
    d["date"] = loc.dt.date
    d["hour"] = loc.dt.hour
    g = d.groupby(["date", "hour"])["price_eur"].agg(["mean", "count"]).reset_index()

    ok_rows, dropped = [], {}
    for date, grp in g.groupby("date"):
        if len(grp) == 24 and (grp["count"] == 4).all() and sorted(grp["hour"]) == list(range(24)):
            ok_rows.append((date, grp.sort_values("hour")["mean"].to_numpy()))
        else:
            n_q = int(grp["count"].sum())
            dropped[date] = f"{n_q} quarter-hours ({len(grp)} local hours)"
    days = pd.DataFrame([v for _, v in ok_rows], index=pd.Index([k for k, _ in ok_rows], name="date"),
                        columns=[f"h{h:02d}" for h in range(24)])
    return days, dropped


def main():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL first.")
    import psycopg2
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("select time_utc, price_eur from public.day_ahead_prices "
                "where zone='DK1' and source='dayahead_real' order by time_utc")
    df = pd.DataFrame(cur.fetchall(), columns=["time_utc", "price_eur"])
    days, dropped = build_hourly(df)
    days.to_csv("dk1_hourly_da.csv")

    p = days.to_numpy()
    daily_mean, daily_min, daily_max = p.mean(1), p.min(1), p.max(1)
    print(f"raw quarter-hour rows : {len(df)}  ({df.time_utc.min()} .. {df.time_utc.max()})")
    print(f"complete local days   : {len(days)}  ({days.index.min()} .. {days.index.max()})")
    print(f"dropped days          : {len(dropped)}")
    for k, v in dropped.items():
        print(f"   {k}: {v}")
    print(f"days with a negative hour : {int((daily_min < 0).sum())}   (lowest {p.min():.2f} EUR/MWh)")
    print(f"days with a price > 300   : {int((daily_max > 300).sum())}   (highest {p.max():.0f} EUR/MWh)")
    print(f"daily mean price          : mean {daily_mean.mean():.1f}, std {daily_mean.std():.1f}, "
          f"min {daily_mean.min():.1f}, max {daily_mean.max():.1f}")
    print(f"intraday max-min spread   : mean {(daily_max - daily_min).mean():.1f}, max {(daily_max - daily_min).max():.0f}")
    top = days.assign(spread=daily_max - daily_min).sort_values("spread", ascending=False).head(5)
    print("5 largest-spread days:", [(str(i), round(float(s))) for i, s in zip(top.index, top.spread)])
    m = pd.Series(daily_mean, index=pd.to_datetime(days.index)).groupby(lambda x: x.strftime("%Y-%m")).mean().round(1)
    print("monthly mean:", m.to_dict())
    print("saved dk1_hourly_da.csv")


if __name__ == "__main__":
    main()
