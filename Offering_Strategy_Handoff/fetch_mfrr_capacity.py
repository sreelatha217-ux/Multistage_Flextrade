"""Download DK1 mFRR capacity-market results (Energi Data Service) and compare them with DA prices.

  uv run python fetch_mfrr_capacity.py
  uv run python fetch_mfrr_capacity.py --start 2025-10-01 --end 2026-10-16

Writes mfrr_capacity_dk1.csv (TimeUTC, TimeDK, UpDemandMW, UpProcuredMW, UpPriceEUR, ...) and prints
  - price statistics (EUR per MW per hour, assumed unit: check the dataset description),
  - the ratio  mFRR up capacity price / DA price  (this is what `--sr-ratio` assumes to be 0.12),
  - the average price per hour of day.
Uses only the standard library + pandas.
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

URL = "https://api.energidataservice.dk/dataset/mFRRCapacityMarket"
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(start, end, area="DK1", page=20000):
    rows, off = [], 0
    while True:
        q = dict(start=start, end=end, filter=json.dumps({"PriceArea": [area]}),
                 sort="TimeUTC asc", limit=page, offset=off)
        with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(q), timeout=120) as r:
            recs = json.load(r)["records"]
        rows += recs
        if len(recs) < page:
            break
        off += page
    return pd.DataFrame(rows)


def summarize(df, da_csv=None):
    df = df.copy()
    df["TimeDK"] = pd.to_datetime(df["TimeDK"])
    p = df["UpPriceEUR"].astype(float)
    print(f"rows: {len(df)}   {df.TimeDK.min()} -> {df.TimeDK.max()}")
    print(f"UpPriceEUR: mean {p.mean():.2f}  median {p.median():.2f}  p95 {p.quantile(.95):.2f}  "
          f"max {p.max():.2f}  zero-price hours {100 * (p == 0).mean():.1f}%")
    print(f"UpProcuredMW: mean {df.UpProcuredMW.mean():.0f}   UpDemandMW: mean {df.UpDemandMW.mean():.0f}")
    print("\nmean UpPriceEUR by hour of day (local):")
    print(p.groupby(df.TimeDK.dt.hour).mean().round(2).to_string())
    if da_csv and os.path.exists(da_csv):
        da = pd.read_csv(da_csv, parse_dates=["date"]).set_index("date")
        long = da.stack().rename("da").reset_index()
        long.columns = ["date", "col", "da"]
        long["hour"] = long["col"].str[1:].astype(int)
        m = pd.DataFrame({"date": df.TimeDK.dt.normalize(), "hour": df.TimeDK.dt.hour, "mfrr": p})
        m = m.merge(long[["date", "hour", "da"]], on=["date", "hour"])
        m = m[m.da > 1.0]
        if len(m):
            r = m.mfrr / m.da
            print(f"\nmatched with DA prices: {len(m)} hours")
            print(f"ratio of means  (mean mFRR / mean DA): {m.mfrr.mean() / m.da.mean():.4f}")
            print(f"per-hour ratio: median {r.median():.4f}  mean {r.mean():.4f}  p95 {r.quantile(.95):.4f}")
            print("model assumption was --sr-ratio 0.12")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--area", default="DK1")
    ap.add_argument("--out", default=os.path.join(HERE, "mfrr_capacity_dk1.csv"))
    ap.add_argument("--da", default=os.path.join(HERE, "dk1_hourly_da.csv"))
    a = ap.parse_args()
    d = fetch(a.start, a.end, a.area)
    if d.empty:
        raise SystemExit("no records returned: check --start/--end")
    d.to_csv(a.out, index=False)
    print(f"saved {a.out}")
    summarize(d, a.da)
