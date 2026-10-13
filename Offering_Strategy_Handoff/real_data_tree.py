"""Pure numpy/pandas helpers: turn real DK1 day-ahead prices into scenario-tree arrays.

No Pyomo needed, so this module can be tested on its own:  python real_data_tree.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_prices(path: str = "dk1_hourly_da.csv") -> pd.DataFrame:
    """Day x 24 table (EUR/MWh), index = date, columns h00..h23."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    cols = [f"h{h:02d}" for h in range(24)]
    df = df[cols].astype(float)
    if df.isna().any().any():
        raise ValueError("dk1_hourly_da.csv contains missing values")
    return df


def day_types(df: pd.DataFrame, end_date, window: int, n: int, seed: int = 2024):
    """k-means day types from the `window` days up to and including end_date.

    Returns (prices [n,24], probs [n]) sorted by daily mean price (cheap -> expensive).
    """
    hist = df.loc[:pd.Timestamp(end_date)].iloc[-window:]
    if len(hist) < max(n, 5):
        raise ValueError(f"only {len(hist)} history days before {end_date}")
    if n == 1:
        return hist.values.mean(0, keepdims=True), np.array([1.0])
    km = KMeans(n_clusters=n, n_init=20, random_state=seed).fit(hist.values)
    probs = np.bincount(km.labels_, minlength=n) / len(hist)
    order = np.argsort(km.cluster_centers_.mean(1))
    return km.cluster_centers_[order], probs[order]


def tree_arrays(da_price: np.ndarray, probs: np.ndarray, *, floor: float = 0.0, sell_ratio: float = 0.55,
                sr_ratio: float = 0.12, id_sigma: float = 0.08, sr_sigma: float = 0.20, seed: int = 7):
    """DA/ID/SR arrays [S,T] from DA price paths [S,T] (same recipe as OfferingScenarioSet.generate).

    The repo validator rejects negative prices, so prices are floored at `floor` (default 0).
    ID = DA * lognormal(mean 1, id_sigma); SR = sr_ratio * DA_buy * lognormal(mean 1, sr_sigma).
    """
    rng = np.random.default_rng(seed)
    S, T = da_price.shape
    da_buy = np.maximum(da_price, floor)
    da_sell = sell_ratio * da_buy
    g_id = np.exp(id_sigma * rng.standard_normal((S, T)) - id_sigma ** 2 / 2)
    f_sr = np.exp(sr_sigma * rng.standard_normal((S, T)) - sr_sigma ** 2 / 2)
    return dict(prob=np.asarray(probs, float), da_buy=da_buy, da_sell=da_sell, sr_up=sr_ratio * da_buy * f_sr,
                id_buy=da_buy * g_id, id_sell=da_sell * g_id, load_dev_mw=np.zeros((S, T)))


def actual_day_tree(day_prices: np.ndarray, n_id: int = 3, **kw):
    """Evaluation tree: ONE real DA price path, n_id ID/SR shocks around it (all share the same DA price)."""
    return tree_arrays(np.tile(day_prices, (n_id, 1)), np.full(n_id, 1.0 / n_id), **kw)


if __name__ == "__main__":
    df = load_prices()
    P, pi = day_types(df, df.index[-1], 90, 3)
    a = tree_arrays(P, pi)
    print("days:", len(df), df.index[0].date(), "->", df.index[-1].date())
    print("neg-price hours:", int((df.values < 0).sum()), "of", df.size)
    print("3 day types, probs", pi.round(3), "daily means", P.mean(1).round(1))
    for k, v in a.items():
        assert np.all(np.isfinite(v))
    assert abs(a["prob"].sum() - 1) < 1e-12 and (a["da_sell"] <= a["da_buy"] + 1e-9).all()
    assert (a["da_buy"] >= 0).all() and (a["sr_up"] >= 0).all() and (a["id_sell"] <= a["id_buy"] + 1e-9).all()
    e = actual_day_tree(df.iloc[-1].values)
    assert np.allclose(e["da_buy"], e["da_buy"][0])
    print("OK: arrays valid for OfferingScenarioSet.validate (>=0, sell<=buy, probs sum to 1)")
