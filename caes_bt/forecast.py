"""Day-ahead price forecaster (direct 24-hour LightGBM model), trained walk-forward.

Design choices, all fixed before looking at the test period:
 1. Direct forecast of the 24 hours of day D from information available on D-1
    (one row per delivery hour), instead of an hour-by-hour recursion that would feed
    its own errors back over 24 steps.
 2. Point forecast of the expected price, with no simulated noise: the dispatch needs
    the expectation. Recent volatility enters as a feature.
 3. Normalised target and features: z = (price - mean(D-1)) / std(last 7 days).
    The model learns the SHAPE of the day, which is robust to level shifts (the 2022
    crisis averaged ~276 EUR/MWh, 2024 ~58 EUR/MWh) that tree models cannot extrapolate.
 4. No mean-reversion prior and no price floor: negative prices exist and are a major
    source of storage P&L.
 5. Calendar features in LOCAL time, plus French public holidays.
 6. Full history since 2015, exponentially weighted (half-life one year); every
    available day is used for training.
 7. LightGBM: fast enough to retrain every quarter.
"""
from __future__ import annotations

import time

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import TZ, day_hour_matrix

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, objective="l2",
                  n_jobs=1, verbose=-1, random_state=0)
HALF_LIFE_DAYS = 365
SCALE_FLOOR = 5.0  # EUR/MWh, avoids exploding z-scores in very calm weeks

FEATURES = ["hour", "dow", "month", "doy_sin", "doy_cos", "hol", "hol_prev", "hol_next",
            "level", "log_scale", "d1_min", "d1_max", "d1_std", "d1_last", "d7_mean",
            "w_mean", "lag1", "lag2", "lag7", "ma7"]


def build_dataset(M: pd.DataFrame) -> pd.DataFrame:
    """One row per (delivery day D, local hour h). Every feature uses D-1 and earlier only."""
    A = M.values
    fr_hol = holidays.France(years=range(M.index[0].year, M.index[-1].year + 2))
    recs = []
    for i in range(8, A.shape[0]):
        d = M.index[i]
        prev = A[i - 1]
        m1 = prev.mean()
        scale = max(A[i - 7:i].std(), SCALE_FLOOR)
        ma7 = A[i - 7:i].mean(axis=0)
        day_feats = dict(
            dow=d.dayofweek, month=d.month,
            doy_sin=np.sin(2 * np.pi * d.dayofyear / 365.25),
            doy_cos=np.cos(2 * np.pi * d.dayofyear / 365.25),
            hol=int(d in fr_hol),
            hol_prev=int((d - pd.Timedelta(days=1)) in fr_hol),
            hol_next=int((d + pd.Timedelta(days=1)) in fr_hol),
            level=m1, log_scale=np.log(scale),
            d1_min=(prev.min() - m1) / scale, d1_max=(prev.max() - m1) / scale,
            d1_std=prev.std() / scale, d1_last=(prev[23] - m1) / scale,
            d7_mean=(A[i - 7].mean() - m1) / scale,
            w_mean=(A[i - 7:i].mean() - m1) / scale,
        )
        for h in range(24):
            r = dict(day_feats)
            r.update(day=d, hour=h,
                     lag1=(prev[h] - m1) / scale,
                     lag2=(A[i - 2, h] - m1) / scale,
                     lag7=(A[i - 7, h] - m1) / scale,
                     ma7=(ma7[h] - m1) / scale,
                     m1=m1, scale=scale,
                     y=(A[i, h] - m1) / scale)
            recs.append(r)
    return pd.DataFrame(recs)


def fit_model(ds: pd.DataFrame, cutoff: pd.Timestamp) -> lgb.LGBMRegressor:
    """Train on every delivery day strictly before `cutoff`, with exponential time decay."""
    tr = ds[ds["day"] < cutoff]
    age = (cutoff - tr["day"]).dt.days.values
    w = 0.5 ** (age / HALF_LIFE_DAYS)
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[FEATURES], tr["y"], sample_weight=w)


def predict(model: lgb.LGBMRegressor, rows: pd.DataFrame) -> np.ndarray:
    """De-normalise the model output back to EUR/MWh."""
    return rows["m1"].values + model.predict(rows[FEATURES]) * rows["scale"].values


def walk_forward(prices: pd.Series, start: str, end: str, retrain: str = "QS", verbose: bool = True):
    """Out-of-sample forecasts for every delivery day in [start, end].

    The model is retrained at each `retrain` date on days strictly before it. The
    decision for D is taken on D-1, when D-1 prices are known (D-1's auction cleared on
    D-2), so features may use prices up to the end of D-1 and nothing later.

    Returns (hourly forecasts on the true UTC hours, feature importance of the last model).
    Columns: model forecast, D-1 persistence, D-7 persistence.
    """
    M = day_hour_matrix(prices)
    ds = build_dataset(M)
    test_days = pd.date_range(start, end, freq="D")
    cutoffs = list(pd.date_range(start, end, freq=retrain))
    if not cutoffs or cutoffs[0] > test_days[0]:
        cutoffs = [test_days[0]] + cutoffs
    out, t0, model = [], time.time(), None
    for k, cut in enumerate(cutoffs):
        nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else test_days[-1] + pd.Timedelta(days=1)
        model = fit_model(ds, cut)
        te = ds[(ds["day"] >= cut) & (ds["day"] < nxt)].copy()
        te["forecast"] = predict(model, te)
        te["naive_d1"] = te["m1"] + te["lag1"] * te["scale"]
        te["naive_d7"] = te["m1"] + te["lag7"] * te["scale"]
        out.append(te[["day", "hour", "forecast", "naive_d1", "naive_d7"]])
        if verbose:
            print(f"retrain {cut.date()}: {int((ds['day'] < cut).sum())} training rows "
                  f"-> {len(te)} forecasts | {time.time() - t0:.0f}s", flush=True)
    fc = pd.concat(out)

    # map local (day, hour) back to the true UTC hours (handles 23 / 25-hour days)
    loc = prices.index.tz_localize("UTC").tz_convert(TZ)
    key = pd.DataFrame({"day": pd.to_datetime(loc.date), "hour": loc.hour}, index=prices.index)
    key = key[(key["day"] >= test_days[0]) & (key["day"] <= test_days[-1])]
    key.index.name = "datetime_utc"
    hourly = key.reset_index().merge(fc, on=["day", "hour"], how="left").set_index("datetime_utc")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return hourly[["forecast", "naive_d1", "naive_d7"]], importance
