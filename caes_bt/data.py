"""Price data: loading, local delivery days, day x hour matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Paris"


def load_prices(path_csv: str, country: str = "France") -> pd.Series:
    """Hourly day-ahead prices (Ember format), indexed by naive UTC timestamps.

    Duplicated timestamps keep the last value; isolated gaps are interpolated so that
    every UTC hour exists.
    """
    df = pd.read_csv(path_csv)
    df = df[df["Country"] == country]
    s = df.set_index(pd.to_datetime(df["Datetime (UTC)"]))["Price (EUR/MWhe)"]
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.asfreq("h").interpolate()
    s.name = "price"
    return s


def local_dates(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """Local (Europe/Paris) calendar date of each naive-UTC timestamp."""
    return index_utc.tz_localize("UTC").tz_convert(TZ).date


def delivery_days(prices: pd.Series, start: str, end: str):
    """List of (local date, UTC index of the hours of that delivery day).

    A delivery day runs from 00:00 to 24:00 Paris time, so it has 23, 24 or 25 hours.
    """
    loc_date = pd.Series(local_dates(prices.index), index=prices.index)
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return [(d, grp.index) for d, grp in loc_date.groupby(loc_date) if lo <= d <= hi]


def day_hour_matrix(prices: pd.Series) -> pd.DataFrame:
    """Matrix of local days x 24 local hours.

    On the 23-hour day the missing hour is filled from its neighbours; on the 25-hour
    day the repeated hour is averaged. This matrix is only used to build forecasting
    features; settlement always uses the true UTC hours.
    """
    loc = prices.tz_localize("UTC").tz_convert(TZ)
    df = pd.DataFrame({"p": loc.values, "d": loc.index.date, "h": loc.index.hour})
    M = df.pivot_table(index="d", columns="h", values="p", aggfunc="mean")
    M = M.reindex(columns=range(24)).ffill(axis=1).bfill(axis=1)
    M.index = pd.to_datetime(M.index)
    return M
