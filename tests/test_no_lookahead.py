"""Look-ahead control: poison every price from day D onwards and check that the forecast
of D does not move. If any feature used a price of D or later, it would."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caes_bt.data import TZ, day_hour_matrix, load_prices  
from caes_bt.forecast import build_dataset, fit_model, predict  

DAY, CUTOFF = pd.Timestamp("2024-05-15"), pd.Timestamp("2024-04-01")


def _forecast_day(prices):
    ds = build_dataset(day_hour_matrix(prices))
    model = fit_model(ds, CUTOFF)
    return predict(model, ds[ds["day"] == DAY])


def test_forecast_of_day_d_ignores_prices_from_d_onwards():
    prices = load_prices(os.path.join(ROOT, "data", "France.csv"))
    ref = _forecast_day(prices)
    first_hour_utc = DAY.tz_localize(TZ).tz_convert("UTC").tz_localize(None)
    poisoned = prices.copy()
    poisoned.loc[first_hour_utc:] = 99_999.0
    assert np.allclose(ref, _forecast_day(poisoned)), "look-ahead detected"

    stored = os.path.join(ROOT, "outputs", "forecasts.csv")
    if os.path.exists(stored):  # the backtest file holds exactly this out-of-sample forecast
        fc = pd.read_csv(stored, index_col=0, parse_dates=True)
        loc = fc.index.tz_localize("UTC").tz_convert(TZ)
        s = fc[loc.date == DAY.date()]["forecast"].values
        assert np.allclose(np.sort(s), np.sort(ref))
