"""Trading and forecasting metrics."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .asset import CAESParams
from .data import local_dates


def max_drawdown(cum: pd.Series):
    """Maximum drawdown (EUR) and its duration in days, from the prior peak to recovery."""
    dd = cum - cum.cummax()
    trough = dd.idxmin()
    mdd = float(-dd.min())
    start = cum.loc[:trough].idxmax()
    after = cum.loc[trough:]
    rec = after[after >= cum.loc[start]]
    end = rec.index[0] if len(rec) else cum.index[-1]
    return mdd, (pd.Timestamp(end) - pd.Timestamp(start)).days


def bootstrap_sharpe_ci(x: np.ndarray, n_boot: int = 2000, block: int = 20, seed: int = 0):
    """95% interval of the annualised Sharpe ratio by moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        s = x[(starts[:, None] + np.arange(block)[None, :]).ravel()[:n]]
        sd = s.std(ddof=1)
        out.append(s.mean() / sd * np.sqrt(365) if sd > 0 else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])


def trading_metrics(daily: pd.DataFrame, p: CAESParams, upper_bound_total: float | None = None) -> dict:
    """`daily` columns: net, gross, costs, ch_mwh, dis_mwh, buy_eur, sell_eur."""
    pnl = daily["net"]
    n = len(pnl)
    years = n / 365.25
    sd = pnl.std(ddof=1)
    dsd = np.sqrt((pnl[pnl < 0] ** 2).sum() / max(n - 1, 1))
    mdd, dd_days = max_drawdown(pnl.cumsum())
    ann = pnl.sum() / years
    ci = bootstrap_sharpe_ci(pnl.values)
    m = {
        "Net P&L total (kEUR)": pnl.sum() / 1e3,
        "Net P&L per year (kEUR/yr)": ann / 1e3,
        "EUR/kW-yr (120 MW turbine)": ann / (p.p_turb_mw * 1e3),
        "Sharpe (daily, annualised)": pnl.mean() / sd * np.sqrt(365) if sd > 0 else np.nan,
        "Sharpe 95% CI low": ci[0],
        "Sharpe 95% CI high": ci[1],
        "Sortino": pnl.mean() / dsd * np.sqrt(365) if dsd > 0 else np.nan,
        "Max drawdown (kEUR)": mdd / 1e3,
        "Max drawdown duration (days)": dd_days,
        "Calmar (annual P&L / MDD)": ann / mdd if mdd > 0 else np.nan,
        "Winning days (%)": 100 * (pnl > 0).mean(),
        "Losing days (%)": 100 * (pnl < 0).mean(),
        "Worst day (kEUR)": pnl.min() / 1e3,
        "Cycles per year": daily["dis_mwh"].sum() / p.eta_d / p.capacity_mwh / years,
        "Average buy price (EUR/MWh)": daily["buy_eur"].sum() / max(daily["ch_mwh"].sum(), 1e-9),
        "Average sell price (EUR/MWh)": daily["sell_eur"].sum() / max(daily["dis_mwh"].sum(), 1e-9),
        "Costs / gross P&L (%)": 100 * daily["costs"].sum() / max(abs(daily["gross"].sum()), 1e-9),
    }
    if upper_bound_total is not None:
        m["Capture rate vs perfect foresight (%)"] = 100 * pnl.sum() / upper_bound_total
    return m


def forecast_metrics(fc: pd.Series, real: pd.Series, naive: pd.Series) -> dict:
    """Day-ahead errors, plus how well the intraday SHAPE is ranked (what dispatch needs)."""
    idx = fc.dropna().index.intersection(real.index)
    f, r, nv = fc.loc[idx], real.loc[idx], naive.loc[idx]
    err = f - r
    mae = err.abs().mean()
    df = pd.DataFrame({"f": f, "r": r, "d": local_dates(idx)})
    with warnings.catch_warnings():  # days with a flat price profile have no rank correlation
        warnings.simplefilter("ignore")
        rho = df.groupby("d").apply(lambda g: g["f"].corr(g["r"], method="spearman"), include_groups=False)
    return {
        "MAE (EUR/MWh)": mae,
        "RMSE (EUR/MWh)": float(np.sqrt((err ** 2).mean())),
        "rMAE vs D-1 persistence": mae / (nv - r).abs().mean(),
        "Intraday rank correlation (Spearman)": float(rho.mean()),
    }
