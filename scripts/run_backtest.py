"""Settle every strategy on realised prices and compute the metrics.

Strategies (forecast x decision rule):
  * Perfect foresight          : DP on realised prices over the whole period (upper bound)
  * Strategy                   : LightGBM forecast + DP dispatch (rolling 48 h)
  * D-1 persistence + DP       : same dispatch, naive forecast    -> value of the forecast
  * D-7 persistence + DP       : same dispatch, weekly naive forecast
  * LightGBM + mean threshold  : same forecast, heuristic rule   -> value of the decision rule
  * Perfect forecast + mean threshold : the heuristic with an oracle forecast
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caes_bt.asset import CAESParams, Costs  
from caes_bt.backtest import run_perfect_foresight, run_strategy
from caes_bt.data import delivery_days, load_prices  
from caes_bt.dispatch import DPDispatcher, mean_threshold_rule  
from caes_bt.metrics import forecast_metrics, trading_metrics  

STRATEGY = "LightGBM forecast + DP dispatch"
UPPER = "Perfect foresight (upper bound)"


def main(csv, start, end, outdir):
    P, C = CAESParams(), Costs()
    real = load_prices(csv)
    days = delivery_days(real, start, end)
    test_idx = pd.DatetimeIndex(np.concatenate([h.values for _, h in days]))
    fc = pd.read_csv(os.path.join(outdir, "forecasts.csv"), index_col=0, parse_dates=True)
    assert fc.loc[test_idx, "forecast"].notna().all(), "missing forecasts"

    def dp_rule(costs):
        dp = DPDispatcher(P, costs)
        return lambda f, e: dp.plan_day(f, e)

    threshold = lambda f, e: mean_threshold_rule(f, P)  
    plan = [
        (STRATEGY, fc["forecast"], dp_rule(C)),
        ("D-1 persistence + DP dispatch", fc["naive_d1"], dp_rule(C)),
        ("D-7 persistence + DP dispatch", fc["naive_d7"], dp_rule(C)),
        ("LightGBM forecast + mean-threshold rule", fc["forecast"], threshold),
        ("Perfect forecast + mean-threshold rule", real, threshold),
    ]
    res = {UPPER: run_perfect_foresight(days, real, P, C)}
    for name, f, rule in plan:
        res[name] = run_strategy(days, real, f, rule, P, C)
        print(f"settled: {name}", flush=True)

    ub = res[UPPER]["net"].sum()
    tm = pd.DataFrame({k: trading_metrics(v, P, ub) for k, v in res.items()}).T
    tm.to_csv(os.path.join(outdir, "metrics_trading.csv"))
    daily = pd.DataFrame({k: v["net"] for k, v in res.items()})
    daily.to_csv(os.path.join(outdir, "pnl_daily.csv"))
    pd.DataFrame({k: v["soc_end"] for k, v in res.items()}).to_csv(os.path.join(outdir, "soc_daily_end.csv"))
    yearly = daily.groupby(daily.index.year).sum() / 1e3
    yearly.T.to_csv(os.path.join(outdir, "pnl_by_year_keur.csv"))

    # forecast accuracy
    r = real.loc[test_idx]
    nv = fc.loc[test_idx, "naive_d1"]
    fm = pd.DataFrame({
        "D-1 persistence": forecast_metrics(nv, r, nv),
        "D-7 persistence": forecast_metrics(fc.loc[test_idx, "naive_d7"], r, nv),
        "LightGBM (direct 24 h)": forecast_metrics(fc.loc[test_idx, "forecast"], r, nv),
    }).T
    fm.to_csv(os.path.join(outdir, "metrics_forecast.csv"))

    # cost stress: fees and O&M x2, x3 
    stress = {}
    for k in (1, 2, 3):
        Ck = C.scaled(k)
        pf = run_perfect_foresight(days, real, P, Ck)
        st = run_strategy(days, real, fc["forecast"], dp_rule(Ck), P, Ck)
        nv_k = run_strategy(days, real, fc["naive_d1"], dp_rule(Ck), P, Ck)
        yrs = len(st) / 365.25
        stress[f"costs x{k}"] = {
            "Strategy net P&L (kEUR/yr)": st["net"].sum() / yrs / 1e3,
            "Strategy Sharpe": st["net"].mean() / st["net"].std() * np.sqrt(365),
            "Strategy capture rate (%)": 100 * st["net"].sum() / pf["net"].sum(),
            "D-1 persistence + DP net P&L (kEUR/yr)": nv_k["net"].sum() / yrs / 1e3,
            "Perfect foresight (kEUR/yr)": pf["net"].sum() / yrs / 1e3,
        }
    sdf = pd.DataFrame(stress)
    sdf.to_csv(os.path.join(outdir, "cost_stress_test.csv"))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
    print("\n=== FORECAST ===\n", fm)
    print("\n=== TRADING ===\n", tm.T)
    print("\n=== P&L BY YEAR (kEUR) ===\n", yearly.T)
    print("\n=== COST STRESS ===\n", sdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "France.csv"))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-19")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args()
    main(a.csv, a.start, a.end, a.outdir)
