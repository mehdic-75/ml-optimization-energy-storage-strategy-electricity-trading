"""Backtest charts (PNG), built from the outputs of run_forecast.py and run_backtest.py."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates 
import matplotlib.pyplot as plt  
import pandas as pd  

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
from caes_bt.data import load_prices  

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

daily = pd.read_csv(os.path.join(OUT, "pnl_daily.csv"), index_col=0, parse_dates=True)

# 1) cumulative P&L 
fig, ax = plt.subplots(figsize=(11, 5.6))
cum = daily.cumsum() / 1e6
series = [("Perfect foresight (upper bound)", GREY, "--", "Perfect foresight"),
          ("LightGBM forecast + DP dispatch", BLUE, "-", "Strategy (LightGBM + DP)"),
          ("D-1 persistence + DP dispatch", AQUA, "-", "D-1 persistence + DP"),
          ("LightGBM forecast + mean-threshold rule", ORANGE, "-", "LightGBM + threshold rule")]
for col, color, ls, label in series:
    y = cum[col]
    ax.plot(cum.index, y, color=color, lw=2, ls=ls, label=label)
    ax.annotate(f"{label}  {y.iloc[-1]:+.1f} MEUR", (cum.index[-1], y.iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel("Cumulative net P&L (MEUR)")
ax.set_title("CAES 120 MW / 200 MWh: cumulative net P&L, walk-forward, French day-ahead, Jan 2023 to Jan 2026",
             loc="left", color=INK, fontsize=12)
ax.legend(loc="upper left", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=360))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_cumulative_pnl.png"), dpi=160)
plt.close(fig)

# 2) decomposition: annual P&L per forecast x rule 
tm = pd.read_csv(os.path.join(OUT, "metrics_trading.csv"), index_col=0)
s = tm["Net P&L per year (kEUR/yr)"].sort_values()
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.set_axisbelow(True)
colors = [BLUE if "LightGBM forecast + DP" in n else GREY for n in s.index]
ax.barh(range(len(s)), s.values, color=colors, height=0.62)
for i, v in enumerate(s.values):
    ax.annotate(f"{v:,.0f}", (v, i), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(s)))
ax.set_yticklabels(s.index, color=INK)
ax.axvline(0, color=MUTED, lw=0.8)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Average net P&L (kEUR per year)")
ax.set_title("Forecast x decision rule: where the value comes from", loc="left", color=INK, fontsize=12)
ax.set_xlim(min(0, s.min() * 1.25), s.max() * 1.18)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_decomposition.png"), dpi=160)
plt.close(fig)

# 3) a sample week: forecasts vs realised 
real = load_prices(os.path.join(ROOT, "data", "France.csv"))
fc = pd.read_csv(os.path.join(OUT, "forecasts.csv"), index_col=0, parse_dates=True)
wk = slice("2024-05-06", "2024-05-12 23:00")
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(real.loc[wk].index, real.loc[wk], color=INK, lw=2, label="Realised price")
ax.plot(fc.loc[wk].index, fc.loc[wk, "forecast"], color=BLUE, lw=1.6, label="LightGBM forecast (made on D-1)")
ax.plot(fc.loc[wk].index, fc.loc[wk, "naive_d1"], color=ORANGE, lw=1.4, ls="--", label="D-1 persistence")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel("EUR/MWh")
ax.set_title("Week of 6 May 2024: day-ahead forecasts vs realised prices", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, loc="upper left", ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_week_forecast.png"), dpi=160)
plt.close(fig)
print("charts OK")
