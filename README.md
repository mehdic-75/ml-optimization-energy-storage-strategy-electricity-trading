# CAES day-ahead trading: walk-forward backtest on French power prices

This repository backtests a compressed-air energy storage (CAES) plant that trades the French day-ahead market, from January 2023 to January 2026 (1,115 delivery days, fully out of sample). The plant is rated 120 MW turbine, 70 MW compressor and 200 MWh.

The strategy has two components:
- a **direct 24-hour price forecaster** (LightGBM) retrained every quarter;
- an **optimal dispatch** solved by dynamic programming, which accounts for efficiencies, power limits, storage capacity and costs.

Every decision for delivery day D is taken with information available on D-1. P&L is settled on realised prices through a physical simulator, net of fees and variable O&M.

**Headline:** 3.65 MEUR per year net (30 EUR/kW-yr), i.e. **82% of the perfect-foresight upper bound**. The strategy wins on 93% of days, with a maximum drawdown of 18 kEUR. It keeps 80% capture when costs are tripled.

---

## Run

```bash
pip install -r requirements.txt
python scripts/run_forecast.py     # walk-forward forecasts            
python -m pytest -q tests          # look-ahead and unit tests         
python scripts/run_backtest.py     # settlement, metrics, cost stress  
python scripts/make_charts.py      # charts
```

Or simply `bash run_all.sh`.

```
caes_dayahead_backtest/
├── caes_bt/
│   ├── data.py        # price loading, local delivery days, day x hour matrix
│   ├── forecast.py    # features, LightGBM model, walk-forward training
│   ├── dispatch.py    # DP dispatcher (rolling 48 h) and heuristic benchmark
│   ├── asset.py       # plant parameters, costs, physical simulator = settlement
│   ├── backtest.py    # day-by-day loop: decide on D-1, settle on D
│   └── metrics.py     # trading and forecasting metrics, block bootstrap
├── scripts/           # run_forecast, run_backtest, make_charts
├── tests/             # look-ahead control, simulator and DP tests
├── data/France.csv    # Ember hourly day-ahead prices, France, 2015 – Jan 2026
└── outputs/           # forecasts, daily P&L, metric tables, charts
```

---

## Method

### Data and timing
- **Prices**: hourly French day-ahead prices (Ember), from January 2015 to January 2026.
- **Delivery days**: a delivery day runs from 00:00 to 24:00 Paris time, so it has 23, 24 or 25 hours. Settlement always uses the true UTC hours.
- **Information set**: to deliver on day D, only prices up to the end of D-1 are known, because D-1's auction cleared on D-2.
- **Look-ahead control**: `tests/test_no_lookahead.py` overwrites every price from D onwards with 99,999 EUR/MWh. It then checks that the forecast of D is unchanged (difference 0.0) and identical to the one stored in the backtest.

### Forecaster (`caes_bt/forecast.py`)
- **Direct model**: one row per delivery hour. All 24 hours of D are predicted at once from D-1 information, so errors do not compound through a recursion.
- **Normalised target**: `z = (price − mean(D-1)) / std(last 7 days)`. The model learns the *shape* of the day. This makes it robust to level shifts (the 2022 crisis averaged ~276 EUR/MWh, 2024 ~58 EUR/MWh), which tree models cannot extrapolate.
- **Features**:
  - lags D-1, D-2, D-7 and the 7-day mean at the same hour;
  - D-1 min, max, std and last hour;
  - price level and volatility;
  - local hour, weekday, month, day-of-year harmonics;
  - French public holidays (the day itself, the day before, the day after).
- **Training**: on all history since 2015, with exponential sample weights (half-life one year). The model is retrained at the start of every quarter on days strictly before the cutoff.
- **No price floor or mean-reversion prior**: negative prices are kept (about 450 hours in 2025) because they are a major source of storage P&L.
- **Hyperparameters**: all fixed a priori. None were tuned on the test period.

### Dispatch (`caes_bt/dispatch.py`)
- **Dynamic programming** on a 1 MWh state-of-charge grid. The objective is `Σ price·(discharge − charge) − fees − O&M`, subject to compressor and turbine limits, capacity and efficiencies.
- **Round-trip losses**: selling 1 MWh requires buying 1/0.748 = 1.34 MWh, so a spread that does not cover these losses and the costs is never traded.
- **Rolling 48-hour horizon**: day D plus a proxy for D+1 (D's forecast repeated). This stops the optimiser from emptying the store every evening. Only the 24 hours of D are executed, and the stored energy carries over to the next day.

### Settlement (`caes_bt/asset.py`)
- **Plant**: 70 MW charging, 120 MW discharging, 200 MWh, efficiencies 0.85 / 0.88 (round trip 0.748), self-discharge 0.01% per hour.
- **Feasibility caps**: the physical simulator caps exchanged power by available energy and headroom, so only feasible energy is bought or sold.
- **Costs**: 0.10 EUR/MWh exchange fee on every MWh traded, plus 2 EUR/MWh variable O&M on every MWh discharged.

### Benchmarks
The backtest uses a forecast × decision-rule grid, so the value of each component can be isolated:

| Benchmark | What it isolates |
|---|---|
| Perfect foresight: DP on realised prices over the whole period | Upper bound for this plant |
| D-1 and D-7 persistence + DP | Value of the forecaster |
| LightGBM + mean-threshold rule (charge below the daily mean, discharge above, full power) | Value of the decision rule |
| Perfect forecast + mean-threshold rule | Best case for the heuristic |

---

## Results (1,115 days, 2023-01-01 → 2026-01-19)

### Forecast accuracy

| Model | MAE (EUR/MWh) | RMSE | rMAE vs D-1 | Intraday rank correlation |
|---|---|---|---|---|
| D-1 persistence | 20.8 | 29.8 | 1.00 | 0.75 |
| D-7 persistence | 28.8 | 39.4 | 1.39 | 0.78 |
| **LightGBM (direct 24 h)** | **15.8** | **21.8** | **0.76** | **0.86** |

The most important features are, in order: D-1's last hour, price level, seasonality, hour of day and D-1's maximum.

### Trading

| Strategy | Net P&L (kEUR/yr) | Sharpe [95% CI] | Max DD (kEUR) | Winning days | Capture vs upper bound |
|---|---|---|---|---|---|
| Perfect foresight (upper bound) | 4,443 | 23.5 | 32 | 95% | 100% |
| **LightGBM forecast + DP dispatch** | **3,649** | **23.4** [21.1, 26.1] | **18** | **93%** | **82%** |
| D-1 persistence + DP dispatch | 2,928 | 16.4 | 35 | 84% | 66% |
| D-7 persistence + DP dispatch | 2,924 | 14.9 | 49 | 83% | 66% |
| Perfect forecast + mean-threshold rule | 668 | 3.8 | 421 | 64% | 15% |
| LightGBM forecast + mean-threshold rule | 176 | 1.1 | 897 | 54% | 4% |

Further strategy statistics:
- 30.4 EUR/kW-yr and about 547 cycles per year;
- average buy price 45.8 EUR/MWh, average sell price 101.4 EUR/MWh;
- costs equal 5.6% of gross P&L.

**By year (kEUR):** 3,071 (2023) · 3,699 (2024) · 4,310 (2025).

**Cost stress** (fees and O&M multiplied, with the optimiser seeing the stressed costs):

| | x1 | x2 | x3 |
|---|---|---|---|
| Strategy net P&L (kEUR/yr) | 3,649 | 3,436 | 3,214 |
| Capture rate | 82.1% | 81.2% | 79.7% |

### Reading

1. **The decision rule matters more than the forecast.**
   - With the same LightGBM forecast, moving from the threshold heuristic to the DP dispatch takes P&L from 0.18 to 3.65 MEUR per year.
   - Even with a *perfect* forecast, the heuristic captures only 15% of the available value. It ignores the 75% round-trip efficiency, so it cycles on spreads that do not pay for the losses.
2. **Once the decision is right, the forecast adds real value.** Against D-1 persistence under the same DP dispatch:
   - P&L rises by 721 kEUR per year (+25%);
   - capture rises from 66% to 82%;
   - the maximum drawdown halves (35 to 18 kEUR).
3. **What matters for storage is the intraday shape, not the price level.** The DP only needs to know which hours are cheap and which are expensive. The forecaster's intraday rank correlation (0.86 vs 0.75 for persistence) is the metric that translates into P&L, more than MAE.
4. **The result is robust to costs**: capture remains 80% when fees and O&M are tripled.

**A note on the Sharpe ratio.** For a physical arbitrage asset with good dispatch, daily P&L is almost always positive, so a Sharpe ratio above 20 is normal. It is **not comparable** to the Sharpe ratio of a financial trading strategy: there is no capital at risk in the denominator, and the revenue is largely structural. The relevant metrics here are the **capture rate against perfect foresight** and **EUR/kW-yr**.

---

## Tests

| Test | What it proves |
|---|---|
| `test_no_lookahead` | Poisoning every price from D onwards leaves the forecast of D unchanged, and the result matches the stored backtest forecast. |
| `test_cannot_sell_from_an_empty_store_or_buy_into_a_full_one` | The simulator never settles energy that could not physically flow. |
| `test_energy_balance_and_bounds_under_random_commands` | Under 500 random set-points: power limits hold, energy balance closes exactly, and P&L and costs are settled correctly. |
| `test_dp_matches_brute_force_on_a_small_instance` | The DP finds the same optimum as exhaustive enumeration of all paths. |
| `test_dp_does_not_trade_a_spread_that_does_not_cover_losses` | A 60 → 75 EUR spread is not traded (it costs 80 EUR to sell 75 EUR); a 20 → 120 spread is. |

---

## Assumptions and limitations

- **Price taker.** 120 MW is small relative to the French market, so the plant is assumed not to move prices.
- **Day-ahead only.** Execution is at the day-ahead clearing price, with no intraday re-optimisation and no ancillary services. Both would add revenue.
- **Point forecast.** The DP uses a point forecast. A stochastic DP on forecast quantiles would value the option to wait.
- **Optimistic efficiency.** The plant parameters correspond to an advanced adiabatic CAES. A conventional diabatic CAES has a round-trip efficiency of about 45–55%, which would reduce both the upper bound and the strategy's P&L.
- **Hourly products.** The backtest uses hourly products. European day-ahead coupling moved to 15-minute products in October 2025.

## Next steps

1. Add fundamental drivers to the forecaster:
   - ENTSO-E load, wind and solar forecasts;
   - TTF gas and EUA carbon prices.
2. Produce quantile forecasts and solve a stochastic DP.
3. Re-optimise intraday against the continuous market.
4. Move to 15-minute resolution.
