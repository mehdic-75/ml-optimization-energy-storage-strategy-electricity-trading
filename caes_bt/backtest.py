"""Day-by-day backtest loop: decide on D-1, execute and settle on D at realised prices."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .asset import CAESParams, Costs, simulate_and_settle
from .dispatch import DPDispatcher

# rule(forecast prices of day D, stored energy at the start of D) -> hourly set-points (MW)
Rule = Callable[[np.ndarray, float], np.ndarray]


def run_strategy(days, real: pd.Series, forecast: pd.Series, rule: Rule,
                 p: CAESParams, c: Costs) -> pd.DataFrame:
    """For each delivery day D:
      1. take the forecast of D's prices made with information available on D-1;
      2. the decision rule turns it into hourly set-points;
      3. the physical simulator executes them and P&L is settled on D's REALISED prices,
         net of fees and variable O&M;
      4. the stored energy at the end of D becomes the starting point of D+1.
    """
    e = p.e0_mwh
    recs = []
    for d, hours in days:
        price = real.loc[hours].values
        cmd = rule(forecast.loc[hours].values, e)
        e, ch, dis, gross, costs = simulate_and_settle(cmd, price, e, p, c)
        recs.append(dict(day=pd.Timestamp(d), gross=gross, costs=costs, net=gross - costs,
                         ch_mwh=ch.sum(), dis_mwh=dis.sum(), buy_eur=(ch * price).sum(),
                         sell_eur=(dis * price).sum(), soc_end=e))
    return pd.DataFrame(recs).set_index("day")


def run_perfect_foresight(days, real: pd.Series, p: CAESParams, c: Costs) -> pd.DataFrame:
    """Single DP over the whole period on realised prices: the upper bound for this plant."""
    hours = pd.DatetimeIndex(np.concatenate([h.values for _, h in days]))
    cmd = pd.Series(DPDispatcher(p, c).solve(real.loc[hours].values, p.e0_mwh), index=hours)
    return run_strategy(days, real, cmd, lambda f, e: f, p, c)
