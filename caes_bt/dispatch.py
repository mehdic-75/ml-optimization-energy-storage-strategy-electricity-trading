"""Decision rules: optimal dispatch by dynamic programming, and a heuristic benchmark."""
from __future__ import annotations

import numpy as np

from .asset import CAESParams, Costs


class DPDispatcher:
    """Maximise sum_t [price_t * (discharge_t - charge_t) - costs_t] under the plant constraints.

    State = stored energy on a grid of 'step_mwh'. One transition per hour; a transition
    is feasible if the implied grid power respects the compressor / turbine limits.
    Efficiencies and costs are inside the objective, so a price spread that does not
    cover the round-trip losses and costs is never traded. Self-discharge (1e-4 per hour)
    is ignored in the optimisation but applied by the simulator.
    """

    def __init__(self, p: CAESParams, c: Costs, step_mwh: float = 1.0):
        self.p, self.c = p, c
        self.grid = np.arange(0.0, p.capacity_mwh + 1e-9, step_mwh)
        delta = self.grid[None, :] - self.grid[:, None]                 # energy change from -> to
        self.buy_mw = np.where(delta > 0, delta / p.eta_c, 0.0)          # power bought
        self.sell_mw = np.where(delta < 0, -delta * p.eta_d, 0.0)        # power sold
        self.feasible = (self.buy_mw <= p.p_comp_mw + 1e-9) & (self.sell_mw <= p.p_turb_mw + 1e-9)

    def rewards(self, price: float) -> np.ndarray:
        c = self.c
        r = (price - c.fee_eur_mwh - c.vom_eur_mwh) * self.sell_mw - (price + c.fee_eur_mwh) * self.buy_mw
        return np.where(self.feasible, r, -np.inf)

    def solve(self, prices: np.ndarray, e0: float, terminal_value: float = 0.0) -> np.ndarray:
        """Optimal grid set-points (MW) for the whole price sequence, starting from e0."""
        T, G = len(prices), len(self.grid)
        V = terminal_value * self.grid
        policy = np.empty((T, G), dtype=np.int32)
        for t in range(T - 1, -1, -1):
            Q = self.rewards(prices[t]) + V[None, :]
            policy[t] = np.argmax(Q, axis=1)
            V = Q[np.arange(G), policy[t]]
        i = int(np.argmin(np.abs(self.grid - e0)))
        cmd = np.zeros(T)
        for t in range(T):
            j = policy[t, i]
            cmd[t] = self.buy_mw[i, j] - self.sell_mw[i, j]
            i = j
        return cmd

    def plan_day(self, forecast_day: np.ndarray, e0: float) -> np.ndarray:
        """Rolling 48-hour horizon: day D plus a proxy for D+1 (the forecast of D repeated).

        Looking past midnight stops the optimiser from emptying the store at the end of
        every day; only the 24 hours of D are executed.
        """
        f = np.asarray(forecast_day, dtype=float)
        return self.solve(np.concatenate([f, f]), e0)[: len(f)]


def mean_threshold_rule(forecast_day: np.ndarray, p: CAESParams) -> np.ndarray:
    """Heuristic benchmark: charge at full power when the forecast is below its daily mean,
    discharge at full power when it is above.

    It ignores efficiencies and costs. It is used only as an ablation, to show how much
    of the value comes from the decision rule rather than from the forecast.
    """
    f = np.asarray(forecast_day, dtype=float)
    m = f.mean()
    return np.where(f < m, p.p_comp_mw, np.where(f > m, -p.p_turb_mw, 0.0))
