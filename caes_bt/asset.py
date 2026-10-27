"""CAES asset: technical parameters, costs, and the physical simulator used for settlement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CAESParams:
    """Adiabatic CAES plant. Powers are measured at the grid connection."""
    p_comp_mw: float = 70.0        # maximum compressor (charging) power
    p_turb_mw: float = 120.0       # maximum turbine (discharging) power
    capacity_mwh: float = 200.0    # storable energy
    eta_c: float = 0.85            # charging efficiency
    eta_d: float = 0.88            # discharging efficiency  -> round trip 0.748
    loss_per_h: float = 1e-4       # self-discharge, fraction of stored energy per hour
    e0_mwh: float = 50.0           # initial stored energy

    @property
    def round_trip(self) -> float:
        return self.eta_c * self.eta_d


@dataclass(frozen=True)
class Costs:
    fee_eur_mwh: float = 0.10      # exchange fee per MWh traded (buy and sell)
    vom_eur_mwh: float = 2.0       # variable O&M per MWh discharged

    def scaled(self, k: float) -> "Costs":
        return Costs(self.fee_eur_mwh * k, self.vom_eur_mwh * k)


def simulate_and_settle(cmd_mw: np.ndarray, price: np.ndarray, e0: float,
                        p: CAESParams, c: Costs):
    """Execute hourly grid set-points (MW, + = charge, - = discharge) and settle them.

    Physical rules:
      - the power actually exchanged is capped by the power limits, by the energy
        available (discharge) and by the remaining headroom (charge), so only feasible
        energy is bought or sold;
      - self-discharge is applied to the stored energy every hour.

    Returns (end energy, charge MW per hour, discharge MW per hour, gross P&L, costs).
    """
    n = len(cmd_mw)
    e = float(e0)
    ch = np.zeros(n)
    dis = np.zeros(n)
    for t in range(n):
        e *= (1.0 - p.loss_per_h)
        u = cmd_mw[t]
        if u > 0:
            pw = min(u, p.p_comp_mw, max(p.capacity_mwh - e, 0.0) / p.eta_c)
            e += pw * p.eta_c
            ch[t] = pw
        elif u < 0:
            pw = min(-u, p.p_turb_mw, max(e, 0.0) * p.eta_d)
            e -= pw / p.eta_d
            dis[t] = pw
        e = min(max(e, 0.0), p.capacity_mwh)
    gross = float(np.sum(price * (dis - ch)))
    costs = float(np.sum(c.fee_eur_mwh * (ch + dis) + c.vom_eur_mwh * dis))
    return e, ch, dis, gross, costs
