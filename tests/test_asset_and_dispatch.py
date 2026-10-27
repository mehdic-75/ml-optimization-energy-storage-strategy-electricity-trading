"""Unit tests for the physical simulator and the DP dispatcher."""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caes_bt.asset import CAESParams, Costs, simulate_and_settle  
from caes_bt.dispatch import DPDispatcher  

P = CAESParams()
C = Costs()


def test_cannot_sell_from_an_empty_store_or_buy_into_a_full_one():
    e, ch, dis, gross, costs = simulate_and_settle(np.array([-120.0] * 3), np.array([500.0] * 3), 0.0, P, C)
    assert dis.sum() == 0 and gross == 0 and costs == 0 and e == 0
    p = CAESParams(loss_per_h=0.0)  # without self-discharge a full store has no headroom at all
    e, ch, dis, gross, costs = simulate_and_settle(np.array([70.0] * 3), np.array([-50.0] * 3), 200.0, p, C)
    assert ch.sum() == 0 and gross == 0 and e == 200.0


def test_energy_balance_and_bounds_under_random_commands():
    rng = np.random.default_rng(1)
    cmd = rng.uniform(-200, 150, 500)
    price = rng.normal(80, 40, 500)
    p = CAESParams(loss_per_h=0.0)
    e_end, ch, dis, gross, costs = simulate_and_settle(cmd, price, p.e0_mwh, p, C)
    assert (ch <= p.p_comp_mw + 1e-9).all() and (dis <= p.p_turb_mw + 1e-9).all()
    assert np.isclose(e_end, p.e0_mwh + p.eta_c * ch.sum() - dis.sum() / p.eta_d)
    assert np.isclose(gross, np.sum(price * (dis - ch)))
    assert np.isclose(costs, np.sum(C.fee_eur_mwh * (ch + dis) + C.vom_eur_mwh * dis))


def test_dp_matches_brute_force_on_a_small_instance():
    p = CAESParams(p_comp_mw=2.5, p_turb_mw=2.0, capacity_mwh=4.0, eta_c=0.9, eta_d=0.9, e0_mwh=1.0)
    dp = DPDispatcher(p, C, step_mwh=1.0)
    prices = np.array([20.0, -5.0, 90.0, 40.0, 150.0, 10.0])
    best = -np.inf
    for path in itertools.product(range(len(dp.grid)), repeat=len(prices)):
        i, v = 1, 0.0
        for t, j in enumerate(path):
            v += dp.rewards(prices[t])[i, j]
            i = j
        best = max(best, v)
    cmd = dp.solve(prices, p.e0_mwh)
    buy, sell = np.clip(cmd, 0, None), np.clip(-cmd, 0, None)
    value = np.sum((prices - C.fee_eur_mwh - C.vom_eur_mwh) * sell - (prices + C.fee_eur_mwh) * buy)
    assert np.isclose(value, best)


def test_dp_does_not_trade_a_spread_that_does_not_cover_losses():
    dp = DPDispatcher(P, C)
    # spread 60 -> 75: selling 1 MWh needs 1/0.748 = 1.34 MWh bought, i.e. 80 EUR > 75 EUR
    prices = np.array([60.0] * 12 + [75.0] * 12)
    assert np.allclose(dp.solve(prices, 0.0), 0.0)
    # a wide enough spread is traded
    prices = np.array([20.0] * 12 + [120.0] * 12)
    assert dp.solve(prices, 0.0).max() > 0
