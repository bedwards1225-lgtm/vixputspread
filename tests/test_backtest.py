"""Backtest end-to-end determinism with the BSM provider."""
from __future__ import annotations

from datetime import date

import pandas as pd

from vrp.backtest.engine import run_backtest
from vrp.config import Config
from vrp.data.providers import BlackScholesProvider


def test_backtest_deterministic_same_seed(fake_history_lookup):
    cfg = Config()
    # Restrict universe to one ticker to make the test fast.
    cfg.universe.tickers = ["SPY"]

    def make_provider():
        # Always inject a "VIX" series of 16 vol via the same fake lookup.
        def _lookup(ticker, start, end):
            df = fake_history_lookup(ticker, start, end)
            if ticker == "^VIX":
                df = df.copy()
                df["Close"] = 16.0
            return df
        return BlackScholesProvider(
            rate=cfg.pricing.risk_free_rate,
            dividend_yields=cfg.universe.dividend_yields,
            vix_scale=cfg.universe.vix_scale,
            skew_slope=cfg.pricing.bsm_skew_slope,
            history_lookup=_lookup,
        )

    start = date(2019, 1, 2)
    end = date(2019, 3, 1)

    r1 = run_backtest(cfg=cfg, provider=make_provider(), start=start, end=end, seed=42)
    r2 = run_backtest(cfg=cfg, provider=make_provider(), start=start, end=end, seed=42)

    assert len(r1.snapshots) == len(r2.snapshots)
    # Same number of trades.
    assert len(r1.trades) == len(r2.trades)
    # Equity curves match.
    eq1 = [s.portfolio_value for s in r1.snapshots]
    eq2 = [s.portfolio_value for s in r2.snapshots]
    assert eq1 == eq2
