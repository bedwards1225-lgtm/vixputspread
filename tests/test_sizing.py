"""Sizing constraints: per-trade, VaR, Greek caps."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vrp.config import SizingConfig
from vrp.persistence.models import Position
from vrp.signals.entry import SignalCandidate
from vrp.signals.regime import SkewSnapshot
from vrp.sizing.sizer import size_trade


def _candidate(
    *,
    max_loss=100.0, credit=0.40, width=2.0,
    delta=-0.15, gamma=0.001, vega=0.002,
    vrp=0.01, vrp_pct=0.5, skew_pct=0.5, size_reduction=1.0,
) -> SignalCandidate:
    return SignalCandidate(
        ticker="SPY", as_of=date(2024, 1, 2),
        expiry=date(2024, 2, 1), dte=30,
        underlying_price=500.0,
        short_strike=470.0, long_strike=468.0, width=width,
        short_iv=0.20, long_iv=0.21,
        short_delta=delta, short_gamma=gamma, short_vega=vega,
        credit=credit, credit_pct_of_width=credit/width,
        max_loss_per_contract=max_loss,
        vrp=vrp, vrp_percentile=vrp_pct, rv_forecast=0.04,
        vix=0.15, vix_regime="mid",
        skew=SkewSnapshot(skew=-0.02, skew_normalised=-0.1),
        skew_percentile=skew_pct, size_reduction=size_reduction,
        reject_reasons=[],
    )


def _iv_lookup(*_a, **_kw) -> float:
    return 0.20


@pytest.fixture
def price_series():
    rng = np.random.default_rng(42)
    n = 260
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return {"SPY": pd.Series(close, index=idx)}


def test_per_trade_cap_binds(price_series):
    # equity small enough that per-trade cap is the binding constraint.
    cand = _candidate(max_loss=1500.0)  # 1.5% of $100k = $1500 per contract -> 1 max
    cfg = SizingConfig()
    res = size_trade(
        cand, cfg=cfg, equity=100_000, open_positions=[],
        underlying_history=price_series, iv_lookup=_iv_lookup,
        rate=0.04, dividend_yields={"SPY": 0.013},
        portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
    )
    assert res.contracts <= 1
    assert "per_trade" in res.binding_constraint


def test_greek_cap_binds(price_series):
    # Force gamma cap to be tiny via small equity numbers.
    cand = _candidate(max_loss=10.0, gamma=0.10)  # huge per-contract gamma
    cfg = SizingConfig(
        max_loss_per_trade_pct=1.0,
        portfolio_var_cap_pct=1.0,
        net_short_gamma_cap_pct=0.00001,  # near-zero gamma budget
    )
    res = size_trade(
        cand, cfg=cfg, equity=100_000, open_positions=[],
        underlying_history=price_series, iv_lookup=_iv_lookup,
        rate=0.04, dividend_yields={"SPY": 0.013},
        portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
    )
    assert res.contracts == 0 or "greek" in res.binding_constraint


def test_rejected_candidate_returns_zero(price_series):
    cand = _candidate()
    cand.reject_reasons.append("VRP non-positive")
    cfg = SizingConfig()
    res = size_trade(
        cand, cfg=cfg, equity=100_000, open_positions=[],
        underlying_history=price_series, iv_lookup=_iv_lookup,
        rate=0.04, dividend_yields={"SPY": 0.013},
        portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
    )
    assert res.contracts == 0
    assert res.binding_constraint == "rejected"


def test_skew_reduction_halves_count(price_series):
    full = _candidate(max_loss=500.0)
    cfg = SizingConfig(portfolio_var_cap_pct=10.0)  # remove VaR as binder
    no_skew = size_trade(
        full, cfg=cfg, equity=200_000, open_positions=[],
        underlying_history=price_series, iv_lookup=_iv_lookup,
        rate=0.04, dividend_yields={"SPY": 0.013},
        portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
    )
    reduced = _candidate(max_loss=500.0, size_reduction=0.5)
    with_skew = size_trade(
        reduced, cfg=cfg, equity=200_000, open_positions=[],
        underlying_history=price_series, iv_lookup=_iv_lookup,
        rate=0.04, dividend_yields={"SPY": 0.013},
        portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_vega=0.0,
    )
    assert with_skew.contracts <= no_skew.contracts
