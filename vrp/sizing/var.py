"""Portfolio 1-day Value-at-Risk via historical simulation.

Each open position is repriced under each historical 1-day shock to its
underlying; positions on different tickers are aggregated via the
empirical correlation in their shocks (which is implicit in the joint
historical sample). The portfolio's 1-day P&L distribution is then taken
as the empirical distribution; VaR is the (1 - confidence) quantile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from vrp.persistence.models import Position
from vrp.pricing import black_scholes as bsm

logger = logging.getLogger(__name__)


def _spread_value(
    position: Position,
    *,
    underlying_price: float,
    iv_short: float,
    iv_long: float,
    as_of: date,
    rate: float,
    div_yield: float,
) -> float:
    """Mid-value (debit to close) of one credit-spread contract per share."""
    T = max((position.expiry - as_of).days / 365.0, 1e-6)
    p_short = float(bsm.price(underlying_price, position.short_strike, T, rate, div_yield, iv_short, "put"))
    p_long = float(bsm.price(underlying_price, position.long_strike, T, rate, div_yield, iv_long, "put"))
    return p_short - p_long


def historical_var(
    positions: list[Position],
    *,
    as_of: date,
    underlying_history: dict[str, pd.Series],
    iv_lookup,
    rate: float,
    dividend_yields: dict[str, float],
    confidence: float = 0.99,
    lookback: int = 252,
) -> float:
    """1-day historical-simulation VaR in dollars.

    Args:
        positions: Open positions to include.
        as_of: Valuation date.
        underlying_history: Map ticker -> close-price Series.
        iv_lookup: Callable ``(ticker, strike, expiry, as_of) -> iv``. The
            simulated 1-day shock keeps IV constant (a conservative-ish
            simplification: realistic VaR would shock IV too, but the
            spec asks for underlying-shock historical sim).
        rate: Risk-free rate.
        dividend_yields: Per-ticker dividend yield.
        confidence: Confidence level (e.g. 0.99 for 99% VaR).
        lookback: Number of trailing returns to use.

    Returns:
        VaR magnitude in dollars (positive). 0.0 if no positions or no data.
    """
    if not positions:
        return 0.0

    # Build the per-ticker return matrix (aligned to inner join).
    tickers = sorted({p.ticker for p in positions})
    pivot = pd.concat(
        {t: underlying_history[t] for t in tickers if t in underlying_history},
        axis=1,
    ).dropna()
    if pivot.empty:
        return 0.0
    rets = np.log(pivot / pivot.shift(1)).dropna().tail(lookback)
    if rets.empty:
        return 0.0

    # For each historical day, simulate underlying shock and price each position.
    base_prices: dict[str, float] = {}
    base_values: dict[int, float] = {}
    for p in positions:
        if p.ticker not in pivot.columns:
            continue
        base_prices[p.ticker] = float(pivot[p.ticker].iloc[-1])
        iv_s = iv_lookup(p.ticker, p.short_strike, p.expiry, as_of)
        iv_l = iv_lookup(p.ticker, p.long_strike, p.expiry, as_of)
        base_values[id(p)] = _spread_value(
            p,
            underlying_price=base_prices[p.ticker],
            iv_short=iv_s,
            iv_long=iv_l,
            as_of=as_of,
            rate=rate,
            div_yield=dividend_yields.get(p.ticker, 0.0),
        )

    pnls = np.zeros(len(rets))
    for i, day_returns in enumerate(rets.itertuples(index=False)):
        ret_by_t = dict(zip(rets.columns, day_returns))
        day_pnl = 0.0
        for p in positions:
            if p.ticker not in base_prices:
                continue
            shocked = base_prices[p.ticker] * np.exp(ret_by_t[p.ticker])
            iv_s = iv_lookup(p.ticker, p.short_strike, p.expiry, as_of)
            iv_l = iv_lookup(p.ticker, p.long_strike, p.expiry, as_of)
            shocked_val = _spread_value(
                p,
                underlying_price=shocked,
                iv_short=iv_s,
                iv_long=iv_l,
                as_of=as_of,
                rate=rate,
                div_yield=dividend_yields.get(p.ticker, 0.0),
            )
            # Short credit spread: P&L = (base_value - shocked_value) * 100 * contracts.
            pnl = (base_values[id(p)] - shocked_val) * 100.0 * p.contracts
            day_pnl += pnl
        pnls[i] = day_pnl

    q = float(np.quantile(pnls, 1.0 - confidence))
    return max(-q, 0.0)
