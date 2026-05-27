"""HAR-RV parameter recovery and no-look-ahead causality."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vrp.forecasting import har_rv


def _ohlc_with_vol(daily_sigma: np.ndarray, *, seed: int = 0) -> pd.DataFrame:
    """Synthesize OHLC where each day's intraday range tracks ``daily_sigma[t]``.

    We simulate ``steps`` intraday increments per day with std = sigma/sqrt(steps)
    so the close-to-close return has the target variance AND the high/low
    realistically reflect the realised path. Yang-Zhang can then estimate sigma_d.
    """
    rng = np.random.default_rng(seed)
    n = len(daily_sigma)
    steps = 78  # ~ 5-minute bars in 6.5h
    dates = pd.bdate_range("2018-01-01", periods=n)
    closes = np.empty(n)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    prev_close = 100.0
    for t in range(n):
        step_std = daily_sigma[t] / np.sqrt(steps)
        increments = rng.normal(0.0, step_std, steps)
        log_path = np.cumsum(increments)
        intraday = prev_close * np.exp(log_path)
        opens[t] = prev_close   # assume open == prior close (no overnight gap)
        closes[t] = intraday[-1]
        highs[t] = max(intraday.max(), opens[t], closes[t])
        lows[t] = min(intraday.min(), opens[t], closes[t])
        prev_close = closes[t]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 0},
        index=dates,
    )


def test_har_recovers_known_parameters():
    """HAR-RV fit on synthetic data with known structure: betas should be positive
    and meaningfully > 0, R^2 should be positive."""
    rng = np.random.default_rng(0)
    n = 700
    sigma_d = np.full(n, 0.012)  # ~12 vol per day baseline
    # Persistent vol regime: shock at day 200 and 450.
    for t in range(22, n):
        rv_d = sigma_d[t - 1] ** 2
        rv_w = (sigma_d[t - 5:t] ** 2).mean()
        rv_m = (sigma_d[t - 22:t] ** 2).mean()
        true_var = 0.00005 + 0.40 * rv_d + 0.30 * rv_w + 0.20 * rv_m
        sigma_d[t] = np.sqrt(max(true_var, 1e-8)) * np.exp(rng.normal(0, 0.1))
    ohlc = _ohlc_with_vol(sigma_d, seed=0)

    fit = har_rv.fit(ohlc, as_of=ohlc.index[-1].date(), horizon_days=5, window_days=500)
    assert fit is not None
    # Persistent vol -> daily / weekly / monthly betas should each be > 0.
    total_beta = fit.beta_daily + fit.beta_weekly + fit.beta_monthly
    assert total_beta > 0.3, f"sum of HAR betas too low: {total_beta:.3f}"
    assert fit.r_squared > 0.05, f"R^2 too low: {fit.r_squared:.3f}"


def test_no_look_ahead_truncates_features():
    """fit() on day T must not see any data after T."""
    rng = np.random.default_rng(1)
    n = 600
    sigma_d = np.full(n, 0.012)
    ohlc = _ohlc_with_vol(sigma_d, seed=1)
    full_last = ohlc.index[-1].date()
    earlier = ohlc.index[-100].date()

    fit_full = har_rv.fit(ohlc, as_of=full_last, horizon_days=5, window_days=252)
    fit_early = har_rv.fit(ohlc, as_of=earlier, horizon_days=5, window_days=252)

    assert fit_full is not None and fit_early is not None
    # Verify the early fit literally cannot have used post-`earlier` data by
    # rerunning it with truncated OHLC and checking exact equality.
    truncated = ohlc.loc[ohlc.index <= pd.Timestamp(earlier)]
    fit_truncated = har_rv.fit(truncated, as_of=earlier, horizon_days=5, window_days=252)
    assert fit_truncated is not None
    assert fit_early.beta_daily == fit_truncated.beta_daily
    assert fit_early.beta_weekly == fit_truncated.beta_weekly
    assert fit_early.beta_monthly == fit_truncated.beta_monthly


def test_fit_returns_none_when_insufficient_data():
    rng = np.random.default_rng(2)
    n = 50
    sigma_d = np.full(n, 0.012)
    ohlc = _ohlc_with_vol(sigma_d, seed=2)
    fit = har_rv.fit(ohlc, as_of=ohlc.index[-1].date(), horizon_days=5, window_days=252)
    assert fit is None
