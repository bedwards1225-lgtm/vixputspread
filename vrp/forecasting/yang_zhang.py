"""Yang-Zhang (2000) volatility estimator on OHLC bars.

The Yang-Zhang estimator combines overnight, opening-jump, and Rogers-Satchell
intraday variance into a single drift-and-jump-robust estimate. It is the
most efficient estimator obtainable from daily OHLC data alone.

Reference:
    Yang, Dennis & Zhang, Qiang (2000). "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices."
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def yang_zhang_variance(ohlc: pd.DataFrame, window: int = 22) -> pd.Series:
    """Rolling-window Yang-Zhang variance, annualised.

    Args:
        ohlc: DataFrame indexed by date with columns ``Open``, ``High``,
            ``Low``, ``Close``.
        window: Rolling window length in trading days (>= 2 for the full
            estimator). When ``window == 1`` the function returns the
            Rogers-Satchell single-day variance, annualised, since the
            multi-day open/close variances are undefined.

    Returns:
        Series of annualised variance estimates, indexed by date.

    Raises:
        KeyError: If required OHLC columns are missing.
        ValueError: If ``window < 1``.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(ohlc.columns)
    if missing:
        raise KeyError(f"missing OHLC columns: {sorted(missing)}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    o = ohlc["Open"].astype(float)
    h = ohlc["High"].astype(float)
    l = ohlc["Low"].astype(float)
    c = ohlc["Close"].astype(float)

    # Rogers-Satchell intraday variance (drift-free), per-day.
    rs = (np.log(h / c) * np.log(h / o)) + (np.log(l / c) * np.log(l / o))

    if window == 1:
        # Fall back to single-day Rogers-Satchell — open/close variances are
        # undefined with a 1-observation rolling window.
        return rs * _TRADING_DAYS

    c_prev = c.shift(1)
    on = np.log(o / c_prev)
    open_var = on.rolling(window).var(ddof=1)
    co = np.log(c / o)
    close_var = co.rolling(window).var(ddof=1)
    rs_var = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    daily_var = open_var + k * close_var + (1.0 - k) * rs_var
    return daily_var * _TRADING_DAYS


def yang_zhang_vol(ohlc: pd.DataFrame, window: int = 22) -> pd.Series:
    """Yang-Zhang volatility (sqrt of variance), annualised."""
    return np.sqrt(yang_zhang_variance(ohlc, window=window))
