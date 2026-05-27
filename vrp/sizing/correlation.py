"""Rolling correlation matrix of underlying log-returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns_matrix(prices_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    """Build an aligned DataFrame of daily log-returns.

    Args:
        prices_by_ticker: Map ticker -> close-price Series (datetime index).

    Returns:
        DataFrame whose columns are tickers and rows are dates where ALL
        tickers have a return (inner join).
    """
    aligned = pd.concat(prices_by_ticker, axis=1).dropna()
    rets = np.log(aligned / aligned.shift(1)).dropna()
    return rets


def correlation_matrix(
    prices_by_ticker: dict[str, pd.Series], *, lookback: int = 60
) -> pd.DataFrame:
    """Compute the most recent ``lookback``-day correlation matrix.

    Args:
        prices_by_ticker: Map ticker -> close-price Series.
        lookback: Number of trailing days to use.

    Returns:
        DataFrame of pairwise Pearson correlations. Returns 1x1 identity
        if only one ticker is supplied.
    """
    if len(prices_by_ticker) == 0:
        return pd.DataFrame()
    if len(prices_by_ticker) == 1:
        t = next(iter(prices_by_ticker))
        return pd.DataFrame([[1.0]], index=[t], columns=[t])
    rets = returns_matrix(prices_by_ticker).tail(lookback)
    return rets.corr()
