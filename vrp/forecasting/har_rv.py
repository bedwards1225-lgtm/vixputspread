"""HAR-RV (Corsi 2009) realized variance forecasting.

Model:

    RV_{t+h} = beta_0 + beta_d * RV_d(t) + beta_w * RV_w(t) + beta_m * RV_m(t)

where ``RV_d`` is daily RV, ``RV_w`` is the 5-day mean, and ``RV_m`` is the
22-day mean. Fitting uses OLS on a rolling window. Strict causality is
enforced: the fit on day ``T`` uses only data with index <= ``T``.

For daily-data users we approximate RV with the Yang-Zhang variance
estimator over a short window (default 1 trading day). This is the
substitute the spec calls out when 5-minute data is unavailable.

Reference:
    Corsi, Fulvio (2009). "A Simple Approximate Long-Memory Model of
    Realized Volatility." Journal of Financial Econometrics, 7(2), 174-196.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from vrp.forecasting.yang_zhang import yang_zhang_variance
from vrp.persistence.models import HarRvFit

logger = logging.getLogger(__name__)


@dataclass
class HarFeatures:
    """Lagged HAR-RV regressors and target at each date."""

    rv_daily: pd.Series
    rv_weekly: pd.Series
    rv_monthly: pd.Series


def build_features(ohlc: pd.DataFrame, daily_window: int = 1) -> HarFeatures:
    """Compute (RV_d, RV_w, RV_m) for each date, using Yang-Zhang as the RV proxy.

    Args:
        ohlc: OHLC DataFrame indexed by date.
        daily_window: Window for the "daily" RV (1 = pure single-day YZ; using
            5 smooths some noise but trades against the spec's RV_d definition).

    Returns:
        :class:`HarFeatures` with three aligned Series.
    """
    rv_d = yang_zhang_variance(ohlc, window=max(daily_window, 1))
    rv_w = rv_d.rolling(5).mean()
    rv_m = rv_d.rolling(22).mean()
    return HarFeatures(rv_daily=rv_d, rv_weekly=rv_w, rv_monthly=rv_m)


def fit(
    ohlc: pd.DataFrame,
    *,
    as_of: date,
    horizon_days: int,
    window_days: int = 252,
) -> Optional[HarRvFit]:
    """Fit HAR-RV via OLS over a rolling window ending on or before ``as_of``.

    The regression target is the realized variance over the *next* ``horizon_days``,
    aligned so that the regression at index ``t`` predicts ``mean(RV_{t+1..t+h})``.

    Args:
        ohlc: OHLC DataFrame indexed by ``date``.
        as_of: Last date allowed in the feature set (strict no-look-ahead).
        horizon_days: Forecast horizon ``h`` in trading days.
        window_days: Rolling window length for the OLS fit.

    Returns:
        A :class:`HarRvFit` if enough observations exist, else ``None``.
    """
    if "Close" not in ohlc.columns:
        raise KeyError("ohlc must contain a 'Close' column")

    # Filter strictly to data on or before as_of.
    work = ohlc.loc[ohlc.index <= pd.Timestamp(as_of)]
    if len(work) < window_days + horizon_days + 22:
        return None

    feats = build_features(work)
    rv_daily = feats.rv_daily

    # Target: mean RV over the next ``horizon_days`` (shift -h then rolling mean
    # of size h, taken from the same point so it's strictly forward-looking).
    forward_mean = rv_daily.shift(-horizon_days).rolling(horizon_days).mean()
    # The above is the mean of RV_{t-h+1..t} shifted to t-h -> i.e. mean of RV
    # over (t+1, t+h). Equivalent and simpler:
    forward_mean = (
        rv_daily.shift(-1).rolling(horizon_days).mean().shift(-(horizon_days - 1))
    )

    df = pd.DataFrame(
        {
            "y": forward_mean,
            "rv_d": feats.rv_daily,
            "rv_w": feats.rv_weekly,
            "rv_m": feats.rv_monthly,
        }
    ).dropna()

    # Use the last ``window_days`` rows that have all features AND a target.
    # Crucially: the most recent rows where ``forward_mean`` is still defined.
    df = df.iloc[-window_days:]
    if len(df) < 30:
        return None

    X = np.column_stack(
        [np.ones(len(df)), df["rv_d"].to_numpy(), df["rv_w"].to_numpy(), df["rv_m"].to_numpy()]
    )
    y = df["y"].to_numpy()
    # OLS via lstsq for numerical stability.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return HarRvFit(
        fit_date=as_of,
        beta_0=float(beta[0]),
        beta_daily=float(beta[1]),
        beta_weekly=float(beta[2]),
        beta_monthly=float(beta[3]),
        r_squared=float(r2),
        n_obs=int(len(df)),
    )


def forecast(
    fit_result: HarRvFit,
    ohlc: pd.DataFrame,
    as_of: date,
) -> float:
    """Apply a fitted HAR-RV model to the latest features.

    Args:
        fit_result: Output of :func:`fit`.
        ohlc: OHLC DataFrame indexed by date.
        as_of: Date at which to compute the features (uses data <= as_of).

    Returns:
        Forecasted annualised realized variance (decimal squared).
    """
    work = ohlc.loc[ohlc.index <= pd.Timestamp(as_of)]
    feats = build_features(work)
    rv_d = float(feats.rv_daily.iloc[-1])
    rv_w = float(feats.rv_weekly.iloc[-1])
    rv_m = float(feats.rv_monthly.iloc[-1])
    pred = (
        fit_result.beta_0
        + fit_result.beta_daily * rv_d
        + fit_result.beta_weekly * rv_w
        + fit_result.beta_monthly * rv_m
    )
    return max(pred, 0.0)
