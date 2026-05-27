"""Regime classification helpers.

The three classifications used by entry logic:

- VIX percentile over its last 252 sessions, used to pick the short-delta target.
- VRP percentile over its last ``vrp_lookback_days`` days, used as an entry gate.
- 25-delta skew percentile over its last ``skew_lookback_days`` days, used for
  size reduction and exit triggering.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from vrp.pricing.surface import VolSurface

VolRegime = Literal["low", "mid", "high"]


def percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of ``series`` values less than ``value`` (in [0, 1]).

    Uses strict less-than so the percentile of the series's max is < 1.
    NaNs are dropped before the comparison.

    Args:
        series: Reference distribution.
        value: Value to rank.

    Returns:
        Percentile rank in [0, 1]. Returns 0.0 for an empty series.
    """
    arr = series.dropna().to_numpy()
    if arr.size == 0:
        return 0.0
    return float((arr < value).mean())


def classify_vix(vix_history: pd.Series, current_vix: float) -> VolRegime:
    """Map current VIX into a low/mid/high regime by 252d percentile.

    Cutoffs: ``< 25th -> low``, ``25-75 -> mid``, ``> 75th -> high``.
    """
    pct = percentile_rank(vix_history.tail(252), current_vix)
    if pct < 0.25:
        return "low"
    if pct > 0.75:
        return "high"
    return "mid"


@dataclass
class SkewSnapshot:
    """The 25-delta skew on a given date."""

    skew: float            # IV(25d put) - IV(25d call)
    skew_normalised: float # ... / IV(ATM)


def compute_25d_skew(
    surface: VolSurface,
    expiry: date,
    *,
    target_delta: float = 0.25,
) -> SkewSnapshot:
    """Compute (IV_{25d put} - IV_{25d call}) / IV_atm for one expiry.

    Strikes for the 25-delta wings are found numerically by scanning the
    surface's stored slice for the closest call/put strikes whose absolute
    delta matches ``target_delta``. The result is order-of-magnitude robust
    even if the spline is wiggly at the wings.

    Args:
        surface: Fitted vol surface.
        expiry: Expiry whose smile is being measured.
        target_delta: Target |delta| for the wings.

    Returns:
        :class:`SkewSnapshot` with raw and ATM-normalised skew.
    """
    from vrp.pricing import black_scholes as bsm  # local to avoid cycle in __init__

    spot = surface.spot
    r = surface.rate
    q = surface.div_yield
    T = max((expiry - surface.as_of).days / 365.0, 1e-6)

    iv_atm = surface.iv(spot, expiry)

    # Search a grid of moneyness for the strikes closest to target |delta|.
    moneyness = np.linspace(0.70, 1.30, 121)
    strikes = spot * moneyness
    put_ivs = np.array([surface.iv(k, expiry) for k in strikes])
    call_ivs = put_ivs
    put_deltas = np.array(
        [float(bsm.delta(spot, k, T, r, q, iv, "put")) for k, iv in zip(strikes, put_ivs)]
    )
    call_deltas = np.array(
        [float(bsm.delta(spot, k, T, r, q, iv, "call")) for k, iv in zip(strikes, call_ivs)]
    )

    p_idx = int(np.argmin(np.abs(put_deltas - (-target_delta))))
    c_idx = int(np.argmin(np.abs(call_deltas - target_delta)))
    iv_put_25 = float(put_ivs[p_idx])
    iv_call_25 = float(call_ivs[c_idx])
    skew = iv_put_25 - iv_call_25
    norm = skew / iv_atm if iv_atm > 0 else 0.0
    return SkewSnapshot(skew=skew, skew_normalised=norm)
