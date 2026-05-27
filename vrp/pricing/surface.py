"""Implied vol surface fitting.

Per-expiry: cubic spline in log-moneyness ``log(K/F)``.
Across expiries: linear in time-to-expiry, extrapolating flat at the wings.

The resulting :class:`VolSurface` answers ``iv(strike, expiry)`` queries used
by the entry / monitor / backtest engines, including at strikes that may not
have liquid quotes on the underlying date.

Surfaces are persistable to disk via pickle for backtest reuse — they are
small (KB-scale) and recomputation is the slow path.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


@dataclass
class _Slice:
    """One expiry's vol smile, interpolated in log-moneyness."""

    expiry: date
    T: float                  # years from valuation date
    forward: float
    spline: CubicSpline
    k_min: float
    k_max: float
    iv_at_k_min: float
    iv_at_k_max: float

    def iv_at_strike(self, strike: float) -> float:
        k = np.log(strike / self.forward)
        if k <= self.k_min:
            return self.iv_at_k_min
        if k >= self.k_max:
            return self.iv_at_k_max
        return float(self.spline(k))


@dataclass
class VolSurface:
    """Composite IV surface across expiries for one underlying on one date.

    Attributes:
        as_of: Valuation date.
        spot: Underlying spot used in forward construction.
        rate: Risk-free rate used in forward construction.
        div_yield: Dividend yield used in forward construction.
        slices: One per expiry, sorted ascending by T.
    """

    as_of: date
    spot: float
    rate: float
    div_yield: float
    slices: list[_Slice]

    def iv(self, strike: float, expiry: date) -> float:
        """IV for an arbitrary strike at an arbitrary expiry.

        Strikes outside any slice's log-moneyness range hold flat.
        Expiries outside the slice range hold flat at the nearest slice.

        Args:
            strike: Option strike.
            expiry: Option expiry (cash settle assumed).

        Returns:
            Interpolated implied volatility.

        Raises:
            ValueError: If the surface has no slices.
        """
        if not self.slices:
            raise ValueError("empty surface")
        T = max((expiry - self.as_of).days / 365.0, 1e-6)
        Ts = np.array([s.T for s in self.slices])
        if T <= Ts[0]:
            return self.slices[0].iv_at_strike(strike)
        if T >= Ts[-1]:
            return self.slices[-1].iv_at_strike(strike)
        # Bracket and linearly interpolate the two slice IVs at this strike.
        idx = int(np.searchsorted(Ts, T))
        s0, s1 = self.slices[idx - 1], self.slices[idx]
        iv0 = s0.iv_at_strike(strike)
        iv1 = s1.iv_at_strike(strike)
        w = (T - s0.T) / (s1.T - s0.T)
        return iv0 + w * (iv1 - iv0)


def fit_surface(
    chain: pd.DataFrame,
    *,
    as_of: date,
    spot: float,
    rate: float,
    div_yield: float,
) -> VolSurface:
    """Fit a vol surface from an options chain.

    Args:
        chain: DataFrame with at least columns ``strike``, ``expiry``, ``iv``.
            ``expiry`` may be ``datetime.date`` or pandas datetime.
        as_of: Valuation date.
        spot: Underlying spot.
        rate: Risk-free rate.
        div_yield: Dividend yield.

    Returns:
        Fitted :class:`VolSurface`.

    Raises:
        ValueError: If the chain is empty or has no valid IVs.
    """
    if chain.empty:
        raise ValueError("cannot fit surface from empty chain")

    work = chain.copy()
    work["expiry"] = pd.to_datetime(work["expiry"]).dt.date
    work = work.dropna(subset=["iv", "strike"])
    work = work[(work["iv"] > 0) & (work["iv"] < 5.0)]
    if work.empty:
        raise ValueError("no valid IV quotes in chain")

    slices: list[_Slice] = []
    for expiry, group in work.groupby("expiry"):
        T = max((expiry - as_of).days / 365.0, 1e-6)
        forward = spot * np.exp((rate - div_yield) * T)
        group = group.sort_values("strike").drop_duplicates("strike")
        ks = np.log(group["strike"].to_numpy(dtype=float) / forward)
        ivs = group["iv"].to_numpy(dtype=float)
        if len(ks) < 3:
            # Not enough points for cubic spline; pad with flat extrapolation.
            iv_const = float(ivs.mean())
            ks_pad = np.array([-0.5, 0.0, 0.5])
            ivs_pad = np.array([iv_const] * 3)
            spline = CubicSpline(ks_pad, ivs_pad, extrapolate=False)
            k_min, k_max = ks_pad[0], ks_pad[-1]
            iv_lo, iv_hi = iv_const, iv_const
        else:
            spline = CubicSpline(ks, ivs, extrapolate=False)
            k_min, k_max = float(ks[0]), float(ks[-1])
            iv_lo, iv_hi = float(ivs[0]), float(ivs[-1])
        slices.append(
            _Slice(
                expiry=expiry,
                T=T,
                forward=forward,
                spline=spline,
                k_min=k_min,
                k_max=k_max,
                iv_at_k_min=iv_lo,
                iv_at_k_max=iv_hi,
            )
        )

    slices.sort(key=lambda s: s.T)
    return VolSurface(as_of=as_of, spot=spot, rate=rate, div_yield=div_yield, slices=slices)


def save_surface(surface: VolSurface, path: Path) -> None:
    """Pickle a surface to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(surface, fh)


def load_surface(path: Path) -> VolSurface:
    """Unpickle a surface from disk."""
    with path.open("rb") as fh:
        return pickle.load(fh)
