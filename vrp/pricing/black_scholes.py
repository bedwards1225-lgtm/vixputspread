"""Black-Scholes-Merton pricing and Greeks for European options on a
dividend-paying underlying.

All functions are vectorised over the inputs (numpy broadcasting). The
return-type convention is:

- ``delta`` is per share (range [-1, 1]).
- ``gamma`` is per share, per $1 underlying move.
- ``vega`` is per share, per 1.00 vol unit (i.e. per 100 vol points). Multiply
  by 0.01 to get the conventional "per vol point" quote.
- ``theta`` is per share, per calendar year. Divide by 365 for per-day theta.

Tenor ``T`` is in years (act/365).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]
_EPS = 1e-12


def _d1_d2(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    q: float,
    sigma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    q: float,
    sigma: float | np.ndarray,
    option_type: OptionType,
) -> np.ndarray:
    """Black-Scholes-Merton option price per share.

    Args:
        S: Spot.
        K: Strike.
        T: Time to expiry in years.
        r: Continuously-compounded risk-free rate.
        q: Continuous dividend yield.
        sigma: Implied vol (decimal, e.g. 0.20).
        option_type: ``"call"`` or ``"put"``.

    Returns:
        Option price per share as a numpy array (0-d for scalar inputs).
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    if option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError(f"unknown option_type {option_type!r}")


def delta(S, K, T, r, q, sigma, option_type: OptionType) -> np.ndarray:
    """Black-Scholes delta per share."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    if option_type == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError(f"unknown option_type {option_type!r}")


def gamma(S, K, T, r, q, sigma) -> np.ndarray:
    """Black-Scholes gamma per share (identical for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma) -> np.ndarray:
    """Black-Scholes vega per share, per 1.00 vol unit.

    Multiply by 0.01 for per-vol-point.
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, option_type: OptionType) -> np.ndarray:
    """Black-Scholes theta per share, per calendar year. Divide by 365 for daily."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    first = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return first - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    if option_type == "put":
        return first + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    raise ValueError(f"unknown option_type {option_type!r}")


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Recover implied vol from an option price via bisection.

    Bisection is slower than Newton but more robust at the wings, where
    Newton can step into negative vol.

    Args:
        target_price: Observed option price per share.
        S, K, T, r, q: BSM inputs.
        option_type: ``"call"`` or ``"put"``.
        tol: Absolute price tolerance.
        max_iter: Maximum bisection iterations.

    Returns:
        Implied volatility. Returns ``float('nan')`` if target is outside
        the no-arb bounds.
    """
    intrinsic = max(
        (S * np.exp(-q * T) - K * np.exp(-r * T)) if option_type == "call"
        else (K * np.exp(-r * T) - S * np.exp(-q * T)),
        0.0,
    )
    if target_price < intrinsic - tol:
        return float("nan")

    lo, hi = 1e-4, 5.0
    p_lo = float(price(S, K, T, r, q, lo, option_type))
    p_hi = float(price(S, K, T, r, q, hi, option_type))
    if not (p_lo - tol <= target_price <= p_hi + tol):
        if target_price > p_hi:
            return hi
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = float(price(S, K, T, r, q, mid, option_type))
        if abs(p_mid - target_price) < tol:
            return mid
        if p_mid < target_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
