"""Entry signal generation: VRP-conditional put credit spread proposals.

The pipeline (per ticker, per expiry):

  1. Forecast realized variance for the trade horizon (HAR-RV).
  2. Pick a candidate short-delta target from VIX regime.
  3. Find the short strike on the surface that matches the delta target.
  4. Pick a long strike width such that credit >= target % of width.
  5. Compute the VRP at the short strike. Reject if VRP <= 0 or < 30th pct.
  6. Apply skew filter: if 25d skew is in top 10%, halve the size.
  7. Reject if credit < min_credit_pct_of_width.

Output is a :class:`SignalCandidate` — sizing is applied separately by
:mod:`vrp.sizing.sizer`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from vrp.config import Config
from vrp.persistence.models import HarRvFit
from vrp.pricing import black_scholes as bsm
from vrp.pricing.surface import VolSurface
from vrp.signals.regime import (
    SkewSnapshot,
    VolRegime,
    classify_vix,
    compute_25d_skew,
    percentile_rank,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalCandidate:
    """A single proposed credit spread, fully diagnosed but not yet sized."""

    ticker: str
    as_of: date
    expiry: date
    dte: int
    underlying_price: float
    short_strike: float
    long_strike: float
    width: float
    short_iv: float
    long_iv: float
    short_delta: float
    short_gamma: float
    short_vega: float
    credit: float
    credit_pct_of_width: float
    max_loss_per_contract: float
    vrp: float
    vrp_percentile: float
    rv_forecast: float
    vix: float
    vix_regime: VolRegime
    skew: SkewSnapshot
    skew_percentile: float
    size_reduction: float
    reject_reasons: list[str]

    @property
    def accepted(self) -> bool:
        """True iff no reject reasons were appended."""
        return not self.reject_reasons


def _delta_target_band(cfg: Config, regime: VolRegime) -> tuple[float, float]:
    if regime == "low":
        return cfg.entry.delta_target_low_vol
    if regime == "high":
        return cfg.entry.delta_target_high_vol
    return cfg.entry.delta_target_mid_vol


def _short_strike_for_delta(
    surface: VolSurface,
    expiry: date,
    target_delta: float,
    *,
    rate: float,
    div_yield: float,
) -> tuple[float, float]:
    """Find the put strike whose |delta| equals ``target_delta``.

    Returns:
        ``(strike, iv)`` tuple.

    Raises:
        ValueError: If no bracketing strike pair can be found.
    """
    spot = surface.spot
    T = max((expiry - surface.as_of).days / 365.0, 1e-6)

    def f(k: float) -> float:
        iv = surface.iv(k, expiry)
        d = float(bsm.delta(spot, k, T, rate, div_yield, iv, "put"))
        return d + target_delta   # delta + target = 0 at the right strike

    lo = spot * 0.50
    hi = spot * 1.00
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # Edge case: surface so flat that target delta is outside range.
        # Return whichever endpoint is closer.
        if abs(f_lo) < abs(f_hi):
            return lo, surface.iv(lo, expiry)
        return hi, surface.iv(hi, expiry)
    k = brentq(f, lo, hi, xtol=1e-3)
    return k, surface.iv(k, expiry)


def _spread_credit(
    surface: VolSurface,
    expiry: date,
    short_k: float,
    long_k: float,
    *,
    rate: float,
    div_yield: float,
) -> tuple[float, float, float]:
    """Compute credit and the two leg IVs for a put credit spread.

    Returns:
        ``(credit, iv_short, iv_long)`` per share (multiply by 100 for $).
    """
    spot = surface.spot
    T = max((expiry - surface.as_of).days / 365.0, 1e-6)
    iv_s = surface.iv(short_k, expiry)
    iv_l = surface.iv(long_k, expiry)
    p_short = float(bsm.price(spot, short_k, T, rate, div_yield, iv_s, "put"))
    p_long = float(bsm.price(spot, long_k, T, rate, div_yield, iv_l, "put"))
    return p_short - p_long, iv_s, iv_l


def propose_candidate(
    *,
    cfg: Config,
    ticker: str,
    as_of: date,
    surface: VolSurface,
    underlying_history: pd.DataFrame,
    vix_history: pd.Series,
    vrp_history: pd.Series,
    skew_history: pd.Series,
    har_fit: HarRvFit,
    rate: float,
    div_yield: float,
) -> Optional[SignalCandidate]:
    """Build the best candidate spread on ``ticker`` for ``as_of``, or None.

    Args:
        cfg: Full strategy config.
        ticker: Underlying symbol.
        as_of: Valuation date.
        surface: Fitted vol surface on ``as_of``.
        underlying_history: OHLC up to and including ``as_of`` (causal).
        vix_history: Daily VIX close, up to and including ``as_of``.
        vrp_history: Recent VRP values (squared vol units) for percentile gate.
        skew_history: Recent 25d skew values for percentile and reversal logic.
        har_fit: Most recent valid HAR-RV fit.
        rate: Risk-free rate.
        div_yield: Continuous dividend yield.

    Returns:
        A :class:`SignalCandidate` (which may be ``accepted=False`` with
        ``reject_reasons`` populated), or ``None`` if no valid expiry exists.
    """
    current_vix = float(vix_history.iloc[-1]) / 100.0
    regime = classify_vix(vix_history.tail(252) / 100.0, current_vix)
    delta_lo, delta_hi = _delta_target_band(cfg, regime)
    target_delta = 0.5 * (delta_lo + delta_hi)
    spot = surface.spot

    # Pick the expiry whose DTE is closest to midpoint of [min_dte, max_dte].
    target_dte = 0.5 * (cfg.entry.min_dte + cfg.entry.max_dte)
    candidates_exp = [
        s.expiry for s in surface.slices
        if cfg.entry.min_dte <= (s.expiry - as_of).days <= cfg.entry.max_dte
    ]
    if not candidates_exp:
        return None
    expiry = min(candidates_exp, key=lambda e: abs((e - as_of).days - target_dte))
    dte = (expiry - as_of).days

    # Short strike from delta target.
    try:
        short_k, short_iv = _short_strike_for_delta(
            surface, expiry, target_delta, rate=rate, div_yield=div_yield
        )
    except ValueError as exc:
        logger.warning("strike root-find failed for %s %s: %s", ticker, expiry, exc)
        return None

    # Pick a width: walk candidate widths (~3% steps of spot) downward,
    # accept the smallest width whose credit fraction >= target.
    width_grid = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.075]) * spot
    width_grid = np.round(width_grid, 2)
    best = None
    for w in width_grid:
        long_k = round(short_k - w, 2)
        if long_k <= 0:
            continue
        credit, iv_s, iv_l = _spread_credit(
            surface, expiry, short_k, long_k, rate=rate, div_yield=div_yield
        )
        if credit <= 0:
            continue
        pct = credit / w
        if pct >= cfg.entry.target_credit_pct_of_width:
            best = (w, long_k, credit, iv_s, iv_l)
            break
    if best is None:
        # Fall back to widest considered if no width hit target.
        w = float(width_grid[-1])
        long_k = round(short_k - w, 2)
        credit, iv_s, iv_l = _spread_credit(
            surface, expiry, short_k, long_k, rate=rate, div_yield=div_yield
        )
        best = (w, long_k, credit, iv_s, iv_l)
    width, long_k, credit, iv_s, iv_l = best
    credit_pct = credit / width if width > 0 else 0.0
    max_loss = max(width - credit, 0.0)

    # Greeks at the short strike.
    T = max(dte / 365.0, 1e-6)
    d_short = float(bsm.delta(spot, short_k, T, rate, div_yield, iv_s, "put"))
    g_short = float(bsm.gamma(spot, short_k, T, rate, div_yield, iv_s))
    v_short = float(bsm.vega(spot, short_k, T, rate, div_yield, iv_s)) * 0.01

    # VRP: IV^2 - RV_forecast^2 at the short strike's IV.
    rv_forecast = max(0.0, float(har_fit.beta_0
                                  + har_fit.beta_daily * underlying_history.attrs.get("rv_d", 0.0)
                                  + har_fit.beta_weekly * underlying_history.attrs.get("rv_w", 0.0)
                                  + har_fit.beta_monthly * underlying_history.attrs.get("rv_m", 0.0)))
    # Note: underlying_history.attrs is populated by the caller with current
    # feature values; this avoids re-computing them inside the loop.
    vrp = iv_s ** 2 - rv_forecast

    vrp_pct = percentile_rank(vrp_history.tail(cfg.entry.vrp_lookback_days), vrp)

    # Skew snapshot + percentile.
    skew = compute_25d_skew(surface, expiry)
    skew_pct = percentile_rank(skew_history.tail(cfg.entry.skew_lookback_days), skew.skew_normalised)
    size_reduction = 1.0
    if skew_pct >= cfg.entry.skew_high_percentile:
        size_reduction = cfg.entry.skew_size_reduction

    reasons: list[str] = []
    if vrp <= 0:
        reasons.append(f"VRP non-positive ({vrp:.4f})")
    if vrp_pct < cfg.entry.vrp_percentile_min:
        reasons.append(f"VRP percentile {vrp_pct:.0%} below {cfg.entry.vrp_percentile_min:.0%}")
    if credit_pct < cfg.entry.min_credit_pct_of_width:
        reasons.append(f"credit {credit_pct:.0%} below floor {cfg.entry.min_credit_pct_of_width:.0%}")
    if max_loss <= 0:
        reasons.append("non-positive max loss")

    return SignalCandidate(
        ticker=ticker,
        as_of=as_of,
        expiry=expiry,
        dte=dte,
        underlying_price=spot,
        short_strike=short_k,
        long_strike=long_k,
        width=width,
        short_iv=iv_s,
        long_iv=iv_l,
        short_delta=d_short,
        short_gamma=g_short,
        short_vega=v_short,
        credit=credit,
        credit_pct_of_width=credit_pct,
        max_loss_per_contract=max_loss * 100,
        vrp=vrp,
        vrp_percentile=vrp_pct,
        rv_forecast=rv_forecast,
        vix=current_vix,
        vix_regime=regime,
        skew=skew,
        skew_percentile=skew_pct,
        size_reduction=size_reduction,
        reject_reasons=reasons,
    )
