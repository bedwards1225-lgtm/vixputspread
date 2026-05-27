"""Portfolio-level Greek aggregation.

A position's Greek is computed at current spot/IV/T and scaled by signed
contracts * 100 (US equity option multiplier).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from vrp.persistence.models import Position
from vrp.pricing import black_scholes as bsm

_MULTIPLIER = 100  # US equity options


@dataclass
class PositionGreeks:
    """Greeks for a single position, expressed as $ per unit move."""

    delta: float        # $ change per $1 underlying move
    gamma: float        # $ change in delta per $1 underlying move
    vega: float         # $ change per 1 vol POINT (e.g. VIX 15->16)
    theta: float        # $ change per calendar day


@dataclass
class PortfolioGreeks:
    """Sum of position Greeks across an arbitrary set of positions."""

    delta: float
    gamma: float
    vega: float
    theta: float


def position_greeks(
    position: Position,
    *,
    underlying_price: float,
    iv_short: float,
    iv_long: float,
    as_of: date,
    rate: float,
    div_yield: float,
) -> PositionGreeks:
    """Compute current dollar Greeks for a two-leg vertical spread or hedge.

    Args:
        position: Position with two legs (short + long for spreads, single
            long leg replicated for tail-hedge "spreads" is also supported by
            using ``contracts=0`` on the second leg).
        underlying_price: Current spot.
        iv_short: Implied vol of the short leg.
        iv_long: Implied vol of the long leg.
        as_of: Date used to compute time to expiry.
        rate: Risk-free rate.
        div_yield: Continuous dividend yield.

    Returns:
        Dollar-scaled Greeks for the whole position.
    """
    T = max((position.expiry - as_of).days / 365.0, 1e-6)
    contracts = position.contracts
    sign_short = -1.0   # short leg
    sign_long = +1.0    # long leg

    # Determine option types from legs if available; default to puts.
    short_type = "put"
    long_type = "put"
    if position.legs:
        for leg in position.legs:
            if leg.side == "short":
                short_type = leg.option_type
            else:
                long_type = leg.option_type

    d_short = float(bsm.delta(underlying_price, position.short_strike, T, rate, div_yield, iv_short, short_type))
    g_short = float(bsm.gamma(underlying_price, position.short_strike, T, rate, div_yield, iv_short))
    v_short = float(bsm.vega(underlying_price, position.short_strike, T, rate, div_yield, iv_short))
    t_short = float(bsm.theta(underlying_price, position.short_strike, T, rate, div_yield, iv_short, short_type))

    d_long = float(bsm.delta(underlying_price, position.long_strike, T, rate, div_yield, iv_long, long_type))
    g_long = float(bsm.gamma(underlying_price, position.long_strike, T, rate, div_yield, iv_long))
    v_long = float(bsm.vega(underlying_price, position.long_strike, T, rate, div_yield, iv_long))
    t_long = float(bsm.theta(underlying_price, position.long_strike, T, rate, div_yield, iv_long, long_type))

    scale = contracts * _MULTIPLIER
    delta = scale * (sign_short * d_short + sign_long * d_long)
    gamma = scale * (sign_short * g_short + sign_long * g_long)
    # vega: per vol POINT, so multiply numpy vega (per 1.00 vol unit) by 0.01.
    vega = scale * 0.01 * (sign_short * v_short + sign_long * v_long)
    # theta: per day, so divide annualised by 365.
    theta = scale * (sign_short * t_short + sign_long * t_long) / 365.0

    return PositionGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta)


def aggregate(greeks: list[PositionGreeks]) -> PortfolioGreeks:
    """Sum a list of :class:`PositionGreeks`."""
    if not greeks:
        return PortfolioGreeks(0.0, 0.0, 0.0, 0.0)
    return PortfolioGreeks(
        delta=sum(g.delta for g in greeks),
        gamma=sum(g.gamma for g in greeks),
        vega=sum(g.vega for g in greeks),
        theta=sum(g.theta for g in greeks),
    )
