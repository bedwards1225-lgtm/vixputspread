"""Dataclasses that mirror the database tables.

Conversion to/from sqlite rows is handled in :mod:`vrp.persistence.db`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

PositionType = Literal["credit_spread", "tail_hedge"]
PositionStatus = Literal["open", "closed", "expired"]
OptionType = Literal["put", "call"]
LegSide = Literal["short", "long"]


@dataclass
class Leg:
    """One option leg within a position."""

    side: LegSide
    option_type: OptionType
    strike: float
    expiry: date
    contracts: int
    entry_price: float
    entry_iv: float
    entry_delta: float
    entry_gamma: float
    entry_vega: float
    entry_theta: float
    id: Optional[int] = None
    position_id: Optional[int] = None


@dataclass
class Position:
    """A multi-leg position (credit spread or tail hedge)."""

    ticker: str
    type: PositionType
    entry_date: date
    expiry: date
    short_strike: float
    long_strike: float
    contracts: int
    credit_or_debit: float
    iv_at_entry: float
    vrp_at_entry: float
    regime: str
    status: PositionStatus = "open"
    legs: list[Leg] = field(default_factory=list)
    entry_underlying_price: float = 0.0
    entry_vix: float = 0.0
    entry_delta_short: float = 0.0
    exit_date: Optional[date] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    commissions: float = 0.0
    notes: Optional[str] = None
    id: Optional[int] = None


@dataclass
class DailySnapshot:
    """End-of-day portfolio snapshot."""

    date: date
    vix: float
    vrp: float
    portfolio_value: float
    portfolio_var: float
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    open_positions: int
    hedge_coverage: float
    realized_vol_22d: float = 0.0
    iv_atm: float = 0.0
    skew_25d: float = 0.0


@dataclass
class HarRvFit:
    """A fitted HAR-RV model."""

    fit_date: date
    beta_0: float
    beta_daily: float
    beta_weekly: float
    beta_monthly: float
    r_squared: float
    n_obs: int
