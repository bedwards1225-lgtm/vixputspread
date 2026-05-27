"""Stress scenarios produce expected directional P&L."""
from __future__ import annotations

from datetime import date, timedelta

from vrp.backtest.stress import SCENARIOS, reprice_under
from vrp.persistence.models import Position


def _credit_spread() -> Position:
    return Position(
        ticker="SPY", type="credit_spread",
        entry_date=date(2024, 1, 2),
        expiry=date(2024, 2, 2),
        short_strike=470.0, long_strike=460.0, contracts=10,
        credit_or_debit=1.20, iv_at_entry=0.18, vrp_at_entry=0.01,
        regime="mid",
    )


def test_credit_spread_loses_under_gfc_scenario():
    pos = _credit_spread()
    result = reprice_under(
        SCENARIOS["gfc"],
        positions=[pos],
        as_of=date(2024, 1, 2),
        spot_lookup=lambda t: 500.0,
        rate=0.04,
        dividend_yields={"SPY": 0.013},
        vix_scale={"SPY": 1.0},
    )
    assert result.spread_pnl < 0
    assert result.net_pnl < 0


def test_tail_hedge_gains_under_gfc_scenario():
    hedge = Position(
        ticker="SPY", type="tail_hedge",
        entry_date=date(2024, 1, 2),
        expiry=date(2024, 4, 2),
        short_strike=470.0, long_strike=470.0, contracts=2,
        credit_or_debit=5.00, iv_at_entry=0.20, vrp_at_entry=0.0,
        regime="hedge",
    )
    result = reprice_under(
        SCENARIOS["gfc"],
        positions=[hedge],
        as_of=date(2024, 1, 2),
        spot_lookup=lambda t: 500.0,
        rate=0.04,
        dividend_yields={"SPY": 0.013},
        vix_scale={"SPY": 1.0},
    )
    assert result.hedge_pnl > 0
