"""Boundary tests for the 5 exit rules."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from vrp.config import ExitConfig
from vrp.persistence.models import Position
from vrp.signals.exit import evaluate


def _position(credit: float = 1.00, expiry_days_out: int = 30) -> Position:
    today = date(2024, 1, 2)
    return Position(
        ticker="SPY", type="credit_spread",
        entry_date=today, expiry=today + timedelta(days=expiry_days_out),
        short_strike=470, long_strike=468, contracts=1,
        credit_or_debit=credit, iv_at_entry=0.2, vrp_at_entry=0.01,
        regime="mid",
    )


def test_profit_target_at_exact_half():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=0.50,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert d.triggered and d.reason == "PROFIT_TARGET"


def test_profit_target_just_above_does_not_trigger():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=0.501,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert not d.triggered


def test_stop_loss_at_exact_2x():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=2.00,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert d.triggered and d.reason == "STOP_LOSS"


def test_stop_loss_just_under_does_not_trigger():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=1.999,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert not d.triggered


def test_gamma_stop_at_exact_3x():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=1.00,
                 current_gamma=-30, entry_gamma=-10,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert d.triggered and d.reason == "GAMMA_STOP"


def test_skew_reversal_triggers_at_2_sigma():
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=1.00,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0.05, skew_yesterday=0.01, skew_rolling_std=0.02,
                 cfg=cfg)
    assert d.triggered and d.reason == "SKEW_REVERSAL"


def test_time_stop_at_exact_5_dte():
    cfg = ExitConfig()
    pos = _position(credit=1.00, expiry_days_out=5)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=1.00,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert d.triggered and d.reason == "HARD_TIME_STOP"


def test_time_stop_does_not_trigger_at_6_dte():
    cfg = ExitConfig()
    pos = _position(credit=1.00, expiry_days_out=6)
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=1.00,
                 current_gamma=0, entry_gamma=0,
                 skew_today=0, skew_yesterday=0, skew_rolling_std=0,
                 cfg=cfg)
    assert not d.triggered


def test_priority_profit_target_before_stop_loss():
    """If both rules could fire, profit target wins."""
    cfg = ExitConfig()
    pos = _position(credit=1.00)
    # Impossible state but tests priority.
    d = evaluate(pos, as_of=pos.entry_date, debit_to_close=0.40,
                 current_gamma=-100, entry_gamma=-1,
                 skew_today=10, skew_yesterday=0, skew_rolling_std=1,
                 cfg=cfg)
    assert d.reason == "PROFIT_TARGET"
