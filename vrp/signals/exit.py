"""Daily exit-rule evaluation for open positions.

Rules in order; first match wins:

  1. PROFIT_TARGET   : debit to close <= 50% of credit received
  2. STOP_LOSS       : debit to close >= 200% of credit received
  3. GAMMA_STOP      : position gamma >= 3x entry gamma
  4. SKEW_REVERSAL   : 25d skew jumped > 2 sigma in one day
  5. HARD_TIME_STOP  : DTE <= 5

Each rule has a dedicated boundary so off-by-one tests are unambiguous:
the inequality is ``>=`` / ``<=`` exactly as listed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

from vrp.config import ExitConfig
from vrp.persistence.models import Position

ExitReason = Literal[
    "PROFIT_TARGET", "STOP_LOSS", "GAMMA_STOP", "SKEW_REVERSAL", "HARD_TIME_STOP"
]


@dataclass
class ExitDecision:
    """Outcome of checking the rules against one position."""

    triggered: bool
    reason: Optional[ExitReason] = None
    detail: str = ""


def evaluate(
    position: Position,
    *,
    as_of: date,
    debit_to_close: float,
    current_gamma: float,
    entry_gamma: float,
    skew_today: float,
    skew_yesterday: float,
    skew_rolling_std: float,
    cfg: ExitConfig,
) -> ExitDecision:
    """Apply exit rules to a single position.

    Args:
        position: Open position being checked.
        as_of: Evaluation date.
        debit_to_close: Cost (per share, summed over the spread) to close.
        current_gamma: Current $-scaled gamma of the position.
        entry_gamma: $-scaled gamma at entry.
        skew_today: 25d normalised skew today.
        skew_yesterday: 25d normalised skew yesterday.
        skew_rolling_std: Rolling std of *one-day changes* in skew.
        cfg: Exit configuration thresholds.

    Returns:
        :class:`ExitDecision`. ``triggered=False`` if no rule matched.
    """
    credit = position.credit_or_debit
    if credit <= 0:
        return ExitDecision(triggered=False)

    # Rule 1: profit target.
    if debit_to_close <= cfg.profit_target_pct * credit:
        return ExitDecision(
            triggered=True,
            reason="PROFIT_TARGET",
            detail=f"debit {debit_to_close:.2f} <= {cfg.profit_target_pct:.0%} of credit {credit:.2f}",
        )

    # Rule 2: stop loss.
    if debit_to_close >= cfg.stop_loss_multiple * credit:
        return ExitDecision(
            triggered=True,
            reason="STOP_LOSS",
            detail=f"debit {debit_to_close:.2f} >= {cfg.stop_loss_multiple:.0f}x credit {credit:.2f}",
        )

    # Rule 3: gamma stop. Only applies if entry_gamma magnitude is non-trivial.
    if abs(entry_gamma) > 1e-9 and abs(current_gamma) >= cfg.gamma_stop_multiple * abs(entry_gamma):
        return ExitDecision(
            triggered=True,
            reason="GAMMA_STOP",
            detail=f"|gamma| {abs(current_gamma):.4f} >= {cfg.gamma_stop_multiple:.0f}x entry {abs(entry_gamma):.4f}",
        )

    # Rule 4: skew reversal (one-day jump > N sigma).
    if skew_rolling_std > 0:
        z = (skew_today - skew_yesterday) / skew_rolling_std
        if abs(z) >= cfg.skew_reversal_sigmas:
            return ExitDecision(
                triggered=True,
                reason="SKEW_REVERSAL",
                detail=f"skew change z={z:+.2f} >= {cfg.skew_reversal_sigmas:.1f}",
            )

    # Rule 5: hard time stop.
    dte = (position.expiry - as_of).days
    if dte <= cfg.hard_time_stop_dte:
        return ExitDecision(
            triggered=True,
            reason="HARD_TIME_STOP",
            detail=f"DTE {dte} <= {cfg.hard_time_stop_dte}",
        )

    return ExitDecision(triggered=False)
