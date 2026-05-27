"""Combined sizing logic.

The number of contracts taken is the minimum across:
  - Per-trade max-loss cap (1.5% of equity by default).
  - Portfolio VaR cap (5% of equity).
  - Portfolio Greek caps (delta, gamma, vega).

A skew-driven size reduction (from the entry signal) is then applied as a
multiplicative shrink. The result is rounded according to ``round_contracts``
config (floor / nearest) and may be zero, in which case the trade is skipped.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from vrp.config import SizingConfig
from vrp.persistence.models import Position
from vrp.signals.entry import SignalCandidate
from vrp.sizing.var import historical_var

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """How many contracts to take and which constraint was binding."""

    contracts: int
    binding_constraint: str
    per_trade_max: int
    var_max: int
    greek_max: int
    notes: str = ""


def size_trade(
    candidate: SignalCandidate,
    *,
    cfg: SizingConfig,
    equity: float,
    open_positions: list[Position],
    underlying_history: dict[str, pd.Series],
    iv_lookup,
    rate: float,
    dividend_yields: dict[str, float],
    portfolio_delta: float,
    portfolio_gamma: float,
    portfolio_vega: float,
) -> SizingResult:
    """Compute contract count respecting per-trade, VaR, and Greek limits.

    Args:
        candidate: Output of :func:`vrp.signals.entry.propose_candidate`.
        cfg: Sizing configuration.
        equity: Current account equity (dollar).
        open_positions: Currently open positions for VaR aggregation.
        underlying_history: Map ticker -> close-price Series for VaR.
        iv_lookup: Callable ``(ticker, strike, expiry, as_of) -> iv``.
        rate: Risk-free rate.
        dividend_yields: Per-ticker dividend yield.
        portfolio_delta: Current $-scaled portfolio delta.
        portfolio_gamma: Current $-scaled portfolio gamma.
        portfolio_vega: Current $-scaled portfolio vega per vol point.

    Returns:
        :class:`SizingResult`.
    """
    if not candidate.accepted:
        return SizingResult(
            contracts=0,
            binding_constraint="rejected",
            per_trade_max=0,
            var_max=0,
            greek_max=0,
            notes="; ".join(candidate.reject_reasons),
        )

    # --- per-trade cap ---
    max_loss_per_contract = candidate.max_loss_per_contract  # already in $
    per_trade_budget = cfg.max_loss_per_trade_pct * equity
    per_trade_max = (
        math.floor(per_trade_budget / max_loss_per_contract)
        if max_loss_per_contract > 0
        else 0
    )

    # --- VaR cap (binary-search the largest contract count keeping VaR < cap) ---
    var_cap = cfg.portfolio_var_cap_pct * equity
    var_max = _max_contracts_under_var(
        candidate,
        open_positions=open_positions,
        as_of=candidate.as_of,
        underlying_history=underlying_history,
        iv_lookup=iv_lookup,
        rate=rate,
        dividend_yields=dividend_yields,
        confidence=cfg.var_confidence,
        lookback=cfg.var_lookback_days,
        var_cap=var_cap,
        upper_hint=max(per_trade_max, 1),
    )

    # --- Greek caps (per-trade contribution) ---
    # New trade contributes negative delta / negative gamma / negative vega (short put spread).
    # Caps are expressed as $ per 1 unit of underlying / vol-point. We convert
    # to "max additional |Greek| we can take" given existing portfolio Greek.
    spot = candidate.underlying_price
    # Per-contract Greek contributions (signed, short put spread is negative).
    delta_per_contract = -candidate.short_delta * 100 - (
        # long leg delta contribution: long put has negative delta too;
        # treat |long leg| as the long-strike delta. As an approximation,
        # the long leg gamma/vega largely offsets the short leg, so we focus
        # on the short leg's marginal contribution as a conservative bound.
        0
    )
    gamma_per_contract = -candidate.short_gamma * 100
    vega_per_contract = -candidate.short_vega * 100

    delta_budget = cfg.net_delta_cap_pct * equity - abs(portfolio_delta)
    gamma_budget = cfg.net_short_gamma_cap_pct * equity - max(-portfolio_gamma, 0.0)
    vega_budget = cfg.net_short_vega_cap_pct * equity - max(-portfolio_vega, 0.0)

    delta_max = (
        math.floor(delta_budget / abs(delta_per_contract))
        if abs(delta_per_contract) > 1e-9
        else 10_000
    )
    gamma_max = (
        math.floor(gamma_budget / abs(gamma_per_contract))
        if abs(gamma_per_contract) > 1e-9 and gamma_per_contract < 0
        else 10_000
    )
    vega_max = (
        math.floor(vega_budget / abs(vega_per_contract))
        if abs(vega_per_contract) > 1e-9 and vega_per_contract < 0
        else 10_000
    )
    greek_max = max(0, min(delta_max, gamma_max, vega_max))

    raw = min(per_trade_max, var_max, greek_max)
    # Apply skew-driven reduction.
    raw = int(math.floor(raw * candidate.size_reduction))

    binding = _binding_constraint(per_trade_max, var_max, greek_max, raw)

    contracts = max(raw, 0)
    if contracts == 0 and not cfg.skip_if_zero:
        contracts = 1
        binding = f"forced minimum 1 (would have skipped, binding={binding})"

    return SizingResult(
        contracts=contracts,
        binding_constraint=binding,
        per_trade_max=per_trade_max,
        var_max=var_max,
        greek_max=greek_max,
    )


def _binding_constraint(per_trade_max: int, var_max: int, greek_max: int, raw: int) -> str:
    vals = {
        "per_trade": per_trade_max,
        "var": var_max,
        "greek": greek_max,
    }
    binding = min(vals, key=lambda k: vals[k])
    if vals[binding] != raw:
        return f"{binding}+skew_reduction"
    return binding


def _max_contracts_under_var(
    candidate: SignalCandidate,
    *,
    open_positions: list[Position],
    as_of: date,
    underlying_history: dict[str, pd.Series],
    iv_lookup,
    rate: float,
    dividend_yields: dict[str, float],
    confidence: float,
    lookback: int,
    var_cap: float,
    upper_hint: int,
) -> int:
    """Largest integer ``n`` such that adding ``n`` contracts keeps VaR < cap.

    Uses upper-hint-doubling + binary search. If even 1 contract violates,
    returns 0.
    """
    if upper_hint <= 0:
        upper_hint = 1

    def var_with(n: int) -> float:
        if n <= 0:
            return historical_var(
                open_positions,
                as_of=as_of,
                underlying_history=underlying_history,
                iv_lookup=iv_lookup,
                rate=rate,
                dividend_yields=dividend_yields,
                confidence=confidence,
                lookback=lookback,
            )
        # Synthesize a Position object for the candidate at n contracts.
        from vrp.persistence.models import Position
        cand_pos = Position(
            ticker=candidate.ticker,
            type="credit_spread",
            entry_date=as_of,
            expiry=candidate.expiry,
            short_strike=candidate.short_strike,
            long_strike=candidate.long_strike,
            contracts=n,
            credit_or_debit=candidate.credit,
            iv_at_entry=candidate.short_iv,
            vrp_at_entry=candidate.vrp,
            regime=str(candidate.vix_regime),
        )
        return historical_var(
            open_positions + [cand_pos],
            as_of=as_of,
            underlying_history=underlying_history,
            iv_lookup=iv_lookup,
            rate=rate,
            dividend_yields=dividend_yields,
            confidence=confidence,
            lookback=lookback,
        )

    # Find an upper bound where VaR exceeds cap.
    hi = upper_hint
    while hi <= 10_000 and var_with(hi) <= var_cap:
        hi *= 2
    if hi > 10_000:
        return 10_000
    if var_with(1) > var_cap:
        return 0

    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if var_with(mid) <= var_cap:
            lo = mid
        else:
            hi = mid - 1
    return lo
