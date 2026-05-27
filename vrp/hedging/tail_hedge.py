"""Tail-hedge construction and sizing.

The hedge is a long put on the configured underlying (default SPY) at
~6% OTM, 60-90 DTE. Sizing is driven by a stress test:

  Stress = (SPX down by ``stress_spx_drop``)
         + (VIX moves to ``stress_vix_level``)
         + (skew moves to ``stress_skew_percentile`` quantile of history)

Under stress, every credit spread is repriced and the loss aggregated.
Number of hedge contracts is the smallest integer ``n`` such that the
hedge gain under stress >= ``hedge_coverage_target`` * loss.

If satisfying the target would exceed the annual premium budget
(``annual_premium_budget_pct`` of equity), the hedge is capped at the
budget and the gap is logged. We never overpay.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

from vrp.config import HedgingConfig
from vrp.persistence.models import Position
from vrp.pricing import black_scholes as bsm

logger = logging.getLogger(__name__)

_MULTIPLIER = 100


@dataclass
class HedgeProposal:
    """A proposed (new or roll) tail-hedge position."""

    ticker: str
    expiry: date
    strike: float
    contracts: int
    estimated_debit: float        # per share, mid
    estimated_cost: float         # total $ cost
    coverage_ratio: float         # hedge_gain / spread_loss under stress
    binding_constraint: str       # "coverage" | "premium_budget"
    notes: str = ""


def stress_spread_loss(
    positions: list[Position],
    *,
    as_of: date,
    spot_lookup,
    iv_lookup_stressed,
    rate: float,
    dividend_yields: dict[str, float],
    stress_drop: float,
) -> float:
    """Aggregate $ loss across credit spreads under the stress scenario.

    Args:
        positions: Open credit-spread positions (type ``credit_spread``).
        as_of: Valuation date (for time-to-expiry).
        spot_lookup: Callable ``ticker -> current_spot``.
        iv_lookup_stressed: Callable ``(ticker, strike, expiry, as_of) -> iv``
            that already encodes the stressed surface.
        rate: Risk-free rate.
        dividend_yields: Per-ticker dividend yield.
        stress_drop: Fractional underlying drop (e.g. 0.08).

    Returns:
        Positive number = $ loss.
    """
    total_loss = 0.0
    for p in positions:
        if p.type != "credit_spread":
            continue
        q = dividend_yields.get(p.ticker, 0.0)
        spot = spot_lookup(p.ticker)
        stressed_spot = spot * (1.0 - stress_drop)
        T = max((p.expiry - as_of).days / 365.0, 1e-6)
        iv_s = iv_lookup_stressed(p.ticker, p.short_strike, p.expiry, as_of)
        iv_l = iv_lookup_stressed(p.ticker, p.long_strike, p.expiry, as_of)
        p_short = float(bsm.price(stressed_spot, p.short_strike, T, rate, q, iv_s, "put"))
        p_long = float(bsm.price(stressed_spot, p.long_strike, T, rate, q, iv_l, "put"))
        stressed_debit = p_short - p_long
        loss_per_contract = max(stressed_debit - p.credit_or_debit, 0.0) * _MULTIPLIER
        total_loss += loss_per_contract * p.contracts
    return total_loss


def propose_hedge(
    *,
    cfg: HedgingConfig,
    as_of: date,
    equity: float,
    open_spreads: list[Position],
    spot_lookup,
    iv_lookup_today,
    iv_lookup_stressed,
    rate: float,
    dividend_yields: dict[str, float],
    existing_hedge: Position | None = None,
) -> HedgeProposal | None:
    """Compute a tail-hedge proposal for the configured ticker.

    Args:
        cfg: Hedging config.
        as_of: Valuation date.
        equity: Account equity.
        open_spreads: Open credit-spread positions.
        spot_lookup: Callable ``ticker -> current_spot``.
        iv_lookup_today: Callable for today's surface.
        iv_lookup_stressed: Callable for the stressed surface.
        rate: Risk-free rate.
        dividend_yields: Per-ticker dividend yield.
        existing_hedge: Currently held hedge (used only to detect roll).

    Returns:
        :class:`HedgeProposal`, or ``None`` if the hedge is disabled or
        no spreads are open.
    """
    if not cfg.enabled or not open_spreads:
        return None

    ticker = cfg.ticker
    q = dividend_yields.get(ticker, 0.0)
    spot = spot_lookup(ticker)
    target_strike = round(spot * (1.0 - cfg.otm_pct), 2)
    expiry = as_of + timedelta(days=int(0.5 * (cfg.min_dte + cfg.max_dte)))

    iv = iv_lookup_today(ticker, target_strike, expiry, as_of)
    T = max((expiry - as_of).days / 365.0, 1e-6)
    debit_per_share = float(bsm.price(spot, target_strike, T, rate, q, iv, "put"))
    cost_per_contract = debit_per_share * _MULTIPLIER

    # Loss the spreads take in the stress scenario.
    loss = stress_spread_loss(
        open_spreads,
        as_of=as_of,
        spot_lookup=spot_lookup,
        iv_lookup_stressed=iv_lookup_stressed,
        rate=rate,
        dividend_yields=dividend_yields,
        stress_drop=cfg.stress_spx_drop,
    )
    if loss <= 0:
        return None

    # Gain per hedge contract under the stress scenario.
    stressed_spot = spot * (1.0 - cfg.stress_spx_drop)
    stressed_iv = iv_lookup_stressed(ticker, target_strike, expiry, as_of)
    stressed_T = max(((expiry - as_of).days - 1) / 365.0, 1e-6)  # one day forward
    stressed_value = float(bsm.price(stressed_spot, target_strike, stressed_T, rate, q, stressed_iv, "put"))
    gain_per_contract = max((stressed_value - debit_per_share) * _MULTIPLIER, 0.0)

    if gain_per_contract <= 0:
        return None

    target_total_gain = cfg.hedge_coverage_target * loss
    contracts_for_coverage = math.ceil(target_total_gain / gain_per_contract)

    # Premium budget cap.
    premium_budget = cfg.annual_premium_budget_pct * equity
    # Crude annualisation: assume ~4 hedge rolls per year (90-DTE).
    rolls_per_year = max(1.0, 365.0 / (0.5 * (cfg.min_dte + cfg.max_dte)))
    max_cost_per_roll = premium_budget / rolls_per_year
    contracts_for_budget = (
        math.floor(max_cost_per_roll / cost_per_contract)
        if cost_per_contract > 0 else 0
    )

    if contracts_for_coverage <= contracts_for_budget:
        contracts = contracts_for_coverage
        binding = "coverage"
        notes = ""
    else:
        contracts = max(contracts_for_budget, 0)
        binding = "premium_budget"
        notes = (
            f"coverage target requires {contracts_for_coverage} contracts but "
            f"budget allows only {contracts_for_budget}; underhedged"
        )
        logger.warning("hedge underhedged: %s", notes)

    coverage_ratio = (contracts * gain_per_contract) / loss if loss > 0 else 0.0
    return HedgeProposal(
        ticker=ticker,
        expiry=expiry,
        strike=target_strike,
        contracts=contracts,
        estimated_debit=debit_per_share,
        estimated_cost=contracts * cost_per_contract,
        coverage_ratio=coverage_ratio,
        binding_constraint=binding,
        notes=notes,
    )


def needs_roll(existing_hedge: Position | None, *, as_of: date, roll_dte: int) -> bool:
    """True if the hedge should be rolled (no hedge, or DTE below threshold)."""
    if existing_hedge is None:
        return True
    return (existing_hedge.expiry - as_of).days < roll_dte
