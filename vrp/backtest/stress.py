"""Historical stress-scenario repricer.

Three canonical scenarios:

  GFC          : SPX -10%, VIX to 80
  VOLMAGEDDON  : SPX -4%, VIX 13 -> 37 in one day
  COVID        : SPX -12% over 3 days, VIX to 75

Each scenario shocks the underlying spot and VIX (proxy for ATM IV via
``vix_scale``) and reprices every open position. Used both inside the
backtest (logged daily) and as a standalone ``vrp stress`` CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Literal

from vrp.persistence.models import Position
from vrp.pricing import black_scholes as bsm

_MULTIPLIER = 100

ScenarioName = Literal["gfc", "volmageddon", "covid"]


@dataclass
class StressScenario:
    """A single named stress scenario."""

    name: ScenarioName
    spx_return: float          # e.g. -0.10
    vix_level: float           # e.g. 80
    days: int                  # span (used to scale theta if needed)


SCENARIOS: dict[ScenarioName, StressScenario] = {
    "gfc": StressScenario(name="gfc", spx_return=-0.10, vix_level=80.0, days=1),
    "volmageddon": StressScenario(name="volmageddon", spx_return=-0.04, vix_level=37.0, days=1),
    "covid": StressScenario(name="covid", spx_return=-0.12, vix_level=75.0, days=3),
}


@dataclass
class StressResult:
    """Per-scenario P&L summary."""

    scenario: ScenarioName
    spread_pnl: float
    hedge_pnl: float
    net_pnl: float


def reprice_under(
    scenario: StressScenario,
    *,
    positions: list[Position],
    as_of: date,
    spot_lookup: Callable[[str], float],
    rate: float,
    dividend_yields: dict[str, float],
    vix_scale: dict[str, float],
) -> StressResult:
    """Reprice every position under the shocked spot / IV state.

    IVs are approximated by scaling the new VIX level by ``vix_scale[ticker]``.
    A more sophisticated model would also shock skew; the spec's three
    scenarios encode the worst plausible move in level which is the dominant
    driver of credit-spread P&L.

    Args:
        scenario: Stress definition.
        positions: All positions (credit spreads + tail hedge).
        as_of: Valuation date.
        spot_lookup: Map ticker -> current spot.
        rate: Risk-free rate.
        dividend_yields: Per-ticker dividend yield.
        vix_scale: Per-ticker VIX scaling.

    Returns:
        :class:`StressResult` with spread / hedge / net P&L in dollars.
    """
    spread_pnl = 0.0
    hedge_pnl = 0.0
    new_iv_atm = scenario.vix_level / 100.0
    days_forward = max(scenario.days, 1)
    for p in positions:
        spot = spot_lookup(p.ticker)
        stressed_spot = spot * (1.0 + scenario.spx_return)
        q = dividend_yields.get(p.ticker, 0.0)
        scale = vix_scale.get(p.ticker, 1.0)
        iv = new_iv_atm * scale
        T = max((p.expiry - as_of).days - days_forward, 1) / 365.0

        if p.type == "credit_spread":
            p_short = float(bsm.price(stressed_spot, p.short_strike, T, rate, q, iv, "put"))
            p_long = float(bsm.price(stressed_spot, p.long_strike, T, rate, q, iv, "put"))
            stressed_debit = p_short - p_long
            pnl = (p.credit_or_debit - stressed_debit) * _MULTIPLIER * p.contracts
            spread_pnl += pnl
        elif p.type == "tail_hedge":
            stressed_val = float(bsm.price(stressed_spot, p.short_strike, T, rate, q, iv, "put"))
            pnl = (stressed_val - p.credit_or_debit) * _MULTIPLIER * p.contracts
            hedge_pnl += pnl
    return StressResult(
        scenario=scenario.name,
        spread_pnl=spread_pnl,
        hedge_pnl=hedge_pnl,
        net_pnl=spread_pnl + hedge_pnl,
    )


def run_all(
    *,
    positions: list[Position],
    as_of: date,
    spot_lookup: Callable[[str], float],
    rate: float,
    dividend_yields: dict[str, float],
    vix_scale: dict[str, float],
) -> dict[ScenarioName, StressResult]:
    """Reprice ``positions`` under all three canonical scenarios."""
    return {
        name: reprice_under(
            sc,
            positions=positions,
            as_of=as_of,
            spot_lookup=spot_lookup,
            rate=rate,
            dividend_yields=dividend_yields,
            vix_scale=vix_scale,
        )
        for name, sc in SCENARIOS.items()
    }
