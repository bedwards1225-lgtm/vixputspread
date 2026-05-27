"""``vrp hedge-status`` — show current hedge coverage and the gap to target."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from vrp.config import Config
from vrp.data import live
from vrp.data.providers import ChainSnapshot, build_surface
from vrp.hedging.tail_hedge import needs_roll, propose_hedge
from vrp.persistence.db import connect, list_positions

logger = logging.getLogger(__name__)


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    positions = list_positions(conn, status="open")
    spreads = [p for p in positions if p.type == "credit_spread"]
    hedge = next((p for p in positions if p.type == "tail_hedge"), None)
    today = date.today()

    if hedge:
        print(
            f"Current hedge: #{hedge.id} {hedge.ticker} strike={hedge.short_strike:.2f} "
            f"exp={hedge.expiry} contracts={hedge.contracts} cost={hedge.credit_or_debit:.2f}"
        )
    else:
        print("No tail hedge open.")

    if not spreads:
        print("No credit spreads open; nothing to hedge against.")
        return 0

    spots = {t: live.get_spot(t) for t in {p.ticker for p in spreads} | {cfg.hedging.ticker}}

    def iv_today(ticker, strike, expiry, as_of):
        try:
            chain = live.get_chain(ticker, cache_dir=Path(cfg.data.cache_dir))
            snap = ChainSnapshot(ticker=ticker, as_of=chain.as_of, spot=chain.spot, chain=chain.chain)
            surface = build_surface(snap, rate=cfg.pricing.risk_free_rate,
                                    div_yield=cfg.universe.dividend_yields.get(ticker, 0.0))
            return surface.iv(strike, expiry)
        except Exception as exc:
            logger.warning("iv lookup failed for %s: %s", ticker, exc)
            return 0.2

    def iv_stressed(ticker, strike, expiry, as_of):
        return iv_today(ticker, strike, expiry, as_of) * 2.0

    proposal = propose_hedge(
        cfg=cfg.hedging,
        as_of=today,
        equity=cfg.sizing.starting_equity,  # informational; user's actual equity may differ
        open_spreads=spreads,
        spot_lookup=lambda t: spots[t],
        iv_lookup_today=iv_today,
        iv_lookup_stressed=iv_stressed,
        rate=cfg.pricing.risk_free_rate,
        dividend_yields=cfg.universe.dividend_yields,
        existing_hedge=hedge,
    )
    if proposal is None:
        print("No hedge needed.")
        return 0
    needs = needs_roll(hedge, as_of=today, roll_dte=cfg.hedging.roll_dte)
    print(
        f"Proposed: {proposal.ticker} {proposal.strike} exp={proposal.expiry} "
        f"x{proposal.contracts} debit=${proposal.estimated_debit:.2f} "
        f"cost=${proposal.estimated_cost:,.0f} coverage={proposal.coverage_ratio:.0%} "
        f"binding={proposal.binding_constraint}"
    )
    if proposal.notes:
        print(f"  note: {proposal.notes}")
    print(f"  roll needed: {needs}")
    return 0
