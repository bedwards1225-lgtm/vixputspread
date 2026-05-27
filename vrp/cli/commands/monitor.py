"""``vrp monitor`` — check open positions for exit signals."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from vrp.config import Config
from vrp.alerts.notifier import build_notifier
from vrp.data import live
from vrp.data.providers import ChainSnapshot, build_surface
from vrp.persistence.db import connect, list_positions
from vrp.pricing import black_scholes as bsm
from vrp.pricing.greeks import position_greeks
from vrp.signals.exit import evaluate as evaluate_exit
from vrp.signals.regime import compute_25d_skew

logger = logging.getLogger(__name__)


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    notifier = build_notifier(cfg.alerts)
    positions = list_positions(conn, status="open")
    if not positions:
        print("No open positions.")
        return 0

    today = date.today()
    flagged = 0
    for pos in positions:
        try:
            chain = live.get_chain(
                pos.ticker,
                cache_dir=Path(cfg.data.cache_dir),
                min_dte=0, max_dte=180,
            )
        except live.LiveDataError as exc:
            logger.warning("skip %s: %s", pos.ticker, exc)
            continue
        snap = ChainSnapshot(ticker=pos.ticker, as_of=chain.as_of, spot=chain.spot, chain=chain.chain)
        try:
            surface = build_surface(
                snap, rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(pos.ticker, 0.0),
            )
        except ValueError:
            continue
        iv_s = surface.iv(pos.short_strike, pos.expiry)
        iv_l = surface.iv(pos.long_strike, pos.expiry)
        T = max((pos.expiry - today).days / 365.0, 1e-6)
        q = cfg.universe.dividend_yields.get(pos.ticker, 0.0)
        p_short = float(bsm.price(chain.spot, pos.short_strike, T, cfg.pricing.risk_free_rate, q, iv_s, "put"))
        p_long = float(bsm.price(chain.spot, pos.long_strike, T, cfg.pricing.risk_free_rate, q, iv_l, "put"))
        debit = p_short - p_long if pos.type == "credit_spread" else p_short
        greeks = position_greeks(
            pos, underlying_price=chain.spot, iv_short=iv_s, iv_long=iv_l,
            as_of=today, rate=cfg.pricing.risk_free_rate, div_yield=q,
        )
        try:
            skew_today = compute_25d_skew(surface, pos.expiry).skew_normalised
        except Exception:
            skew_today = 0.0

        decision = evaluate_exit(
            pos,
            as_of=today,
            debit_to_close=debit,
            current_gamma=greeks.gamma,
            entry_gamma=pos.entry_delta_short or 0.0,
            skew_today=skew_today,
            skew_yesterday=skew_today,
            skew_rolling_std=0.0,
            cfg=cfg.exit,
        )
        pnl = (pos.credit_or_debit - debit) * 100 * pos.contracts
        line = (
            f"#{pos.id} {pos.ticker} {pos.short_strike:.2f}/{pos.long_strike:.2f} "
            f"exp={pos.expiry} contracts={pos.contracts} debit=${debit:.2f} "
            f"open_pnl=${pnl:.0f} dte={(pos.expiry - today).days}"
        )
        if decision.triggered:
            flagged += 1
            notifier.notify(f"EXIT: {decision.reason}", f"{line}\n{decision.detail}")
        else:
            print(f"HOLD  {line}")
    print(f"\n{flagged} of {len(positions)} positions flagged for exit.")
    return 0
