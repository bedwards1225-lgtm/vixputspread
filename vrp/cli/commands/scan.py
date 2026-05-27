"""``vrp scan`` — scan for new entry signals across the universe."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from vrp.config import Config
from vrp.data import live
from vrp.data.providers import build_surface
from vrp.data.providers import ChainSnapshot
from vrp.forecasting import har_rv
from vrp.persistence.db import connect, latest_har_fit, list_positions, upsert_har_fit
from vrp.signals.entry import propose_candidate

logger = logging.getLogger(__name__)


def run(args, cfg: Config) -> int:
    tickers = args.ticker if args.ticker else cfg.universe.tickers
    cache_dir = Path(cfg.data.cache_dir)

    conn = connect(Path(cfg.persistence.db_path))

    print(f"{'Ticker':<6} {'DTE':>4} {'Short':>8} {'Long':>8} {'Width':>6} {'Credit':>8} "
          f"{'C/W':>5} {'VRP':>7} {'VRP%':>5} {'Skew%':>6} {'Regime':<6} {'IV':>5} {'RV^.5':>5} Status")
    print("-" * 100)

    today = date.today()
    fit = latest_har_fit(conn, today)
    if fit is None:
        # Fit on SPY history.
        spy_hist = live.get_history("SPY", days=cfg.data.history_days_for_har)
        fit = har_rv.fit(
            spy_hist,
            as_of=today,
            horizon_days=int(0.5 * (cfg.entry.min_dte + cfg.entry.max_dte)),
            window_days=cfg.forecasting.har_window_days,
        )
        if fit is None:
            logger.error("not enough history to fit HAR-RV; aborting scan")
            return 2
        upsert_har_fit(conn, fit)

    vix_series = live.get_history("^VIX", days=cfg.entry.skew_lookback_days + 30)["Close"]

    stale_tickers: list[str] = []
    n_accepted = 0
    for ticker in tickers:
        try:
            chain = live.get_chain(
                ticker,
                cache_dir=cache_dir,
                retries=cfg.data.yfinance_retries,
                backoff=cfg.data.yfinance_backoff_seconds,
                min_dte=cfg.entry.min_dte - 5,
                max_dte=cfg.entry.max_dte + 5,
            )
        except live.LiveDataError as exc:
            logger.warning("skipping %s: %s", ticker, exc)
            continue
        if live.looks_like_stale_chain(chain.chain):
            stale_tickers.append(ticker)
            continue
        snap = ChainSnapshot(
            ticker=ticker, as_of=chain.as_of, spot=chain.spot, chain=chain.chain
        )
        try:
            surface = build_surface(
                snap,
                rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(ticker, 0.0),
            )
        except ValueError as exc:
            logger.warning("surface fit failed for %s: %s", ticker, exc)
            continue
        history = live.get_history(ticker, days=cfg.data.history_days_for_har)
        feats = har_rv.build_features(history)
        history.attrs["rv_d"] = float(feats.rv_daily.iloc[-1])
        history.attrs["rv_w"] = float(feats.rv_weekly.iloc[-1])
        history.attrs["rv_m"] = float(feats.rv_monthly.iloc[-1])

        candidate = propose_candidate(
            cfg=cfg,
            ticker=ticker,
            as_of=today,
            surface=surface,
            underlying_history=history,
            vix_history=vix_series,
            vrp_history=pd.Series(dtype=float),
            skew_history=pd.Series(dtype=float),
            har_fit=fit,
            rate=cfg.pricing.risk_free_rate,
            div_yield=cfg.universe.dividend_yields.get(ticker, 0.0),
        )
        if candidate is None:
            continue
        status = "ACCEPT" if candidate.accepted else "REJECT: " + "; ".join(candidate.reject_reasons[:1])
        if candidate.accepted:
            n_accepted += 1
        print(
            f"{ticker:<6} {candidate.dte:>4d} "
            f"{candidate.short_strike:>8.2f} {candidate.long_strike:>8.2f} "
            f"{candidate.width:>6.2f} {candidate.credit:>8.2f} "
            f"{candidate.credit_pct_of_width:>5.0%} {candidate.vrp:>7.4f} "
            f"{candidate.vrp_percentile:>5.0%} {candidate.skew_percentile:>6.0%} "
            f"{candidate.vix_regime:<6} {candidate.short_iv:>5.2f} "
            f"{candidate.rv_forecast**0.5:>5.2f} {status}"
        )

    print(f"\n{n_accepted} accepted out of {len(tickers)} scanned.")
    if stale_tickers:
        print()
        print("=" * 72)
        print("WARNING: yfinance returned stale (bucketed) IV data for: "
              f"{', '.join(stale_tickers)}")
        print("This usually means the US options market is closed (weekends,")
        print("after 4pm ET, holidays). Live signals require market-hours data.")
        print("Run again between 9:30am-4:00pm ET on a US trading day.")
        print("=" * 72)
    return 0
