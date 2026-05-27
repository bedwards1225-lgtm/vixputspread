"""``vrp backtest`` — run the historical backtest."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from vrp.backtest.engine import run_backtest
from vrp.backtest.reporting import write_report
from vrp.config import Config
from vrp.data import live
from vrp.data.providers import (
    BlackScholesProvider,
    CboeDatashopProvider,
    OratsProvider,
    PolygonProvider,
)

logger = logging.getLogger(__name__)

_BSM_WARNING = (
    "\n" + "!" * 70 + "\n"
    "!  WARNING: --data-provider=bsm uses synthetic option chains.\n"
    "!  Results are APPROXIMATIONS only and should NOT be used to\n"
    "!  validate the strategy for real-money deployment.\n"
    "!  See README.md for paid data sources.\n"
    + "!" * 70 + "\n"
)


def _make_provider(cfg: Config, name: str, root: Path | None):
    if name == "bsm":
        print(_BSM_WARNING)
        return BlackScholesProvider(
            rate=cfg.pricing.risk_free_rate,
            dividend_yields=cfg.universe.dividend_yields,
            vix_scale=cfg.universe.vix_scale,
            skew_slope=cfg.pricing.bsm_skew_slope,
            history_lookup=_history_lookup_via_yfinance,
        )
    if name == "orats":
        if root is None:
            raise ValueError("--data-root required for orats")
        return OratsProvider(root)
    if name == "polygon":
        if root is None:
            raise ValueError("--data-root required for polygon")
        return PolygonProvider(root)
    if name == "cboe":
        if root is None:
            raise ValueError("--data-root required for cboe")
        return CboeDatashopProvider(root)
    raise ValueError(f"unknown provider {name!r}")


def _history_lookup_via_yfinance(ticker: str, start: date, end: date):
    """Pull OHLC for the BSM provider via yfinance."""
    days = (end - start).days + 60
    return live.get_history(ticker, days=days)


def run(args, cfg: Config) -> int:
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    provider = _make_provider(cfg, args.data_provider, args.data_root)

    logger.info("backtest %s -> %s using %s", start, end, args.data_provider)
    result = run_backtest(cfg=cfg, provider=provider, start=start, end=end, seed=args.seed)

    out_dir = Path(cfg.backtest.results_dir)
    tag = args.tag or f"{args.data_provider}_{start}_{end}"
    run_dir = write_report(result, out_dir=out_dir, tag=tag)
    print(f"\nReport written to: {run_dir}")
    return 0
