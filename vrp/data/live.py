"""yfinance live-data wrapper with retry, backoff, and on-disk fallback.

End-of-day data is assumed; a 15-minute live-data delay is fine since the
strategy doesn't make intraday decisions.

This module never raises on a transient failure — it falls back to the most
recent cached payload and logs a warning. Hard failures (no cache and no
network) propagate as exceptions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from vrp.data.cache import load_chain, save_chain

logger = logging.getLogger(__name__)


@dataclass
class LiveChain:
    """A normalised live option chain snapshot for one ticker."""

    ticker: str
    as_of: date
    spot: float
    chain: pd.DataFrame   # columns: option_type, strike, expiry, bid, ask, mid, iv, volume, oi


class LiveDataError(RuntimeError):
    """Raised when live data cannot be obtained and no cache exists."""


def _retry(func, retries: int, backoff: float, *args, **kwargs):
    """Call ``func`` with exponential backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError) as exc:
            last = exc
            wait = backoff * (2 ** attempt)
            logger.warning("retry %d/%d after %.1fs: %s", attempt + 1, retries, wait, exc)
            time.sleep(wait)
    raise LiveDataError(f"giving up after {retries} retries: {last}")


def get_history(
    ticker: str, *, days: int, retries: int = 3, backoff: float = 2.0
) -> pd.DataFrame:
    """Fetch daily OHLC history for the last ``days`` days.

    Args:
        ticker: Yahoo symbol.
        days: Calendar days of history to request.
        retries: Number of retry attempts on transient failures.
        backoff: Initial backoff in seconds, doubled per retry.

    Returns:
        DataFrame indexed by datetime with columns ``Open``, ``High``,
        ``Low``, ``Close``, ``Volume``.

    Raises:
        LiveDataError: If all retries fail.
    """
    def _fetch() -> pd.DataFrame:
        df = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=False)
        if df.empty:
            raise ConnectionError(f"empty history for {ticker}")
        # yfinance returns a tz-aware index ("America/New_York"). Strip the
        # timezone so the rest of the codebase can use tz-naive timestamps.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    return _retry(_fetch, retries, backoff)


def get_spot(ticker: str, *, retries: int = 3, backoff: float = 2.0) -> float:
    """Most recent close for ``ticker``."""
    df = get_history(ticker, days=5, retries=retries, backoff=backoff)
    return float(df["Close"].iloc[-1])


def get_vix(*, retries: int = 3, backoff: float = 2.0) -> float:
    """Latest VIX close (^VIX)."""
    df = get_history("^VIX", days=5, retries=retries, backoff=backoff)
    return float(df["Close"].iloc[-1])


def looks_like_stale_chain(chain: pd.DataFrame) -> bool:
    """Detect yfinance's after-hours "bucketed IV" pattern.

    yfinance returns IV values like 0.0625, 0.125, 0.25, 0.50 — exact
    fractions of one — when the underlying market is closed and it can't
    compute live IVs. If a high fraction of IVs cluster on these buckets,
    the chain is essentially useless for live signal generation.

    Returns:
        True if the chain appears to be after-hours placeholder data.
    """
    if chain.empty:
        return True
    buckets = {0.0625, 0.125, 0.25, 0.50, 1.0}
    rounded = chain["iv"].round(3)
    matches = rounded.apply(lambda v: any(abs(v - b) < 0.001 for b in buckets))
    return matches.mean() > 0.5


def get_chain(
    ticker: str,
    *,
    cache_dir: Path | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    min_dte: int = 0,
    max_dte: int = 120,
) -> LiveChain:
    """Pull and normalise a yfinance options chain.

    On failure, returns the most recent cached chain if one exists. Pulls
    every expiry within ``[min_dte, max_dte]`` and bundles into a long-form
    DataFrame.

    Args:
        ticker: Yahoo symbol.
        cache_dir: Directory to read/write cached chains. ``None`` disables caching.
        retries: Retries per HTTP call.
        backoff: Initial backoff seconds.
        min_dte: Minimum DTE to include.
        max_dte: Maximum DTE to include.

    Returns:
        A :class:`LiveChain`.

    Raises:
        LiveDataError: If live fetch fails and no cache is available.
    """
    as_of = date.today()
    spot = get_spot(ticker, retries=retries, backoff=backoff)

    def _fetch_chain() -> pd.DataFrame:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)
        rows: list[pd.DataFrame] = []
        for exp_str in expirations:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp - as_of).days
            if dte < min_dte or dte > max_dte:
                continue
            opt = tk.option_chain(exp_str)
            for kind, frame in (("call", opt.calls), ("put", opt.puts)):
                if frame is None or frame.empty:
                    continue
                f = frame.copy()
                f["option_type"] = kind
                f["expiry"] = exp
                rows.append(f)
        if not rows:
            raise ConnectionError(f"no options data for {ticker}")
        wide = pd.concat(rows, ignore_index=True)
        return _normalise_chain(wide)

    try:
        chain = _retry(_fetch_chain, retries, backoff)
        if cache_dir is not None:
            save_chain(chain, cache_dir, ticker, as_of)
    except LiveDataError as exc:
        if cache_dir is None:
            raise
        for back in range(1, 8):
            cached_date = as_of - timedelta(days=back)
            cached = load_chain(cache_dir, ticker, cached_date)
            if cached is not None:
                logger.warning("using cached chain from %s for %s", cached_date, ticker)
                return LiveChain(ticker=ticker, as_of=cached_date, spot=spot, chain=cached)
        raise LiveDataError(f"no cache available for {ticker}: {exc}") from exc

    return LiveChain(ticker=ticker, as_of=as_of, spot=spot, chain=chain)


def _normalise_chain(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance chain into a uniform schema.

    yfinance behaviour varies depending on whether the market is open:
      - During session: ``bid`` and ``ask`` are populated, ``impliedVolatility``
        is computed at last price (often stale).
      - After session: ``bid`` and ``ask`` are commonly zero; only
        ``lastPrice`` and ``impliedVolatility`` are populated.

    We accept rows that satisfy ALL of:
      - ``impliedVolatility`` in [0.05, 1.50] (anything else is a placeholder
        like 1e-5 or a clearly broken outlier).
      - A non-zero price source (mid if bid/ask are real, else lastPrice).
      - For rows with a real two-sided market, sanity-check the spread isn't
        wider than the mid (a sign of stale quotes).
    """
    out = pd.DataFrame()
    out["option_type"] = raw["option_type"]
    out["strike"] = raw["strike"].astype(float)
    out["expiry"] = pd.to_datetime(raw["expiry"]).dt.date
    out["bid"] = raw.get("bid", 0.0).astype(float)
    out["ask"] = raw.get("ask", 0.0).astype(float)
    last = raw.get("lastPrice", 0.0).astype(float)
    real_mid = (out["bid"] + out["ask"]) / 2.0
    has_two_sided = (out["bid"] > 0) & (out["ask"] > 0)
    out["mid"] = real_mid.where(has_two_sided, last)
    out["iv"] = raw.get("impliedVolatility", 0.0).astype(float)
    out["volume"] = raw.get("volume", 0).fillna(0).astype(int)
    out["oi"] = raw.get("openInterest", 0).fillna(0).astype(int)
    # Sane IV band (drops the 1e-5 placeholders) + non-zero price.
    keep = (out["iv"] >= 0.05) & (out["iv"] <= 1.50) & (out["mid"] > 0)
    # When bid/ask are real, also require the spread not to exceed the mid.
    sane_spread = ~has_two_sided | ((out["ask"] - out["bid"]) < out["mid"])
    out = out[keep & sane_spread]
    return out.reset_index(drop=True)
