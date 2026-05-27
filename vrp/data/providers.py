"""Backtest data providers.

Two flavours of provider:

1. *Real-data* providers (ORATS / Polygon / CBOE) load historical option
   chains from CSV dumps on disk. The exact column names differ by vendor;
   subclasses normalise into the canonical schema.

2. The :class:`BlackScholesProvider` synthesizes a chain from historical
   VIX and underlying OHLC. It is intended only as a fallback for users
   without a paid data subscription; backtest results are approximations
   and a warning is printed by the CLI when this provider is selected.

Canonical chain schema (DataFrame columns):
    option_type   : "call" | "put"
    strike        : float
    expiry        : datetime.date
    bid, ask, mid : float
    iv            : float (decimal, e.g. 0.20)
    underlying    : float (spot at as_of)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vrp.pricing import black_scholes as bsm
from vrp.pricing.surface import VolSurface, fit_surface

logger = logging.getLogger(__name__)


@dataclass
class ChainSnapshot:
    """A historical chain snapshot for one ticker on one date."""

    ticker: str
    as_of: date
    spot: float
    chain: pd.DataFrame


class OptionsDataProvider(ABC):
    """Abstract base for historical options data providers."""

    is_approximation: bool = False

    @abstractmethod
    def trading_days(self, start: date, end: date) -> list[date]:
        """Return the list of trading days available in ``[start, end]``."""

    @abstractmethod
    def get_underlying(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """OHLC DataFrame indexed by date for ``ticker`` in ``[start, end]``."""

    @abstractmethod
    def get_chain(self, ticker: str, as_of: date) -> ChainSnapshot:
        """Return the full option chain snapshot for ``ticker`` on ``as_of``."""

    def get_vix(self, start: date, end: date) -> pd.Series:
        """VIX close series; default impl pulls from underlying handler."""
        df = self.get_underlying("^VIX", start, end)
        return df["Close"]


class BlackScholesProvider(OptionsDataProvider):
    """Synthetic-IV provider for users without a paid options data feed.

    The IV surface on each historical date is constructed from:
      - ATM IV  = VIX(t) * scale[ticker]
      - Skew    = linear-in-delta slope (config-driven), constant across T
      - Term    = flat (no term structure)

    This is documented to the user as a CRUDE approximation.
    """

    is_approximation = True

    def __init__(
        self,
        *,
        rate: float,
        dividend_yields: dict[str, float],
        vix_scale: dict[str, float],
        skew_slope: float,
        history_lookup,
        strike_step_pct: float = 0.005,
        n_strikes: int = 81,
        expiries_dte: tuple[int, ...] = (7, 14, 21, 30, 37, 44, 60, 90),
    ) -> None:
        """
        Args:
            rate: Risk-free rate.
            dividend_yields: Map ticker -> q.
            vix_scale: Per-ticker scaling vs VIX.
            skew_slope: Linear-in-delta skew slope (positive => OTM puts richer).
            history_lookup: Callable ``(ticker, start, end) -> DataFrame``.
                Decouples the provider from a specific data source so
                tests can inject fixture data.
            strike_step_pct: Strike grid spacing as fraction of spot.
            n_strikes: Total number of strikes per expiry (odd => spot-centred).
            expiries_dte: DTEs at which to generate synthetic chains.
        """
        self.rate = rate
        self.dividend_yields = dividend_yields
        self.vix_scale = vix_scale
        self.skew_slope = skew_slope
        self._history_lookup = history_lookup
        self.strike_step_pct = strike_step_pct
        self.n_strikes = n_strikes
        self.expiries_dte = expiries_dte
        self._underlying_cache: dict[str, pd.DataFrame] = {}

    def trading_days(self, start: date, end: date) -> list[date]:
        spy = self.get_underlying("SPY", start, end)
        return [d.date() if hasattr(d, "date") else d for d in spy.index.to_pydatetime()]

    def get_underlying(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        key = ticker
        if key not in self._underlying_cache:
            df = self._history_lookup(ticker, start, end)
            df.index = pd.to_datetime(df.index)
            self._underlying_cache[key] = df
        df = self._underlying_cache[key]
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    def get_chain(self, ticker: str, as_of: date) -> ChainSnapshot:
        start = as_of - timedelta(days=10)
        und = self.get_underlying(ticker, start, as_of)
        if und.empty:
            raise ValueError(f"no underlying data for {ticker} on {as_of}")
        spot = float(und["Close"].iloc[-1])
        vix_series = self.get_vix(start, as_of)
        if vix_series.empty:
            raise ValueError(f"no VIX data on {as_of}")
        vix_close = float(vix_series.iloc[-1]) / 100.0  # decimal
        scale = self.vix_scale.get(ticker, 1.0)
        atm_iv = vix_close * scale
        q = self.dividend_yields.get(ticker, 0.0)

        rows: list[dict] = []
        half = self.n_strikes // 2
        for dte in self.expiries_dte:
            expiry = as_of + timedelta(days=int(dte))
            T = max(dte / 365.0, 1e-6)
            for i in range(-half, half + 1):
                strike = round(spot * (1.0 + i * self.strike_step_pct), 2)
                if strike <= 0:
                    continue
                for opt_type in ("put", "call"):
                    iv = self._iv_at_strike(spot, strike, T, atm_iv, opt_type, q)
                    mid = float(bsm.price(spot, strike, T, self.rate, q, iv, opt_type))
                    spread = max(0.05, 0.02 * mid)  # synthetic bid/ask
                    bid = max(mid - 0.5 * spread, 0.0)
                    ask = mid + 0.5 * spread
                    rows.append(
                        {
                            "option_type": opt_type,
                            "strike": strike,
                            "expiry": expiry,
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "iv": iv,
                            "underlying": spot,
                        }
                    )
        return ChainSnapshot(ticker=ticker, as_of=as_of, spot=spot, chain=pd.DataFrame(rows))

    def _iv_at_strike(
        self,
        spot: float,
        strike: float,
        T: float,
        atm_iv: float,
        option_type: str,
        q: float,
    ) -> float:
        """Compute IV at ``strike`` from ATM IV via a linear-in-delta skew.

        Index-equity smile shape:
          - OTM puts  (delta close to 0):   IV > ATM    (left-tail premium)
          - ATM       (|delta| ~ 0.5):      IV = ATM
          - ITM puts  (delta close to -1):  IV < ATM

        Linear-in-delta parameterisation:
          puts:  IV = ATM * (1 + slope * (d + 0.5))    d in [-1, 0]
          calls: IV = ATM * (1 - slope * (d - 0.5))    d in [0, 1]

        With slope > 0 this gives OTM puts the highest IV (correct for SPX
        smile) and OTM calls slightly below ATM.
        """
        d_atm = float(
            bsm.delta(spot, strike, T, self.rate, q, atm_iv, option_type)
        )
        if option_type == "put":
            # d_atm in [-1, 0]. For OTM puts d_atm close to 0 -> (d+0.5) close to +0.5
            # -> IV pumped up. For ITM puts d_atm close to -1 -> IV pushed down.
            adj = self.skew_slope * (d_atm + 0.5)
        else:
            # d_atm in [0, 1]. For OTM calls d_atm close to 0 -> (d-0.5) close to -0.5
            # -> IV pushed down (mild). For ITM calls d_atm close to 1 -> IV pumped up.
            adj = -self.skew_slope * (d_atm - 0.5)
        iv = atm_iv * (1.0 + adj)
        return float(np.clip(iv, 0.05, 2.0))


class CsvBackedProvider(OptionsDataProvider):
    """Common base for providers that read pre-saved CSV chains.

    Expects:
        ``{root}/underlying/{ticker}.csv`` with columns ``date,Open,High,Low,Close,Volume``
        ``{root}/chains/{ticker}/{YYYY-MM-DD}.csv`` with provider-specific cols.
    """

    is_approximation = False

    def __init__(self, root: Path) -> None:
        self.root = root
        self._underlying_cache: dict[str, pd.DataFrame] = {}

    def trading_days(self, start: date, end: date) -> list[date]:
        spy = self.get_underlying("SPY", start, end)
        return [d.date() for d in spy.index.to_pydatetime()]

    def get_underlying(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if ticker not in self._underlying_cache:
            path = self.root / "underlying" / f"{ticker}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing underlying file: {path}")
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            self._underlying_cache[ticker] = df
        df = self._underlying_cache[ticker]
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    def get_chain(self, ticker: str, as_of: date) -> ChainSnapshot:
        path = self.root / "chains" / ticker / f"{as_of.isoformat()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing chain: {path}")
        raw = pd.read_csv(path)
        chain = self._normalise(raw)
        spot = float(chain["underlying"].iloc[0])
        return ChainSnapshot(ticker=ticker, as_of=as_of, spot=spot, chain=chain)

    @abstractmethod
    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map provider-specific columns to the canonical schema."""


class OratsProvider(CsvBackedProvider):
    """ORATS one-minute snapshots, end-of-day flavoured. See README for cols."""

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["option_type"] = raw["type"].str.lower().map({"c": "call", "p": "put", "call": "call", "put": "put"})
        out["strike"] = raw["strike"].astype(float)
        out["expiry"] = pd.to_datetime(raw["expirDate"]).dt.date
        out["bid"] = raw["bidPrice"].astype(float)
        out["ask"] = raw["askPrice"].astype(float)
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out["iv"] = raw["smvVol"].astype(float)
        out["underlying"] = raw["stockPrice"].astype(float)
        return out.dropna(subset=["iv"]).reset_index(drop=True)


class PolygonProvider(CsvBackedProvider):
    """Polygon.io flat-files schema."""

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["option_type"] = raw["contract_type"].str.lower()
        out["strike"] = raw["strike_price"].astype(float)
        out["expiry"] = pd.to_datetime(raw["expiration_date"]).dt.date
        out["bid"] = raw["bid"].astype(float)
        out["ask"] = raw["ask"].astype(float)
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out["iv"] = raw["implied_volatility"].astype(float)
        out["underlying"] = raw["underlying_price"].astype(float)
        return out.dropna(subset=["iv"]).reset_index(drop=True)


class CboeDatashopProvider(CsvBackedProvider):
    """CBOE DataShop end-of-day option chain CSV format."""

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["option_type"] = raw["option_type"].str.lower()
        out["strike"] = raw["strike"].astype(float)
        out["expiry"] = pd.to_datetime(raw["expiration"]).dt.date
        out["bid"] = raw["bid_eod"].astype(float)
        out["ask"] = raw["ask_eod"].astype(float)
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out["iv"] = raw["iv"].astype(float)
        out["underlying"] = raw["underlying_last"].astype(float)
        return out.dropna(subset=["iv"]).reset_index(drop=True)


def build_surface(snapshot: ChainSnapshot, rate: float, div_yield: float) -> VolSurface:
    """Convenience: fit a VolSurface from a ChainSnapshot."""
    return fit_surface(
        snapshot.chain[["strike", "expiry", "iv"]],
        as_of=snapshot.as_of,
        spot=snapshot.spot,
        rate=rate,
        div_yield=div_yield,
    )


PROVIDERS = {
    "bsm": BlackScholesProvider,
    "orats": OratsProvider,
    "polygon": PolygonProvider,
    "cboe": CboeDatashopProvider,
}
