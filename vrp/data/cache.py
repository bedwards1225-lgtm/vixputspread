"""On-disk caching for chains (parquet) and vol surfaces (pickle).

Cache keys are derived from ticker + as_of date so the same backtest day
hits a cached chain instead of re-downloading or re-fitting.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from vrp.pricing.surface import VolSurface, load_surface, save_surface

logger = logging.getLogger(__name__)


def chain_cache_path(cache_dir: Path, ticker: str, as_of: date) -> Path:
    """Resolve the parquet path for ``(ticker, as_of)``."""
    return cache_dir / "chains" / ticker / f"{as_of.isoformat()}.parquet"


def surface_cache_path(cache_dir: Path, ticker: str, as_of: date) -> Path:
    """Resolve the pickle path for a cached surface."""
    return cache_dir / ticker / f"{as_of.isoformat()}.pkl"


def save_chain(df: pd.DataFrame, cache_dir: Path, ticker: str, as_of: date) -> Path:
    """Write a chain DataFrame to parquet under ``cache_dir``."""
    path = chain_cache_path(cache_dir, ticker, as_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_chain(cache_dir: Path, ticker: str, as_of: date) -> pd.DataFrame | None:
    """Load a cached chain DataFrame, or ``None`` if absent."""
    path = chain_cache_path(cache_dir, ticker, as_of)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_surface_to_cache(
    surface: VolSurface, cache_dir: Path, ticker: str, as_of: date
) -> Path:
    """Pickle a fitted surface to the cache directory."""
    path = surface_cache_path(cache_dir, ticker, as_of)
    save_surface(surface, path)
    return path


def load_surface_from_cache(
    cache_dir: Path, ticker: str, as_of: date
) -> VolSurface | None:
    """Load a cached fitted surface, or ``None`` if absent."""
    path = surface_cache_path(cache_dir, ticker, as_of)
    if not path.exists():
        return None
    return load_surface(path)
