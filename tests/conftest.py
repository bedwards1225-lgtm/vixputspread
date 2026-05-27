"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vrp.config import Config


@pytest.fixture
def cfg() -> Config:
    """Default config with built-in defaults."""
    return Config()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """Synthetic OHLC for ~600 days with mild GBM dynamics."""
    rng = np.random.default_rng(123)
    n = 600
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    intraday = rng.normal(0, 0.008, n)
    open_ = close * np.exp(-intraday)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 0},
        index=dates,
    )
    return df


@pytest.fixture
def fake_history_lookup(synthetic_ohlc):
    """A history_lookup compatible with BlackScholesProvider for testing."""
    def lookup(ticker: str, start, end):
        df = synthetic_ohlc.copy()
        df.index.name = "date"
        return df
    return lookup
