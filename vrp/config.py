"""Strongly-typed configuration loaded from YAML.

All tunable strategy parameters live here. The runtime never reads magic
numbers directly; modules accept a typed config and operate on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class UniverseConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "IWM", "QQQ"])
    dividend_yields: dict[str, float] = Field(
        default_factory=lambda: {"SPY": 0.013, "IWM": 0.014, "QQQ": 0.006}
    )
    vix_scale: dict[str, float] = Field(
        default_factory=lambda: {"SPY": 1.00, "IWM": 1.15, "QQQ": 1.10}
    )


class PricingConfig(BaseModel):
    risk_free_rate: float = 0.043
    bsm_skew_slope: float = 0.80


class ForecastingConfig(BaseModel):
    har_window_days: int = 252
    refit_every_days: int = 5
    yang_zhang_window: int = 22


class EntryConfig(BaseModel):
    min_dte: int = 30
    max_dte: int = 45
    delta_target_low_vol: tuple[float, float] = (0.20, 0.25)
    delta_target_mid_vol: tuple[float, float] = (0.15, 0.20)
    delta_target_high_vol: tuple[float, float] = (0.10, 0.15)
    min_credit_pct_of_width: float = 0.20
    target_credit_pct_of_width: float = 0.33
    vrp_lookback_days: int = 90
    vrp_percentile_min: float = 0.30
    skew_lookback_days: int = 252
    skew_high_percentile: float = 0.90
    skew_size_reduction: float = 0.50
    min_days_between_entries_same_ticker: int = 7
    max_entries_per_day: int = 1


class ExitConfig(BaseModel):
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    gamma_stop_multiple: float = 3.0
    skew_reversal_sigmas: float = 2.0
    hard_time_stop_dte: int = 5


class SizingConfig(BaseModel):
    starting_equity: float = 100_000.0
    max_loss_per_trade_pct: float = 0.015
    portfolio_var_cap_pct: float = 0.05
    var_confidence: float = 0.99
    var_lookback_days: int = 252
    correlation_lookback_days: int = 60
    net_short_vega_cap_pct: float = 0.005
    net_short_gamma_cap_pct: float = 0.001
    net_delta_cap_pct: float = 0.05
    round_contracts: Literal["floor", "nearest"] = "floor"
    skip_if_zero: bool = True


class HedgingConfig(BaseModel):
    enabled: bool = True
    ticker: str = "SPY"
    otm_pct: float = 0.06
    min_dte: int = 60
    max_dte: int = 90
    roll_dte: int = 30
    stress_spx_drop: float = 0.08
    stress_vix_level: float = 50.0
    stress_skew_percentile: float = 0.99
    hedge_coverage_target: float = 0.50
    annual_premium_budget_pct: float = 0.04


class BacktestConfig(BaseModel):
    commission_per_contract: float = 0.65
    entry_slippage_pct_of_spread: float = 0.50
    exit_slippage_pct_of_spread: float = 0.50
    hedge_slippage_pct_of_spread: float = 0.60
    results_dir: str = "backtest_results"


class DataConfig(BaseModel):
    cache_dir: str = "data_cache"
    surface_cache_dir: str = "surface_cache"
    yfinance_retries: int = 3
    yfinance_backoff_seconds: float = 2.0
    history_days_for_har: int = 400


class PersistenceConfig(BaseModel):
    db_path: str = "vrp.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "vrp.log"

    @field_validator("level")
    @classmethod
    def _level_upper(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {v!r}")
        return v


class AlertsConfig(BaseModel):
    discord_webhook_url: str | None = None
    discord_username: str = "VRP"


class Config(BaseModel):
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


def load_config(path: Path | None = None) -> Config:
    """Load a YAML config file, returning defaults when the file is absent.

    Args:
        path: Path to a YAML config file. ``None`` returns defaults.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If the YAML is malformed or fails validation.
    """
    if path is None:
        return Config()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
