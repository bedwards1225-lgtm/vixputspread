"""CLI smoke tests: subcommands run on a fresh empty database without crashing."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from vrp.cli.main import build_parser, main
from vrp.config import Config


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch):
    """Create a config pointing all paths into tmp_path."""
    cfg = Config()
    cfg.persistence.db_path = str(tmp_path / "smoke.db")
    cfg.data.cache_dir = str(tmp_path / "cache")
    cfg.data.surface_cache_dir = str(tmp_path / "surfaces")
    cfg.logging.log_dir = str(tmp_path / "logs")
    cfg.backtest.results_dir = str(tmp_path / "results")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump()))
    return cfg_path


def test_positions_empty(cli_env):
    rc = main(["--config", str(cli_env), "positions", "--status", "open"])
    assert rc == 0


def test_stats_empty(cli_env):
    rc = main(["--config", str(cli_env), "stats", "--period", "all"])
    assert rc == 0


def test_add_then_close(cli_env):
    rc = main([
        "--config", str(cli_env), "add",
        "--ticker", "SPY", "--short", "470", "--long", "468",
        "--expiry", "2099-01-15", "--credit", "0.85", "--contracts", "2",
    ])
    assert rc == 0
    rc = main([
        "--config", str(cli_env), "close",
        "--id", "1", "--exit-price", "0.40", "--reason", "MANUAL",
    ])
    assert rc == 0
    rc = main(["--config", str(cli_env), "positions", "--status", "closed"])
    assert rc == 0


def test_stress_no_positions(cli_env):
    rc = main(["--config", str(cli_env), "stress", "--scenario", "all"])
    assert rc == 0


def test_help_builds():
    parser = build_parser()
    assert parser is not None
