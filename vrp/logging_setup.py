"""Centralised logging configuration.

Call :func:`configure_logging` once at process start (from the CLI). Library
modules just do ``logger = logging.getLogger(__name__)``.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from vrp.config import LoggingConfig

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(cfg: LoggingConfig, *, console: bool = True) -> None:
    """Install console and rotating-file handlers on the root logger.

    Safe to call more than once; subsequent calls are no-ops to avoid
    duplicate handlers under pytest or repeated CLI invocations.

    Args:
        cfg: Logging configuration.
        console: Whether to attach a stderr console handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / cfg.log_file

    root = logging.getLogger()
    root.setLevel(cfg.level)

    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # Quiet noisy third-party loggers.
    for noisy in ("yfinance", "urllib3", "matplotlib", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
