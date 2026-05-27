"""``vrp daily`` — scan + monitor + hedge check; intended for cron."""
from __future__ import annotations

from vrp.config import Config
from vrp.cli.commands import scan, monitor, hedge_status


def run(args, cfg: Config) -> int:
    print("=== SCAN ===")
    rc = scan.run(args, cfg)
    print("\n=== MONITOR ===")
    rc = monitor.run(args, cfg) or rc
    print("\n=== HEDGE ===")
    rc = hedge_status.run(args, cfg) or rc
    return rc
