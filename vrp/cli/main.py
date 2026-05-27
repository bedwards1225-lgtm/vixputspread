"""argparse dispatcher for the ``vrp`` CLI.

Each subcommand lives in its own module under :mod:`vrp.cli.commands`. The
dispatcher loads global config (via ``--config``), configures logging, then
hands ``args`` to the subcommand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vrp.config import Config, load_config
from vrp.logging_setup import configure_logging


def _add_global(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config (defaults: built-in defaults)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrp", description="Variance Risk Premium harvesting CLI.")
    _add_global(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Scan for new entry signals.")
    s.add_argument("--ticker", action="append", help="Limit scan to one or more tickers.")

    sub.add_parser("monitor", help="Check open positions for exit signals.")
    sub.add_parser("daily", help="Run scan + monitor + hedge check (suitable for cron).")

    b = sub.add_parser("backtest", help="Run a historical backtest.")
    b.add_argument("--start", required=True, help="YYYY-MM-DD")
    b.add_argument("--end", required=True, help="YYYY-MM-DD")
    b.add_argument(
        "--data-provider", default="bsm",
        choices=["bsm", "orats", "polygon", "cboe"],
        help="Historical data backend.",
    )
    b.add_argument("--data-root", type=Path, default=None,
                   help="Root directory for paid-data CSVs (required for non-bsm).")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--tag", default=None, help="Subdirectory name for results.")

    st = sub.add_parser("stress", help="Run stress scenarios on currently open positions.")
    st.add_argument(
        "--scenario", choices=["gfc", "volmageddon", "covid", "all"], default="all"
    )

    a = sub.add_parser("add", help="Manually record a new credit-spread position.")
    a.add_argument("--ticker", required=True)
    a.add_argument("--short", type=float, required=True)
    a.add_argument("--long", type=float, required=True)
    a.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    a.add_argument("--credit", type=float, required=True)
    a.add_argument("--contracts", type=int, required=True)

    c = sub.add_parser("close", help="Manually close an open position.")
    c.add_argument("--id", type=int, required=True)
    c.add_argument("--exit-price", type=float, required=True)
    c.add_argument("--reason", default="MANUAL")

    p = sub.add_parser("positions", help="List positions.")
    p.add_argument("--status", choices=["open", "closed", "expired", "all"], default="open")

    stats = sub.add_parser("stats", help="Show portfolio statistics.")
    stats.add_argument("--period", default="all", choices=["30d", "90d", "ytd", "all"])

    sub.add_parser("hedge-status", help="Show current hedge coverage and gap to target.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a Unix exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.verbose:
        cfg.logging.level = "DEBUG"
    configure_logging(cfg.logging)

    # Dispatch.
    if args.command == "scan":
        from vrp.cli.commands import scan
        return scan.run(args, cfg)
    if args.command == "monitor":
        from vrp.cli.commands import monitor
        return monitor.run(args, cfg)
    if args.command == "daily":
        from vrp.cli.commands import daily
        return daily.run(args, cfg)
    if args.command == "backtest":
        from vrp.cli.commands import backtest
        return backtest.run(args, cfg)
    if args.command == "stress":
        from vrp.cli.commands import stress
        return stress.run(args, cfg)
    if args.command == "add":
        from vrp.cli.commands import add
        return add.run(args, cfg)
    if args.command == "close":
        from vrp.cli.commands import close
        return close.run(args, cfg)
    if args.command == "positions":
        from vrp.cli.commands import positions
        return positions.run(args, cfg)
    if args.command == "stats":
        from vrp.cli.commands import stats as stats_cmd
        return stats_cmd.run(args, cfg)
    if args.command == "hedge-status":
        from vrp.cli.commands import hedge_status
        return hedge_status.run(args, cfg)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
