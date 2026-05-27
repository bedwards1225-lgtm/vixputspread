"""``vrp positions`` — list positions filtered by status."""
from __future__ import annotations

from pathlib import Path

from vrp.config import Config
from vrp.persistence.db import connect, list_positions


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    status = None if args.status == "all" else args.status
    rows = list_positions(conn, status=status)
    if not rows:
        print("No positions match.")
        return 0
    print(f"{'ID':>4} {'Ticker':<6} {'Type':<14} {'Entry':<10} {'Expiry':<10} "
          f"{'Short':>8} {'Long':>8} {'N':>4} {'Cred':>7} {'P&L':>9} {'Status':<8} Reason")
    print("-" * 110)
    for p in rows:
        pnl = "" if p.pnl is None else f"{p.pnl:>9,.0f}"
        reason = p.exit_reason or ""
        print(
            f"{p.id:>4d} {p.ticker:<6} {p.type:<14} {p.entry_date} {p.expiry} "
            f"{p.short_strike:>8.2f} {p.long_strike:>8.2f} {p.contracts:>4d} "
            f"{p.credit_or_debit:>7.2f} {pnl:>9} {p.status:<8} {reason}"
        )
    return 0
