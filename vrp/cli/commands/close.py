"""``vrp close`` — manually close an open position."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from vrp.config import Config
from vrp.persistence.db import connect, get_position, update_position_exit


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    pos = get_position(conn, args.id)
    if pos is None:
        print(f"Position #{args.id} not found.")
        return 2
    if pos.status != "open":
        print(f"Position #{args.id} is not open (status={pos.status}).")
        return 2
    pnl = (pos.credit_or_debit - args.exit_price) * 100 * pos.contracts
    update_position_exit(
        conn,
        args.id,
        exit_date=date.today(),
        exit_reason=args.reason,
        exit_price=args.exit_price,
        pnl=pnl,
        commissions=pos.commissions,
    )
    print(f"Closed #{args.id} @ ${args.exit_price:.2f} -> P&L ${pnl:,.0f} ({args.reason})")
    return 0
