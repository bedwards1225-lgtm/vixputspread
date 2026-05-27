"""``vrp stats`` — summarise closed-trade performance."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from vrp.config import Config
from vrp.persistence.db import connect, list_positions


def _period_start(period: str) -> date | None:
    today = date.today()
    if period == "all":
        return None
    if period == "30d":
        return today - timedelta(days=30)
    if period == "90d":
        return today - timedelta(days=90)
    if period == "ytd":
        return date(today.year, 1, 1)
    return None


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    rows = list_positions(conn, status="closed")
    cutoff = _period_start(args.period)
    if cutoff:
        rows = [p for p in rows if p.exit_date and p.exit_date >= cutoff]
    if not rows:
        print(f"No closed trades in period={args.period}.")
        return 0

    pnls = [p.pnl or 0.0 for p in rows]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    total = sum(pnls)
    print(f"Trades         : {len(rows)}")
    print(f"Win rate       : {len(wins)/len(rows):.1%}")
    print(f"Total P&L      : ${total:,.0f}")
    print(f"Avg win        : ${(sum(wins)/len(wins)) if wins else 0:,.0f}")
    print(f"Avg loss       : ${(sum(losses)/len(losses)) if losses else 0:,.0f}")
    print(f"Profit factor  : {(sum(wins)/abs(sum(losses))) if losses else float('inf'):.2f}")
    by_reason: dict[str, int] = {}
    for p in rows:
        r = p.exit_reason or "UNKNOWN"
        by_reason[r] = by_reason.get(r, 0) + 1
    print("Exit reasons:")
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:>4d}")
    return 0
