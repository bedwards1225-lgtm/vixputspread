"""``vrp add`` — record a manually placed credit spread."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from vrp.config import Config
from vrp.persistence.db import connect, insert_position
from vrp.persistence.models import Leg, Position


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    expiry = datetime.strptime(args.expiry, "%Y-%m-%d").date()
    pos = Position(
        ticker=args.ticker,
        type="credit_spread",
        entry_date=date.today(),
        expiry=expiry,
        short_strike=args.short,
        long_strike=args.long,
        contracts=args.contracts,
        credit_or_debit=args.credit,
        iv_at_entry=0.0,
        vrp_at_entry=0.0,
        regime="manual",
        status="open",
        legs=[
            Leg(side="short", option_type="put", strike=args.short, expiry=expiry,
                contracts=args.contracts, entry_price=args.credit, entry_iv=0.0,
                entry_delta=0.0, entry_gamma=0.0, entry_vega=0.0, entry_theta=0.0),
            Leg(side="long", option_type="put", strike=args.long, expiry=expiry,
                contracts=args.contracts, entry_price=0.0, entry_iv=0.0,
                entry_delta=0.0, entry_gamma=0.0, entry_vega=0.0, entry_theta=0.0),
        ],
    )
    pid = insert_position(conn, pos)
    print(f"Added position #{pid}: {args.ticker} {args.short}/{args.long} x{args.contracts} credit={args.credit:.2f}")
    return 0
