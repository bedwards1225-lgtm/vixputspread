"""``vrp stress`` — reprice currently open positions under stress scenarios."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from vrp.backtest.stress import SCENARIOS, run_all
from vrp.config import Config
from vrp.data import live
from vrp.persistence.db import connect, list_positions


def run(args, cfg: Config) -> int:
    conn = connect(Path(cfg.persistence.db_path))
    positions = list_positions(conn, status="open")
    if not positions:
        print("No open positions to stress.")
        return 0
    spots: dict[str, float] = {}
    for t in {p.ticker for p in positions}:
        spots[t] = live.get_spot(t)
    today = date.today()

    if args.scenario == "all":
        results = run_all(
            positions=positions, as_of=today,
            spot_lookup=lambda t: spots[t],
            rate=cfg.pricing.risk_free_rate,
            dividend_yields=cfg.universe.dividend_yields,
            vix_scale=cfg.universe.vix_scale,
        )
    else:
        sc = SCENARIOS[args.scenario]
        from vrp.backtest.stress import reprice_under
        results = {
            args.scenario: reprice_under(
                sc, positions=positions, as_of=today,
                spot_lookup=lambda t: spots[t],
                rate=cfg.pricing.risk_free_rate,
                dividend_yields=cfg.universe.dividend_yields,
                vix_scale=cfg.universe.vix_scale,
            )
        }
    print(f"{'Scenario':<14} {'Spread P&L':>14} {'Hedge P&L':>14} {'Net P&L':>14}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<14} ${r.spread_pnl:>12,.0f} ${r.hedge_pnl:>12,.0f} ${r.net_pnl:>12,.0f}")
    return 0
