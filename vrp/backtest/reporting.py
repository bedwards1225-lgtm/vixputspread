"""Backtest reporting: stats, plots, CSV trade ledger.

Plots use ``matplotlib`` with the Agg backend so they work headless. The
``write_report`` function deposits everything under
``backtest.results_dir / {run_tag}/``.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp.backtest.engine import BacktestResult, TradeRecord
from vrp.persistence.models import DailySnapshot

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


@dataclass
class Stats:
    """Headline performance statistics."""

    cagr: float
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_drawdown: float
    time_underwater_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int


def compute_stats(snapshots: list[DailySnapshot], trades: list[TradeRecord]) -> Stats:
    """Compute the headline performance stats.

    Args:
        snapshots: Daily equity curve.
        trades: Closed trades for win/loss stats.

    Returns:
        :class:`Stats`.
    """
    if not snapshots:
        return Stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    eq = pd.Series(
        [s.portfolio_value for s in snapshots],
        index=pd.to_datetime([s.date for s in snapshots]),
    )
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0 if start_eq > 0 else 0.0
    total_return = end_eq / start_eq - 1.0 if start_eq > 0 else 0.0

    rets = eq.pct_change().dropna()
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(_TRADING_DAYS))
    else:
        sharpe = 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(_TRADING_DAYS)) if len(downside) > 1 and downside.std() > 0 else 0.0

    cummax = eq.cummax()
    drawdowns = (eq - cummax) / cummax
    max_dd = float(drawdowns.min())
    underwater = drawdowns < 0
    time_underwater = float(underwater.mean())
    avg_dd = float(drawdowns[drawdowns < 0].mean()) if underwater.any() else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    n_trades = len(trades)
    if n_trades:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = float((pnls > 0).mean())
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0

    return Stats(
        cagr=cagr,
        total_return=total_return,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        time_underwater_pct=time_underwater,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=n_trades,
    )


def exit_reason_distribution(trades: list[TradeRecord]) -> dict[str, int]:
    """Count trades by exit reason."""
    out: dict[str, int] = {}
    for t in trades:
        out[t.exit_reason] = out.get(t.exit_reason, 0) + 1
    return out


def monthly_pnl_heatmap(snapshots: list[DailySnapshot]) -> pd.DataFrame:
    """Return a year-by-month P&L heatmap (rows=year, cols=month, values=$)."""
    if not snapshots:
        return pd.DataFrame()
    eq = pd.Series(
        [s.portfolio_value for s in snapshots],
        index=pd.to_datetime([s.date for s in snapshots]),
    )
    monthly = eq.resample("ME").last().diff().dropna()
    df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "pnl": monthly.values,
        }
    )
    return df.pivot(index="year", columns="month", values="pnl").fillna(0.0)


def realized_vs_implied(trades: list[TradeRecord]) -> pd.DataFrame:
    """Build a DataFrame of (IV at entry, realized vol over trade life)."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "iv_at_entry": [t.iv_at_entry for t in trades],
            "vrp_at_entry": [t.vrp_at_entry for t in trades],
            "pnl": [t.pnl for t in trades],
            "exit_reason": [t.exit_reason for t in trades],
        }
    )


def write_report(result: BacktestResult, *, out_dir: Path, tag: str) -> Path:
    """Persist stats, plots, and trade CSVs to ``out_dir/tag/``.

    Returns the run-specific directory path.
    """
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_stats(result.snapshots, result.trades)
    dist = exit_reason_distribution(result.trades)
    heatmap = monthly_pnl_heatmap(result.snapshots)

    # --- Stats text ---
    text = _format_stats(stats, dist)
    (run_dir / "stats.txt").write_text(text)
    logger.info("\n%s", text)

    # --- Equity curve ---
    if result.snapshots:
        eq = pd.Series(
            [s.portfolio_value for s in result.snapshots],
            index=pd.to_datetime([s.date for s in result.snapshots]),
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        eq.plot(ax=ax)
        ax.set_title("Equity curve")
        ax.set_ylabel("Account value ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "equity_curve.png", dpi=120)
        plt.close(fig)

        # Drawdown
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax
        fig, ax = plt.subplots(figsize=(10, 3))
        dd.plot(ax=ax, color="crimson")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "drawdown.png", dpi=120)
        plt.close(fig)

    # --- Exit reason bar ---
    if dist:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(dist.keys()), list(dist.values()))
        ax.set_title("Exit reason distribution")
        ax.set_ylabel("Trade count")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(run_dir / "exit_reasons.png", dpi=120)
        plt.close(fig)

    # --- Monthly heatmap ---
    if not heatmap.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns)
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index)
        ax.set_title("Monthly P&L ($)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / "monthly_heatmap.png", dpi=120)
        plt.close(fig)

    # --- Realized vs implied scatter ---
    rvi = realized_vs_implied(result.trades)
    if not rvi.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(rvi["iv_at_entry"], rvi["pnl"], alpha=0.6, s=20)
        ax.axhline(0, color="grey", linestyle=":")
        ax.set_xlabel("IV at entry")
        ax.set_ylabel("Trade P&L ($)")
        ax.set_title("IV at entry vs trade P&L")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "iv_vs_pnl.png", dpi=120)
        plt.close(fig)

    # --- CSV ledger ---
    _write_trades_csv(result.trades, run_dir / "trades.csv")
    _write_snapshots_csv(result.snapshots, run_dir / "snapshots.csv")

    return run_dir


def _format_stats(stats: Stats, dist: dict[str, int]) -> str:
    """Render the stats block for stats.txt and console output."""
    lines = [
        "=" * 60,
        "BACKTEST SUMMARY",
        "=" * 60,
        f"CAGR              : {stats.cagr:>9.2%}",
        f"Total return      : {stats.total_return:>9.2%}",
        f"Sharpe (annual)   : {stats.sharpe:>9.2f}",
        f"Sortino           : {stats.sortino:>9.2f}",
        f"Calmar            : {stats.calmar:>9.2f}",
        f"Max drawdown      : {stats.max_drawdown:>9.2%}",
        f"Avg drawdown      : {stats.avg_drawdown:>9.2%}",
        f"Time underwater   : {stats.time_underwater_pct:>9.2%}",
        f"Win rate          : {stats.win_rate:>9.2%}",
        f"Avg win           : ${stats.avg_win:>9,.0f}",
        f"Avg loss          : ${stats.avg_loss:>9,.0f}",
        f"Profit factor     : {stats.profit_factor:>9.2f}",
        f"# trades          : {stats.n_trades:>9d}",
        "",
        "EXIT REASON DISTRIBUTION",
    ]
    total = sum(dist.values()) or 1
    for reason, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason:<18} {count:>5d}  ({count/total:>5.1%})")
    lines.append("=" * 60)
    return "\n".join(lines)


def _write_trades_csv(trades: list[TradeRecord], path: Path) -> None:
    if not trades:
        path.write_text("")
        return
    fieldnames = list(asdict(trades[0]).keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))


def _write_snapshots_csv(snapshots: list[DailySnapshot], path: Path) -> None:
    if not snapshots:
        path.write_text("")
        return
    fieldnames = list(asdict(snapshots[0]).keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in snapshots:
            writer.writerow(asdict(s))
