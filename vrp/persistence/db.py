"""SQLite schema, migrations, and CRUD.

The DB owns one schema version. :func:`connect` opens (creating if absent)
the file at the configured path, applies migrations, and returns a
``sqlite3.Connection`` with row-factory and FK pragmas enabled.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

from vrp.persistence.models import (
    DailySnapshot,
    HarRvFit,
    Leg,
    Position,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        type TEXT NOT NULL,
        entry_date DATE NOT NULL,
        expiry DATE NOT NULL,
        short_strike REAL NOT NULL,
        long_strike REAL NOT NULL,
        contracts INTEGER NOT NULL,
        credit_or_debit REAL NOT NULL,
        iv_at_entry REAL NOT NULL,
        vrp_at_entry REAL NOT NULL,
        regime TEXT NOT NULL,
        status TEXT NOT NULL,
        entry_underlying_price REAL DEFAULT 0,
        entry_vix REAL DEFAULT 0,
        entry_delta_short REAL DEFAULT 0,
        exit_date DATE,
        exit_reason TEXT,
        exit_price REAL,
        pnl REAL,
        commissions REAL DEFAULT 0,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS legs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
        side TEXT NOT NULL,
        option_type TEXT NOT NULL,
        strike REAL NOT NULL,
        expiry DATE NOT NULL,
        contracts INTEGER NOT NULL,
        entry_price REAL,
        entry_iv REAL,
        entry_delta REAL,
        entry_gamma REAL,
        entry_vega REAL,
        entry_theta REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_snapshots (
        date DATE PRIMARY KEY,
        vix REAL,
        vrp REAL,
        portfolio_value REAL,
        portfolio_var REAL,
        portfolio_delta REAL,
        portfolio_gamma REAL,
        portfolio_vega REAL,
        open_positions INTEGER,
        hedge_coverage REAL,
        realized_vol_22d REAL,
        iv_atm REAL,
        skew_25d REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS har_rv_models (
        fit_date DATE PRIMARY KEY,
        beta_0 REAL,
        beta_daily REAL,
        beta_weekly REAL,
        beta_monthly REAL,
        r_squared REAL,
        n_obs INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)",
    "CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_positions_entry  ON positions(entry_date)",
    "CREATE INDEX IF NOT EXISTS idx_legs_position    ON legs(position_id)",
)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection, applying schema and migrations.

    Args:
        db_path: Path to the SQLite file. Created if missing.

    Returns:
        An open connection with foreign keys enabled and ``Row`` factory.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Create tables if absent and write the current schema version."""
    with conn:
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )


# ----- Position CRUD --------------------------------------------------------

def insert_position(conn: sqlite3.Connection, p: Position) -> int:
    """Insert a position and its legs; return the new position id."""
    with conn:
        cur = conn.execute(
            """
            INSERT INTO positions(
                ticker, type, entry_date, expiry, short_strike, long_strike,
                contracts, credit_or_debit, iv_at_entry, vrp_at_entry, regime,
                status, entry_underlying_price, entry_vix, entry_delta_short,
                exit_date, exit_reason, exit_price, pnl, commissions, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.ticker, p.type, p.entry_date, p.expiry,
                p.short_strike, p.long_strike, p.contracts, p.credit_or_debit,
                p.iv_at_entry, p.vrp_at_entry, p.regime, p.status,
                p.entry_underlying_price, p.entry_vix, p.entry_delta_short,
                p.exit_date, p.exit_reason, p.exit_price, p.pnl,
                p.commissions, p.notes,
            ),
        )
        pid = cur.lastrowid
        for leg in p.legs:
            conn.execute(
                """
                INSERT INTO legs(
                    position_id, side, option_type, strike, expiry, contracts,
                    entry_price, entry_iv, entry_delta, entry_gamma,
                    entry_vega, entry_theta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid, leg.side, leg.option_type, leg.strike, leg.expiry,
                    leg.contracts, leg.entry_price, leg.entry_iv,
                    leg.entry_delta, leg.entry_gamma, leg.entry_vega,
                    leg.entry_theta,
                ),
            )
    p.id = pid
    return pid


def update_position_exit(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    exit_date: date,
    exit_reason: str,
    exit_price: float,
    pnl: float,
    commissions: float,
    status: str = "closed",
) -> None:
    """Record exit fields on an existing position."""
    with conn:
        conn.execute(
            """
            UPDATE positions
               SET status = ?, exit_date = ?, exit_reason = ?,
                   exit_price = ?, pnl = ?, commissions = ?
             WHERE id = ?
            """,
            (status, exit_date, exit_reason, exit_price, pnl, commissions, position_id),
        )


def list_positions(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    ticker: str | None = None,
    type_: str | None = None,
) -> list[Position]:
    """Return positions filtered by optional status / ticker / type."""
    sql = "SELECT * FROM positions WHERE 1=1"
    args: list = []
    if status is not None:
        sql += " AND status = ?"
        args.append(status)
    if ticker is not None:
        sql += " AND ticker = ?"
        args.append(ticker)
    if type_ is not None:
        sql += " AND type = ?"
        args.append(type_)
    sql += " ORDER BY entry_date, id"
    rows = conn.execute(sql, args).fetchall()
    out: list[Position] = []
    for row in rows:
        legs = _legs_for(conn, row["id"])
        out.append(_row_to_position(row, legs))
    return out


def get_position(conn: sqlite3.Connection, position_id: int) -> Position | None:
    """Fetch a single position by id, with legs."""
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    if row is None:
        return None
    return _row_to_position(row, _legs_for(conn, position_id))


def _legs_for(conn: sqlite3.Connection, position_id: int) -> list[Leg]:
    rows = conn.execute(
        "SELECT * FROM legs WHERE position_id = ? ORDER BY id", (position_id,)
    ).fetchall()
    return [
        Leg(
            id=r["id"],
            position_id=r["position_id"],
            side=r["side"],
            option_type=r["option_type"],
            strike=r["strike"],
            expiry=_to_date(r["expiry"]),
            contracts=r["contracts"],
            entry_price=r["entry_price"],
            entry_iv=r["entry_iv"],
            entry_delta=r["entry_delta"],
            entry_gamma=r["entry_gamma"],
            entry_vega=r["entry_vega"],
            entry_theta=r["entry_theta"],
        )
        for r in rows
    ]


def _row_to_position(row: sqlite3.Row, legs: list[Leg]) -> Position:
    return Position(
        id=row["id"],
        ticker=row["ticker"],
        type=row["type"],
        entry_date=_to_date(row["entry_date"]),
        expiry=_to_date(row["expiry"]),
        short_strike=row["short_strike"],
        long_strike=row["long_strike"],
        contracts=row["contracts"],
        credit_or_debit=row["credit_or_debit"],
        iv_at_entry=row["iv_at_entry"],
        vrp_at_entry=row["vrp_at_entry"],
        regime=row["regime"],
        status=row["status"],
        entry_underlying_price=row["entry_underlying_price"] or 0.0,
        entry_vix=row["entry_vix"] or 0.0,
        entry_delta_short=row["entry_delta_short"] or 0.0,
        exit_date=_to_date(row["exit_date"]) if row["exit_date"] else None,
        exit_reason=row["exit_reason"],
        exit_price=row["exit_price"],
        pnl=row["pnl"],
        commissions=row["commissions"] or 0.0,
        notes=row["notes"],
        legs=legs,
    )


# ----- Snapshots and HAR-RV fits --------------------------------------------

def upsert_snapshot(conn: sqlite3.Connection, s: DailySnapshot) -> None:
    """Insert or replace a daily snapshot keyed by date."""
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_snapshots(
                date, vix, vrp, portfolio_value, portfolio_var,
                portfolio_delta, portfolio_gamma, portfolio_vega,
                open_positions, hedge_coverage, realized_vol_22d,
                iv_atm, skew_25d
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s.date, s.vix, s.vrp, s.portfolio_value, s.portfolio_var,
                s.portfolio_delta, s.portfolio_gamma, s.portfolio_vega,
                s.open_positions, s.hedge_coverage, s.realized_vol_22d,
                s.iv_atm, s.skew_25d,
            ),
        )


def list_snapshots(conn: sqlite3.Connection) -> list[DailySnapshot]:
    rows = conn.execute("SELECT * FROM daily_snapshots ORDER BY date").fetchall()
    return [
        DailySnapshot(
            date=_to_date(r["date"]),
            vix=r["vix"] or 0.0,
            vrp=r["vrp"] or 0.0,
            portfolio_value=r["portfolio_value"] or 0.0,
            portfolio_var=r["portfolio_var"] or 0.0,
            portfolio_delta=r["portfolio_delta"] or 0.0,
            portfolio_gamma=r["portfolio_gamma"] or 0.0,
            portfolio_vega=r["portfolio_vega"] or 0.0,
            open_positions=r["open_positions"] or 0,
            hedge_coverage=r["hedge_coverage"] or 0.0,
            realized_vol_22d=r["realized_vol_22d"] or 0.0,
            iv_atm=r["iv_atm"] or 0.0,
            skew_25d=r["skew_25d"] or 0.0,
        )
        for r in rows
    ]


def upsert_har_fit(conn: sqlite3.Connection, fit: HarRvFit) -> None:
    """Insert or replace a HAR-RV model row keyed by fit_date."""
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO har_rv_models(
                fit_date, beta_0, beta_daily, beta_weekly, beta_monthly,
                r_squared, n_obs
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fit.fit_date, fit.beta_0, fit.beta_daily, fit.beta_weekly,
                fit.beta_monthly, fit.r_squared, fit.n_obs,
            ),
        )


def latest_har_fit(conn: sqlite3.Connection, on_or_before: date) -> HarRvFit | None:
    """Most recent HAR-RV fit with fit_date <= ``on_or_before``."""
    row = conn.execute(
        "SELECT * FROM har_rv_models WHERE fit_date <= ? ORDER BY fit_date DESC LIMIT 1",
        (on_or_before,),
    ).fetchone()
    if row is None:
        return None
    return HarRvFit(
        fit_date=_to_date(row["fit_date"]),
        beta_0=row["beta_0"],
        beta_daily=row["beta_daily"],
        beta_weekly=row["beta_weekly"],
        beta_monthly=row["beta_monthly"],
        r_squared=row["r_squared"],
        n_obs=row["n_obs"],
    )


def schema_version(conn: sqlite3.Connection) -> str:
    """Return the schema version stored in ``schema_meta``."""
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return row["value"] if row else ""


def _to_date(value) -> date:
    """Coerce a sqlite-returned date-ish value to ``datetime.date``."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"cannot coerce {value!r} to date")
