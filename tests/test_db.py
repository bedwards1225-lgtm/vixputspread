"""Database migration and CRUD round-trip."""
from __future__ import annotations

from datetime import date, timedelta

from vrp.persistence.db import (
    SCHEMA_VERSION,
    connect,
    get_position,
    insert_position,
    latest_har_fit,
    list_positions,
    schema_version,
    update_position_exit,
    upsert_har_fit,
    upsert_snapshot,
    list_snapshots,
)
from vrp.persistence.models import DailySnapshot, HarRvFit, Leg, Position


def test_schema_creates_and_versions(tmp_db):
    conn = connect(tmp_db)
    assert schema_version(conn) == SCHEMA_VERSION


def test_position_round_trip(tmp_db):
    conn = connect(tmp_db)
    pos = Position(
        ticker="SPY", type="credit_spread",
        entry_date=date(2024, 1, 2),
        expiry=date(2024, 2, 2),
        short_strike=470.0, long_strike=468.0, contracts=2,
        credit_or_debit=0.85, iv_at_entry=0.18, vrp_at_entry=0.01,
        regime="mid", status="open",
        legs=[
            Leg(side="short", option_type="put", strike=470, expiry=date(2024,2,2),
                contracts=2, entry_price=0.85, entry_iv=0.18,
                entry_delta=-0.16, entry_gamma=0.01, entry_vega=0.04, entry_theta=-0.02),
            Leg(side="long", option_type="put", strike=468, expiry=date(2024,2,2),
                contracts=2, entry_price=0.0, entry_iv=0.19,
                entry_delta=-0.14, entry_gamma=0.009, entry_vega=0.038, entry_theta=-0.018),
        ],
    )
    pid = insert_position(conn, pos)
    loaded = get_position(conn, pid)
    assert loaded is not None
    assert loaded.ticker == "SPY"
    assert loaded.contracts == 2
    assert len(loaded.legs) == 2
    assert loaded.legs[0].side == "short"


def test_position_exit_update(tmp_db):
    conn = connect(tmp_db)
    pos = Position(
        ticker="SPY", type="credit_spread",
        entry_date=date(2024, 1, 2), expiry=date(2024, 2, 2),
        short_strike=470.0, long_strike=468.0, contracts=1,
        credit_or_debit=0.85, iv_at_entry=0.18, vrp_at_entry=0.01,
        regime="mid",
    )
    pid = insert_position(conn, pos)
    update_position_exit(
        conn, pid,
        exit_date=date(2024, 1, 20),
        exit_reason="PROFIT_TARGET",
        exit_price=0.40,
        pnl=45.0,
        commissions=2.60,
    )
    loaded = get_position(conn, pid)
    assert loaded.status == "closed"
    assert loaded.exit_reason == "PROFIT_TARGET"
    assert loaded.pnl == 45.0


def test_snapshot_upsert(tmp_db):
    conn = connect(tmp_db)
    s = DailySnapshot(
        date=date(2024, 1, 2), vix=14.0, vrp=0.02,
        portfolio_value=100_000.0, portfolio_var=2000.0,
        portfolio_delta=0.0, portfolio_gamma=-1.0, portfolio_vega=-50.0,
        open_positions=3, hedge_coverage=0.5,
    )
    upsert_snapshot(conn, s)
    s2 = DailySnapshot(
        date=date(2024, 1, 2), vix=14.5, vrp=0.025,
        portfolio_value=100_100.0, portfolio_var=2050.0,
        portfolio_delta=0.0, portfolio_gamma=-1.1, portfolio_vega=-52.0,
        open_positions=3, hedge_coverage=0.5,
    )
    upsert_snapshot(conn, s2)
    rows = list_snapshots(conn)
    assert len(rows) == 1
    assert rows[0].vix == 14.5


def test_har_fit_latest(tmp_db):
    conn = connect(tmp_db)
    fits = [
        HarRvFit(fit_date=date(2024, 1, 1), beta_0=0.0, beta_daily=0.3,
                 beta_weekly=0.3, beta_monthly=0.3, r_squared=0.4, n_obs=252),
        HarRvFit(fit_date=date(2024, 1, 8), beta_0=0.0, beta_daily=0.35,
                 beta_weekly=0.3, beta_monthly=0.25, r_squared=0.45, n_obs=252),
    ]
    for f in fits:
        upsert_har_fit(conn, f)
    latest = latest_har_fit(conn, date(2024, 1, 10))
    assert latest.fit_date == date(2024, 1, 8)
    older = latest_har_fit(conn, date(2024, 1, 5))
    assert older.fit_date == date(2024, 1, 1)
