"""Event-driven backtest engine.

Walks ``[start, end]`` day by day. Each day:

  1. Refit HAR-RV if it's a refit week (per ``forecasting.refit_every_days``).
  2. For every open position, evaluate exit rules; close if triggered.
  3. For every ticker, propose an entry candidate; size it; open if accepted
     (subject to per-day entry cap and same-ticker cooldown).
  4. Maintain the tail hedge (roll if needed).
  5. Mark-to-market the book; snapshot the day's stats.

All randomness — none currently — is seeded. The loop is strictly causal:
data inputs to day ``T`` only ever reference dates ``<= T``.

Transaction costs:
  - Per-leg commission.
  - Entry slippage: pay ``entry_slippage_pct_of_spread`` * (ask - bid) on
    each leg (debit side gets a worse fill).
  - Exit slippage: symmetric on the way out.
  - Hedge slippage: separate, configured higher.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vrp.config import Config
from vrp.data.providers import OptionsDataProvider, build_surface
from vrp.forecasting import har_rv
from vrp.persistence.models import (
    DailySnapshot,
    HarRvFit,
    Leg,
    Position,
)
from vrp.pricing import black_scholes as bsm
from vrp.pricing.greeks import (
    PortfolioGreeks,
    aggregate,
    position_greeks,
)
from vrp.signals.entry import SignalCandidate, propose_candidate
from vrp.signals.exit import ExitDecision, evaluate as evaluate_exit
from vrp.signals.regime import compute_25d_skew
from vrp.sizing.sizer import size_trade
from vrp.sizing.var import historical_var
from vrp.hedging.tail_hedge import (
    HedgeProposal,
    needs_roll,
    propose_hedge,
)

logger = logging.getLogger(__name__)

_MULTIPLIER = 100


@dataclass
class TradeRecord:
    """One closed trade for CSV export."""

    id: int
    ticker: str
    type: str
    entry_date: date
    exit_date: date
    expiry: date
    short_strike: float
    long_strike: float
    contracts: int
    credit: float
    exit_debit: float
    pnl: float
    commissions: float
    exit_reason: str
    regime: str
    vrp_at_entry: float
    iv_at_entry: float


@dataclass
class BacktestResult:
    """Full backtest output: snapshots + trades."""

    snapshots: list[DailySnapshot] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    har_fits: list[HarRvFit] = field(default_factory=list)
    config: Optional[Config] = None
    start: Optional[date] = None
    end: Optional[date] = None


def run_backtest(
    *,
    cfg: Config,
    provider: OptionsDataProvider,
    start: date,
    end: date,
    seed: int = 42,
) -> BacktestResult:
    """Run the event-driven backtest.

    Args:
        cfg: Strategy config.
        provider: Historical data provider.
        start: First date to evaluate (must be a trading day).
        end: Last date to evaluate (inclusive).
        seed: Random seed (used for any tie-breaking; deterministic).

    Returns:
        :class:`BacktestResult`.
    """
    random.seed(seed)
    np.random.seed(seed)

    equity = cfg.sizing.starting_equity
    open_positions: list[Position] = []
    closed_trades: list[TradeRecord] = []
    snapshots: list[DailySnapshot] = []
    har_fits: list[HarRvFit] = []
    realized_pnl_total = 0.0
    last_entry_by_ticker: dict[str, date] = {}
    last_har_fit: Optional[HarRvFit] = None
    last_refit_date: Optional[date] = None
    vrp_history_by_ticker: dict[str, list[float]] = {t: [] for t in cfg.universe.tickers}
    skew_history_by_ticker: dict[str, list[float]] = {t: [] for t in cfg.universe.tickers}

    # Pre-load historical OHLC for every ticker once.
    warmup = timedelta(days=cfg.data.history_days_for_har)
    ohlc: dict[str, pd.DataFrame] = {}
    for t in cfg.universe.tickers:
        ohlc[t] = provider.get_underlying(t, start - warmup, end)
    vix_series = provider.get_vix(start - warmup, end)
    vix_series.index = pd.to_datetime(vix_series.index)

    trading_days = provider.trading_days(start, end)
    next_trade_id = 1
    next_position_id = 1

    for day in trading_days:
        # --- 1. HAR refit if due ---
        if last_refit_date is None or (day - last_refit_date).days >= cfg.forecasting.refit_every_days:
            fit = har_rv.fit(
                ohlc[cfg.universe.tickers[0]],
                as_of=day,
                horizon_days=int(0.5 * (cfg.entry.min_dte + cfg.entry.max_dte)),
                window_days=cfg.forecasting.har_window_days,
            )
            if fit is not None:
                last_har_fit = fit
                last_refit_date = day
                har_fits.append(fit)
                logger.debug("HAR refit on %s: R^2=%.3f", day, fit.r_squared)

        # --- 2. Exit checks for each open position ---
        kept: list[Position] = []
        for pos in open_positions:
            try:
                snapshot = provider.get_chain(pos.ticker, day)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("missing chain for %s on %s, holding: %s", pos.ticker, day, exc)
                kept.append(pos)
                continue
            surface = build_surface(
                snapshot,
                rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(pos.ticker, 0.0),
            )
            iv_s = surface.iv(pos.short_strike, pos.expiry)
            iv_l = surface.iv(pos.long_strike, pos.expiry)
            T = max((pos.expiry - day).days / 365.0, 1e-6)
            q = cfg.universe.dividend_yields.get(pos.ticker, 0.0)
            p_short = float(bsm.price(snapshot.spot, pos.short_strike, T, cfg.pricing.risk_free_rate, q, iv_s, "put"))
            p_long = float(bsm.price(snapshot.spot, pos.long_strike, T, cfg.pricing.risk_free_rate, q, iv_l, "put"))
            debit_to_close = p_short - p_long if pos.type == "credit_spread" else p_short

            greeks = position_greeks(
                pos,
                underlying_price=snapshot.spot,
                iv_short=iv_s,
                iv_long=iv_l,
                as_of=day,
                rate=cfg.pricing.risk_free_rate,
                div_yield=q,
            )
            # Skew for the position's expiry, today and yesterday.
            try:
                skew_today = compute_25d_skew(surface, pos.expiry).skew_normalised
            except Exception:
                skew_today = 0.0
            history = skew_history_by_ticker.get(pos.ticker, [])
            skew_yesterday = history[-1] if history else skew_today
            skew_changes = pd.Series(history).diff().dropna()
            skew_std = float(skew_changes.tail(60).std()) if len(skew_changes) > 1 else 0.0

            decision: ExitDecision = evaluate_exit(
                pos,
                as_of=day,
                debit_to_close=debit_to_close,
                current_gamma=greeks.gamma,
                entry_gamma=pos.entry_delta_short and pos.entry_delta_short or 0.0,
                skew_today=skew_today,
                skew_yesterday=skew_yesterday,
                skew_rolling_std=skew_std,
                cfg=cfg.exit,
            )

            if decision.triggered or (pos.expiry - day).days <= 0:
                reason = decision.reason or "EXPIRED"
                # Pay slippage on exit.
                half_spread = 0.5 * (
                    max(0.05, 0.02 * max(p_short + p_long, 0.05))
                )
                slip_pct = (
                    cfg.backtest.hedge_slippage_pct_of_spread
                    if pos.type == "tail_hedge"
                    else cfg.backtest.exit_slippage_pct_of_spread
                )
                slip = 2 * slip_pct * half_spread
                effective_debit = debit_to_close + slip
                if pos.type == "credit_spread":
                    pnl = (pos.credit_or_debit - effective_debit) * _MULTIPLIER * pos.contracts
                else:
                    pnl = (effective_debit - pos.credit_or_debit) * _MULTIPLIER * pos.contracts
                comm = 2 * cfg.backtest.commission_per_contract * pos.contracts
                pnl -= comm
                realized_pnl_total += pnl
                equity += pnl
                closed_trades.append(
                    TradeRecord(
                        id=next_trade_id,
                        ticker=pos.ticker,
                        type=pos.type,
                        entry_date=pos.entry_date,
                        exit_date=day,
                        expiry=pos.expiry,
                        short_strike=pos.short_strike,
                        long_strike=pos.long_strike,
                        contracts=pos.contracts,
                        credit=pos.credit_or_debit,
                        exit_debit=effective_debit,
                        pnl=pnl,
                        commissions=comm,
                        exit_reason=reason,
                        regime=pos.regime,
                        vrp_at_entry=pos.vrp_at_entry,
                        iv_at_entry=pos.iv_at_entry,
                    )
                )
                next_trade_id += 1
                logger.info(
                    "EXIT %s %s contracts=%d pnl=$%.0f reason=%s",
                    pos.ticker, pos.expiry, pos.contracts, pnl, reason,
                )
            else:
                kept.append(pos)
        open_positions = kept

        # --- 3. Entry scans ---
        if last_har_fit is None:
            snapshots.append(_snapshot(day, vix_series, 0.0, equity, 0.0, 0.0, 0.0, 0.0, len(open_positions), 0.0))
            continue

        # VRP scan must happen on SPY as the canonical underlying for HAR.
        entries_today = 0
        ranked: list[tuple[SignalCandidate, str]] = []
        for ticker in cfg.universe.tickers:
            last_entry = last_entry_by_ticker.get(ticker)
            if last_entry is not None and (day - last_entry).days < cfg.entry.min_days_between_entries_same_ticker:
                continue
            try:
                snap = provider.get_chain(ticker, day)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("no chain for %s on %s: %s", ticker, day, exc)
                continue
            surface = build_surface(
                snap,
                rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(ticker, 0.0),
            )
            # Build a HAR feature snapshot for this ticker.
            feats = har_rv.build_features(ohlc[ticker].loc[ohlc[ticker].index <= pd.Timestamp(day)])
            try:
                rv_d = float(feats.rv_daily.iloc[-1])
                rv_w = float(feats.rv_weekly.iloc[-1])
                rv_m = float(feats.rv_monthly.iloc[-1])
            except IndexError:
                continue
            ohlc[ticker].attrs["rv_d"] = rv_d
            ohlc[ticker].attrs["rv_w"] = rv_w
            ohlc[ticker].attrs["rv_m"] = rv_m

            vrp_hist = pd.Series(vrp_history_by_ticker[ticker][-cfg.entry.vrp_lookback_days:])
            skew_hist = pd.Series(skew_history_by_ticker[ticker][-cfg.entry.skew_lookback_days:])
            vix_to_day = vix_series.loc[vix_series.index <= pd.Timestamp(day)]

            candidate = propose_candidate(
                cfg=cfg,
                ticker=ticker,
                as_of=day,
                surface=surface,
                underlying_history=ohlc[ticker],
                vix_history=vix_to_day,
                vrp_history=vrp_hist,
                skew_history=skew_hist,
                har_fit=last_har_fit,
                rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(ticker, 0.0),
            )
            if candidate is None:
                continue
            # Always record VRP & skew history (even on reject) so percentile gates have something to chew on.
            vrp_history_by_ticker[ticker].append(candidate.vrp)
            skew_history_by_ticker[ticker].append(candidate.skew.skew_normalised)
            if candidate.accepted:
                ranked.append((candidate, ticker))

        # Sort by VRP descending; respect per-day cap.
        ranked.sort(key=lambda x: -x[0].vrp)
        port_greeks = _portfolio_greeks(
            open_positions,
            cfg=cfg,
            as_of=day,
            spot_lookup=lambda t: ohlc[t].loc[ohlc[t].index <= pd.Timestamp(day)]["Close"].iloc[-1] if not ohlc[t].empty else 0.0,
            iv_lookup=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d),
        )
        for candidate, ticker in ranked:
            if entries_today >= cfg.entry.max_entries_per_day:
                break
            sizing = size_trade(
                candidate,
                cfg=cfg.sizing,
                equity=equity,
                open_positions=open_positions,
                underlying_history={t: ohlc[t]["Close"] for t in cfg.universe.tickers},
                iv_lookup=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d),
                rate=cfg.pricing.risk_free_rate,
                dividend_yields=cfg.universe.dividend_yields,
                portfolio_delta=port_greeks.delta,
                portfolio_gamma=port_greeks.gamma,
                portfolio_vega=port_greeks.vega,
            )
            if sizing.contracts <= 0:
                logger.debug("skip %s: %s", ticker, sizing.binding_constraint)
                continue
            # Open position. Pay entry slippage.
            half_spread = max(0.05, 0.02 * candidate.credit)
            slip = 2 * cfg.backtest.entry_slippage_pct_of_spread * half_spread
            net_credit = max(candidate.credit - slip, 0.0)
            comm = 2 * cfg.backtest.commission_per_contract * sizing.contracts
            equity -= comm   # commissions are realised cash outflow
            pos = Position(
                id=next_position_id,
                ticker=ticker,
                type="credit_spread",
                entry_date=day,
                expiry=candidate.expiry,
                short_strike=candidate.short_strike,
                long_strike=candidate.long_strike,
                contracts=sizing.contracts,
                credit_or_debit=net_credit,
                iv_at_entry=candidate.short_iv,
                vrp_at_entry=candidate.vrp,
                regime=str(candidate.vix_regime),
                status="open",
                entry_underlying_price=candidate.underlying_price,
                entry_vix=candidate.vix,
                entry_delta_short=candidate.short_delta,
                commissions=comm,
                legs=[
                    Leg(
                        side="short", option_type="put",
                        strike=candidate.short_strike, expiry=candidate.expiry,
                        contracts=sizing.contracts,
                        entry_price=candidate.credit,
                        entry_iv=candidate.short_iv,
                        entry_delta=candidate.short_delta,
                        entry_gamma=candidate.short_gamma,
                        entry_vega=candidate.short_vega,
                        entry_theta=0.0,
                    ),
                    Leg(
                        side="long", option_type="put",
                        strike=candidate.long_strike, expiry=candidate.expiry,
                        contracts=sizing.contracts,
                        entry_price=0.0,
                        entry_iv=candidate.long_iv,
                        entry_delta=0.0, entry_gamma=0.0, entry_vega=0.0,
                        entry_theta=0.0,
                    ),
                ],
            )
            next_position_id += 1
            open_positions.append(pos)
            last_entry_by_ticker[ticker] = day
            entries_today += 1
            logger.info(
                "ENTRY %s %s short=%.2f long=%.2f n=%d credit=$%.2f vrp=%.4f",
                ticker, candidate.expiry, candidate.short_strike, candidate.long_strike,
                sizing.contracts, net_credit, candidate.vrp,
            )

        # --- 4. Hedge maintenance (weekly on Mondays to limit churn). ---
        if cfg.hedging.enabled and day.weekday() == 0:
            existing_hedge = next((p for p in open_positions if p.type == "tail_hedge"), None)
            credit_spreads = [p for p in open_positions if p.type == "credit_spread"]
            if needs_roll(existing_hedge, as_of=day, roll_dte=cfg.hedging.roll_dte) and credit_spreads:
                proposal = propose_hedge(
                    cfg=cfg.hedging,
                    as_of=day,
                    equity=equity,
                    open_spreads=credit_spreads,
                    spot_lookup=lambda t: ohlc[t].loc[ohlc[t].index <= pd.Timestamp(day)]["Close"].iloc[-1],
                    iv_lookup_today=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d),
                    iv_lookup_stressed=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d) * 2.0,
                    rate=cfg.pricing.risk_free_rate,
                    dividend_yields=cfg.universe.dividend_yields,
                    existing_hedge=existing_hedge,
                )
                if proposal is not None and proposal.contracts > 0:
                    # Close any existing hedge first.
                    if existing_hedge is not None:
                        equity = _close_hedge(equity, existing_hedge, ohlc, provider, cfg, day, closed_trades, next_trade_id)
                        next_trade_id += 1
                        open_positions = [p for p in open_positions if p is not existing_hedge]
                    debit_cost = proposal.contracts * proposal.estimated_debit
                    slip = 2 * cfg.backtest.hedge_slippage_pct_of_spread * max(0.05, 0.02 * proposal.estimated_debit)
                    debit_cost += slip * proposal.contracts
                    comm = 2 * cfg.backtest.commission_per_contract * proposal.contracts
                    equity -= comm
                    equity -= debit_cost * _MULTIPLIER
                    hedge_pos = Position(
                        id=next_position_id,
                        ticker=proposal.ticker,
                        type="tail_hedge",
                        entry_date=day,
                        expiry=proposal.expiry,
                        short_strike=proposal.strike,    # store the long-put strike here for symmetry
                        long_strike=proposal.strike,
                        contracts=proposal.contracts,
                        credit_or_debit=proposal.estimated_debit + slip,
                        iv_at_entry=0.0,
                        vrp_at_entry=0.0,
                        regime="hedge",
                        status="open",
                        commissions=comm,
                    )
                    next_position_id += 1
                    open_positions.append(hedge_pos)
                    logger.info(
                        "HEDGE ROLL contracts=%d strike=%.2f expiry=%s coverage=%.0f%%",
                        proposal.contracts, proposal.strike, proposal.expiry,
                        100 * proposal.coverage_ratio,
                    )

        # --- 5. Snapshot end-of-day stats ---
        port_greeks = _portfolio_greeks(
            open_positions,
            cfg=cfg,
            as_of=day,
            spot_lookup=lambda t: ohlc[t].loc[ohlc[t].index <= pd.Timestamp(day)]["Close"].iloc[-1] if not ohlc[t].empty else 0.0,
            iv_lookup=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d),
        )
        portfolio_var = historical_var(
            [p for p in open_positions if p.type == "credit_spread"],
            as_of=day,
            underlying_history={t: ohlc[t]["Close"] for t in cfg.universe.tickers},
            iv_lookup=lambda t, k, e, d: _iv_for(provider, cfg, t, k, e, d),
            rate=cfg.pricing.risk_free_rate,
            dividend_yields=cfg.universe.dividend_yields,
            confidence=cfg.sizing.var_confidence,
            lookback=cfg.sizing.var_lookback_days,
        )
        hedge_pos = next((p for p in open_positions if p.type == "tail_hedge"), None)
        spread_loss = sum(
            max(p.credit_or_debit * _MULTIPLIER * p.contracts, 0.0)
            for p in open_positions
            if p.type == "credit_spread"
        )
        hedge_notional = (hedge_pos.credit_or_debit * _MULTIPLIER * hedge_pos.contracts) if hedge_pos else 0.0
        coverage = hedge_notional / spread_loss if spread_loss > 0 else 0.0
        # Average across-ticker VRP from history.
        avg_vrp = float(
            np.mean(
                [vrp_history_by_ticker[t][-1] for t in cfg.universe.tickers if vrp_history_by_ticker[t]]
            )
        ) if any(vrp_history_by_ticker[t] for t in cfg.universe.tickers) else 0.0
        snapshots.append(
            _snapshot(
                day, vix_series, avg_vrp, equity,
                portfolio_var, port_greeks.delta, port_greeks.gamma,
                port_greeks.vega, len(open_positions), coverage,
            )
        )

    result = BacktestResult(
        snapshots=snapshots,
        trades=closed_trades,
        har_fits=har_fits,
        config=cfg,
        start=start,
        end=end,
    )
    return result


def _portfolio_greeks(
    positions: list[Position],
    *,
    cfg: Config,
    as_of: date,
    spot_lookup,
    iv_lookup,
) -> PortfolioGreeks:
    """Sum dollar Greeks across open positions at the current surface."""
    greeks = []
    for p in positions:
        spot = spot_lookup(p.ticker)
        iv_s = iv_lookup(p.ticker, p.short_strike, p.expiry, as_of)
        iv_l = iv_lookup(p.ticker, p.long_strike, p.expiry, as_of)
        greeks.append(
            position_greeks(
                p,
                underlying_price=spot,
                iv_short=iv_s,
                iv_long=iv_l,
                as_of=as_of,
                rate=cfg.pricing.risk_free_rate,
                div_yield=cfg.universe.dividend_yields.get(p.ticker, 0.0),
            )
        )
    return aggregate(greeks)


def _iv_for(
    provider: OptionsDataProvider, cfg: Config, ticker: str, strike: float, expiry: date, as_of: date
) -> float:
    """Cheap IV lookup: refit surface only if not cached for this call site."""
    try:
        snap = provider.get_chain(ticker, as_of)
    except (FileNotFoundError, ValueError):
        return 0.2
    surface = build_surface(
        snap,
        rate=cfg.pricing.risk_free_rate,
        div_yield=cfg.universe.dividend_yields.get(ticker, 0.0),
    )
    try:
        return surface.iv(strike, expiry)
    except Exception:
        return 0.2


def _snapshot(
    day: date,
    vix_series: pd.Series,
    avg_vrp: float,
    equity: float,
    var: float,
    delta: float,
    gamma: float,
    vega: float,
    n_open: int,
    coverage: float,
) -> DailySnapshot:
    vix_today = float(vix_series.asof(pd.Timestamp(day)))
    return DailySnapshot(
        date=day,
        vix=vix_today,
        vrp=avg_vrp,
        portfolio_value=equity,
        portfolio_var=var,
        portfolio_delta=delta,
        portfolio_gamma=gamma,
        portfolio_vega=vega,
        open_positions=n_open,
        hedge_coverage=coverage,
    )


def _close_hedge(
    equity: float,
    hedge: Position,
    ohlc: dict[str, pd.DataFrame],
    provider: OptionsDataProvider,
    cfg: Config,
    day: date,
    closed_trades: list[TradeRecord],
    next_id: int,
) -> float:
    """Close an existing hedge at mid and add a trade record."""
    spot = float(ohlc[hedge.ticker].loc[ohlc[hedge.ticker].index <= pd.Timestamp(day)]["Close"].iloc[-1])
    try:
        snap = provider.get_chain(hedge.ticker, day)
        surface = build_surface(snap, rate=cfg.pricing.risk_free_rate,
                                div_yield=cfg.universe.dividend_yields.get(hedge.ticker, 0.0))
        iv = surface.iv(hedge.short_strike, hedge.expiry)
    except (FileNotFoundError, ValueError):
        iv = 0.2
    T = max((hedge.expiry - day).days / 365.0, 1e-6)
    q = cfg.universe.dividend_yields.get(hedge.ticker, 0.0)
    value = float(bsm.price(spot, hedge.short_strike, T, cfg.pricing.risk_free_rate, q, iv, "put"))
    slip = 2 * cfg.backtest.hedge_slippage_pct_of_spread * max(0.05, 0.02 * value)
    effective = max(value - slip, 0.0)
    pnl = (effective - hedge.credit_or_debit) * _MULTIPLIER * hedge.contracts
    comm = 2 * cfg.backtest.commission_per_contract * hedge.contracts
    pnl -= comm
    equity += pnl
    closed_trades.append(
        TradeRecord(
            id=next_id, ticker=hedge.ticker, type="tail_hedge",
            entry_date=hedge.entry_date, exit_date=day, expiry=hedge.expiry,
            short_strike=hedge.short_strike, long_strike=hedge.long_strike,
            contracts=hedge.contracts, credit=hedge.credit_or_debit,
            exit_debit=effective, pnl=pnl, commissions=comm,
            exit_reason="HEDGE_ROLL", regime="hedge",
            vrp_at_entry=0.0, iv_at_entry=0.0,
        )
    )
    return equity
