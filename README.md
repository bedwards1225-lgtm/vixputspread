# VRP — Variance Risk Premium harvesting

A recommendation engine for systematically harvesting the variance risk premium on US equity index ETFs (SPY, IWM, QQQ) via defined-risk put credit spreads, with a properly-sized long-dated tail hedge.

This is a **recommendation engine**, not an auto-trader. It generates entry signals, monitors open positions for exits, sizes a tail hedge against a stress scenario, and writes everything to a local SQLite database. You place the orders manually.

> **This is educational code. It is not financial advice and not a solicitation. The author makes no guarantees of profitability. Paper trade for at least 3 months before deploying real capital. Never deploy capital you cannot afford to lose.**

---

## What this is, briefly

SPX implied variance has historically traded above subsequent realized variance — documented since 1986 (CBOE PUT index), supported by Carr & Wu (2009) and Bondarenko (2014). The premium persists because structural insurance demand (pensions, asset managers, 401(k) overlays) exceeds the supply of capital willing to short variance. Selling defined-risk put credit spreads is the most retail-accessible way to harvest it; a long-dated OTM put hedge offsets the catastrophic-tail exposure that the short spreads carry.

**This is not a directional bet on the market.** It's a fee on insurance demand. It will underperform a simple SPX buy-and-hold during raging bull markets (2017, 2019, 2021, parts of 2023). The point is risk-adjusted return, not absolute return.

---

## What this is *not*

- No ML models. The edge is the variance risk premium, not prediction.
- No directional signals (RSI, MACD, sentiment).
- No auto-execution to a broker.
- No leverage beyond what defined-risk spreads naturally provide.
- No reliance on real-time data — end-of-day is fine.
- No web UI. CLI only.

---

## Strategy details

### Entry
1. Forecast realized variance for the trade horizon with a HAR-RV (Corsi 2009) model fit on a rolling 252-day window. With daily OHLC data, realized variance is approximated by the Yang-Zhang (2000) estimator, the most efficient drift-and-jump-robust estimator obtainable from OHLC alone.
2. Compute VRP at the short strike: `IV_strike^2 - RV_forecast^2`.
3. Gate: enter only when `VRP > 0` AND `VRP > 30th percentile of trailing 90 days`.
4. Strike selection by VIX regime (252d percentile): low vol (<25th) → short delta 0.20–0.25; mid → 0.15–0.20; high vol (>75th) → 0.10–0.15.
5. Skew filter: if 25-delta skew is in the top 10% of its 252-day distribution, halve the position size.
6. Spread width chosen so target credit ≥ 33% of width; reject if credit < 20% of width.
7. DTE 30–45; minimum 7 days between entries on the same ticker; max 1 entry/day.

### Exit
Checked daily, in priority order; first match wins:
1. **PROFIT_TARGET** — close at ≤ 50% of max profit.
2. **STOP_LOSS** — close at ≥ 2× credit received.
3. **GAMMA_STOP** — close when position gamma ≥ 3× entry gamma.
4. **SKEW_REVERSAL** — close on a 1-day jump > 2σ in normalised 25-delta skew.
5. **HARD_TIME_STOP** — close at 5 DTE.

### Sizing
The number of contracts is the minimum across:
- Per-trade max loss ≤ 1.5% of equity.
- Portfolio 99% 1-day VaR ≤ 5% of equity (historical simulation, 252-day window).
- Net portfolio Greeks: vega ≤ 0.5% per vol point, short gamma ≤ 0.1% per $1 move, |delta| ≤ 5% of equity.
- Correlation-adjusted via a 60-day rolling matrix.

### Tail hedge
- Long SPY puts ~6% OTM, 60–90 DTE.
- Sized under a stress scenario: SPX -8% in one day + VIX → 50.
- Target: hedge gain ≥ 50% of credit-spread loss under stress.
- Capped at 4% of equity per year in premium; if the coverage target requires more, the hedge is underhedged and logged (better than overpaying).
- Rolled when DTE drops below 30.

---

## Install

Requires Python 3.10+.

```bash
git clone <this-repo>
cd VolatilityPutSpreads
pip install -e .
```

This installs the `vrp` CLI globally (within your active environment).

For development:

```bash
pip install -e ".[dev]"
pytest
```

---

## Configuration

Copy `config.yaml.example` to `config.yaml` and edit any values you want to change. Pass `--config config.yaml` to all CLI commands, e.g.:

```bash
vrp --config config.yaml scan
```

If you don't pass `--config`, built-in defaults are used (matching the YAML example).

---

## CLI usage

| Command | What it does |
|---|---|
| `vrp scan [--ticker SPY ...]` | Scan for new entry signals across the universe. Prints a ranked diagnostic table. |
| `vrp monitor` | Check every open position against the 5 exit rules; alert if any triggered. |
| `vrp daily` | `scan` + `monitor` + `hedge-status`. Suitable for cron. |
| `vrp backtest --start YYYY-MM-DD --end YYYY-MM-DD [--data-provider bsm\|orats\|polygon\|cboe]` | Run a historical backtest; writes stats, plots, and CSV ledger to `backtest_results/`. |
| `vrp stress [--scenario gfc\|volmageddon\|covid\|all]` | Reprice currently open positions under historical stress scenarios. |
| `vrp add --ticker SPY --short 470 --long 468 --expiry 2024-02-15 --credit 0.85 --contracts 2` | Record a manually placed credit spread. |
| `vrp close --id N --exit-price X [--reason MANUAL]` | Mark a position closed. |
| `vrp positions [--status open\|closed\|all]` | List positions. |
| `vrp stats [--period 30d\|90d\|ytd\|all]` | Summary stats over closed trades. |
| `vrp hedge-status` | Show current hedge coverage and any gap to target. |

### Recommended cron entry

```cron
30 16 * * 1-5  /usr/local/bin/vrp --config /path/to/config.yaml daily >> /path/to/cron.log 2>&1
```

(Runs each weekday at 4:30pm — 15 minutes after market close.)

---

## Backtesting

```bash
# Synthetic-IV mode (no paid data needed; results are approximations only).
vrp backtest --start 2019-01-01 --end 2024-12-31 --data-provider bsm

# Real historical chains (recommended for serious validation).
vrp backtest --start 2019-01-01 --end 2024-12-31 \
             --data-provider orats --data-root /path/to/orats_csvs
```

### Output

Each run writes to `backtest_results/<tag>/`:
- `stats.txt` — headline stats.
- `equity_curve.png`, `drawdown.png`, `exit_reasons.png`, `monthly_heatmap.png`, `iv_vs_pnl.png`.
- `trades.csv` — every closed trade, fully diagnosed.
- `snapshots.csv` — daily end-of-day portfolio snapshot.

### A word on the BSM data provider

> **The `bsm` provider synthesises a historical IV surface from VIX history. It uses VIX as ATM IV (scaled per ticker), a fixed linear-in-delta skew calibrated to long-run SPX averages, and a flat term structure. This is a crude approximation. Backtest results from this provider should NOT be used to validate the strategy for real-money deployment. They are useful for sanity-checking code paths and getting a rough sense of behaviour, nothing more.**

When you run `vrp backtest --data-provider bsm`, a prominent warning is printed. Don't ignore it.

---

## Paid data sources

For trustworthy backtests you need historical option chains. The provider abstraction (`vrp/data/providers.py`) supports three formats out of the box:

| Provider | URL | Approx cost | Notes |
|---|---|---|---|
| ORATS | https://orats.com | $99–$499/mo | Smoothed (SMV) IV, EOD chains, deep history. Good for daily systematic. |
| Polygon.io | https://polygon.io | $79–$199/mo (options) | Per-trade and quote-level data, multiple data tiers. |
| CBOE DataShop | https://datashop.cboe.com | $$$, per-file | Highest fidelity, but expensive and per-file pricing. |

Each provider expects CSV files laid out under a root directory:

```
{root}/
  underlying/{TICKER}.csv     # OHLC with columns: date, Open, High, Low, Close, Volume
  chains/{TICKER}/{YYYY-MM-DD}.csv    # Provider-specific schema (see vrp/data/providers.py)
```

You'll need to ETL the raw vendor exports into this layout once. Custom providers can be added by subclassing `CsvBackedProvider` and implementing `_normalise()`.

---

## What to realistically expect

- **CAGR**: 8–15% with proper execution (not 20%+; that would require leverage beyond the defined-risk structure or assumptions that don't survive transaction costs).
- **Sharpe**: 0.7–1.1 net of costs.
- **Max drawdown**: 12–20%.
- **Win rate per trade**: 70–80%. Lossy trades are larger than wins by design (selling premium).
- **Underperforms SPX in raging bull markets.** Period. The point is risk-adjusted return.
- **Tail hedge costs ~3–4% annually in premium.** It is a drag in calm years. Do not remove it.

If a backtest is showing 30%+ CAGR with low drawdowns, something is wrong — most likely look-ahead bias, missing transaction costs, or the BSM provider painting a rosier IV surface than reality. The engine is strictly causal and uses configurable transaction costs (commission + 50% of half-spread slippage on entry/exit; 60% for the hedge), but verify behavior independently before deploying real capital.

---

## Architecture

```
vrp/
├── config.py                   # Pydantic-validated YAML config
├── logging_setup.py
├── pricing/
│   ├── black_scholes.py        # BSM pricing + Greeks (vectorised numpy)
│   ├── greeks.py               # Portfolio Greek aggregation
│   └── surface.py              # Cubic-spline-in-log-money / linear-in-T surface
├── forecasting/
│   ├── yang_zhang.py
│   └── har_rv.py               # Strictly causal rolling fit
├── signals/
│   ├── regime.py               # VIX %ile, skew %ile classification
│   ├── entry.py                # VRP-based candidate generator
│   └── exit.py                 # 5 rules, priority order
├── sizing/
│   ├── correlation.py
│   ├── var.py                  # Historical-simulation VaR
│   └── sizer.py                # Combined per-trade / VaR / Greek caps
├── hedging/
│   └── tail_hedge.py           # Stress-driven hedge sizing
├── data/
│   ├── live.py                 # yfinance + retry/backoff/cache
│   ├── providers.py            # ABC + BSM/ORATS/Polygon/CBOE
│   └── cache.py                # Parquet chains, pickle surfaces
├── persistence/
│   ├── db.py                   # SQLite schema, migrations, CRUD
│   └── models.py               # Dataclasses
├── backtest/
│   ├── engine.py               # Event-driven daily loop
│   ├── stress.py               # 3 historical scenarios
│   └── reporting.py            # Stats, plots, CSV
├── alerts/
│   └── notifier.py             # Console + file + Discord (stub by default)
└── cli/
    ├── main.py
    └── commands/               # One subcommand per file
```

### Database

SQLite at `vrp.db` (configurable). Tables: `positions`, `legs`, `daily_snapshots`, `har_rv_models`, `schema_meta`. See `vrp/persistence/db.py` for the full schema and migrations.

---

## Testing

```bash
pytest                          # all tests
pytest --cov=vrp                # with coverage
pytest tests/test_pricing.py    # one module
```

The test suite covers:
- Black-Scholes pricing parity and Greek finite-difference checks.
- HAR-RV parameter recovery on synthetic data and strict no-look-ahead.
- Vol surface fit/evaluate round-trip (< 0.5 vol-point error).
- Strike selection monotonicity in target delta.
- Each sizing constraint (per-trade, VaR, Greeks) binding.
- Boundary cases for every exit rule.
- Backtest determinism (same seed → identical equity curves).
- Stress scenarios produce correctly-signed P&L.
- Database schema creates from scratch and round-trips data.
- CLI smoke tests for each subcommand on an empty database.

---

## License

MIT. See `LICENSE`.
