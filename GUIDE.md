# Simple Guide

A plain-English walkthrough for using this thing without needing to know what HAR-RV or "25-delta skew" means.

---

## What this actually does

Imagine you sell car insurance. Most years, most drivers don't crash. You collect premiums, pay out the occasional claim, and keep the difference. Over time, insurance companies make money because they charge slightly more than the average claim is worth.

This tool does the same thing, but for the stock market. You sell "insurance" to other investors who are worried about a crash. They pay you a small premium. Most weeks, the crash they're worried about doesn't happen, and you keep the premium. Occasionally there's a sharp drop and you take a loss — but you also hold a separate "tail hedge" that pays off in a big crash, so you can't be wiped out.

The technical name is "selling put credit spreads with a long-dated tail hedge." This tool tells you **when** to sell and **at what strikes**, **monitors** your open positions, and **tells you when to close them**.

**It does not place trades for you.** You read the signal, then you go into your broker (Schwab, Fidelity, Robinhood, IBKR, etc.) and click the buttons yourself. Then you tell this tool what you did so it can keep track.

---

## The basic loop

Every weekday, after the market closes:

1. **Run the scanner.** It tells you if there's a trade worth taking today.
2. **If yes,** open your broker, place the trade, then tell this tool you did it.
3. **Run the monitor.** It checks all your open trades and tells you if any need to be closed.
4. **If yes,** close that trade in your broker, then tell this tool you closed it.

That's it. The whole thing takes 5 minutes a day.

---

## One-time setup

You only do this once. If `vrp` is already installed (it is — I tested it), skip to "Your first day."

```bash
cd /Users/bruceedwards/Desktop/VolatilityPutSpreads
pip3 install -e .
```

Test it works:
```bash
vrp --help
```

You should see a list of commands. If you see "command not found," tell me and we'll fix the PATH.

---

## Your first day

### Step 1: Look for a signal

```bash
vrp scan
```

It'll think for ~30 seconds (it's downloading live data), then show you a table like:

```
Ticker  DTE    Short    Long  Width  Credit   C/W     VRP  VRP%  Skew% Regime    IV RV^.5 Status
SPY      37   570.00  565.00   5.00    1.65   33%  0.0125   65%   42% mid     0.16  0.12 ACCEPT
QQQ      37   495.00  490.00   5.00    1.20   24%  0.0080   45%   55% mid     0.18  0.15 REJECT: credit 24% below floor 20%
IWM      37   215.00  213.00   2.00    0.35   17%  0.0050   30%   60% mid     0.20  0.18 REJECT: credit 17% below floor 20%

1 accepted out of 3 scanned.
```

**Reading this:**
- **Ticker** — which ETF (SPY = S&P 500, QQQ = Nasdaq, IWM = small caps).
- **DTE** — days until the options expire. ~37 days out is normal.
- **Short / Long** — the two strike prices that make the spread.
- **Width** — distance between strikes (your max risk per contract is this minus the credit).
- **Credit** — how much money the trade pays you per contract per share. Multiply by 100 to get dollars per contract. (Credit of 1.65 = $165 per contract.)
- **C/W** — credit as % of width. Higher = better paid for the risk.
- **VRP** — the actual edge. Positive = options are overpriced relative to expected risk. This is what we're harvesting.
- **VRP% / Skew%** — where these values rank in recent history.
- **Status** — ACCEPT means take the trade. REJECT means skip it.

### Step 2: If you see an ACCEPT, place the trade

Open your broker. Find SPY options. Place a "**put credit spread**" with:
- The short strike from the table (e.g. 570)
- The long strike from the table (e.g. 565)
- The expiry that matches the DTE (e.g. 37 days out)
- 1 contract to start (more on sizing below)
- Limit order at or near the credit shown (e.g. $1.65)

Your broker will explain "this trade is risk-defined; max loss is $X." Make sure you understand it before clicking.

### Step 3: Tell this tool what you did

After your broker fills the order:

```bash
vrp add --ticker SPY --short 570 --long 565 --expiry 2025-03-21 --credit 1.65 --contracts 1
```

(Use the actual fill price your broker gave you, which might be slightly different from 1.65.)

You'll see:
```
Added position #1: SPY 570.0/565.0 x1 credit=1.65
```

You're done for today.

---

## Every day after that

### Morning or after close: run monitor

```bash
vrp monitor
```

This checks every open position and tells you whether to:
- **HOLD** — leave it alone.
- **EXIT** — close the trade today.

If you see `EXIT: PROFIT_TARGET` or `EXIT: STOP_LOSS` or anything similar:

1. Open your broker.
2. Close the spread (your broker calls this "Buy to close" — you're buying back what you sold).
3. Tell the tool:

```bash
vrp close --id 1 --exit-price 0.80 --reason PROFIT_TARGET
```

(Use the position ID from the monitor output, the actual debit you paid to close, and the reason the tool flagged.)

### Or just one command:

```bash
vrp daily
```

Runs scan + monitor + hedge check all together. Good for a cron job.

---

## What the exit reasons mean

| Reason | What happened | What to do |
|---|---|---|
| `PROFIT_TARGET` | You captured 50% of max profit. Time to lock it in. | Close. Smile. |
| `STOP_LOSS` | The trade has lost 2× the credit you collected. Cut it before it gets worse. | Close. It's fine — losses are part of this. |
| `GAMMA_STOP` | The market moved against you enough that risk is spiking. | Close. |
| `SKEW_REVERSAL` | Market is suddenly pricing in more crash risk. Get out. | Close. |
| `HARD_TIME_STOP` | Trade has 5 days left. Stop pushing your luck. | Close. |

---

## How much money to actually risk

The tool's defaults assume a $100,000 account. If you have less, you need to manually scale down.

**Quick rule of thumb:** never put more than **1.5% of your account on one trade**. For a $25,000 account that's $375 max loss per trade. The spreads above (5-point width, $1.65 credit) have ~$335 max loss per contract, so 1 contract is safe.

To make the tool aware of your actual account size, edit `config.yaml`:

```bash
cp config.yaml.example config.yaml
```

Open the file in any editor and change:
```yaml
sizing:
  starting_equity: 25000.0    # your real number
```

Then run every command with `--config config.yaml`:
```bash
vrp --config config.yaml scan
```

---

## When NOT to take a signal

Even if the scanner says ACCEPT, **skip the trade** if:

1. You don't understand what the trade is. Read about put credit spreads on Investopedia first.
2. You can't afford to lose the max-loss amount. Period.
3. There's a Fed meeting, CPI release, or major earnings tomorrow. Vol changes are unpredictable on those days.
4. You feel emotional about it. The tool doesn't care about your feelings. Bad signal to trade on.

---

## The tail hedge (important)

The strategy includes a **long-dated put as insurance** against a market crash. The tool doesn't open it automatically — you'd need to add it manually based on `vrp hedge-status` output.

For a small account ($25K or less), running the hedge is expensive relative to the spreads you can sell. You have two honest options:

1. **Trade smaller and run the hedge.** Sell 1 contract at a time, keep one tail hedge open.
2. **Skip the hedge and limit risk by other means.** Only ever have 1-2 open spreads at once. Stop trading entirely if VIX > 30.

Option 2 is what most retail traders actually do. It's not wrong, but understand that **a 2020 COVID-style crash will hurt** without the hedge.

---

## Running a backtest

Before you trade real money, run a backtest to see how the strategy would have done historically:

```bash
vrp backtest --start 2023-01-01 --end 2024-12-31 --data-provider bsm
```

Takes 1-5 minutes. When it's done it'll print where the report went, something like:
```
Report written to: backtest_results/bsm_2023-01-01_2024-12-31
```

Open that folder. Look at:
- `stats.txt` — headline numbers.
- `equity_curve.png` — did the account grow over time?
- `drawdown.png` — how bad did it get during rough patches?
- `exit_reasons.png` — are most exits profit targets (good) or stop losses (bad)?

**Big caveat:** the `bsm` backtest uses synthetic option prices, not real historical chains. Real prices would give you different (probably worse) results. Treat the backtest as a sanity check, not a forecast.

---

## Common things that will go wrong

- **`vrp scan` shows nothing accepted.** Normal. Some days/weeks there just isn't enough variance premium to harvest. Patience is the strategy.
- **yfinance is slow or returns errors.** Normal. Run it again in a few minutes.
- **The scanner suggests a strike that doesn't exist on your broker.** Pick the nearest available strike. The exact strike isn't sacred; the delta target is what matters.
- **You lost money on a trade.** Normal. The strategy wins ~75% of trades but losers are bigger than winners. Over 50+ trades it should be positive. Don't change anything after one loss.
- **The market crashed and you have open trades.** This is what tail hedges are for. If you didn't set one up, you eat the loss. Lesson for next time.

---

## The one-line summary

> **Run `vrp daily` after market close. If it says ACCEPT, place the trade in your broker, then `vrp add` to record it. If it says EXIT, close the trade in your broker, then `vrp close` to record it. Repeat.**

That's the whole job.

---

## When to stop

Stop trading and reassess if:
- You've lost more than 10% of your account over any 30-day period.
- You don't understand why a trade lost money.
- VIX is above 35 and you don't have a hedge.
- You're trading bigger size because you "feel lucky."

The tool will keep generating signals. That doesn't mean you have to take them.

---

## Reminder

This is educational software. Not financial advice. Not a guarantee of profit. The author is not your broker, not your fiduciary, not responsible for your trades. Paper-trade for at least 3 months before risking real money, and never risk what you can't afford to lose.
