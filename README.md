# NQ Custom Strategy Backtest

A transparent, **day-by-day** backtesting toolkit for NQ (Nasdaq 100 futures). The
strategy trades RTH long setups: it builds 1-hour "bias-flip" levels, waits for a
pullback into a previous-day bullish level, and enters on a 5-minute breakout with
a fixed 2.5R target. Everything is plain `pandas`/`numpy` (no backtesting
framework) so the logic stays auditable and easy to modify.

> ## ⚠️ Disclaimer
> - 🚧 **Still in development** — expect rough edges and breaking changes.
> - 🤖 **Developed using AI.**
> - 📉 **Not financial advice.** This is for research and educational purposes
>   only. Nothing here is a recommendation to trade. Use at your own risk.

## What's inside

| File | Purpose |
|------|---------|
| `Backtest.py` | The strategy backtester. Writes `trade_log.csv` and prints summary stats; can chart the equity curve and individual trades with Plotly. |
| `Convert_Data.py` | Resamples a 5-minute CSV into 1-hour and RTH-only 1-hour CSVs. |
| `DrawDay.py` | Plots a single day of candles (5-min / 1-hour) with VWAP and volume. |

## Requirements

- Python 3.9+ (developed on 3.14)
- `pip install pandas numpy plotly`

## Data is not included

The market-data CSVs are **not** part of this repository (they're git-ignored).
Supply your own NQ data in the project root as `NQ_5Min.csv` and `NQ_1Hour.csv`.

**Expected format** — semicolon-delimited, one title row then a header row:

```
Time Series;NQH26;;;;;
Date;Symbol;Open;High;Low;Close;Volume
1/30/2026 3:55 PM;NQH26;25.640,75;25.648,50;25.634,25;25.639,25;739
```

- **Dates:** `M/D/YYYY h:mm AM/PM`, **Chicago (CT) exchange time**, bar-start
  stamped, rows newest-first (descending).
- **Numbers:** European format (`.` thousands separator, `,` decimal).

If your data is in a different timezone/format, adjust `load_ohlc` and the `Config`
fields at the top of `Backtest.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Usage

```bash
# (optional) build 1-hour + RTH 1-hour data from your 5-minute file
python Convert_Data.py

# run the backtest -> writes trade_log.csv and prints a summary
python Backtest.py

# chart specific trades by their row number in trade_log.csv (ranges + "all" work)
python Backtest.py 5 12 30-34

# plot one day of candles
python DrawDay.py NQ_5Min.csv 1/29/2026
```

## Configuration

All tunables live in the `Config` dataclass at the top of `Backtest.py`:
session hours, level-touch tolerance, profit target (R), `start_year` / `end_year`
window, `skip_first_30min` opening-range filter, and more. Every strategy
assumption is documented and numbered (A1–A19) in the file's docstring.
