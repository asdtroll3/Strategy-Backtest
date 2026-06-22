"""
Draw a single day of OHLCV candlesticks (with VWAP + volume) from a CSV.

Works with any of the NQ exports (they share one format): NQ_1Min.csv,
NQ_5Min.csv, NQ_1Hour.csv.  The bar timeframe in the title is detected
automatically from the spacing between bars.

Usage
─────
    python DrawDay.py                          # uses the defaults below
    python DrawDay.py NQ_1Hour.csv 1/29/2026   # file + date as CLI args

File format
───────────
Delimiter  : semicolon (;)
Layout     : title row, then header row  Date;Symbol;Open;High;Low;Close;Volume
Date field : M/D/YYYY H:MM AM  (e.g. 1/21/2026 4:40 AM)
Number fmt : European  –  dot as thousands sep, comma as decimal sep
             e.g.  25.181,75  →  25181.75

Requirements
────────────
    pip install pandas plotly
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

# ╔══════════════════════════════════════════════════════════════════╗
# ║       ▶  EDIT THESE DEFAULTS  (or pass them as CLI args)  ◀        ║
# ║     e.g.   python DrawDay.py NQ_1Hour.csv 1/29/2026               ║
DEFAULT_CSV_FILE    = "NQ_5Min.csv"   # NQ_5Min.csv / NQ_1Hour.csv / NQ_1Min.csv
DEFAULT_TARGET_DATE = "1/29/2026"     # day to plot  (M/D/YYYY)
# ╚══════════════════════════════════════════════════════════════════╝

CSV_FILE    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_FILE
TARGET_DATE = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET_DATE

# Resolve the CSV path relative to this script so it works from any directory.
csv_path = Path(CSV_FILE)
if not csv_path.is_absolute():
    csv_path = BASE_DIR / csv_path


# ── helpers ──────────────────────────────────────────────────────────────────

def eu_float(s: str) -> float:
    """'25.181,75'  →  25181.75"""
    return float(str(s).strip().replace(".", "").replace(",", "."))

def eu_int(s) -> int:
    """'1.234' or '1234'  →  1234"""
    return int(str(s).strip().replace(".", "").replace(",", ""))

def infer_timeframe_label(timestamps) -> str:
    """Infer a bar-size label like '5-min' or '1-hour' from the bar spacing."""
    diffs = timestamps.sort_values().diff()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return ""
    minutes = int(round(diffs.min().total_seconds() / 60))
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}-hour"
    return f"{minutes}-min"


# ── 1.  Load & parse ──────────────────────────────────────────────────────────

# skiprows=1 drops the "Time Series;SYMBOL;;;;;" title row so the real header
# (Date;Symbol;Open;High;Low;Close;Volume) is used.
df = pd.read_csv(csv_path, sep=";", skiprows=1, dtype=str, encoding="utf-8-sig")

# Normalise column names: strip whitespace + any invisible/BOM characters
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

# Debug: uncomment the line below to see the exact column names that were parsed
# print("Columns found:", df.columns.tolist())

# Parse datetime – pandas handles single-digit month/day without explicit format
df["Datetime"] = pd.to_datetime(
    df["Date"].str.strip(),
    format="%m/%d/%Y %I:%M %p"
)

for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].apply(eu_float)

df["Volume"] = df["Volume"].apply(eu_int)


# ── 2.  Filter to the requested day ──────────────────────────────────────────

target_date = pd.to_datetime(TARGET_DATE, dayfirst=False).date()
day = (
    df[df["Datetime"].dt.date == target_date]
    .sort_values("Datetime")
    .reset_index(drop=True)
)

if day.empty:
    sys.exit(
        f"\n  No bars found for  {TARGET_DATE}\n"
        f"  Check the date or the CSV path: {csv_path}\n"
    )

symbol   = day["Symbol"].iloc[0] if "Symbol" in day.columns else ""
date_str = target_date.strftime("%B %d, %Y")
print(f"  Loaded {len(day)} bars  |  {symbol}  |  {date_str}")


# ── 3.  Compute a few stats for the title ────────────────────────────────────

day_open  = day["Open"].iloc[0]
day_close = day["Close"].iloc[-1]
day_high  = day["High"].max()
day_low   = day["Low"].min()
day_range = day_high - day_low
chg       = day_close - day_open
chg_pct   = chg / day_open * 100
chg_sign  = "▲" if chg >= 0 else "▼"
chg_color = "#26a69a" if chg >= 0 else "#ef5350"

tf_label = infer_timeframe_label(day["Datetime"])
tf_text  = f"({tf_label})   " if tf_label else ""

title_text = (
    f"{symbol}  ·  {date_str}  {tf_text}"
    f"O {day_open:,.2f}  H {day_high:,.2f}  L {day_low:,.2f}  C {day_close:,.2f}  "
    f"Range {day_range:,.2f}"
)


# ── 4.  Build the chart ───────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.02,
)

# --- candlesticks ---
fig.add_trace(
    go.Candlestick(
        x=day["Datetime"],
        open=day["Open"],
        high=day["High"],
        low=day["Low"],
        close=day["Close"],
        name=symbol,
        increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        whiskerwidth=0.5,
    ),
    row=1, col=1,
)

# --- VWAP (simple running average weighted by volume) ---
day["TypicalPrice"] = (day["High"] + day["Low"] + day["Close"]) / 3
day["CumTPV"]       = (day["TypicalPrice"] * day["Volume"]).cumsum()
day["CumVol"]       = day["Volume"].cumsum()
day["VWAP"]         = day["CumTPV"] / day["CumVol"]

fig.add_trace(
    go.Scatter(
        x=day["Datetime"],
        y=day["VWAP"],
        name="VWAP",
        line=dict(color="#f0c040", width=1.2, dash="dot"),
        hovertemplate="VWAP: %{y:,.2f}<extra></extra>",
    ),
    row=1, col=1,
)

# --- volume bars (colour-matched to candle direction) ---
bar_colors = [
    "#26a69a" if c >= o else "#ef5350"
    for c, o in zip(day["Close"], day["Open"])
]

fig.add_trace(
    go.Bar(
        x=day["Datetime"],
        y=day["Volume"],
        name="Volume",
        marker_color=bar_colors,
        opacity=0.7,
        showlegend=False,
    ),
    row=2, col=1,
)

# ── 5.  Styling ───────────────────────────────────────────────────────────────

BG      = "#131722"
GRID    = "#1e222d"
TEXT    = "#d1d4dc"
BORDER  = "#2a2e39"

fig.update_layout(
    title=dict(
        text=title_text,
        x=0.5,
        font=dict(size=13, color=TEXT),
    ),
    height=700,
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(color=TEXT, size=11, family="'Courier New', monospace"),
    legend=dict(
        bgcolor=BORDER,
        bordercolor=BORDER,
        borderwidth=1,
        x=0.01,
        y=0.99,
    ),
    xaxis_rangeslider_visible=False,
    margin=dict(l=70, r=40, t=60, b=40),
    hovermode="x unified",
)

# Price pane
fig.update_yaxes(
    gridcolor=GRID,
    gridwidth=0.5,
    zeroline=False,
    tickformat=",.2f",
    showline=True,
    linecolor=BORDER,
    row=1, col=1,
)
fig.update_xaxes(
    gridcolor=GRID,
    showgrid=False,
    showline=True,
    linecolor=BORDER,
    row=1, col=1,
)

# Volume pane
fig.update_yaxes(
    title_text="Volume",
    title_font=dict(size=10),
    gridcolor=GRID,
    gridwidth=0.5,
    zeroline=False,
    showline=True,
    linecolor=BORDER,
    row=2, col=1,
)
fig.update_xaxes(
    gridcolor=GRID,
    gridwidth=0.5,
    showgrid=True,
    showline=True,
    linecolor=BORDER,
    tickformat="%H:%M",
    row=2, col=1,
)

# ── 6.  Show ──────────────────────────────────────────────────────────────────

fig.show()