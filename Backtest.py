"""
================================================================================
NQ (Nasdaq 100 futures) — RTH Pullback-to-Bias-Flip Long Backtester
================================================================================

A transparent, *day-by-day* backtest. No backtesting framework is used on
purpose: every decision is made with plain pandas/numpy so the logic is fully
auditable and easy to modify. vectorbt / backtesting.py are intentionally
avoided (the spec asked to prioritise transparency over framework abstraction).

Strategy in one paragraph
-------------------------
1-hour candles define market structure via "bias flip" levels (the open of the
candle that flips direction). During the RTH session we only take LONGS, only
when price is above the 9:30 open and in an uptrend, and only when price pulls
back into a *bullish* bias-flip level created on a *previous* day and still
active. A 5-minute reaction candle tags the level; the next candle is the
confirmation candle; we enter on a break above the confirmation candle's high.
Stop = low of the first bearish pullback candle that wicked below the level.
Target = entry + 2.5R. Stop-first if a candle straddles both.

--------------------------------------------------------------------------------
ASSUMPTIONS  (every ambiguous point in the spec is resolved here and surfaced
in the CONFIG dataclass below so it can be changed in one place)
--------------------------------------------------------------------------------
A1.  SOURCE TIMEZONE.  The spec says data must end up in America/New_York, but
     the provided NQ CSVs are stamped in *Chicago* exchange time (established
     earlier in this project).  We therefore localise to `source_tz`
     (America/Chicago) and convert to `target_tz` (America/New_York).  Because
     both zones share US DST, the offset is a constant +1h and 9:30 NY == 8:30
     Chicago all year.

A2.  BAR TIMESTAMPS ARE BAR-START times (confirmed from the data: the 9:30 NY /
     8:30 CT bar carries the opening-auction volume spike).  So a 5-min bar
     stamped 09:30 covers [09:30, 09:35).

A3.  "9:30 opening candle" = the 5-min bar whose start is exactly 09:30 NY.  Its
     OPEN is the daily reference.  Days without a 09:30 bar (holidays/late
     opens) are skipped.

A4.  RTH SESSION = [rth_start, rth_end) = [09:30, 16:00) NY (standard cash
     session).  NOTE: earlier in this project you used a custom 09:30–15:30
     window — set `rth_end = time(15, 30)` to reproduce that.

A5.  LEVELS LIVE ON THE RTH-ANCHORED 1-HOUR CANDLES (NQ_1Hour_RTH.csv): the
     session bars anchored to the 09:30 open (09:30, 10:30, … each day).  BOTH
     bias-flip creation and invalidation use this RTH series — a level is the
     OPEN of an RTH hourly candle that flips direction, and it dies when a later
     RTH hourly candle's BODY closes through it: open/close straddle (or pass)
     the level in EITHER direction (bull dies on min(open,close)<level, bear on
     max(open,close)>level); wicks alone don't count.  (Earlier versions detected
     on the 24h clock-hour file and used a one-sided close rule.)

A6.  "Current price" for the trend / above-open checks = the close of the 5-min
     candle being evaluated (the reaction candle).

A7.  "Previous day's high" = previous trading day's RTH high (toggle
     `prev_day_high_rth_only`).

A8.  "Nearest bullish bias flip level" (trend rule) = active bull level closest
     in absolute price to the current price.  For speed this nearest lookup
     ignores rare *intraday* invalidations (it only affects the OR-trend
     filter, never which level is actually traded — that one is fully
     invalidation-checked).

A9.  REACTION CANDLE: low <= level + tol AND close >= level - tol  (wick tags or
     pierces the level, close holds at/above it). `tol = level_touch_tolerance`.

A10. CONFIRMATION = the bar immediately after the reaction bar. TRIGGER = a
     later bar whose HIGH exceeds the confirmation high, within
     `trigger_timeout_bars` bars; entry fills at the confirmation high (gaps
     above are filled at the confirmation high — optimistic by the gap).

A11. STOP = low of the FIRST bearish (close<open) candle in the pullback run
     (the consecutive bars ending at the reaction bar that tagged the level)
     whose low is below the level.  Fallback: the lowest low in the run that is
     below the level.  If nothing wicked below the level, the setup is rejected.

A12. TRADE MANAGEMENT runs on candles AFTER the entry bar (we do not model the
     unknown intrabar path of the entry bar). Stop-first when a bar hits both.

A13. EXIT-ON-SESSION-END: if neither stop nor target is hit by the last RTH bar
     the trade is closed at that bar's close. (Multi-day holds are out of scope
     for this version; management is per-session.)

A14. ONE TRADE PER DAY by default (`max_trades_per_day`), the first valid setup.

A15. CONTINUOUS CONTRACT: the CSV stitches contract months (NQH26, NQZ25, ...).
     Roll gaps are NOT back-adjusted; a roll could in theory spawn a stray
     level. Use back-adjusted data for production. Symbol is ignored.

A16. Risk/PnL are reported in INDEX POINTS and R multiples (per the spec). A
     `point_value` ($/point) is provided for optional dollar PnL.

A17. LEVEL VALIDITY is checked when the reaction bar forms (the "setup") AND
     re-confirmed at entry: if an RTH 1-hour candle closed through the level in
     between, the setup is cancelled — we never enter on an invalidated level.

A18. YEAR WINDOW.  `start_year` / `end_year` (NY-local, inclusive) clip BOTH the
     1h and 5m data right after load; either set to None means "no bound on that
     side" (the whole file).  Because levels are detected only within the loaded
     window, expect a short warm-up at the start of `start_year` while the first
     levels form (negligible over multi-year windows).

A19. OPENING-RANGE FILTER.  When `skip_first_30min` is True, no ENTRY may occur
     within `open_skip_minutes` (default 30) of the open — entries are blocked in
     [09:30, 10:00) NY.  A setup may still *form* in that window; it is only
     taken if its trigger fills at/after the cutoff.
================================================================================
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NEVER_NS = np.iinfo(np.int64).max  # sentinel: "level never invalidated"


# ============================================================================
# CONFIGURATION  — every tunable / assumption lives here
# ============================================================================
@dataclass
class Config:
    # --- data ---
    file_1h: Path = BASE_DIR / "NQ_1Hour.csv"
    file_5m: Path = BASE_DIR / "NQ_5Min.csv"
    file_1h_rth: Path = BASE_DIR / "NQ_1Hour_RTH.csv"  # RTH-only hourly, for level charting
    source_tz: str = "America/Chicago"           # A1
    target_tz: str = "America/New_York"
    date_format: str = "%m/%d/%Y %I:%M %p"        # CSV date format
    start_year: int | None = 2020                 # A18: None = from the start of the file
    end_year: int | None = 2025                   # A18: None = to the end of the file

    # --- session (NY time) ---
    rth_start: time = time(9, 30)                 # A4
    rth_end: time = time(16, 0)
    open_candle: time = time(9, 30)               # A3 daily reference candle
    skip_first_30min: bool = True                # A19 block entries in the opening window
    open_skip_minutes: int = 30                   # A19 size of that window (minutes after open)

    # --- setup / execution tolerances ---
    level_touch_tolerance: float = 5.0            # A9  points around a level
    trigger_timeout_bars: int = 3                 # A10 bars after confirmation
    pullback_max_lookback: int = 6                # A11 max bars in pullback run
    profit_target_R: float = 2.5                  # Step 7 fixed target
    max_trades_per_day: int = 1                   # A14

    # --- trend rule ---
    prev_day_high_rth_only: bool = True           # A7

    # --- accounting ---
    point_value: float = 20.0                     # NQ = $20 / point (A16)
    force_exit_at_session_end: bool = True        # A13 (only True implemented: no overnight holds)

    # --- output ---
    trade_log_csv: Path = BASE_DIR / "trade_log.csv"
    show_plots: bool = True                       # open Plotly figures in browser
    plot_equity: bool = True                      # include the equity-curve figure
    plot_trade_indices: tuple = (251,252,253,254)              # which trade rows to chart (CSV row numbers)
    max_charts: int = 20                          # safety cap on how many trade tabs to open


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ohlc(path: Path, cfg: Config) -> pd.DataFrame:
    """Load one of the project's NQ CSVs into a tz-aware (NY) OHLCV frame.

    Source format: ';'-delimited, a title row then a header row, European
    numbers ('25.640,75' -> 25640.75) and 'M/D/YYYY h:mm AM/PM' dates in
    Chicago time.  Returns a frame indexed by NY datetime, sorted ascending,
    with columns [open, high, low, close, volume].
    """
    df = pd.read_csv(
        path, sep=";", skiprows=1, decimal=",", thousands=".", encoding="utf-8-sig",
    )
    df = df.rename(columns={
        "Date": "datetime", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], format=cfg.date_format)
    # Localise as Chicago, then convert to New York (A1).
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize(cfg.source_tz, ambiguous=True, nonexistent="shift_forward")
        .dt.tz_convert(cfg.target_tz)
    )
    keep = ["datetime", "open", "high", "low", "close"]
    if "volume" in df.columns:
        keep.append("volume")
    df = (df[keep]
          .dropna()
          .sort_values("datetime")
          .drop_duplicates("datetime")
          .set_index("datetime"))
    # A18: optional inclusive year window (None = no bound on that side).
    if cfg.start_year is not None:
        df = df[df.index.year >= cfg.start_year]
    if cfg.end_year is not None:
        df = df[df.index.year <= cfg.end_year]
    return df


def minutes_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorised minute-of-day (0..1439) for a DatetimeIndex."""
    return index.hour.to_numpy() * 60 + index.minute.to_numpy()


# ============================================================================
# STEP 1 — BIAS FLIP DETECTION
# ============================================================================
def detect_bias_flips(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Find bias-flip levels from consecutive opposite 1-hour candles.

    Bear->Bull (prev close<open, this close>open): bullish level at THIS open.
    Bull->Bear (prev close>open, this close<open): bearish level at THIS open.
    Returns columns: created_pos, created_time, price, type ('bull'/'bear').
    """
    o = df_1h["open"].to_numpy()
    c = df_1h["close"].to_numpy()
    direction = np.sign(c - o)                     # +1 bull, -1 bear, 0 doji
    prev, cur = direction[:-1], direction[1:]      # cur aligns to candle i (i>=1)
    bull = (prev < 0) & (cur > 0)
    bear = (prev > 0) & (cur < 0)

    rows = []
    for i in range(1, len(df_1h)):
        if bull[i - 1]:
            rows.append((i, df_1h.index[i], o[i], "bull"))
        elif bear[i - 1]:
            rows.append((i, df_1h.index[i], o[i], "bear"))
    return pd.DataFrame(rows, columns=["created_pos", "created_time", "price", "type"])


# ============================================================================
# LEVEL INVALIDATION  (event-driven chronological sweep)
# ============================================================================
def compute_invalidations(df_1h: pd.DataFrame, levels: pd.DataFrame,
                          cfg: Config) -> pd.DataFrame:
    """Stamp each level with `invalidated_at` (NY Timestamp or NaT).

    A level dies the moment an *RTH* 1-hour candle's BODY trades through it — the
    open/close straddle (or fully pass) the level, in either direction; wicks
    don't count.  Concretely: a bull level dies when min(open, close) < level, a
    bear level dies when max(open, close) > level.  We sweep all 1-hour candles
    in time order, maintaining the active bull/bear levels sorted by price; each
    candle kills the whole block of levels its body reaches, and we record the
    candle's CLOSE time as the invalidation time.
    """
    o = df_1h["open"].to_numpy()
    c = df_1h["close"].to_numpy()
    index = df_1h.index
    step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(hours=1)
    close_times = index + step                         # bar-start -> bar-close (A2)
    mod = minutes_of_day(index)
    rth_start_m = cfg.rth_start.hour * 60 + cfg.rth_start.minute
    rth_end_m = cfg.rth_end.hour * 60 + cfg.rth_end.minute
    # A5: a 1h candle is "RTH" if it OVERLAPS [rth_start, rth_end).
    is_rth = (mod < rth_end_m) & (mod + 60 > rth_start_m)

    # Map 1h candle position -> level row id (each flip is created on one bar).
    pos2level = {int(r.created_pos): idx for idx, r in levels.iterrows()}
    invalidated_at = [pd.NaT] * len(levels)

    # Active levels held as price-sorted parallel lists (prices + level ids).
    bull_p: list[float] = []
    bull_id: list[int] = []
    bear_p: list[float] = []
    bear_id: list[int] = []

    for i in range(len(df_1h)):
        if is_rth[i]:
            lo, hi = (o[i], c[i]) if o[i] <= c[i] else (c[i], o[i])   # candle BODY span
            # Bull levels the body reached BELOW (price > body low) are invalidated.
            cut = bisect.bisect_right(bull_p, lo)
            for k in range(cut, len(bull_p)):
                invalidated_at[bull_id[k]] = close_times[i]
            del bull_p[cut:]
            del bull_id[cut:]
            # Bear levels the body reached ABOVE (price < body high) are invalidated.
            cut = bisect.bisect_left(bear_p, hi)
            for k in range(cut):
                invalidated_at[bear_id[k]] = close_times[i]
            del bear_p[:cut]
            del bear_id[:cut]

        lvl = pos2level.get(i)
        if lvl is not None:                            # create this bar's level
            price, ltype = levels.at[lvl, "price"], levels.at[lvl, "type"]
            if ltype == "bull":
                j = bisect.bisect_left(bull_p, price)
                bull_p.insert(j, price); bull_id.insert(j, lvl)
            else:
                j = bisect.bisect_left(bear_p, price)
                bear_p.insert(j, price); bear_id.insert(j, lvl)

    levels = levels.copy()
    inv = pd.Series(invalidated_at, index=levels.index)
    levels["invalidated_at"] = pd.to_datetime(inv)
    return levels


# ============================================================================
# STEP 3 — TREND DETERMINATION  (modular: swap this function freely)
# ============================================================================
def default_uptrend(price: float, prev_day_high: float | None,
                    nearest_bull_level: float | None) -> bool:
    """Uptrend if price is above the previous day's high OR above the nearest
    bullish bias-flip level.  Replace with your own rule of the same signature.
    """
    above_prev_high = prev_day_high is not None and price > prev_day_high
    above_level = nearest_bull_level is not None and price > nearest_bull_level
    return above_prev_high or above_level


def _nearest_price(sorted_prices: np.ndarray, x: float) -> float | None:
    """Nearest value to x in an ascending array (checks the two neighbours)."""
    if sorted_prices.size == 0:
        return None
    i = int(np.searchsorted(sorted_prices, x))
    cands = []
    if i < sorted_prices.size:
        cands.append(sorted_prices[i])
    if i > 0:
        cands.append(sorted_prices[i - 1])
    return float(min(cands, key=lambda p: abs(p - x)))


# ============================================================================
# STEP 6 — STOP PLACEMENT
# ============================================================================
def compute_stop(o, h, l, c, react_i: int, level: float, cfg: Config) -> float | None:
    """Stop = low of the first bearish pullback candle that wicked below the
    level (A11).  The pullback run is the consecutive bars ending at the
    reaction bar whose low tagged the level (low <= level + tol).
    """
    tol = cfg.level_touch_tolerance
    start = react_i
    while (start - 1 >= 0
           and l[start - 1] <= level + tol
           and (react_i - (start - 1)) <= cfg.pullback_max_lookback):
        start -= 1

    for k in range(start, react_i + 1):              # first bearish wick below level
        if c[k] < o[k] and l[k] < level:
            return float(l[k])

    belows = [l[k] for k in range(start, react_i + 1) if l[k] < level]
    return float(min(belows)) if belows else None    # fallback / reject


# ============================================================================
# STEP 8 — TRADE MANAGEMENT
# ============================================================================
def manage_trade(h, l, entry_i: int, stop: float, target: float,
                 close_last: float, n: int):
    """Walk bars AFTER entry (A12). Returns (exit_pos, exit_price, reason).
    Stop-first when a single bar hits both stop and target.
    """
    for k in range(entry_i + 1, n):
        if l[k] <= stop:                              # stop-first (covers both)
            return k, stop, "stop"
        if h[k] >= target:
            return k, target, "target"
    return n - 1, close_last, "session_end"           # A13


# ============================================================================
# BACKTEST DRIVER  (day-by-day; never looks into the future)
# ============================================================================
def run_backtest(df_5m: pd.DataFrame, levels: pd.DataFrame, cfg: Config,
                 trend_fn=default_uptrend) -> pd.DataFrame:
    tol = cfg.level_touch_tolerance
    rth_start_m = cfg.rth_start.hour * 60 + cfg.rth_start.minute
    rth_end_m = cfg.rth_end.hour * 60 + cfg.rth_end.minute
    open_m = cfg.open_candle.hour * 60 + cfg.open_candle.minute
    entry_cutoff_m = rth_start_m + cfg.open_skip_minutes   # A19 earliest entry minute-of-day

    # Restrict 5-min data to the RTH session once.
    mod_all = minutes_of_day(df_5m.index)
    rth = df_5m[(mod_all >= rth_start_m) & (mod_all < rth_end_m)].copy()
    rth["session_date"] = rth.index.normalize()
    rth["mod"] = minutes_of_day(rth.index)

    # Previous-day RTH high (A7) and per-day groups.
    day_high = rth.groupby("session_date")["high"].max()
    groups = {d: g for d, g in rth.groupby("session_date")}
    dates = list(groups.keys())

    # Bull levels as price-sorted numpy arrays for fast per-day queries.
    bull = levels[levels["type"] == "bull"].sort_values("price")
    bp = bull["price"].to_numpy(dtype=float)
    # NY-local creation date (tz-naive) for the "created on a previous day" test.
    bcd = bull["created_time"].dt.date.to_numpy().astype("datetime64[D]")
    binv = bull["invalidated_at"].astype("int64").to_numpy()         # ns; NaT -> min int
    binv = np.where(binv == np.iinfo(np.int64).min, NEVER_NS, binv)

    trades = []
    for di, date in enumerate(dates):
        day = groups[date]
        n = len(day)
        mod = day["mod"].to_numpy()
        if not (mod == open_m).any():                 # A3: need a 9:30 bar
            continue
        o = day["open"].to_numpy(); h = day["high"].to_numpy()
        l = day["low"].to_numpy(); c = day["close"].to_numpy()
        t_ns = day.index.astype("int64").to_numpy()
        times = day.index
        open_idx = int(np.argmax(mod == open_m))
        ref_open = float(o[open_idx])                 # 9:30 open (A3)

        prev_high = float(day_high.iloc[di - 1]) if di > 0 else None

        # Candidate bull levels: created before today AND still active at the open.
        sess_start_ns = int(t_ns[open_idx])
        D64 = np.datetime64(date.date(), "D")
        coarse = (bcd < D64) & (binv > sess_start_ns)
        cand_price = bp[coarse]
        cand_inv = binv[coarse]
        # Small in-range subset actually reachable by today's price (for reactions).
        lo, hi = float(l.min()), float(h.max())
        rng = (cand_price >= lo - tol) & (cand_price <= hi + tol)
        in_price = cand_price[rng]
        in_inv = cand_inv[rng]

        i = 0
        traded = 0
        while i < n and traded < cfg.max_trades_per_day:
            close_i = c[i]
            if close_i <= ref_open:                   # Step 2: must be above 9:30 open
                i += 1; continue
            t = int(t_ns[i])

            # --- find the bull level this candle reacts into (nearest, still active) ---
            level = None
            level_inv = NEVER_NS                      # chosen level's invalidation time (ns)
            best = np.inf
            for k in range(in_price.size):
                p = in_price[k]
                if in_inv[k] <= t:                    # invalidated by now -> ineligible
                    continue
                if l[i] <= p + tol and close_i >= p - tol:   # A9 reaction
                    d = abs(close_i - p)
                    if d < best:
                        best, level, level_inv = d, float(p), int(in_inv[k])
            if level is None:
                i += 1; continue

            # --- trend filter (Step 3) ---
            nearest = _nearest_price(cand_price, close_i)
            if not trend_fn(close_i, prev_high, nearest):
                i += 1; continue

            # --- stop from the pullback (Step 6) ---
            stop = compute_stop(o, h, l, c, i, level, cfg)
            if stop is None:
                i += 1; continue

            # --- confirmation + trigger (Step 5) ---
            if i + 1 >= n:                            # no confirmation bar left today
                break
            conf_high = h[i + 1]
            entry_i = None
            j_max = min(i + 1 + cfg.trigger_timeout_bars, n - 1)
            for j in range(i + 2, j_max + 1):
                if h[j] > conf_high:                  # trades above confirmation high
                    entry_i = j; break
            if entry_i is None:                       # A10 timeout -> cancel
                i += 1; continue
            if level_inv <= int(t_ns[entry_i]):       # A17: invalidated before entry
                i = entry_i + 1; continue
            if cfg.skip_first_30min and mod[entry_i] < entry_cutoff_m:  # A19 opening filter
                i = entry_i + 1; continue

            entry = float(conf_high)
            risk = entry - stop
            if risk <= 0:
                i = entry_i + 1; continue
            target = entry + cfg.profit_target_R * risk

            # --- manage (Step 8) ---
            exit_i, exit_price, reason = manage_trade(
                h, l, entry_i, stop, target, float(c[-1]), n)

            r_mult = (exit_price - entry) / risk
            trades.append({
                "Date": date.date(),
                "Level Used": round(level, 2),
                "Entry Time": times[entry_i],
                "Entry Price": round(entry, 2),
                "Stop Price": round(stop, 2),
                "Target Price": round(target, 2),
                "Exit Time": times[exit_i],
                "Exit Price": round(exit_price, 2),
                "PnL Points": round(exit_price - entry, 2),
                "R Multiple": round(r_mult, 3),
                "Result": "Win" if r_mult > 0 else ("Loss" if r_mult < 0 else "Scratch"),
                "Exit Reason": reason,
                "PnL $": round((exit_price - entry) * cfg.point_value, 2),
            })
            traded += 1
            i = exit_i + 1                            # resume after the trade closes

    return pd.DataFrame(trades)


# ============================================================================
# STEP 9 — PERFORMANCE STATISTICS
# ============================================================================
def compute_statistics(trades: pd.DataFrame, cfg: Config) -> dict:
    if trades.empty:
        return {"Total Trades": 0}
    r = trades["R Multiple"].to_numpy(dtype=float)
    pnl = trades["PnL Points"].to_numpy(dtype=float)
    wins = r > 0
    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    equity_r = np.cumsum(r)
    drawdown = equity_r - np.maximum.accumulate(equity_r)
    hold = (trades["Exit Time"] - trades["Entry Time"]).mean()

    return {
        "Total Trades": int(len(trades)),
        "Win Rate %": round(100 * wins.mean(), 2),
        "Average R": round(r.mean(), 3),
        "Net R": round(r.sum(), 2),
        "Expectancy (R/trade)": round(r.mean(), 3),
        "Profit Factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else np.inf,
        "Max Drawdown (R)": round(drawdown.min(), 2),
        "Net PnL (points)": round(pnl.sum(), 2),
        "Net PnL ($)": round(pnl.sum() * cfg.point_value, 2),
        "Average Hold": str(hold).split(".")[0],
    }


# ============================================================================
# STEP 10 — VISUALISATION  (Plotly; imported lazily so the core has no GUI dep)
# ============================================================================
def plot_equity_curve(trades: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = trades.sort_values("Exit Time")
    x = t["Exit Time"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Cumulative R", "Cumulative PnL (points)"))
    fig.add_trace(go.Scatter(x=x, y=t["R Multiple"].cumsum(), name="Cum R",
                             line=dict(color="#26a69a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=t["PnL Points"].cumsum(), name="Cum PnL",
                             line=dict(color="#f0c040")), row=2, col=1)
    fig.update_layout(height=650, template="plotly_dark",
                      title="Equity Curve", showlegend=False)
    return fig


def plot_trade(trades: pd.DataFrame, idx: int, df_5m: pd.DataFrame, cfg: Config,
               levels: pd.DataFrame | None = None):
    """Two-panel chart for one trade (panels share the price axis, so levels
    line up across both):
      LEFT  = previous trading day's 1-hour RTH candles + the active bias-flip
              levels (bullish blue, bearish red, the traded level gold).
      RIGHT = the trade day's 5-minute candles + level / entry / stop / target.
    Pass `levels` (from the backtest) to draw the bias-flip lines on the left;
    if `levels` is None or there is no prior session, only the 5-min panel shows.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tr = trades.iloc[idx]
    tz = cfg.target_tz
    rs = cfg.rth_start.hour * 60 + cfg.rth_start.minute
    re = cfg.rth_end.hour * 60 + cfg.rth_end.minute

    def rth_slice(frame):
        m = minutes_of_day(frame.index)
        return frame[(m >= rs) & (m < re)]

    # --- trade day 5-min (RTH) ---
    trade_date = pd.Timestamp(tr["Date"], tz=tz)
    day = rth_slice(df_5m[df_5m.index.normalize() == trade_date])

    # --- previous trading day's RTH 1-hour candles (resampled from 5m) ---
    rmask = (minutes_of_day(df_5m.index) >= rs) & (minutes_of_day(df_5m.index) < re)
    sess_dates = df_5m.index[rmask].normalize().unique()
    earlier = sess_dates[sess_dates < trade_date]
    prev_date = earlier.max() if len(earlier) else None

    prev_1h = None
    if prev_date is not None:
        p = rth_slice(df_5m[df_5m.index.normalize() == prev_date])
        if not p.empty:
            off = pd.Timedelta(minutes=cfg.rth_start.minute)   # anchor buckets to the open
            bucket = (p.index - off).floor("h") + off
            prev_1h = p.groupby(bucket).agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"))

    has_left = prev_1h is not None and not prev_1h.empty
    if has_left:
        fig = make_subplots(
            rows=1, cols=2, shared_yaxes=True, column_widths=[0.32, 0.68],
            horizontal_spacing=0.015,
            subplot_titles=(f"Prev day {prev_date.date()} — 1H RTH",
                            f"Trade day {trade_date.date()} — 5M"))
        rcol = 2
    else:
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=(f"Trade day {trade_date.date()} — 5M",))
        rcol = 1

    # ---- LEFT: prev-day 1H RTH candles + bias-flip levels ----
    if has_left:
        fig.add_trace(go.Candlestick(
            x=prev_1h.index, open=prev_1h["open"], high=prev_1h["high"],
            low=prev_1h["low"], close=prev_1h["close"], showlegend=False),
            row=1, col=1)
        if levels is not None and not levels.empty:
            ymin = float(min(prev_1h["low"].min(), day["low"].min()))
            ymax = float(max(prev_1h["high"].max(), day["high"].max()))
            open_ts = trade_date + pd.Timedelta(minutes=rs)    # trade-day 9:30 open
            active = levels[(levels["created_time"] < open_ts)
                            & (levels["price"].between(ymin, ymax))
                            & (levels["invalidated_at"].isna()
                               | (levels["invalidated_at"] > open_ts))]
            for _, lv in active.iterrows():
                used = abs(lv["price"] - tr["Level Used"]) < 0.01
                fig.add_hline(
                    y=lv["price"], row=1, col=1,
                    line=dict(width=2 if used else 1,
                              dash="solid" if used else "dot",
                              color="#f0c040" if used else
                                    "#42a5f5" if lv["type"] == "bull" else "#ef9a9a"))

    # ---- RIGHT: trade-day 5-min candles + level / entry / stop / target ----
    fig.add_trace(go.Candlestick(
        x=day.index, open=day["open"], high=day["high"],
        low=day["low"], close=day["close"], showlegend=False), row=1, col=rcol)
    for price, color, label in [
        (tr["Level Used"], "#f0c040", "Level"),
        (tr["Entry Price"], "#26a69a", "Entry"),
        (tr["Stop Price"], "#ef5350", "Stop"),
        (tr["Target Price"], "#ab47bc", "Target"),
    ]:
        fig.add_hline(y=price, row=1, col=rcol, line=dict(color=color, dash="dot"),
                      annotation_text=label, annotation_position="right")
    fig.add_trace(go.Scatter(
        x=[tr["Entry Time"]], y=[tr["Entry Price"]], mode="markers", name="Entry",
        marker=dict(color="#26a69a", size=11, symbol="triangle-up")), row=1, col=rcol)
    fig.add_trace(go.Scatter(
        x=[tr["Exit Time"]], y=[tr["Exit Price"]], mode="markers", name="Exit",
        marker=dict(color="#ffffff", size=11, symbol="x")), row=1, col=rcol)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=650, template="plotly_dark", showlegend=False,
                      title=f"Trade {idx}  {tr['Date']}  {tr['Result']}  {tr['R Multiple']}R")
    return fig


def plot_rth_levels(df_1h_rth: pd.DataFrame, levels: pd.DataFrame, cfg: Config,
                    start, end, only_active: bool = True, include_older: bool = False):
    """Plot the NQ_1Hour_RTH candles between `start` and `end` (date-like,
    inclusive) with the bias-flip levels that are in play over that range.

    Candles come straight from the RTH 1-hour file (no resampling) and are drawn
    on a real datetime axis, so overnight/weekend ETH periods show up as gaps —
    like a session-only series.

    "Active" means still valid at the END of the requested range: a level
    invalidated by a close on the last day (e.g. a 15:00 spike) counts as
    invalidated, even though that happens after the last drawn bar's start.

    By DEFAULT only the surviving (active) levels are drawn — solid and
    price-labelled.  Pass `only_active=False` to also show the invalidated ones
    (faded/dotted, stopping where they died).  Bullish = blue, bearish = red.

    Also by default only levels that FORMED within the window are shown; pass
    `include_older=True` to also draw older levels that are still active (they
    persist and the backtest still uses them — they're just hidden here to keep
    the in-range verification clean).
    """
    import plotly.graph_objects as go

    tz = cfg.target_tz
    win_start = pd.Timestamp(start, tz=tz)
    win_end = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)        # inclusive end day
    bars = df_1h_rth[(df_1h_rth.index >= win_start) & (df_1h_rth.index < win_end)]
    if bars.empty:
        raise ValueError(f"No RTH 1-hour candles between {start} and {end} "
                         f"(check the dates / that NQ_1Hour_RTH.csv covers them)")

    t0, t1 = bars.index[0], bars.index[-1]
    fig = go.Figure(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"],
        close=bars["close"], name="1H RTH", showlegend=False))

    # --- bias-flip levels as creation -> invalidation rays, clamped to window ---
    ymin, ymax = float(bars["low"].min()), float(bars["high"].max())
    vis = levels[(levels["price"] >= ymin) & (levels["price"] <= ymax)]
    if not include_older:                       # only levels that FORMED in the window
        vis = vis[vis["created_time"] >= win_start]
    n_drawn = 0
    for _, lv in vis.iterrows():
        ct, iv = lv["created_time"], lv["invalidated_at"]
        if ct > t1:                                                 # created after window
            continue
        if pd.notna(iv) and iv <= t0:                               # died before window
            continue
        active = pd.isna(iv) or iv >= win_end   # survives to the END of the range (not just t1)
        if only_active and not active:
            continue
        x_start = max(ct, t0)                                       # clamp ray into view
        x_end = t1 if active else min(iv, t1)
        if x_end < x_start:
            x_end = x_start
        color = "#42a5f5" if lv["type"] == "bull" else "#ef5350"
        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[lv["price"], lv["price"]], mode="lines",
            line=dict(color=color, width=1.4, dash="solid" if active else "dot"),
            opacity=1.0 if active else 0.45, showlegend=False,
            hovertemplate=f"{lv['type']} {lv['price']:.2f}<extra></extra>"))
        if active:                                                  # label price at right edge
            fig.add_annotation(x=x_end, y=lv["price"], text=f"  {lv['price']:.2f}",
                               xanchor="left", yanchor="middle", showarrow=False,
                               font=dict(size=9, color=color))
        n_drawn += 1

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=700, template="plotly_dark", showlegend=False,
                      title=f"1H RTH bias-flip levels  {t0.date()} .. {t1.date()}  "
                            f"({n_drawn} levels)")
    return fig


def parse_trade_indices(args, n_trades: int) -> list[int]:
    """Turn CLI tokens into trade row numbers. Accepts ints (e.g. 5, -1),
    ranges ('10-15'), and 'all'.  Example:  python Backtest.py 5 12 30-34 -1
    """
    indices: list[int] = []
    for a in args:
        if a.lower() == "all":
            return list(range(n_trades))
        if "-" in a[1:]:                              # range token like 10-15
            lo, hi = a.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(a))
    return indices


def show_trade(idx: int, cfg: Config | None = None):
    """Chart ONE trade by its row number, reusing the saved trade_log.csv so the
    backtest is NOT re-run.  Handy from a REPL for browsing:  show_trade(42).
    (For many trades, load df_5m once and call plot_trade directly — see docs.)
    """
    cfg = cfg or Config()
    trades = pd.read_csv(cfg.trade_log_csv)
    for col in ("Entry Time", "Exit Time"):            # restore tz-aware (NY) times
        trades[col] = pd.to_datetime(trades[col], utc=True).dt.tz_convert(cfg.target_tz)
    df_1h = load_ohlc(cfg.file_1h, cfg)
    levels = compute_invalidations(df_1h, detect_bias_flips(df_1h), cfg)
    df_5m = load_ohlc(cfg.file_5m, cfg)
    plot_trade(trades, idx, df_5m, cfg, levels=levels).show()


def show_levels(start, end, cfg: Config | None = None, only_active: bool = True,
                include_older: bool = False):
    """Convenience: load full history, detect levels, and show the 1H RTH levels
    chart for a date range.  Example:  show_levels("2025-05-21", "2025-06-24")
    Loads the whole file (ignoring the Config year window) so level creation /
    invalidation history is complete; only the requested range is drawn.
    """
    cfg = replace(cfg or Config(), start_year=None, end_year=None)
    df_1h_rth = load_ohlc(cfg.file_1h_rth, cfg)
    levels = compute_invalidations(df_1h_rth, detect_bias_flips(df_1h_rth), cfg)
    plot_rth_levels(df_1h_rth, levels, cfg, start, end, only_active, include_older).show()


# ============================================================================
# MAIN
# ============================================================================
def main(cfg: Config | None = None, plot_args=None) -> pd.DataFrame:
    cfg = cfg or Config()

    yr = f"  [year window: {cfg.start_year or 'start'}..{cfg.end_year or 'end'}]"
    print("Loading data ..." + (yr if (cfg.start_year or cfg.end_year) else ""))
    df_1h_rth = load_ohlc(cfg.file_1h_rth, cfg)   # levels live on the RTH-anchored 1H candles
    df_5m = load_ohlc(cfg.file_5m, cfg)
    if df_5m.empty or df_1h_rth.empty:
        raise SystemExit(f"No data in year window "
                         f"{cfg.start_year}..{cfg.end_year}; check start_year/end_year.")
    print(f"  1H RTH bars: {len(df_1h_rth):,}   5M bars: {len(df_5m):,}   "
          f"range {df_5m.index[0].date()} .. {df_5m.index[-1].date()}")

    print("Detecting bias-flip levels (RTH 1H) ...")
    levels = detect_bias_flips(df_1h_rth)
    levels = compute_invalidations(df_1h_rth, levels, cfg)
    n_bull = int((levels["type"] == "bull").sum())
    n_live = int(levels["invalidated_at"].isna().sum())
    print(f"  {len(levels):,} levels ({n_bull:,} bullish); {n_live:,} still active at end")

    print("Running backtest (day-by-day) ...")
    trades = run_backtest(df_5m, levels, cfg)

    if not trades.empty:
        trades.to_csv(cfg.trade_log_csv, index=False)
        print(f"  Wrote {len(trades):,} trades to {cfg.trade_log_csv.name}")

    print("\n===== SUMMARY =====")
    for k, v in compute_statistics(trades, cfg).items():
        print(f"  {k:<22}: {v}")

    if plot_args:                                     # CLI indices imply "show me"
        cfg.show_plots = True
    if cfg.show_plots and not trades.empty:
        if cfg.plot_equity:
            plot_equity_curve(trades).show()
        wanted = parse_trade_indices(plot_args, len(trades)) if plot_args \
            else list(cfg.plot_trade_indices)
        # keep valid + de-duped (preserve order), then cap the number of tabs.
        seen, indices = set(), []
        for i in wanted:
            if -len(trades) <= i < len(trades) and i not in seen:
                seen.add(i); indices.append(i)
        if len(indices) > cfg.max_charts:
            print(f"  (capping at {cfg.max_charts} charts of {len(indices)} requested)")
            indices = indices[:cfg.max_charts]
        print(f"  charting trade rows: {indices}")
        for i in indices:
            plot_trade(trades, i, df_5m, cfg, levels=levels).show()

    return trades


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "levels":
        # 1H RTH levels for a date range. By default: active levels that FORMED
        # in the range. Optional flags (any order): 'all' also shows invalidated,
        # 'older' also shows still-active levels created before the range.
        #   python Backtest.py levels 2025-12-05 2025-12-10
        #   python Backtest.py levels 2025-12-05 2025-12-10 all older
        if len(args) < 3:
            raise SystemExit("usage: python Backtest.py levels <start> <end> [all] [older]")
        flags = {a.lower() for a in args[3:]}
        show_levels(args[1], args[2], only_active=not ("all" in flags),
                    include_older=("older" in flags))
    else:
        # No args -> uses Config.plot_trade_indices. Otherwise trade rows, e.g.:
        #   python Backtest.py 5 12 30-34 -1      python Backtest.py all
        main(plot_args=args)
