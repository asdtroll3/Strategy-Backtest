"""Convert a 5-minute OHLCV CSV into 1-hour OHLCV CSVs (using pandas).

Produces two files from NQ_5Min.csv:
  * NQ_1Hour.csv      - every hour of the ~24h session, bars anchored on the
                        clock hour (... 1:00, 2:00, 3:00 PM ...).
  * NQ_1Hour_RTH.csv  - Regular Trading Hours only. RTH is 09:30-15:30 New York
                        time; the source is Chicago time (one hour behind), so
                        that window is 08:30-14:30 here. Bars are anchored to
                        the 08:30 open, giving six hourly bars per day:
                        8:30, 9:30, 10:30, 11:30, 12:30, 1:30 PM.

The source file uses:
  * ';' as the field delimiter
  * a title row, then a header row: Date;Symbol;Open;High;Low;Close;Volume
  * dates formatted as  M/D/YYYY h:mm AM/PM   (e.g. 1/30/2026 3:55 PM)
  * European numbers:  25.640,75  ->  25640.75  ('.' thousands, ',' decimal)
  * rows ordered newest-first (descending time)

Each 5-minute bar is grouped into the hour it starts in. Within an hour the
bars are aggregated as:
  Open   = open of the earliest bar in the hour
  High   = highest high
  Low    = lowest low
  Close  = close of the latest bar in the hour
  Volume = sum of volumes
The output keeps the exact same format as the input (delimiter, number format,
date format and newest-first ordering) so it is a drop-in 1-hour equivalent.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "NQ_5Min.csv"
OUTPUT_FILE = BASE_DIR / "NQ_1Hour.csv"
OUTPUT_RTH_FILE = BASE_DIR / "NQ_1Hour_RTH.csv"

DATE_FORMAT = "%m/%d/%Y %I:%M %p"
COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
# European numbers are US-formatted then ',' and '.' are swapped in one pass.
_SWAP = str.maketrans({",": ".", ".": ","})

# RTH window in the source's Chicago time, as minutes past midnight.
# 08:30 Chicago = 09:30 New York (cash open); 14:30 Chicago = 15:30 New York.
RTH_START_MIN = 8 * 60 + 30
RTH_END_MIN = 14 * 60 + 30


def format_prices(prices):
    """Series of floats -> European strings, e.g. 25640.75 -> '25.640,75'."""
    return prices.map("{:,.2f}".format).str.translate(_SWAP)


def format_dates(timestamps):
    """Series of datetimes -> 'M/D/YYYY h:mm AM/PM' (no zero-padding on M/D/h).

    strftime can't strip leading zeros portably, so build the string from parts.
    """
    dt = timestamps.dt
    hour12 = (dt.hour % 12).replace(0, 12)
    meridiem = dt.hour.map(lambda h: "AM" if h < 12 else "PM")
    return (
        dt.month.astype(str) + "/" + dt.day.astype(str) + "/" + dt.year.astype(str)
        + " " + hour12.astype(str) + ":" + dt.minute.astype(str).str.zfill(2)
        + " " + meridiem
    )


def load_five_minute(input_file):
    """Read + parse the 5-minute CSV. Returns (title_row, df, raw_count, skipped).

    The returned frame is sorted ascending so "first"/"last" aggregations pick
    the earliest open / latest close within each bar.
    """
    # Preserve the original title row (line 1) verbatim.
    with open(input_file, encoding="utf-8-sig") as f:
        title_row = f.readline().rstrip("\r\n")

    # decimal/thousands handle the European number format on read.
    df = pd.read_csv(
        input_file, sep=";", skiprows=1, decimal=",", thousands=".",
        encoding="utf-8-sig",
    )
    raw_count = len(df)

    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=COLUMNS).sort_values("Date")
    skipped = raw_count - len(df)
    return title_row, df, raw_count, skipped


def aggregate(df, bucket):
    """Group bars by (Symbol, bucket) into OHLCV bars, newest-first.

    `bucket` is a Series (aligned with df) giving each bar's hour start.
    """
    return (
        df.assign(Bucket=bucket)
        .groupby(["Symbol", "Bucket"], sort=False)
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
        .sort_values("Bucket", ascending=False)  # newest-first, like the source
    )


def write_bars(title_row, bars, output_file):
    """Write aggregated bars back out in the original file format."""
    out = pd.DataFrame({
        "Date": format_dates(bars["Bucket"]),
        "Symbol": bars["Symbol"].to_numpy(),
        "Volume": bars["Volume"].astype("int64").astype(str).to_numpy(),
    })
    for col in PRICE_COLUMNS:
        out[col] = format_prices(bars[col]).to_numpy()
    out = out[COLUMNS]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write(title_row + "\r\n")
        out.to_csv(f, sep=";", index=False, lineterminator="\r\n")
    return len(bars)


def main():
    title_row, df, raw_count, skipped = load_five_minute(INPUT_FILE)

    # Full session: bars anchored on the clock hour.
    hourly = aggregate(df, df["Date"].dt.floor("h"))
    write_bars(title_row, hourly, OUTPUT_FILE)

    # RTH only: keep 08:30-14:30 Chicago, then anchor hourly bars to the 08:30
    # open. Shifting back 30 min, flooring, and shifting forward 30 min maps
    # 08:30->08:30, 09:25->08:30, 09:30->09:30, ... so each bar spans :30 to :30.
    minute_of_day = df["Date"].dt.hour * 60 + df["Date"].dt.minute
    rth = df[(minute_of_day >= RTH_START_MIN) & (minute_of_day < RTH_END_MIN)]
    half_hour = pd.Timedelta(minutes=30)
    rth_bucket = (rth["Date"] - half_hour).dt.floor("h") + half_hour
    rth_hourly = aggregate(rth, rth_bucket)
    write_bars(title_row, rth_hourly, OUTPUT_RTH_FILE)

    print(f"Read {raw_count} five-minute bars from {INPUT_FILE.name}"
          + (f" ({skipped} rows skipped)" if skipped else ""))
    print(f"Wrote {len(hourly)} one-hour bars to {OUTPUT_FILE.name}")
    print(f"Wrote {len(rth_hourly)} one-hour RTH bars to {OUTPUT_RTH_FILE.name}")


if __name__ == "__main__":
    main()
