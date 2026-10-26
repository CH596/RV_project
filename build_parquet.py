"""
Turn the folder of Binance BTCUSDT 1-minute kline zips into one Parquet file,
and print a validation report.

Run once:   python3 build_parquet.py
Needs:      pip install pandas pyarrow
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- 1. Where things live: data/ next to this script ------------------------
DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"
OUT = DATA / "btc_1m.parquet"

# Binance kline columns, in file order (the CSVs have no header row,
# but we guard against one below just in case).
COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# ---- 2. Read every zip into a DataFrame -------------------------------------
zips = sorted(RAW.glob("*.zip"))
print(f"Found {len(zips)} zip files")

frames = []
for zp in zips:
    # pandas can read a CSV straight out of a zip that contains one file
    df = pd.read_csv(zp, header=None, names=COLS, compression="zip")

    # If a file did have a header row, its first column won't be a number: drop it
    t = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[t.notna()].copy()
    t = t[t.notna()].astype("int64")

    # Timestamp units: milliseconds before 2025, microseconds from 2025 on.
    # ms timestamps are ~1.7e12, us timestamps are ~1.7e15, so a threshold works.
    t_ms = np.where(t > 1e14, t // 1000, t)
    df["timestamp"] = pd.to_datetime(t_ms, unit="ms", utc=True)

    # Keep only what the project needs
    df = df[["timestamp", "close", "volume"]].astype({"close": float, "volume": float})
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# ---- 3. Order, de-duplicate, index by time ----------------------------------
n_before = len(data)
data = data.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
n_dupes = n_before - len(data)
data = data.set_index("timestamp")

# ---- 4. Validation report ---------------------------------------------------
print(f"\nRows: {len(data):,}   duplicates removed: {n_dupes}")
print(f"First: {data.index[0]}   Last: {data.index[-1]}")

bad_prices = (data["close"] <= 0).sum()
print(f"Non-positive close prices: {bad_prices}")

# Rows per month vs expected (1440 minutes x days in month)
months = data.index.tz_convert(None).to_period("M")
counts = data.groupby(months).size()
expected = pd.Series(counts.index.days_in_month * 1440, index=counts.index)
short = counts[counts != expected]
print(f"\nMonths with the wrong row count: {len(short)} of {len(counts)}")
if len(short):
    print(pd.DataFrame({"rows": short, "expected": expected[short.index],
                        "missing": expected[short.index] - short}))

# Exact missing minutes
full = pd.date_range(data.index[0], data.index[-1], freq="min", tz="UTC")
missing = full.difference(data.index)
print(f"\nMissing minutes in total: {len(missing)}")
if len(missing):
    # Group consecutive missing minutes into gaps, so outages show as one line
    gap_id = (missing.to_series().diff() != pd.Timedelta("1min")).cumsum()
    gaps = missing.to_series().groupby(gap_id.values).agg(["first", "last", "count"])
    print(gaps.sort_values("count", ascending=False).head(20).to_string(index=False))

# Largest one-minute moves (log returns), to eyeball for bad ticks
r = np.log(data["close"]).diff()
print("\nLargest absolute 1-min log returns:")
print(r.abs().nlargest(10).to_string())

# ---- 5. Save ----------------------------------------------------------------
# Gaps are left as gaps on purpose: decide the missing-minute rule separately.
data.to_parquet(OUT)
print(f"\nSaved {OUT}")