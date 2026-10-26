#!/usr/bin/env python3
"""
Download 5-minute equity bars from Interactive Brokers through the TWS API.

Needs IB Gateway (or TWS) running and logged in on this machine, with the API
enabled. IBKR Desktop does not serve the API. The account also needs a market
data subscription for the stock's exchange, otherwise IB rejects the request.

One request per symbol per trading day, saved as data/raw_ibkr/<SYMBOL>_5m/<YYYY-MM-DD>.parquet:
  * completed days are skipped, so re-running costs nothing
  * holidays come back empty and are saved as empty files so they are not re-asked
  * failed days (timeouts, dropped connection) are left missing; just run it again
When every day is present, the days are combined into data/<symbol>_5m.parquet.

Usage:
    pip install ib_async pandas pyarrow
    python download_ibkr.py --check                       # test the connection only
    python download_ibkr.py SPY                           # defaults: START..END below
    python download_ibkr.py SPY QQQ --start 2024-01-02 --end 2024-12-31
    python download_ibkr.py SPY --port 4002               # paper-trading Gateway

Ports: IB Gateway 4001 (live) / 4002 (paper); TWS 7496 (live) / 7497 (paper).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from ib_async import IB, Stock

# ---------------------------------------------------------------- settings
DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw_ibkr"

START = date(2022, 1, 3)   # inclusive
END = date(2026, 9, 30)    # inclusive
HOST = "127.0.0.1"
PORT = 4001
CLIENT_ID = 17

USE_RTH = True             # regular trading hours only: 09:30-16:00 ET
BAR = "5 mins"             # same sampling interval as the BTC RV estimator
BARS_PER_DAY = 78          # 6.5 hours / 5 minutes (half days close 13:00 ET: 42)
WHAT = "TRADES"            # bar prices from trades (MIDPOINT, BID_ASK also exist)
PAUSE = 1.0                # seconds between requests, to stay clear of pacing limits
TIMEOUT = 60

ET = ZoneInfo("America/New_York")
# Error codes that mean "no point retrying": missing data permissions / unknown contract
FATAL = {200, 354, 10089, 10090, 10167, 10168}
# --------------------------------------------------------------------------


def weekdays(start: date, end: date):
    for d in pd.bdate_range(start, end):
        yield d.date()


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def connect(port: int) -> tuple[IB, list]:
    ib = IB()
    errors: list = []   # (reqId, code, message) since the last clear
    ib.errorEvent += lambda req_id, code, msg, contract: errors.append((req_id, code, msg))
    try:
        ib.connect(HOST, port, clientId=CLIENT_ID, timeout=15)
    except (ConnectionRefusedError, TimeoutError):
        sys.exit(f"Nothing answered on {HOST}:{port}. Start IB Gateway (or TWS), log in, "
                 f"and check the API port under Configure > Settings > API > Settings.")
    return ib, errors


def fetch_day(ib: IB, errors: list, contract, day: date) -> pd.DataFrame | None:
    """One trading day of 5-minute bars. None means the request failed (retry later)."""
    errors.clear()
    end = datetime.combine(day, dtime(23, 59, 59), tzinfo=ET)
    bars = ib.reqHistoricalData(
        contract, endDateTime=end, durationStr="1 D", barSizeSetting=BAR,
        whatToShow=WHAT, useRTH=USE_RTH, formatDate=2, timeout=TIMEOUT,
    )
    fatal = [e for e in errors if e[1] in FATAL]
    if fatal:
        raise RuntimeError(f"IB error {fatal[0][1]}: {fatal[0][2]}")

    df = pd.DataFrame(
        [(b.date, b.open, b.close, b.volume) for b in bars],
        columns=["timestamp", "open", "close", "volume"],
    )
    if df.empty:
        # Error 162 "HMDS query returned no data" is a holiday; no error at all is a timeout
        return df if any(e[1] == 162 for e in errors) else None

    # IB stamps each bar with its START time, the same convention as the Binance
    # klines; 'close' is the price at the end of that 5-minute bar.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"].dt.tz_convert(ET).dt.date == day]   # guard against spill-over
    return df.set_index("timestamp").astype(float)


def download(ib: IB, errors: list, symbol: str, days: list[date]) -> list[date]:
    """Fetch every missing day for one symbol. Returns the days still missing."""
    contract = Stock(symbol, "SMART", "USD")
    if not ib.qualifyContracts(contract):
        raise RuntimeError(f"IB does not recognise {symbol} as a USD stock")

    outdir = RAW / f"{symbol}_5m"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (outdir / f"{d}.parquet").exists()]
    print(f"\n[{stamp()}] {symbol}: {len(days) - len(todo)}/{len(days)} days already "
          f"on disk, {len(todo)} to fetch", flush=True)

    failed = []
    for i, day in enumerate(todo, 1):
        df = fetch_day(ib, errors, contract, day)
        if df is None:
            failed.append(day)
            print(f"  [{stamp()}] failed   {symbol} {day} (will retry on re-run)", flush=True)
        else:
            df.to_parquet(outdir / f"{day}.parquet")
            if i % 20 == 0 or i == len(todo) or len(df) not in (0, BARS_PER_DAY):
                note = "holiday" if df.empty else f"{len(df)} bars"
                print(f"  [{stamp()}] {i}/{len(todo)}  {symbol} {day}  {note}", flush=True)
        time.sleep(PAUSE)
    return failed


def combine(symbol: str, days: list[date]) -> Path:
    """Stack the per-day files into one Parquet file, like btc_1m.parquet."""
    frames = [pd.read_parquet(RAW / f"{symbol}_5m" / f"{d}.parquet") for d in days]
    data = pd.concat([f for f in frames if not f.empty]).sort_index()
    data = data[~data.index.duplicated(keep="first")]
    out = DATA / f"{symbol.lower()}_5m.parquet"
    data.to_parquet(out)

    per_day = data.groupby(data.index.tz_convert(ET).date).size()
    print(f"  {symbol}: {len(data):,} bars over {len(per_day)} trading days "
          f"({data.index[0]} to {data.index[-1]})")
    short = per_day[per_day != BARS_PER_DAY]
    if len(short):
        print(f"  {len(short)} day(s) without {BARS_PER_DAY} bars (half days close at 13:00 ET = 42):")
        print("   " + ", ".join(f"{d} ({n})" for d, n in short.items()))
    print(f"  saved {out}")
    return out


def check(port: int) -> int:
    ib, errors = connect(port)
    print(f"Connected to {HOST}:{port}; server time {ib.reqCurrentTime()}")
    print(f"Accounts: {', '.join(ib.managedAccounts())}")
    contract = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(contract)
    day = pd.bdate_range(end=date.today() - pd.Timedelta(days=1), periods=1)[0].date()
    df = fetch_day(ib, errors, contract, day)
    ib.disconnect()
    if df is None or df.empty:
        print(f"No SPY bars for {day}. IB messages: {errors}")
        return 1
    print(f"SPY {day}: {len(df)} five-minute bars\n{df.head(3)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("symbols", nargs="*", help="e.g. SPY AAPL")
    ap.add_argument("--start", type=date.fromisoformat, default=START)
    ap.add_argument("--end", type=date.fromisoformat, default=END)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--check", action="store_true", help="test the connection and exit")
    args = ap.parse_args()

    if args.check:
        return check(args.port)
    if not args.symbols:
        ap.error("give at least one symbol, or --check")

    days = [d for d in weekdays(args.start, args.end) if d < date.today()]
    ib, errors = connect(args.port)
    incomplete = {}
    try:
        for sym in (s.upper() for s in args.symbols):
            failed = download(ib, errors, sym, days)
            if failed:
                incomplete[sym] = failed
            else:
                combine(sym, days)
    finally:
        ib.disconnect()

    if incomplete:
        print("\nSome days failed; run the same command again to fetch them:")
        for sym, failed in incomplete.items():
            print(f"   {sym}: {len(failed)} day(s), first {failed[0]}")
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
