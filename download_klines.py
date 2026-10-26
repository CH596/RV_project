#!/usr/bin/env python3
"""
Download monthly 1-minute kline archives from Binance's public data site.

No account, no API key, no login — these are plain static files over HTTPS.

Built for an unreliable connection and unattended running:
  * completed files are skipped, so re-running costs nothing
  * partial downloads go to a .part file and are only renamed on success
  * corrupt zips are detected and re-fetched
  * it keeps retrying until everything is present or MAX_HOURS elapses

Usage:
    pip install requests
    python download_klines.py

Safe to kill and restart at any point. If it stops with files still missing,
just run it again.
"""

from __future__ import annotations

import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------- settings
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START = (2022, 1)        # inclusive (year, month)
END = (2026, 9)          # inclusive — last complete month
OUTDIR = Path(__file__).resolve().parent / "data" / "raw"

MAX_HOURS = 8            # give up after this long; re-run to continue
RETRIES_PER_PASS = 3
PASS_PAUSE = 30          # seconds between full passes
TIMEOUT = 60

BASE = "https://data.binance.vision/data/spot/monthly/klines"
# --------------------------------------------------------------------------


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def is_good_zip(path: Path) -> bool:
    """True only if the file exists and is a complete, readable zip."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download(url: str, dest: Path) -> str:
    """Fetch url to dest. Returns 'ok', 'missing', or an error description."""
    part = dest.parent / (dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception as e:
        part.unlink(missing_ok=True)
        return f"{type(e).__name__}: {e}"

    part.replace(dest)
    if not is_good_zip(dest):
        dest.unlink(missing_ok=True)
        return "incomplete or corrupt zip"
    return "ok"


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    wanted = []
    for y, m in months(START, END):
        name = f"{SYMBOL}-{INTERVAL}-{y:04d}-{m:02d}.zip"
        wanted.append((name, f"{BASE}/{SYMBOL}/{INTERVAL}/{name}"))

    print(f"{len(wanted)} month(s) requested -> {OUTDIR.resolve()}")

    not_published = set()
    deadline = time.time() + MAX_HOURS * 3600
    pass_no = 0

    while time.time() < deadline:
        pass_no += 1
        todo = [
            (n, u) for n, u in wanted
            if n not in not_published and not is_good_zip(OUTDIR / n)
        ]
        if not todo:
            break

        print(f"\n[{stamp()}] pass {pass_no}: {len(todo)} file(s) to fetch",
              flush=True)

        for name, url in todo:
            for attempt in range(1, RETRIES_PER_PASS + 1):
                result = download(url, OUTDIR / name)
                if result == "ok":
                    size = (OUTDIR / name).stat().st_size / 1e6
                    print(f"  [{stamp()}] ok       {name}  ({size:.1f} MB)",
                          flush=True)
                    break
                if result == "missing":
                    print(f"  [{stamp()}] no file  {name} (not published)",
                          flush=True)
                    not_published.add(name)
                    break
                print(f"  [{stamp()}] retry {attempt}/{RETRIES_PER_PASS}  "
                      f"{name}: {result}", flush=True)
                time.sleep(5 * attempt)

        if any(n not in not_published and not is_good_zip(OUTDIR / n)
               for n, _ in wanted):
            print(f"[{stamp()}] pausing {PASS_PAUSE}s before the next pass",
                  flush=True)
            time.sleep(PASS_PAUSE)

    available = [n for n, _ in wanted if n not in not_published]
    have = [n for n in available if is_good_zip(OUTDIR / n)]
    still = [n for n in available if n not in have]

    print(f"\n[{stamp()}] {len(have)}/{len(available)} available files "
          f"downloaded into {OUTDIR.resolve()}")

    if not_published:
        print(f"{len(not_published)} month(s) not published by the source "
              f"(normal for very old or very recent months).")

    if still:
        print("Still missing — just run this script again:")
        for n in still:
            print("   " + n)
        return 1

    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())