# BTC Realized Volatility

Daily realized volatility (RV) of BTCUSDT built from 1-minute Binance prices, as groundwork for forecasting volatility with HAR-type models (Corsi, 2009). The sample runs from 2022-01-02 to 2026-09-30: 1,733 complete days.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Download monthly 1-minute klines from Binance's public data site | `download_klines.py` | `data/raw/BTCUSDT-1m-YYYY-MM.zip` |
| 2. Combine into one file and validate | `build_parquet.py` | `data/btc_1m.parquet` |
| 3. 5-minute returns → daily RV, weekday adjustment, diagnostics | `compute_rv.py` | `data/rv_daily_tau0.parquet`, `rv_daily_tau0.png` |
| 4. Periodogram analysis of the weekly cycle | `periodogram_findings.py` | figures for `periodogram_findings.tex` |

No data is in the repository. Every script reads and writes under `data/` next to the scripts, which the steps above rebuild (about 170 MB in total).

### Reproduce

```bash
pip install requests pandas pyarrow numpy scipy matplotlib
python download_klines.py      # resumable; re-run if it stops early
python build_parquet.py
python compute_rv.py           # day starts 00:00 UTC
python compute_rv.py 12        # day starts 12:00 UTC (boundary-shift check)
python periodogram_findings.py
```

Scripts can be run from any directory: paths are relative to the script, not the current directory.

## Equity data from Interactive Brokers

`download_ibkr.py` downloads 5-minute stock bars (TRADES; open, close, volume; regular trading hours, 78 bars a day) through the TWS API and writes `data/<symbol>_5m.parquet`, indexed by bar start time in UTC like `btc_1m.parquet`.

One-time setup:
1. Install **IB Gateway** (stable) from interactivebrokers.com. IBKR Desktop does not serve the API.
2. Log in, then under Configure → Settings → API → Settings: tick *Enable ActiveX and Socket Clients*, tick *Read-Only API*, and note the socket port (Gateway: 4001 live, 4002 paper; TWS: 7496 / 7497).
3. The account needs a US equity market data subscription; without one IB rejects historical requests (error 162/354).

```bash
pip install ib_async
python download_ibkr.py --check                 # connects and fetches one day of SPY
python download_ibkr.py SPY --start 2022-01-03  # resumable; re-run if days fail
```

Per-day files go to `data/raw_ibkr/<SYMBOL>_5m/`; holidays are saved as empty files so they are not requested again.

## Design choices

### Data
- **Source:** Binance spot BTCUSDT 1-minute klines from `data.binance.vision`. These are static monthly files, so no API key is needed.
- **Columns kept:** timestamp, close and volume only.
- **Timestamp units:** Binance switched from milliseconds to microseconds in 2025. Both are converted to UTC.
- **Price timing:** Binance stamps each bar with its *open* time, but `close` is the price at the *end* of the minute. Each close is re-stamped to open time + 1 minute, so the price at 12:05 is the price observed at 12:05.

### Missing minutes
- **Rule: forward-fill.** All missing minutes take the last traded price. Returns inside a gap are zero, and the first return after trading resumes carries the whole jump. The only gap in the sample is the 2023-03-24 Binance trading halt: 80 minutes starting at 12:40 UTC. `build_parquet.py` leaves gaps as gaps, so this rule is applied in one place only.

### Returns
- **Notation follows Corsi (2009):** $p(t)$ is the log price and $r_t = p(t) - p(t-\Delta)$.
- **Sampling interval:** $\Delta$ = 5 minutes, sampled on the clock (xx:00, xx:05, …). This gives 288 returns per day. A return stamped $t$ covers $(t-\Delta, t]$.

### Daily realized volatility
- **Definition:** $RV_t^{(d)} = \sqrt{\sum_{j=0}^{287} r_{t-j\Delta}^2}$, the square root of realized variance.
- **Modelling target:** $\log RV_t^{(d)}$.
- **Day boundary:** a day starts at 00:00 UTC by default. `compute_rv.py <tau>` shifts the start to tau hours UTC, to check that results don't depend on where the day is cut. A day labelled D covers $(D+\tau, D+1\text{ day}+\tau]$, and a return ending exactly on the boundary belongs to the earlier day.
- **Partial days:** days with fewer than 288 returns are dropped. That only happens at the ends of the sample.

### Weekly seasonality
- **Weekday adjustment:** log RV has a strong day-of-week pattern. Monday to Friday are 0.13 to 0.17 above the mean, Saturday is 0.47 below and Sunday 0.24 below. `compute_rv.py` saves `log_rv_ds`, which is log RV minus each weekday's mean deviation. It keeps the overall level of log RV but has no weekday pattern, and the adjustment removes 22% of the variance of log RV.
- **Look-ahead caveat:** the weekday means use the whole sample. That's fine for description, but for out-of-sample forecasting, estimate them on the training window only, or include day-of-week dummies in the regression instead.
- **Calendar horizons:** crypto trades every day, so the natural weekly and monthly horizons are 7 and 30 calendar days. The diagnostics report these alongside Corsi's equity-market horizons of 5, 22 and 66 trading days.

### Diagnostics
- **Autocorrelation:** lags 1 to 70 for log RV and for the weekday-adjusted series, plotted together.
- **Welch spectrum:** computed for both series and shown on log-log axes. It uses 364-day Hann-windowed segments (52 weeks, so the weekly frequencies fall exactly on the grid) with 50% overlap.

## Findings so far

- The only periodic component at periods of 2 to 90 days is the **weekly cycle**. It appears at 7, 3.5 and 2.33 days; the last two are harmonics of the week, produced by the sharp weekend dip. Removing weekday means removes all three.
- After removing the weekday effect, autocorrelation decays smoothly and slowly (0.72 at lag 1, 0.26 at lag 30, 0.14 at lag 66). The spectrum is close to a power law, consistent with long memory: a rough fit gives d ≈ 0.45.
- There is no evidence of other cycles at periods of 2 to 90 days. A 35-day peak in the raw periodogram is borderline and isn't separate from the trend in the Welch spectrum.
- **Not tested:** annual seasonality (under 5 years of data), effects tied to irregular dates (FOMC and CPI releases, holidays, options expiries), and seasonality within the day.

`periodogram_findings.tex` / `.pdf` has the details.

## Reference

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
