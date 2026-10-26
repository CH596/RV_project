"""
Step 2: 1-minute prices -> 5-minute log returns -> daily realized volatility.

Run:   python3 compute_rv.py            (day starts 00:00 UTC)
       python3 compute_rv.py 12         (day starts 12:00 UTC, for the boundary-shift test)
Needs: pip install pandas pyarrow matplotlib scipy
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

# ---- Settings ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
IN_FILE = DATA / "btc_1m.parquet"

TAU_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # day-start offset (UTC hours)
BAR = "5min"                                                    # sampling interval for returns
BARS_PER_DAY = int(pd.Timedelta("1D") / pd.Timedelta(BAR))      # 288 for 5-min

tag = f"tau{TAU_HOURS:g}"
OUT_FILE = DATA / f"rv_daily_{tag}.parquet"
PLOT_FILE = PROJECT / f"rv_daily_{tag}.png"

# ---- 1. Load the Parquet file -----------------------------------------------
# read_parquet gives back exactly the DataFrame that was saved: UTC timestamp index,
# 'close' and 'volume' columns, types intact. No re-parsing needed.
data = pd.read_parquet(IN_FILE)
price = data["close"]

# Binance timestamps are the bar's OPEN time; 'close' is the price at the END of that
# minute. Re-label each close by the moment it was observed (open time + 1 min), so
# "price at 12:05" really means the price at 12:05.
price.index = price.index + pd.Timedelta("1min")

# ---- 2. Missing-minute rule: forward-fill -----------------------------------
# Full minute grid; any missing minute (the 2023-03-24 halt) takes the last traded price.
# Inside the gap returns are zero; the first return after reopening carries the jump.
grid = pd.date_range(price.index[0], price.index[-1], freq="1min", tz="UTC")
n_filled = len(grid.difference(price.index))
price = price.reindex(grid).ffill()
print(f"Minutes forward-filled: {n_filled}")

# ---- 3. 5-minute log returns ------------------------------------------------
# Notation as in Corsi (2009): p(t) is the log price, r_t = p(t) - p(t - Δ), Δ = 5 min.
# Sample on the 5-minute clock (xx:00, xx:05, ...); the return stamped t covers (t - Δ, t].
on_clock = price.index.floor(BAR) == price.index
p = np.log(price[on_clock])
r = p.diff().dropna()

# ---- 4. Daily realized volatility with day-start offset tau -----------------
# A "day" labelled D covers (D + tau, D + 1 day + tau]. Shifting by tau and by 1 ns
# before flooring puts a return ending exactly at the boundary in the earlier day.
shift = pd.Timedelta(hours=TAU_HOURS) + pd.Timedelta("1ns")
day = (r.index - shift).floor("D")

g = (r ** 2).groupby(day)
daily = pd.DataFrame({
    "rvar": g.sum(),            # realized variance  = sum_j r_{t - jΔ}^2
    "n_returns": g.size(),
})
daily["rv"] = np.sqrt(daily["rvar"])          # realized volatility (Corsi's RV^(d))
daily["log_rv"] = np.log(daily["rv"])         # modelling target candidate
# Label each day by the moment its window starts (e.g. 2023-03-24 12:00 when tau = 12)
daily.index = daily.index + pd.Timedelta(hours=TAU_HOURS)
daily.index.name = "window_start_utc"

# Drop partial days at the edges of the sample (fewer than 288 returns).
complete = daily["n_returns"] == BARS_PER_DAY
print(f"Days: {complete.sum()} complete, {(~complete).sum()} partial dropped "
      f"({[str(t) for t in daily.index[~complete]]})")
daily = daily[complete]

# ---- 5. Remove the weekly cycle ---------------------------------------------
# log RV has a weekend dip (periodogram peaks at 7, 3.5, 2.33 days; see
# periodogram_findings.tex). Subtract each weekday's mean deviation from the overall
# mean, so log_rv_ds keeps the level of log_rv but has no day-of-week pattern.
# Weekday = day of the window start. The means use the whole sample: fine for
# description, but for out-of-sample forecasting estimate them on the training window.
weekday = daily.index.dayofweek
daily["weekday_effect"] = (daily.groupby(weekday)["log_rv"].transform("mean")
                           - daily["log_rv"].mean())
daily["log_rv_ds"] = daily["log_rv"] - daily["weekday_effect"]
effects = daily.groupby(weekday)["weekday_effect"].first()
print("\nWeekday effect on log RV: " + ", ".join(
    f"{d} {v:+.3f}" for d, v in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], effects)))
print(f"  share of log RV variance removed: "
      f"{1 - daily['log_rv_ds'].var() / daily['log_rv'].var():.1%}")

# ---- 6. Sanity checks -------------------------------------------------------
ann = np.sqrt(365)   # crypto trades every day
print(f"\nDaily RV (volatility), whole sample:")
print(f"  median {daily['rv'].median():.4f}   "
      f"=> annualized ~{daily['rv'].median() * ann:.0%}")
print(f"  min {daily['rv'].min():.4f} on {daily['rv'].idxmin().date()}   "
      f"max {daily['rv'].max():.4f} on {daily['rv'].idxmax().date()}")
print(f"  log_rv: mean {daily['log_rv'].mean():.3f}, std {daily['log_rv'].std():.3f}")
halt = pd.Timestamp("2023-03-24 13:00", tz="UTC")
halt_day = daily.index[daily.index <= halt]
if len(halt_day):
    print(f"\nDay containing the 2023-03-24 halt (starts {halt_day[-1]}): "
          f"RV {daily.loc[halt_day[-1], 'rv']:.4f}")

# Autocorrelation of log RV at the HAR horizons (5, 22, 66 trading days) and the
# calendar week/month (7, 30 days, since crypto trades every day): should be high and decay slowly
for col in ("log_rv", "log_rv_ds"):
    acf = {k: daily[col].autocorr(k) for k in (1, 5, 7, 22, 30, 66)}
    print(f"Autocorrelation of {col}: " + ", ".join(f"lag {k}: {v:.2f}" for k, v in acf.items()))

daily.to_parquet(OUT_FILE)
print(f"\nSaved {OUT_FILE}")

# ---- 7. Plot ----------------------------------------------------------------
# Panels 1-2 show log RV as measured; the autocorrelation compares log_rv with the
# deseasonalized log_rv_ds, and the two Welch spectra show each separately.
fig, ax = plt.subplots(5, 1, figsize=(11, 15), constrained_layout=True)
ax[0].plot(daily.index, daily["log_rv"], lw=0.6)
ax[0].set_title(f"BTCUSDT daily log realized volatility (5-min returns, day starts {TAU_HOURS:g}:00 UTC)")
ax[0].set_ylabel("log RV")
ax[1].hist(daily["log_rv"], bins=60)
ax[1].set_xlabel("log RV")
ax[1].set_ylabel("days")
lags = np.arange(1, 71)
ax[2].bar(lags, [daily["log_rv"].autocorr(k) for k in lags], width=0.6,
          color="#c4c4c4", label="log RV")
ax[2].plot(lags, [daily["log_rv_ds"].autocorr(k) for k in lags], marker="o", ms=3, lw=1.2,
           label="weekday effect removed")
ax[2].axhline(1.96 / np.sqrt(len(daily)), color="grey", ls="--", lw=0.8)   # 95% white-noise band
ax[2].legend(frameon=False)
ax[2].set_title("Autocorrelation of log RV, before and after removing the weekday effect")
ax[2].set_xlabel("lag (days)")
ax[2].set_ylabel("autocorrelation")

# Welch spectrum: split the series into 364-day segments (52 weeks, so the weekly
# frequencies fall exactly on the grid) overlapping by half, apply a Hann window to each,
# and average their periodograms. Averaging ~8 segments cuts the noise of the raw
# periodogram at the cost of resolution (frequencies 1/364 apart). Log-log axes, so the
# long-memory power law S(f) ~ f^(-2d) shows as a straight line. Panel 4 is log RV as
# measured, panel 5 has the weekday effect removed.
SEG = 364
for a, col, title in ((ax[3], "log_rv", "Welch spectrum of log RV"),
                      (ax[4], "log_rv_ds", "Welch spectrum of log RV, weekday effect removed")):
    freq, power = welch(daily[col].to_numpy(), fs=1.0, window="hann",
                        nperseg=SEG, noverlap=SEG // 2, detrend="constant")
    freq, power = freq[1:], power[1:]
    period = 1 / freq
    keep = (period >= 2) & (period <= 91)
    for t in (7, 3.5, 7 / 3):
        a.axvline(t, color="grey", ls=":", lw=0.8)   # weekly harmonics
    a.loglog(period[keep], power[keep], lw=1.0)
    a.set_title(title)
    a.set_xlabel("period (days)")
    a.set_ylabel("spectral density")
    ticks = [2, 7 / 3, 3.5, 7, 14, 30, 60, 90]
    a.set_xticks(ticks, ["2", "2.33", "3.5", "7", "14", "30", "60", "90"])
    a.minorticks_off()
    peaks, _ = find_peaks(power[keep])
    top = peaks[np.argsort(power[keep][peaks])[::-1][:6]]
    print(f"Welch spectrum peaks of {col} (period in days): " +
          ", ".join(f"{period[keep][i]:.1f}" for i in top))
ax[4].sharey(ax[3])
fig.savefig(PLOT_FILE, dpi=130)
print(f"Saved {PLOT_FILE}")
