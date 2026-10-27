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
# ---- 8. logHAR fits ---------------------------------------------------------
# Corsi (2009) HAR in logs. With windows (1, w, m):
#   y_{t+1} = b0 + b_d y_t + b_w mean(y_{t-w+1..t}) + b_m mean(y_{t-m+1..t}) + e_{t+1}
# Row t holds regressors built from days up to and including t; the target is the next
# day. Rows without a full m-day window or a next day are dropped.
# OLS, with Newey-West (Bartlett kernel) standard errors as in Corsi.
def fit_loghar(log_rv, windows, sample=None, day_of_week_dummy=False):
    """Fit logHAR with (1, w, m) windows; return (coefficient table, residuals by date).

    sample: optional index of dates to fit on (must be a subset of the usable rows).
    day_of_week_dummy: add dummies for the target day's weekday (Tue..Sun, Monday is the
        baseline). Needs weekday_mask from section 10.
    """
    _, w, m = windows
    har = pd.DataFrame({
        "rv_d": log_rv,
        "rv_w": log_rv.rolling(w).mean(),
        "rv_m": log_rv.rolling(m).mean(),
        "target": log_rv.shift(-1),
    }).dropna()
    if sample is not None:
        har = har.loc[sample]
    regressors = ["rv_d", "rv_w", "rv_m"]
    if day_of_week_dummy:
        # Use weekday dummies for the target day. Shift the full mask before picking rows,
        # so each row gets the next calendar day's weekday and no row is lost. Drop Monday
        # as the baseline: all seven dummies would sum to the constant.
        target_weekday = weekday_mask.shift(-1).loc[har.index, WEEKDAYS[1:]].astype(float)
        har = pd.concat([har, target_weekday], axis=1)
        regressors += WEEKDAYS[1:]
    X = np.column_stack([np.ones(len(har)), har[regressors].to_numpy()])
    y = har["target"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # Newey-West covariance: (X'X)^-1 S (X'X)^-1, S = sum of Bartlett-weighted
    # autocovariances of x_t e_t. Lag length from the Newey-West (1994) rule of thumb,
    # so fits on the same number of days use the same lag length.
    nw_lags = int(4 * (len(har) / 100) ** (2 / 9))
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for k in range(1, nw_lags + 1):
        G = Xe[k:].T @ Xe[:-k]
        S += (1 - k / (nw_lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    # Plain OLS standard errors for comparison: sigma^2 (X'X)^-1, which assumes the errors
    # are homoskedastic and serially uncorrelated.
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    ols_se = np.sqrt(np.diag(sigma2 * XtX_inv))

    fit = pd.DataFrame({"coef": beta, "ols_se": ols_se, "nw_se": se, "t": beta / se},
                       index=["const", "b_d", "b_w", "b_m", *regressors[3:]])
    print(f"\nlogHAR{tuple(windows)}{' + weekday dummies' if day_of_week_dummy else ''} "
          f"on {log_rv.name}: {len(har)} obs "
          f"({har.index[0].date()} to {har.index[-1].date()}), Newey-West lags {nw_lags}")
    print(fit.round(4).to_string())
    print(f"  R^2 {1 - resid.var() / y.var():.3f}, residual std {resid.std():.3f}")
    return fit, pd.Series(resid, index=har.index, name="resid")


# Calendar-day horizons (1, 7, 30), since crypto trades every day.
fit_730, resid_730 = fit_loghar(daily["log_rv_ds"], (1, 7, 30))

# ---- 9. logHAR(1, 5, 22): trading-day horizons, for comparison ---------------
# Same 1703 days as (1,7,30) (2022-01-31 to 2026-09-29), hence the same Newey-West lag
# length. The 22-day window fills 8 days sooner than the 30-day one, so without
# sample= it would keep 1711 days.
fit_522, resid_522 = fit_loghar(daily["log_rv_ds"], (1, 5, 22), sample=resid_730.index)
assert resid_522.index.equals(resid_730.index)   # same days, same order

# ---- 10. Weekday masks, for the weekday-dummies variant ----------------------
# Instead of subtracting weekday_effect beforehand, the dummies variant fits logHAR on
# log_rv and lets the regression estimate the weekly cycle. One boolean column per
# weekday (of the window start, as in section 5); exactly one is True on each row.
# In the regression, use the masks of the TARGET day (shift(-1)) and drop one weekday
# as the baseline, since the seven columns sum to the constant.
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_mask = pd.DataFrame({name: daily.index.dayofweek == i for i, name in enumerate(WEEKDAYS)},
                            index=daily.index)
assert (weekday_mask.sum(axis=1) == 1).all()
print("\nDays per weekday: " + ", ".join(f"{d} {n}" for d, n in weekday_mask.sum().items()))

# logHAR(1, 7, 30) on log_rv as measured, with the weekly cycle estimated by the dummies,
# on the same 1703 days as the deseasonalized fit.
fit_dum, resid_dum = fit_loghar(daily["log_rv"], (1, 7, 30), sample=resid_730.index,
                                day_of_week_dummy=True)
fit_raw_730, resid_raw_73 = fit_loghar(daily["log_rv"], (1, 7, 30), sample=resid_730.index)
assert resid_dum.index.equals(resid_730.index)

# ---- 11. Weekday-dummies logHAR(1, 7, 30), one fit per calendar year -----------
# Same days as above, split by the year of row t. The 30-day window still reaches back
# into the previous year, so only the rows are split, not the history behind them.
# 2022 starts on 2022-01-31 and 2026 ends on 2026-09-29, so those years are shorter.
by_year = {}
for year in range(2022, 2027):
    in_year = resid_730.index[resid_730.index.year == year]
    fit, _ = fit_loghar(daily["log_rv"], (1, 7, 30), sample=in_year, day_of_week_dummy=True)
    row = {}
    for name in ["const", "b_d", "b_w", "b_m", *WEEKDAYS[1:]]:
        row[name] = fit.loc[name, "coef"]
        row[f"{name}_se"] = fit.loc[name, "nw_se"]
    row["n"] = len(in_year)
    by_year[year] = row
by_year = pd.DataFrame.from_dict(by_year, orient="index").rename_axis("year")
by_year["n"] = by_year["n"].astype(int)
# One table, printed in two blocks so it fits the terminal: HAR terms, then the
# weekday shifts (target day vs Monday).
har_cols = [c for name in ["const", "b_d", "b_w", "b_m"] for c in (name, f"{name}_se")]
day_cols = [c for name in WEEKDAYS[1:] for c in (name, f"{name}_se")]
print("\nlogHAR(1, 7, 30) + weekday dummies on log_rv, by year (Newey-West standard errors):")
print(by_year[har_cols + ["n"]].round(4).to_string())
print(by_year[day_cols].round(3).to_string())