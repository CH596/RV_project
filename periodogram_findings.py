"""
Figures and numbers for periodogram_findings.tex.

Run:   python3 periodogram_findings.py          (after compute_rv.py, reads rv_daily_tau0.parquet)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
IN_FILE = PROJECT / "data" / "rv_daily_tau0.parquet"
MAX_PERIOD = 90
LINE = "#2a6fdb"
INK = "#444444"

x = pd.read_parquet(IN_FILE)["log_rv"]
x = x - x.mean()
n = len(x)
print(f"Sample: {x.index[0].date()} to {x.index[-1].date()}, {n} days")


def periodogram(series):
    """I(f) = |FFT|^2 / n at the Fourier frequencies f = k/n, returned against period 1/f."""
    freq = np.fft.rfftfreq(len(series), d=1.0)[1:]
    power = np.abs(np.fft.rfft(series.to_numpy()))[1:] ** 2 / len(series)
    return 1 / freq, power


def power_at(period, power, target):
    return power[np.argmin(np.abs(period - target))]


# ---- 1. Day-of-week profile and its Fourier decomposition --------------------
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
profile = x.groupby(x.index.dayofweek).mean()
print("\nMean demeaned log RV by weekday:")
print("  " + ", ".join(f"{d} {v:+.3f}" for d, v in zip(days, profile)))

# Relative power at k = 1, 2, 3 cycles per week (periods 7, 3.5, 2.33 days)
emp = np.abs(np.fft.fft(profile.to_numpy()))[1:4] ** 2
ideal = np.abs(np.fft.fft([0, 0, 0, 0, 0, 1, 1]))[1:4] ** 2    # "low on Sat and Sun" block
print("Relative power at periods 7, 3.5, 2.33 days:")
print("  empirical weekday profile: " + ", ".join(f"{v:.2f}" for v in emp / emp[0]))
print("  idealised weekend block:   " + ", ".join(f"{v:.2f}" for v in ideal / ideal[0]))

# ---- 2. Periodogram before and after removing weekday means ------------------
adj = x - x.groupby(x.index.dayofweek).transform("mean")
print(f"\nShare of log RV variance explained by weekday means: {1 - adj.var() / x.var():.1%}")

period, p_raw = periodogram(x)
_, p_adj = periodogram(adj)
keep = (period >= 2) & (period <= MAX_PERIOD)

print("Power at the weekly frequencies (raw -> weekday-adjusted):")
for t in (7, 3.5, 7 / 3):
    print(f"  {t:4.2f} days: {power_at(period, p_raw, t):6.2f} -> {power_at(period, p_adj, t):.3f}")

# Long-period peaks: compare each ordinate with the mean of its neighbours (excluding
# the immediate ones). Under the null, I(f) / background ~ Exponential(1).
idx = np.flatnonzero(keep & (period >= 20))
print("Largest peaks with period >= 20 days (weekday-adjusted):")
for i in idx[np.argsort(p_adj[idx])[::-1][:6]]:
    nb = np.r_[p_adj[max(i - 6, 0):i - 1], p_adj[i + 2:i + 7]]
    ratio = p_adj[i] / nb.mean()
    print(f"  {period[i]:5.1f} days: power {p_adj[i]:.2f}, {ratio:.1f}x local background, "
          f"P(>ratio) = {np.exp(-ratio):.3f}")
print(f"Fourier frequencies with period 20-{MAX_PERIOD} days: {len(idx)}")

for name, s in (("log RV", x), ("weekday-adjusted log RV", adj)):
    acf = {k: s.autocorr(k) for k in (1, 5, 7, 14, 22, 30, 66)}
    print(f"Autocorrelation of {name}: " + ", ".join(f"lag {k}: {v:.2f}" for k, v in acf.items()))

# ---- 3. Figures --------------------------------------------------------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "font.size": 10})

fig, ax = plt.subplots(figsize=(6.5, 2.8), constrained_layout=True)
ax.bar(days, profile, color=LINE, width=0.6)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("mean log RV (demeaned)")
fig.savefig(PROJECT / "periodogram_weekday.png", dpi=200)

fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True, sharey=True,
                       constrained_layout=True)
for a, p, title in ((ax[0], p_raw, "log RV"),
                    (ax[1], p_adj, "log RV minus weekday means")):
    a.plot(period[keep], p[keep], color=LINE, lw=0.8)
    a.set_yscale("log")
    a.set_title(title, loc="left", fontsize=10, color=INK)
    a.set_ylabel("power")
    a.grid(axis="y", color="#dddddd", lw=0.5)
for t, label in ((7, "7"), (3.5, "3.5"), (7 / 3, "2.33")):
    ax[0].annotate(label, (t, power_at(period, p_raw, t)), xytext=(4, 0),
                   textcoords="offset points", fontsize=8, color=INK, va="center")
ax[1].set_xlabel("period (days)")
ax[1].set_xlim(0, MAX_PERIOD)
fig.savefig(PROJECT / "periodogram_before_after.png", dpi=200)
print("\nSaved periodogram_weekday.png, periodogram_before_after.png")
