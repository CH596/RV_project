"""
Phase 2: out-of-sample evaluation of one-day-ahead forecasts of BTC log RV.

Design fixed on 2026-10-26, before any results (ts_project_plan.md, "Phase 2 design"):
  * test origins t = 2024-01-01 .. 2026-09-29 (targets t+1 = 2024-01-02 .. 2026-09-30)
  * refit at every origin on rows whose target is known at t (row s <= t-1)
  * rolling window of the last 700 usable rows = main; expanding = robustness check
  * six models on the same test days; losses MSE on log RV (primary), QLIKE on
    variance (secondary), MAE / MSE on variance (supplementary)
  * back-transform by smearing over the training residuals; Gaussian factor as diagnostic
  * Diebold-Mariano tests with a Newey-West variance for five fixed pairs

Needs only data/rv_daily_tau0.parquet, so it runs in a cloud session.

Run:   python3 phase2_oos.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import har

# ---- Settings ---------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
IN_FILE = DATA / "rv_daily_tau0.parquet"
OUT_FILE = DATA / "phase2_forecasts.parquet"
PLOT_FILE = PROJECT / "phase2_coef_paths.png"

TEST_START = pd.Timestamp("2024-01-01", tz="UTC")   # first origin
TEST_END = pd.Timestamp("2026-09-29", tz="UTC")     # last origin; its target is 2026-09-30
WINDOW = 700                                        # rolling window, in usable rows

HAR = ["rv_d", "rv_w", "rv_m"]
DUMMIES = har.WEEKDAYS[1:]
# Model -> regressors. The two benchmarks are a single column used as the forecast.
# rv_q / rv_y are the 90- / 365-day averages: the names give the horizon, not a mechanism.
MODELS = {
    "1 random walk": "rw",
    "2 seasonal naive": "sn",
    "3 HAR (d, w, m)": HAR,
    "4 HAR (d, w, m) + DoW": HAR + DUMMIES,
    "5 HAR (d, w, m, q) + DoW": HAR + ["rv_q"] + DUMMIES,
    "6 HAR (d, w, m, y) + DoW": HAR + ["rv_y"] + DUMMIES,
}
CARRIED = "4 HAR (d, w, m) + DoW"   # the model carried forward; its rolling coefficients are plotted
# Diebold-Mariano pairs: ladder neighbours, the dummies headline (4 vs 3), and 4 vs 6.
DM_PAIRS = [("1 random walk", "2 seasonal naive"), ("2 seasonal naive", CARRIED),
            (CARRIED, "5 HAR (d, w, m, q) + DoW"), (CARRIED, "3 HAR (d, w, m)"),
            (CARRIED, "6 HAR (d, w, m, y) + DoW")]

# ---- 1. Regressors ----------------------------------------------------------
log_rv = pd.read_parquet(IN_FILE)["log_rv"]
tab = har.har_table(log_rv, (1, 7, 30), extra={"q": 90, "y": 365})
tab["rw"] = log_rv                  # random walk: tomorrow = today
tab["sn"] = log_rv.shift(6)         # seasonal naive: day t-6 = the target's weekday last week

# Common calendar: the rows usable by HAR(1, 7, 30), 2022-01-31 onwards. The benchmarks
# use the same rows, so their smearing factors come from the same training days.
base = tab[HAR + DUMMIES + ["target"]].dropna()
usable = {}
for name, spec in MODELS.items():
    cols = [spec] if isinstance(spec, str) else spec
    usable[name] = tab.loc[base.index, cols + ["target"]].dropna()
origins = base.index[(base.index >= TEST_START) & (base.index <= TEST_END)]
for name, rows in usable.items():
    assert origins.isin(rows.index).all(), name    # every model forecasts every test day
print(f"Test origins: {len(origins)} ({origins[0].date()} to {origins[-1].date()}), "
      f"targets to {(origins[-1] + pd.Timedelta('1D')).date()}")


# ---- 2. Forecast loop -------------------------------------------------------
def forecast(rows, spec, t, scheme, se=False):
    """Fit on rows before t (their targets are known at t) and forecast row t.

    Returns (forecast, training residuals, number of parameters, fit or None).
    """
    pos = rows.index.searchsorted(t)        # rows.iloc[:pos] have index < t
    train = rows.iloc[max(0, pos - WINDOW) if scheme == "rolling" else 0:pos]
    x = rows.iloc[pos]
    if isinstance(spec, str):               # benchmark: nothing to estimate
        return x[spec], train["target"] - train[spec], 0, None
    fit = har.ols(train, spec, se=se)
    coef, resid = fit[0], fit[1]
    return har.predict(coef, x), resid, len(coef), fit


records, paths = [], []
for scheme in ("rolling", "expanding"):
    for t in origins:
        for name, spec in MODELS.items():
            rows = usable[name]
            want_path = scheme == "rolling" and name == CARRIED
            yhat, resid, k, fit = forecast(rows, spec, t, scheme, se=want_path)
            u = resid.to_numpy()
            records.append({
                "scheme": scheme, "origin": t, "model": name,
                "y": rows.at[t, "target"], "yhat": yhat, "n_train": len(u),
                "smear": np.mean(np.exp(2 * u)),                      # smearing factor
                "gauss": np.exp(2 * (u @ u) / (len(u) - k)),          # e^{2 s^2}
            })
            if want_path:
                coef, _, nw_se = fit
                paths.append(pd.concat([coef, nw_se.add_suffix("_se")]).rename(t))
fc = pd.DataFrame(records)
fc.to_parquet(OUT_FILE)

# ---- 3. Losses --------------------------------------------------------------
# log RV is log volatility, so the variance is e^{2 y} (= rvar of the target day) and the
# variance forecast is h = e^{2 yhat} x the back-transform factor.
fc["sigma2"] = np.exp(2 * fc["y"])
fc["h"] = np.exp(2 * fc["yhat"]) * fc["smear"]
fc["h_gauss"] = np.exp(2 * fc["yhat"]) * fc["gauss"]


def qlike(sigma2, h):
    ratio = sigma2 / h
    return ratio - np.log(ratio) - 1


fc["mse_log"] = (fc["y"] - fc["yhat"]) ** 2
fc["qlike"] = qlike(fc["sigma2"], fc["h"])
fc["qlike_gauss"] = qlike(fc["sigma2"], fc["h_gauss"])
fc["mae_var"] = (fc["sigma2"] - fc["h"]).abs()
fc["mse_var"] = (fc["sigma2"] - fc["h"]) ** 2


def dm_test(loss_a, loss_b):
    """Diebold-Mariano on d = loss_a - loss_b (negative: a is better), Newey-West variance
    with Bartlett weights and the same lag rule as the coefficients. Two-sided normal p."""
    d = loss_a.to_numpy() - loss_b.to_numpy()
    n, dbar = len(d), d.mean()
    u = d - dbar
    lags = har.nw_lags(n)
    var = u @ u / n + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
                              for k in range(1, lags + 1))
    stat = dbar / math.sqrt(var / n)
    return dbar, stat, math.erfc(abs(stat) / math.sqrt(2)), lags


pd.set_option("display.width", 140)
for scheme in ("rolling", "expanding"):
    s = fc[fc["scheme"] == scheme]
    title = (f"rolling window, last {WINDOW} usable rows (main)" if scheme == "rolling"
             else "expanding window (robustness)")
    print(f"\n==== {title} ====")
    first = s[s["origin"] == origins[0]].set_index("model")["n_train"]
    print("Training rows at the first origin: " + ", ".join(f"{m[0]}: {n}" for m, n in first.items()))

    by = s.groupby("model", sort=False)
    table = pd.DataFrame({
        "MSE log": by["mse_log"].mean(),
        "QLIKE": by["qlike"].mean(),
        "QLIKE gauss": by["qlike_gauss"].mean(),
        "smear": by["smear"].mean(),
        "gauss": by["gauss"].mean(),
        "n": by.size(),
    })
    table.insert(1, "MSE rel RW", table["MSE log"] / table.loc["1 random walk", "MSE log"])
    table.insert(3, "QLIKE rel RW", table["QLIKE"] / table.loc["1 random walk", "QLIKE"])
    print("\nPrimary (MSE on log RV) and secondary (QLIKE on variance) losses, test-day means.")
    print("QLIKE gauss uses the Gaussian factor e^{2 s^2} instead of smearing (diagnostic);")
    print("smear and gauss are the mean back-transform factors.")
    print(table.round(4).to_string())

    supp = pd.DataFrame({
        "MAE var x1e4": by["mae_var"].mean() * 1e4,
        "MSE var x1e8": by["mse_var"].mean() * 1e8,
    })
    supp["MAE rel RW"] = supp["MAE var x1e4"] / supp.loc["1 random walk", "MAE var x1e4"]
    supp["MSE rel RW"] = supp["MSE var x1e8"] / supp.loc["1 random walk", "MSE var x1e8"]
    print("\nSupplementary: losses on the variance scale (sigma^2 vs h).")
    print(supp.round(4).to_string())

    print("\nDiebold-Mariano, d = loss(A) - loss(B); negative = A better.")
    for loss in ("mse_log", "qlike"):
        wide = s.pivot(index="origin", columns="model", values=loss)
        for a, b in DM_PAIRS:
            dbar, stat, p, lags = dm_test(wide[a], wide[b])
            print(f"  {loss:8s} {a:24s} vs {b:24s}  mean d {dbar:+.4f}  DM {stat:+6.2f}  "
                  f"p {p:.3f}  (NW lags {lags})")

# ---- 3b. Breakdown by target year and target weekday (rolling, models 3 and 4) ----
# Robustness across years and the mechanism behind the dummies' gain: plain HAR's bias by
# weekday. Grouped by the TARGET day (origin + 1), like the dummies themselves.
r = fc[fc["scheme"] == "rolling"].copy()
target_day = r["origin"] + pd.Timedelta("1D")
r["err"] = r["y"] - r["yhat"]
r["target year"] = target_day.dt.year
r["target weekday"] = pd.Categorical(target_day.dt.dayofweek.map(dict(enumerate(har.WEEKDAYS))),
                                     categories=har.WEEKDAYS, ordered=True)
PLAIN = "3 HAR (d, w, m)"
print("\n==== breakdown by target day, rolling window: model 3 (plain HAR) vs model 4 (+ DoW) ====")
print("err = y - yhat (positive: forecast too low).")
for key in ("target year", "target weekday"):
    g3 = r[r["model"] == PLAIN].groupby(key, observed=True)
    g4 = r[r["model"] == CARRIED].groupby(key, observed=True)
    brk = pd.DataFrame({
        "n": g3.size(),
        "MSE log 3": g3["mse_log"].mean(),
        "MSE log 4": g4["mse_log"].mean(),
        "mean err 3": g3["err"].mean(),
        "mean err 4": g4["err"].mean(),
    })
    brk.insert(3, "change", (brk["MSE log 4"] / brk["MSE log 3"] - 1).map("{:+.0%}".format))
    print()
    print(brk.round(3).to_string())

# ---- 4. Rolling coefficient paths of the carried model -----------------------
path = pd.DataFrame(paths)
fig, ax = plt.subplots(3, 3, figsize=(13, 9), sharex=True, constrained_layout=True)
for a, name in zip(ax.flat, HAR + DUMMIES):
    a.plot(path.index, path[name], lw=1)
    a.fill_between(path.index, path[name] - 2 * path[f"{name}_se"],
                   path[name] + 2 * path[f"{name}_se"], alpha=0.25, lw=0)
    a.axhline(0, color="grey", lw=0.6)
    a.set_title(name if name in HAR else f"{name} vs Monday (target day)")
fig.suptitle(f"HAR (d, w, m) + DoW, horizons 1, 7, 30, rolling {WINDOW}-row fits by origin "
             f"(band: ±2 Newey-West SE)")
fig.autofmt_xdate()
fig.savefig(PLOT_FILE, dpi=130)
print(f"\nSaved {OUT_FILE.name} (in data/) and {PLOT_FILE.name}")
