"""
Shared logHAR code: regressors, OLS, Newey-West standard errors.

Used by phase2_oos.py. Needs only the daily series (data/rv_daily_tau0.parquet), not
the 1-minute data, so anything built on it also runs in a cloud session.

Same model as fit_loghar in compute_rv.py (Corsi 2009, in logs). With windows (1, w, m):
    y_{t+1} = b0 + b_d y_t + b_w mean(y_{t-w+1..t}) + b_m mean(y_{t-m+1..t}) + e_{t+1}
Row t holds regressors built from days up to and including t; the target is the next day.
"""

import numpy as np
import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def har_table(log_rv, windows=(1, 7, 30), extra=None):
    """All regressors and the target, one row per day t. Rows are NOT dropped here.

    windows: (1, w, m) -> columns rv_d, rv_w, rv_m.
    extra:   further averaging windows by name, e.g. {"q": 90, "y": 365} -> columns
             rv_q (quarterly) and rv_y (yearly). The names give the horizon only.
    Weekday dummies Tue..Sun are for the TARGET day t+1 (Monday is the baseline): the
    calendar is known in advance, so they use no future data.
    """
    # The target is "the next row", so the rows must be consecutive calendar days.
    assert (log_rv.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
    _, w, m = windows
    tab = pd.DataFrame({
        "rv_d": log_rv,
        "rv_w": log_rv.rolling(w).mean(),
        "rv_m": log_rv.rolling(m).mean(),
    })
    for name, k in (extra or {}).items():
        tab[f"rv_{name}"] = log_rv.rolling(k).mean()
    target_weekday = (log_rv.index + pd.Timedelta("1D")).dayofweek
    for i, name in enumerate(WEEKDAYS[1:], start=1):
        tab[name] = (target_weekday == i).astype(float)
    tab["target"] = log_rv.shift(-1)
    return tab


def nw_lags(n):
    """Newey-West (1994) rule of thumb, floor(4 (n/100)^(2/9)): 7 lags for 1,703 rows."""
    return int(4 * (n / 100) ** (2 / 9))


def nw_cov(X, resid, lags):
    """Newey-West covariance of the OLS coefficients: (X'X)^-1 S (X'X)^-1, where S sums
    the Bartlett-weighted autocovariances of x_t e_t."""
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for k in range(1, lags + 1):
        G = Xe[k:].T @ Xe[:-k]
        S += (1 - k / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def ols(rows, regressors, se=False):
    """OLS of rows['target'] on a constant and the regressors.

    Returns (coef, resid), plus Newey-West standard errors if se=True. coef and the
    standard errors are Series indexed by "const" and the regressor names.
    """
    X = np.column_stack([np.ones(len(rows)), rows[regressors].to_numpy()])
    y = rows["target"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    names = ["const", *regressors]
    coef = pd.Series(beta, index=names)
    resid = pd.Series(resid, index=rows.index)
    if not se:
        return coef, resid
    nw_se = pd.Series(np.sqrt(np.diag(nw_cov(X, resid.to_numpy(), nw_lags(len(rows))))),
                      index=names)
    return coef, resid, nw_se


def predict(coef, x):
    """Forecast from one row (or a frame of rows) of regressors."""
    return coef["const"] + x[coef.index[1:]] @ coef.iloc[1:]
