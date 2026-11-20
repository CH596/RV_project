# Forecasting Bitcoin Realized Volatility: HAR with a Weekly Season

Daily realized volatility (RV) of BTCUSDT (Binance, 5-min returns, 2022-2026, 1733 days) forecasting(one day ahead) using the log-HAR model (Corsi 2009). Preliminary analysis (Phase 1) shows a strong dip in RV during the weekends. In phase 2 of the project, We investigate whether including the weekly season improves the forecast, out of sample with significance tests.

**Result.** 
We include a weekday dummies in the log-HAR to model the weekend dip in RV.
The log-HAR+weekly dummies model improves the out of sample forecasting significantly compared to the same model without dummies. Specifically, the Mean Square Error (MSE) is 0.075 (with dummies) vs 0.135 (without dummies), which is a 45% reduction in MSE with a Diebold-Mariano (DM) value of -12.5. 



| One-day-ahead forecasts of log RV | MSE (log RV) | QLIKE (variance) |
|---|---|---|
| Random walk (tomorrow = today) | 0.187 | 0.371 |
| HAR (1, 7, 30 days) | 0.135 | 0.274 |
| **HAR + target-day weekday dummies** | **0.075** | **0.192** |

1,003 out-of-sample forecasts (targets 2024-01-02 to 2026-09-30), refit every day on a rolling 700-day window; test period, models and losses fixed before any results were seen.

**Robustness.**

- **Every test year:** The reduction in MSE achieved by the log-HAR+ weekly dummies is 43%, 44%, 49% for yr 2024, yr 2025 and yr 2026 respectively, which is consistent with the overall reduction rate of 45%.
- **Expanding window:** We note that using an expanding window instead of a rolling window (last 700 days) gives the same result (MSE: 0.0750 vs 0.0746).
- **QLIKE:** We consider QLIKE as a distinct loss measure. Both QLIKE and MSE shows that the log-HAR + weekly dummies model is the best overall model (30% reduction in QLIKE with a DM of -7.6). We show both the QLIKE and MSE values for different models in the "out of sample result" section below.
- **Slower terms add nothing:** We also consider the inclusion of a 90- and 365-day terms in the log-HAR+ as an additional model for comparison purpose. Both models show no significant improvement over the (1,7,30) log-HAR +weekly dummies model.

**Status.** Phase 1 (data, seasonality analysis, in-sample HAR fits) and Phase 2 (out-of-sample evaluation) are complete.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Download monthly 1-minute klines from Binance's public data site | `download_klines.py` | `data/raw/BTCUSDT-1m-YYYY-MM.zip` |
| 2. Combine into one file and validate | `build_parquet.py` | `data/btc_1m.parquet` |
| 3. 5-minute returns → daily RV, weekday adjustment, diagnostics, in-sample HAR fits | `compute_rv.py` | `data/rv_daily_tau0.parquet`, `rv_daily_tau0.png`, fit tables printed to the terminal |
| 4. Periodogram analysis of the weekly cycle | `periodogram_findings.py` | figures for `periodogram_findings.tex` |
| 5. Out-of-sample evaluation: daily refits, losses, Diebold–Mariano tests, rolling coefficients (uses the shared HAR code in `har.py`) | `phase2_oos.py` | `phase2_findings.txt` (tables), `phase2_coef_paths.png`, `data/phase2_forecasts.parquet` (every model × test day; not committed) |

Every script reads and writes under `data/` next to the scripts. The only data file in the repository is `data/rv_daily_tau0.parquet` (85 KB, the daily RV series), so the model fits in step 3 and all of step 5 run without the 1-minute data; steps 1–2 rebuild everything else (about 170 MB).

### Reproduce

```bash
pip install -r requirements.txt
python download_klines.py      # resumable; re-run if it stops early
python build_parquet.py
python compute_rv.py           # day starts 00:00 UTC
python compute_rv.py 12        # day starts 12:00 UTC (boundary-shift check)
python periodogram_findings.py
python phase2_oos.py          # about a minute; needs only data/rv_daily_tau0.parquet
```

Scripts can be run from any directory: paths are relative to the script, not the current directory.

## Out-of-sample results (Phase 2)

At every origin $t$ from 2024-01-01 to 2026-09-29, each model is refitted on the 700 most recent days whose target is known at $t$ and forecasts log RV for day $t+1$: 1,003 forecasts, the same test days for every model. The design is in [Out-of-sample evaluation](#out-of-sample-evaluation-fixed-before-any-results) below; all tables are in `phase2_findings.txt`.

**Losses** (test-day means; "vs RW" is the ratio to the random walk):

| Model | MSE (log RV) | vs RW | QLIKE | vs RW |
|---|---|---|---|---|
| 1. Random walk | 0.1872 | 1.00 | 0.3709 | 1.00 |
| 2. Seasonal naive (same weekday last week) | 0.1832 | 0.98 | 0.4591 | 1.24 |
| 3. HAR (1, 7, 30) | 0.1354 | 0.72 | 0.2741 | 0.74 |
| **4. HAR (1, 7, 30) + weekday dummies** | **0.0746** | **0.40** | **0.1918** | **0.52** |
| 5. Model 4 + 90-day average | 0.0747 | 0.40 | 0.1920 | 0.52 |
| 6. Model 4 + 365-day average | 0.0753 | 0.40 | 0.1932 | 0.52 |

**Diebold–Mariano tests** on the daily loss differences $d_t = L_{A,t} - L_{B,t}$ (negative: A better), Newey–West variance with 6 lags:

| A vs B | MSE: DM ($p$) | QLIKE: DM ($p$) | Question |
|---|---|---|---|
| 4 vs 3 | −12.50 (< 0.001) | −7.64 (< 0.001) | Do the weekday dummies help out of sample? |
| 2 vs 4 | +8.94 (< 0.001) | +6.73 (< 0.001) | Does the model beat the naive benchmarks? |
| 4 vs 5 | −1.14 (0.26) | −0.46 (0.65) | Does a 90-day term add anything? |
| 4 vs 6 | −1.49 (0.14) | −1.05 (0.30) | Does a 365-day term add anything? |
| 1 vs 2 | +0.25 (0.80) | −1.92 (0.055) | Random walk vs seasonal naive |

**Why the dummies help: plain HAR is biased on particular weekdays.** Mean forecast error $y - \hat y$ by target weekday:

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| HAR (1, 7, 30) | +0.26 | +0.09 | +0.09 | +0.07 | +0.10 | **−0.57** | −0.06 |
| + weekday dummies | 0.00 | −0.01 | −0.02 | +0.01 | +0.02 | −0.03 | +0.02 |

Plain HAR over-forecasts Saturday volatility by a factor of about $e^{0.57}\approx1.8$ and under-forecasts Monday. The gain is concentrated there: Saturday's MSE falls from 0.44 to 0.12 and Monday's from 0.14 to 0.07, while the other days change by 4–14%. (Per-weekday and per-year tables: section 4 of `phase2_findings.txt`.)

**Checks and diagnostics.**
- **Why the out-of-sample MSE (0.075) is below the in-sample residual variance (0.086):** 2024–26 is calmer. The variance of log RV is 0.212 on the test days against 0.289 before them; relative to that, model 4's out-of-sample $R^2$ is about 0.65, the same as in sample (0.648). Replacing all data after an origin with random values leaves that origin's forecasts unchanged (no look-ahead).
- **Back-transform:** the smearing factor (mean 1.30) is larger than the Gaussian $e^{2s^2}$ (1.19), as fat-tailed residuals imply; QLIKE is slightly better with the Gaussian factor (0.188 vs 0.192). Smearing was pre-registered and is kept.
- **Rolling coefficients** (`phase2_coef_paths.png`): $\beta_d$ 0.39–0.56, $\beta_w$ 0.22–0.38, $\beta_m$ small with a band covering zero. The Saturday shift deepens from −0.72 to −0.91 across the test period, more than the HAR coefficients move, which is why the rolling window is the main design.

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
- **Look-ahead caveat:** the weekday means use the whole sample. That's fine for description, but not for forecasting: Phase 2 uses day-of-week dummies in the regression instead, so the weekday effects are estimated on each training window.
- **Calendar horizons:** crypto trades every day, so the natural weekly and monthly horizons are 7 and 30 calendar days. The diagnostics report these alongside Corsi's equity-market horizons of 5, 22 and 66 trading days.

### HAR model
- **Specification:** log-HAR, $y_{t+1} = c + \beta_d y_t + \beta_w \bar y^{(w)}_t + \beta_m \bar y^{(m)}_t + \varepsilon_{t+1}$ with $y = \log RV$ and $\bar y^{(h)}_t$ the average of $y$ over the $h$ days up to and including $t$.
- **Horizons:** (1, 7, 30) calendar days as the crypto model; (1, 5, 22), Corsi's equity horizons, as the literature baseline.
- **Weekly cycle:** dummies for the weekday of the **target** day $t+1$, with Monday as the baseline. The target day's weekday is known at the forecast origin, so this uses no future information, and in a forecasting setting the weekday effects are estimated on the training window automatically. In-sample this is equivalent to fitting HAR on the weekday-adjusted series.
- **Common sample:** every variant is fitted on the same 1,703 days (2022-01-31 to 2026-09-29), so $R^2$ values are comparable when the target is the same.
- **Standard errors:** Newey–West (Bartlett kernel), 7 lags by the rule $\lfloor 4(n/100)^{2/9}\rfloor$, because the residuals are autocorrelated.

### Out-of-sample evaluation (fixed before any results)
- **Test period:** forecast origins 2024-01-01 to 2026-09-29 (targets to 2026-09-30), 1,003 one-day-ahead forecasts. Fixed in advance; moving it after seeing results would be data snooping.
- **Refitting:** at every origin, on rows whose target is known at the origin.
- **Window:** rolling, the last 700 usable days (about 23 months), as the main design; expanding (all past days) as the robustness check. Rolling because yearly fits showed the Saturday shift drifting (−0.62 in 2022 to −0.91 in 2025) while the HAR coefficients were stable within noise. The 90- and 365-day models start with 640 and 365 training days at the first origin, a small handicap.
- **Models:** two naive benchmarks (random walk; seasonal naive, the same weekday last week), plain HAR, HAR + weekday dummies, and that model plus a 90-day or a 365-day average. The 90-day term is the next rung of the 1 → 7 → 30 ladder (the weekday-adjusted autocorrelation is still about 0.14 at 50–70 days); the 365-day term acts as a slowly moving intercept, a competitor to the rolling window.
- **Losses:** MSE on log RV (primary: the scale the model is fitted on). QLIKE on the variance scale, $\frac{\sigma^2}{h} - \log\frac{\sigma^2}{h} - 1$ averaged over test days (secondary): it is robust to noise in the RV proxy (Patton, 2011), scale-free, penalises under-forecasts more, and also tests the back-transform. MAE and MSE on variance are reported as a supplement.
- **Back-transform to variance:** $h = e^{2\hat y}$ times the smearing factor (Duan, 1983), the mean of $e^{2\hat u}$ over the training residuals; $e^{2\hat y}$ alone is biased low. The Gaussian factor $e^{2s^2}$ is reported as a diagnostic.
- **Significance:** Diebold–Mariano tests on daily loss differences, with a Newey–West variance (6 lags by the same rule, $n = 1003$) because both models miss in the same turbulent weeks. Only the five pairs that answer a question are tested.

### Diagnostics
- **Autocorrelation:** lags 1 to 70 for log RV and for the weekday-adjusted series, plotted together.
- **Welch spectrum:** computed for both series and shown on log-log axes. It uses 364-day Hann-windowed segments (52 weeks, so the weekly frequencies fall exactly on the grid) with 50% overlap.

## In-sample findings (Phase 1)

- The only periodic component at periods of 2 to 90 days is the **weekly cycle**. It appears at 7, 3.5 and 2.33 days; the last two are harmonics of the week, produced by the sharp weekend dip. Removing weekday means removes all three.
- After removing the weekday effect, autocorrelation decays smoothly and slowly (0.72 at lag 1, 0.26 at lag 30, 0.14 at lag 66). The spectrum is close to a power law, consistent with long memory: a rough fit gives d ≈ 0.45.
- There is no evidence of other cycles at periods of 2 to 90 days. A 35-day peak in the raw periodogram is borderline and doesn't stand out from the background in the Welch spectrum.
- **Modelling the weekly cycle matters for HAR, in fit and in coefficients.** On the same 1,703 days and the same target (log RV):

  | | plain HAR (1, 7, 30) | + weekday dummies |
  |---|---|---|
  | $\beta_d$ | 0.388 | 0.485 |
  | $\beta_w$ | 0.427 | 0.312 |
  | $\beta_m$ | 0.051 ($t$ = 1.2) | 0.089 ($t$ = 2.3) |
  | $R^2$ | 0.428 | **0.648** |
  | residual s.d. | 0.373 | **0.293** |

  Six calendar dummies cut the residual standard deviation by about 21%. Without them, yesterday's value carries the weekday swing while the 7-day average cancels it, so the plain fit shifts weight from the daily to the weekly term and the monthly term looks insignificant. The weekday shifts (Sat −0.81, Sun −0.28, Tue–Fri about −0.2, relative to Monday) are conditional on the HAR inputs, not raw weekday means.
- Once the weekly cycle is handled, horizons (1, 7, 30) and (1, 5, 22) fit almost identically.
- **Not tested:** annual seasonality (under 5 years of data), effects tied to irregular dates (FOMC and CPI releases, holidays, options expiries), and seasonality within the day.

`periodogram_findings.tex` has the details: the spectral analysis and the full table of HAR fits (compile it with `pdflatex` in this folder; it uses the two `periodogram_*.png` figures).

## Limitations
- **One test period, and a calm one.** 2024–26 has lower volatility than 2022–23, so absolute loss levels may not transfer; the comparison between models is on the same days.

- **RV is a noisy proxy** for the true daily variance (288 five-minute returns); QLIKE is used partly for this reason. The 5-minute interval follows the literature and has not been checked on this data with a signature plot.

- **The weekday effect itself drifts** (Saturday −0.62 to −0.92 over 2022–26); the rolling window adapts, but the cause is not identified.

- **Forecast accuracy, not economic value.** No trading strategy or cost is evaluated.

## Outlook
- **Replication for different assets:** Currently BTCUSDT on Binance only; an ETH replication is planned for phase 3 of the project. Equities are also potential application targets. 

- **Weekly and monthly forecast:** Currently one-day ahead forecast only, a weekly and monthly forecast will be included in the next phase.

## Other scripts

### Equity data from Interactive Brokers (for a later cross-asset check)

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

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78, 605–610.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
