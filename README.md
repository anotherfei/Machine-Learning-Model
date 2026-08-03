# Spindle Condition Monitoring — Isolation Forest + Kalman + Trend Forecasting

Fully unsupervised replacement for `Predictive_Maintenance.zip`'s
`main.py` (autoencoder + KMeans + `health_index * 90`) and for this
repo's earlier LightGBM+Kalman version. **No label of any kind is used
anywhere in this codebase.** `health_status` exists in the raw CSV
(`data/raw/spindle.csv`) but `preprocessing.load_data()` reads only
`[timestamp, vibration_mps2, temperature_c, current_ampere]` via pandas'
`usecols` — structurally, not just by convention — so it's never even
loaded, let alone referenced downstream. It exists purely for you to
manually compare pipeline output against, outside this code.

## Architecture

```
Sensors (vibration, temperature, current)
        │
        ▼
Feature Engineering        (rolling RMS/kurtosis/crest factor/trend slope)
        │
        ▼
Isolation Forest           (unsupervised anomaly score, fit on a baseline window)
        │
        ▼
Anomaly → Health mapping   (calibrated against that baseline's own distribution)
        │
        ▼
Kalman Filter              (denoise into a smoothed "estimated health state")
        │
        ▼
Trend Forecasting          (linear regression over a lookback window)
        │
        ▼
Remaining Useful Life + Failure Probability   (random-walk-with-drift model)
        │
        ▼
Maintenance Recommendation (rule-based: OK / INSPECT / PLAN / URGENT)
```

Each stage has one job and can be swapped independently — e.g. replace
Isolation Forest with Deep SVDD without touching the Kalman filter,
trend forecasting, or maintenance rules; replace linear trend forecasting
with an exponential fit without touching anything else.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_isolation_forest.py   # preprocess -> features -> fit Isolation Forest -> save
python predict_realtime.py         # replay spindle.csv through the full pipeline, log results
```

## Design decisions (read before changing config.py)

- **The Isolation Forest is fit only on a "reference baseline window"**
  (`config.REFERENCE_WINDOW_MINUTES`, default: first 2 days) — a standard
  industrial-monitoring practice: use an early commissioning/burn-in
  period as the definition of "normal," on the engineering assumption
  that a freshly commissioned asset starts in good condition. **This is
  an assumption, not something derived from data or labels.** If it's
  wrong — the asset was already degrading during that window — every
  downstream health/RUL number is calibrated against a bad reference.
  That's a real, general limitation of unsupervised condition monitoring,
  not something specific to this codebase.
- **The anomaly-score-to-health mapping is calibrated once, at fit time,
  from the baseline window's own score distribution** (mean/std), then
  held fixed (`isolation_forest.py`'s `health_from_score`). It does not
  renormalize as more (possibly degraded) data streams in — a health
  score of 20% should keep meaning "far from the reference condition,"
  not "typical for what we've seen recently."
- **The Kalman filter is deliberately just a denoiser — one state
  (level), no velocity.** An earlier version of this pipeline used a
  2-state (level + velocity) Kalman filter to estimate the sensor trend
  directly, and it was unstable: a velocity estimated from noisy
  per-tick data blows up when divided into a "days remaining" number.
  Splitting denoising (Kalman) from extrapolation (a separate regression
  module) avoids that entirely and matches the modularity this
  architecture is built around.
- **Trend forecasting uses a 12-hour lookback, not 4.** A 4-hour window
  was tried first and produced a real problem: during a clearly healthy
  stretch (vibration ~1.0–1.9, nowhere near any threshold), the
  maintenance recommendation flip-flopped between `OK` and `URGENT`
  tick to tick, because ordinary noise in the smoothed health signal was
  enough to swing a short-window linear fit's slope wildly (350%/day one
  moment, −149%/day a few minutes later). Widening the lookback to 12
  hours fixed this — the escalation through the actual degradation event
  is now smooth and monotonic (`OK → PLAN → URGENT` as vibration climbs)
  instead of noisy. If you shorten `TREND_LOOKBACK_MINUTES`, re-check for
  this exact failure mode during a known-healthy stretch before trusting
  it.
- **Failure probability uses a random-walk-with-drift model**
  (uncertainty grows with √horizon), a standard assumption in RUL
  literature — not an arbitrary confidence band. It still inherits every
  assumption of the linear trend fit underneath it, so treat it as a
  reasonable approximation, not a calibrated guarantee, especially at
  horizons far from where the trend was actually fit.
- **Maintenance thresholds are plain rules, not learned** — see
  `maintenance.py`. This is intentional: business policy for when to act
  belongs to the operator, not a model.
- **Rolling windows with insufficient history produce NaN and are
  dropped, never imputed.**
- **`artifacts/metadata.json` stores a hash of the feature-engineering
  config**; `artifact_utils.load_artifacts()` refuses to load if
  `config.py` has drifted since training.

## Known limitations, stated plainly

- **No labels anywhere means no accuracy/precision/recall can be
  reported in this codebase**, by design. If you want to know how well
  this actually tracks real degradation, compare `predict_realtime.py`'s
  `health_state` / `maintenance` output against `health_status` yourself,
  outside this code — that comparison is exactly what `health_status` is
  for, but it stays external to the pipeline.
- **There's a startup transient**: health readings during the first ~50
  ticks (before the rolling window and Kalman filter both have enough
  history) are less reliable than later readings — expected, not a bug,
  but worth knowing if you see health swing during the first minute of a
  cold start.
- **`REFERENCE_WINDOW_MINUTES`, `HEALTH_SENSITIVITY_STD`,
  `TREND_LOOKBACK_MINUTES`, `KALMAN_PARAMS`, and `FAILURE_HEALTH_THRESHOLD`
  are all starting points**, not calibrated against real failure data —
  there is none in this dataset (see the fully unsupervised framing
  above; that's the whole point, but it also means these knobs were
  tuned by inspecting behavior, not fit to ground truth).
- **Per-tick recomputation is not fast enough for rapid offline
  backtesting** — a full replay of this dataset's ~10,000 rows takes a
  few minutes, since `predict_realtime.py` recomputes rolling features
  from scratch each tick. Fine for real deployment (one reading per
  minute has minutes of slack); if you need to backtest quickly, write a
  vectorized batch-scoring path instead of driving it through the
  real-time loop.
- **`predict_realtime.py` replays logged data instead of reading live
  hardware** — see the module docstring for what to change to point it
  at a real feed.

## Project structure

```
config.py                  # all tunable constants — read the comments before changing
preprocessing.py           # load (excludes health_status structurally), clean, reference-window split
feature_engineering.py     # rolling stats (RMS/kurtosis/crest factor/trend slope)
isolation_forest.py        # unsupervised anomaly scorer + health calibration
kalman.py                  # single-state denoising filter
trend_forecast.py          # linear regression over Kalman output, RUL extrapolation
failure_probability.py     # random-walk-with-drift failure probability table
maintenance.py             # rule-based OK/INSPECT/PLAN/URGENT recommendation
artifact_utils.py          # save/load Isolation Forest + calibration + config hash
train_isolation_forest.py  # fit pipeline end to end
predict_realtime.py        # replaces PM's main.py — full pipeline on streaming data
artifacts/                 # isolation_forest.pkl, calibration.json, metadata.json
data/raw/spindle.csv, data/processed/
```
