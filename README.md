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
Maintenance Recommendation (rule-based: OK / WARN / CRITICAL)
```

Each stage has one job and can be swapped independently — e.g. replace
Isolation Forest with Deep SVDD without touching the Kalman filter,
trend forecasting, or maintenance rules; replace linear trend forecasting
with an exponential fit without touching anything else (tested — see
"Validation & reliability" below for why linear was kept).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_isolation_forest.py   # preprocess -> features -> fit Isolation Forest -> save
python predict_realtime.py         # replay spindle.csv through the full pipeline, log results
python validate.py                 # external accuracy/reliability check — see below
```

## Design decisions (read before changing config.py)

- **The Isolation Forest is fit on rows passing a fixed spec-bound filter**
  (`preprocessing.select_spec_normal_rows()`, `config.SPEC_VIBRATION_MAX/
  TEMPERATURE_MAX/CURRENT_MAX`), not a burn-in time window. This replaced
  an earlier "first N days = normal" assumption after that assumption was
  shown to fail on non-ramp (e.g. cyclic) trajectories — a fixed window
  can't tell a healthy burn-in period from a burn-in period that's
  already 50%+ degraded, but a spec bound doesn't need one to exist.
  **The trade-off:** correctness now depends entirely on `SPEC_*` being
  the genuine rated range for this hardware. These values are currently
  carried over from an internal reference implementation, not confirmed
  against a manufacturer spec sheet — get that wrong and, unlike a bad
  time window, nothing in this pipeline will detect it. Confirm before
  relying on this in production. `preprocessing.split_reference_window()`
  (the old naive-window approach) still exists as `validate.py`'s
  fallback for artifacts trained before this switch — it's not the
  default path anymore.
- **The anomaly-score-to-health mapping is calibrated once, at fit time,
  from the reference set's own score distribution** (mean/std), then
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
- **The Kalman filter seeds from the mean of the first `KALMAN_INIT_SAMPLES`
  (15) raw readings, not a single tick.** Seeding from tick 0 alone let
  one noisy measurement set the entire starting point — confirmed
  directly to cause a multi-hour false CRITICAL/WARN stretch at the start
  of every run, on data later confirmed genuinely healthy throughout. No
  reading is emitted while the buffer fills (`predict_realtime.py`'s
  `SpindleMonitor.update()` returns `None`; `validate.py`'s `run_backtest()`
  drops those rows the same way), so this trades a few minutes of silence
  at startup for not reporting a false emergency during it. Both call
  sites must seed identically — if you touch one, touch the other, or
  `validate.py` stops measuring what actually ships.
- **Trend-based escalation (`remaining_days`, failure probability) is
  gated behind two independent checks, both required
  (`maintenance.recommend(..., trend_trusted=...)`):**
  1. `TREND_SETTLE_TICKS` (60) — ticks since the trend fit first had
     enough points (`TREND_MIN_POINTS`). The first fit's window still
     partly overlaps the tail of the Kalman warm-up's recovery climb, and
     a plain line over a decelerating rise reads as a false negative
     slope.
  2. `trend_forecast.slope_is_significant()` — a standard OLS
     significance test (`TREND_SLOPE_Z_THRESHOLD`, z=2.0) on the fitted
     slope vs. its own standard error, so a noisy/shallow fit doesn't
     drive an escalation just because `remaining_days()` floors at 1 day.
  Neither alone is sufficient — confirmed directly: significance alone
  still left 12 of 27 originally-observed false post-warm-up CRITICALs
  in place, because that specific window is smooth enough to look
  statistically real while still being unrepresentative. `health_percent`-
  based checks (`FAILURE_HEALTH_THRESHOLD`, `MAINTENANCE_HEALTH_INSPECT`)
  are never gated by this — only the trend-derived triggers are.
- **Failure probability uses a random-walk-with-drift model**
  (uncertainty grows with √horizon), a standard assumption in RUL
  literature — not an arbitrary confidence band. `config.FAILURE_PROB_HORIZONS_DAYS`
  is restricted to `[0.25, 0.5, 0.75, 1]` days — an earlier version went
  out to 35 days (copied from an illustrative example without checking
  it against this data), and `validate.py`'s calibration check showed
  why that was wrong: horizons beyond ~1 day measured Brier scores
  *worse than guessing 50/50* (2d: 0.311, 3d: 0.325, vs. 0.25 for an
  uninformative constant). Don't extend this list without rerunning
  `validate.py` and confirming the new horizon still beats that
  baseline. An exponential-decay alternative to the linear trend fit was
  also tested on the hypothesis that it would fix the underconfidence
  (real wear often accelerates) — it measured worse at every horizon,
  so linear was kept; see git history / conversation log for the test if
  you want to retry it differently.
- **Maintenance thresholds are plain rules, not learned** — see
  `maintenance.py`. This is intentional: business policy for when to act
  belongs to the operator, not a model.
- **Rolling windows with insufficient history produce NaN and are
  dropped, never imputed.**
- **`artifacts/metadata.json` stores a hash of the feature-engineering
  config**; `artifact_utils.load_artifacts()` refuses to load if
  `config.py` has drifted since training. It also stores the exact
  reference-set timestamps used to fit the model
  (`reference_timestamps.json`), so `validate.py` tests the model that
  was actually trained instead of re-deriving its own guess at what the
  reference set should have been.

## Validation & reliability

**When to retrain** — three triggers, only one of which needs code:
1. On a schedule (operational policy, not something this repo enforces).
2. On evidence of drift — `validate.py`'s overfitting check (reference-
   set train/holdout split) is the built-in signal for this.
3. **After a known repair.** If a repair happens with your knowledge —
   you're the one servicing the spindle, or there's a maintenance log —
   you don't need the pipeline to detect a new lifecycle automatically;
   you just need the habit of retraining right after, using data from
   after the repair. A repaired component's "normal" baseline can shift
   even when it's genuinely healthy (e.g. a new bearing's vibration
   signature differs from the worn one it replaced) — the same concern
   `select_spec_normal_rows()`/`assess_reference_window`-style checks
   exist for elsewhere in this repo, just triggered by a fact only a
   human knows, not something in the sensor stream. This is a process
   change, not an engineering one — nothing to build, just don't forget
   to run `train_isolation_forest.py` again after a repair.

`validate.py` is the accuracy/reliability test suite — the **only** file
in this repo that reads `health_status`, kept structurally separate from
the pipeline (see the top of this README). Run it after every retrain:

```bash
python validate.py
```

It prints a full report to the console and saves `validation_report.png`
(9-panel graphical summary: score distributions, ROC/PR curves, health
distributions, confusion matrix heatmaps, reliability diagrams, Brier
scores by horizon). What it checks:

- **Overfitting** — splits the reference baseline window in half, fits a
  temporary Isolation Forest on one half, scores the other. Found a real
  ~3.8-std gap between halves — but tracing it back showed the reference
  window itself isn't stationary (mean vibration rises ~51% within the
  2-day window meant to represent "normal"), not pure model overfitting.
  This is a genuine calibration-window problem, documented above under
  "Design decisions," not fixed — shrinking `REFERENCE_WINDOW_MINUTES`
  would need re-validating against this same check.
- **Discrimination** — ROC-AUC and PR-AUC of `health_state` against
  `health_status != normal`. Measured: ROC-AUC 0.98, PR-AUC 0.98 — the
  Kalman+trend layer measurably improves on the raw Isolation Forest
  score alone (0.94 AUC pre-Kalman).
- **Confusion matrices** — raw counts (TP/FP/TN/FN), for both a bare
  `health_state <= FAILURE_HEALTH_THRESHOLD` cutoff and the actual
  `maintenance.recommend()` CRITICAL output. The two differ meaningfully:
  the raw threshold gets 92.7% precision / 99.7% recall; the full
  maintenance rules trade precision down to 81.3% to reach 100% recall
  (catches every labeled problem, at the cost of ~3x more false alarms
  than the threshold alone). Pick whichever operating point matches your
  actual cost of a missed failure vs. an unnecessary inspection.
- **Calibration/reliability** — Brier score and reliability diagrams for
  `failure_probability_table()`'s output. This is what drove the horizon
  restriction described above: probabilities beyond ~1 day were
  systematically underconfident (predicted 30-60%, observed ~100%) and
  scored worse than an uninformative baseline.

**Read the caveats this script prints, not just the numbers.** Every
check here runs against a single trajectory with exactly one escalation
event — good discrimination scores are real and useful, but "validated"
in the sense of tested against one specific ramp, not proven to
generalize to a different unit or failure mode. Two corrections were
considered and rejected for exactly this reason:
- **Isotonic regression** to fix the probability miscalibration — tested,
  and any honest held-out split of this data is either in-sample
  (misleadingly perfect) or lands in the trajectory's constant-critical
  tail (trivially perfect, tells you nothing). No way to validate a
  calibration correction with one event; revisit once you have failure
  data from more than one.
- **Exponential trend model** instead of linear — physically motivated
  (real wear often accelerates) but measured worse at every horizon when
  actually tested. Kept as a documented negative result rather than
  silently discarded.

## Known limitations, stated plainly

- **No labels are used inside the pipeline itself**, by design — but
  `validate.py` DOES report accuracy/discrimination/calibration metrics,
  externally, against `health_status`, and those results are summarized
  above. An earlier version of this README claimed no accuracy could be
  reported at all; that was true before `validate.py` existed and is no
  longer accurate.
- **There's a brief, intentional startup silence, not a startup transient
  anymore**: `predict_realtime.py` emits nothing for the first
  `KALMAN_INIT_SAMPLES` (15) ticks while the Kalman filter's seed value
  is computed from their mean, and trend-based (not health-based)
  escalation stays untrusted for a further `TREND_SETTLE_TICKS` (60)
  ticks after the trend fit first has enough points, gated additionally
  by a statistical significance check on the fitted slope
  (`trend_forecast.slope_is_significant()`). This replaced an earlier
  version that seeded from a single noisy tick and trusted every trend
  fit immediately — confirmed directly to produce a multi-hour false
  CRITICAL/WARN stretch at the start of every run on data that was
  genuinely healthy throughout. If you see a real false alarm shortly
  after startup despite this, that's a signal these two constants need
  raising for your specific deployment, not that the mechanism is wrong.
- **`SPEC_VIBRATION_MAX/TEMPERATURE_MAX/CURRENT_MAX` (the values that now
  define "normal" for training) are carried over from an internal
  reference implementation, not confirmed against this hardware's actual
  spec sheet.** Everything downstream — the reference set, the health
  calibration, all of it — is only as correct as these three numbers.
  This is a bigger single point of failure than the old time-window
  assumption was, precisely because nothing in this pipeline can detect
  if they're wrong the way `assess_reference_window`-style plausibility
  checks could catch a bad time window.
- **`REFERENCE_WINDOW_MINUTES` (now fallback-only — see `SPEC_*` above for
  the active reference-selection knobs), `HEALTH_SENSITIVITY_STD`,
  `TREND_LOOKBACK_MINUTES`, `KALMAN_PARAMS`, `KALMAN_INIT_SAMPLES`,
  `TREND_SETTLE_TICKS`, `TREND_SLOPE_Z_THRESHOLD`, and
  `FAILURE_HEALTH_THRESHOLD` are all starting points**, not calibrated
  against real failure data — there is none in this dataset (see the
  fully unsupervised framing above; that's the whole point, but it also
  means these knobs were tuned by inspecting behavior on one trajectory,
  not fit to ground truth or validated against a second one).
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
maintenance.py             # rule-based OK/WARN/CRITICAL recommendation
artifact_utils.py          # save/load Isolation Forest + calibration + config hash
train_isolation_forest.py  # fit pipeline end to end
predict_realtime.py        # replaces PM's main.py — full pipeline on streaming data
validate.py                # external accuracy/reliability suite — only file reading health_status
validation_report.png      # generated by validate.py
artifacts/                 # isolation_forest.pkl, calibration.json, metadata.json
data/raw/spindle.csv, data/processed/
```
