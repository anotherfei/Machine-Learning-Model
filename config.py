"""
Global configuration for the spindle condition-monitoring pipeline.

Architecture (see README.md for the full walkthrough of why):

    Sensors -> Feature Engineering -> Isolation Forest -> Anomaly Score
        -> Kalman Filter -> Estimated Health State -> Trend Forecasting
        -> Remaining Useful Life -> Failure Probability
        -> Maintenance Recommendation

This is fully UNSUPERVISED. The raw CSV has a `health_status` column
(normal/warning/critical) but it is intentionally never loaded, read, or
referenced anywhere in this codebase — preprocessing.py selects columns
by name at read time specifically to exclude it structurally, not just by
convention. It exists only for the person running this pipeline to
manually spot-check results against, entirely outside this code. Do not
reintroduce it into any module — that would silently turn this back into
the label-dependent pipeline this one replaced.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
# Generated outputs for a human to look at (realtime prediction log,
# validation report) — distinct from ARTIFACTS_DIR, which is what the
# model needs to run, not what running it produced.

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "spindle_train.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")
# ---------------------------------------------------------------------------
# Raw column names
# ---------------------------------------------------------------------------
# Sensor: ifm VVB001 (IO-Link vibration/temperature) — no analog current
# output, so current_ampere (previously sourced from a separate current
# sensor) is dropped from this pipeline entirely, not just renamed.
COL_TIMESTAMP = "timestamp"
COL_A_RMS = "a_rms_mps2"          # acceleration RMS, m/s^2 — same physical
                                   # quantity the old vibration_mps2 held
COL_V_RMS = "v_rms_mms"           # canonical velocity RMS, mm/s
COL_A_PEAK = "a_peak_mps2"        # peak acceleration, m/s^2
COL_CREST_FACTOR = "crest_factor" # a_peak / a_rms, dimensionless
COL_TEMPERATURE = "temperature_c"

RAW_SENSOR_COLS = [COL_A_RMS, COL_V_RMS, COL_A_PEAK, COL_CREST_FACTOR, COL_TEMPERATURE]

# PostgreSQL source values are normalized into the canonical units above at
# one boundary in db.canonical_sensor_reading().  The VVB001 IO-Link process
# value for v-RMS is transmitted in SI m/s with 0.0001 m/s resolution, while
# this application deliberately uses the more conventional display/threshold
# unit mm/s.  Do not apply these scales to the offline CSV path: CSV fixtures
# already use the canonical config.py column units.
POSTGRES_SENSOR_SCALES = {
    COL_A_RMS: 1.0,
    COL_V_RMS: 1000.0,  # m/s -> mm/s
    COL_A_PEAK: 1.0,
    COL_CREST_FACTOR: 1.0,
    COL_TEMPERATURE: 1.0,
}

# ---------------------------------------------------------------------------
# Spec-based "normal operation" bounds (alternative to a time-window-based
# reference period — see preprocessing.select_spec_normal_rows()).
# PLACEHOLDER VALUES: these are Predictive_Maintenance.zip's spec-shaped
# filter (vibration<=3.0, temp<=50), carried over because they were the
# only bounds available at the time this was written. This only produces
# a correct reference set if these are genuinely the manufacturer's rated
# normal range for this spindle. If they're a rough guess, that error is
# silent — select_spec_normal_rows() has no way to detect a wrong bound,
# only a missing one. Confirm against the actual spec sheet before relying
# on this in production.
#
# SPEC_A_RMS_MAX carries over the old vibration threshold unchanged since
# a-RMS in m/s^2 is the same physical quantity vibration_mps2 was. There
# is deliberately no threshold set below for v-RMS, a-Peak, or crest
# factor yet — no rated-normal-range values exist for them in this repo,
# and select_spec_normal_rows() treats a None bound as "don't filter on
# this column" rather than silently guessing one. Fill these in once
# you've determined the right operating range for this spindle; until
# then the reference-window approach (see below) doesn't depend on them.

# SPEC_A_RMS_MAX = 3.0
# SPEC_TEMPERATURE_MAX = 50.0
SPEC_A_RMS_MAX = None
SPEC_TEMPERATURE_MAX = None
SPEC_V_RMS_MAX = None
SPEC_A_PEAK_MAX = None
SPEC_CREST_FACTOR_MAX = None
# Maps each raw column to its spec bound above, so preprocessing.py can
# iterate generically instead of hardcoding which columns have bounds.
# A None value means "don't filter on this column" (see note above).
SPEC_MAX = {
    COL_A_RMS: SPEC_A_RMS_MAX,
    COL_V_RMS: SPEC_V_RMS_MAX,
    COL_A_PEAK: SPEC_A_PEAK_MAX,
    COL_CREST_FACTOR: SPEC_CREST_FACTOR_MAX,
    COL_TEMPERATURE: SPEC_TEMPERATURE_MAX,
}
# NOTE: deliberately no COL_STATUS here — see module docstring.

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# PostgreSQL production evidence is approximately one source row per second.
# Rolling windows are row based, so WINDOW_SIZE=10 represents roughly ten
# seconds at that cadence (not ten minutes).
WINDOW_SIZE = 10
SAMPLING_RATE_HZ = 1.0
MIN_PERIODS = WINDOW_SIZE

# Every engineered feature is centered/scaled against that machine's own
# confirmed-healthy commissioning baseline before it enters the shared model.
# Extreme relative deviations are clipped only to bound numeric leverage.
MACHINE_NORMALIZATION_METHOD = "per_machine_median_iqr_v1"
MACHINE_NORMALIZATION_CLIP = 20.0
CONDITION_SCORE_MIN_SPREAD = 0.005

FEATURE_CONFIG = {
    "window_size": WINDOW_SIZE,
    "min_periods": MIN_PERIODS,
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "sensor_cols": RAW_SENSOR_COLS,
    "postgres_sensor_scales": POSTGRES_SENSOR_SCALES,
    "machine_normalization_method": MACHINE_NORMALIZATION_METHOD,
    "machine_normalization_clip": MACHINE_NORMALIZATION_CLIP,
    "condition_score_min_spread": CONDITION_SCORE_MIN_SPREAD,
}

# ---------------------------------------------------------------------------
# Reference / commissioning baseline window
# ---------------------------------------------------------------------------
# Isolation Forest is unsupervised but still needs SOMETHING to define
# "normal" relative to. Standard industrial practice: use an early
# commissioning/burn-in period as the reference baseline, on the
# engineering assumption that a freshly commissioned or recently serviced
# asset starts in good condition. This is an operational assumption, NOT
# derived from any label in the data — if the assumption is wrong (the
# asset was already degrading during this window), the whole pipeline's
# calibration is off. That's a real limitation of unsupervised monitoring
# in general, not specific to this dataset — flag it, don't hide it.
REFERENCE_WINDOW_MINUTES = 2 * 24 * 60  # first 2 days

# ---------------------------------------------------------------------------
# Where train_isolation_forest.py reads the reference/commissioning
# window FROM.
#
#   "live" - the exact same Postgres table + connection worker.py/db.py
#            already use in production, restricted to
#            [REFERENCE_WINDOW_START, REFERENCE_WINDOW_END). Training and
#            production then share one schema and one source of truth —
#            no separate offline file that can silently drift out of sync
#            with what the live sensor actually emits (this pipeline has
#            already been through one such drift: see the a_rms_mps2 /
#            current_ampere history above). This is also just a literal
#            reading of the commissioning-baseline assumption this file
#            already documents above: an early live burn-in period IS the
#            reference, not a CSV standing in for one.
#   "csv"  - config.RAW_DATA_PATH, the original offline-file behavior.
#            Kept for offline experimentation / CI fixtures that
#            shouldn't depend on a reachable database.
#
# Only train_isolation_forest.py reads this — it has no effect on
# predict_realtime.py or worker.py, which always score the live feed
# regardless of where the model was originally fit.
REFERENCE_SOURCE = "live"

# Commissioning window boundaries, only used when REFERENCE_SOURCE=="live".
# Deliberately left unset: train_isolation_forest.py refuses to guess a
# window and fit on "whatever the table currently holds" — that would
# make every training run pull a different, unreproducible reference set
# (the live table keeps growing; a static CSV never did). Pin these to a
# specific confirmed-healthy stretch once enough live data exists after
# commissioning — e.g.:
#   REFERENCE_WINDOW_START = "2026-01-05T00:00:00Z"
#   REFERENCE_WINDOW_END   = "2026-01-07T00:00:00Z"
# (--start/--end on the train_isolation_forest.py command line override
# these for a one-off run without editing this file.)
REFERENCE_WINDOW_START = None
REFERENCE_WINDOW_END = None

# Large commissioning ranges are scanned automatically in bounded PostgreSQL
# chunks.  Every clean source row is considered, while only a deterministic,
# balanced reservoir is retained for the in-memory Isolation Forest fit and
# forward validation artifacts.  These are implementation/resource controls,
# not live monitoring thresholds and therefore are intentionally not editable
# from the website.
TRAINING_DB_CHUNK_ROWS = 50_000
TRAINING_DB_SLICE_HOURS = 24
TRAINING_DB_STATEMENT_TIMEOUT_MS = 120_000
TRAINING_STATE_PROFILE_ROWS = 200_000
TRAINING_MAX_ROWS_PER_MACHINE = 100_000
TRAINING_RESERVOIR_SEED = 42

# ---------------------------------------------------------------------------
# Web-editable review and shared-model retraining defaults
# ---------------------------------------------------------------------------
# runtime_config.py persists operator overrides, but the defaults live here so
# production, Demo, schedulers, and validators cannot acquire different values.
NEAR_MISS_TREND_WINDOW_HOURS = 6
RETRAIN_BATCH_SIZE = 50
RETRAIN_TIME_CAP_DAYS = 30
REFERENCE_WINDOW_MONTHS = 6
REFERENCE_DEDUP_WINDOW_HOURS = 24
REFERENCE_COSINE_SIMILARITY = 0.98
RETRAIN_CHECK_INTERVAL_MINUTES = 60
RETRAIN_RETRY_COOLDOWN_HOURS = 24
RETRAIN_MAX_FP_RATE_INCREASE = 0.02
AUTO_RETRAIN_ENABLED = True

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    # A larger per-tree sample gives the shared model more opportunity to
    # represent distinct machine-relative operating patterns.  Isolation
    # Forest still samples by design; it never needs every retained row in
    # every tree.
    "max_samples": 4096,
    "contamination": "auto",
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Anomaly score -> health percentage mapping
# ---------------------------------------------------------------------------
# condition score = 100 when the anomaly score is at or above the healthy
# reference median, degrading linearly to 0 at HEALTH_SENSITIVITY_STD robust
# spreads below it. This is a relative condition index, not physical remaining
# life. The minimum spread prevents a nearly constant commissioning score from
# turning harmless floating-point noise into a severe condition change. Tune
# either value only against reviewed faults/maintenance outcomes.
HEALTH_SENSITIVITY_STD = 4.0

# How often the production worker checks PostgreSQL for newly ingested rows.
# This is independent of the sensor's own sample cadence.
WORKER_POLL_SECONDS = 1

# ---------------------------------------------------------------------------
# Kalman filter — denoises the raw health-percentage signal into a
# smoothed "estimated health state". Deliberately a simple constant-level
# (random-walk) filter with NO velocity/trend state — trend estimation is
# a separate, swappable stage (trend_forecast.py), not folded into the
# Kalman filter. Keeping these responsibilities separate is the whole
# point of this architecture (see README).
# ---------------------------------------------------------------------------
KALMAN_PARAMS = {
    "process_var": 0.01,     # how much true health can drift per tick
    "measurement_var": 9.0,  # noise in the raw health-percentage estimate
}

# Cold-start seeding: initializing the filter's level from a single raw
# health measurement lets one noisy first tick set the starting point,
# then makes the filter spend hours dragging that level to where the
# signal actually sits (measurement_var=9.0 means each update only closes
# a fraction of the gap). Seeding from the mean of the first
# KALMAN_INIT_SAMPLES raw ticks instead removes most of that single-point
# noise up front, so the filter starts near the true level rather than
# converging to it over hours. No output (health/status) is emitted until
# this warm-up buffer fills — see SpindleMonitor.update() — so the
# few-tick delay never surfaces as a false reading.
#
# 15, not 5: measured against data/raw/spindle.csv, the mean of the first
# 5 raw health readings is still ~11% (below FAILURE_HEALTH_THRESHOLD=20)
# because the earliest rolling-window features are inherently noisier —
# small-sample statistics on a window that has just reached WINDOW_SIZE —
# so a handful of ticks can still land on a genuinely bad run. The mean
# over 15 lands at ~24%, clear of the CRITICAL cutoff. This is a
# empirically-tuned default from one dataset, not a guarantee for every
# deployment — if a real feed still opens on a false CRITICAL/WARN,
# raise this further; the cost is only a longer (still sub-hour, given
# 1 row/minute) delay before the first reading, not a wrong one.
KALMAN_INIT_SAMPLES = 15

# ---------------------------------------------------------------------------
# Trend forecasting
# ---------------------------------------------------------------------------
# Linear regression over the last TREND_LOOKBACK_MINUTES of Kalman-smoothed
# health values. Long enough to average out noise, short enough to react
# if the degradation rate changes.
TREND_LOOKBACK_MINUTES = 720  # 12 hours — 4 hours was noisy enough to cause
                                # false URGENT flags during flat, healthy periods;
                                # see README for the specific example found
TREND_MIN_POINTS = 60

# How many ticks after the trend fit first starts producing a value (i.e.
# after TREND_MIN_POINTS is reached) before remaining_days/failure_probability
# are trusted enough to drive a WARN/CRITICAL escalation on their own.
# health_percent-based checks (FAILURE_HEALTH_THRESHOLD, MAINTENANCE_HEALTH_
# INSPECT) are NOT gated by this — only the trend-derived triggers are.
#
# Why this exists: the first trend fit's window still partly overlaps the
# tail of the Kalman warm-up's recovery climb (see KALMAN_INIT_SAMPLES). A
# plain least-squares line over a decelerating rise reads as a slightly
# negative slope, and remaining_days() floors at 1 day — so on this
# dataset, that alone was enough to falsely trip MAINTENANCE_REMAINING_
# DAYS_URGENT for 27 consecutive ticks right after warm-up, on a machine
# that was demonstrably healthy and improving (confirmed: health_state
# climbing through the 50s-60s throughout). This was invisible before the
# Kalman warm-up fix, because those same ticks were already CRITICAL from
# the raw health threshold for a different reason — fixing that exposed
# this.
#
# 60, not something closer to the observed 27: same reasoning as
# KALMAN_INIT_SAMPLES — this is measured on one dataset's one recovery
# curve, not a guaranteed bound for every deployment. Tying it to
# TREND_MIN_POINTS's own value (rather than a number derived purely from
# this dataset) gives comfortable margin without inventing a second
# unrelated magic number. If a real feed still shows a false trend-based
# escalation shortly after warm-up, raise this — the cost is only a
# longer delay before trend-based (not health-based) alerts are trusted.
TREND_SETTLE_TICKS = 60

# Statistical significance threshold for trend_forecast.slope_is_significant()
# — see that function's docstring for the empirical validation. z=2.0 is the
# standard two-tailed ~95% convention, not something fit to this dataset.
# This and TREND_SETTLE_TICKS are complementary, not redundant: significance
# alone doesn't catch the post-warm-up window (that fit is often smooth
# enough to look "significant" while still measuring the tail of an
# artificial recovery, not real degradation — confirmed: significance
# alone left 12 of the original 27 false post-warm-up CRITICALs in place).
# Settling alone doesn't catch noise-driven false triggers later in the
# trajectory, since those aren't a warm-up phenomenon. Both are required
# (see predict_realtime.py / validate.py) because each covers what the
# other misses.
TREND_SLOPE_Z_THRESHOLD = 2.0
REMAINING_DAYS_CAP = 90

# ---------------------------------------------------------------------------
# Failure probability
# ---------------------------------------------------------------------------
# Degradation is modeled as Brownian motion with drift. The first-passage
# calculation estimates the chance of crossing the critical condition boundary
# at any point within the horizon. Diffusion is estimated from detrended
# consecutive condition innovations using their real timestamp intervals; the
# regression residual level is deliberately not reused as per-step noise.
#
# Short horizons drive urgent decisions; broader horizons provide planning
# visibility up to one week. These are model-based first-passage risks, not
# empirical failure frequencies. Validate the longer horizons against labelled
# fleet outcomes with Accuracy simulation before relying on their numeric value.
FAILURE_PROB_HORIZONS_DAYS = [0.25, 0.5, 0.75, 1, 7]

# ---------------------------------------------------------------------------
# Maintenance recommendation rules
# ---------------------------------------------------------------------------
MAINTENANCE_HORIZON_DAYS = 7          # planned-maintenance WARN look-ahead
MAINTENANCE_URGENT_HORIZON_DAYS = 1   # CRITICAL remains a near-term decision
MAINTENANCE_URGENT_HORIZON_MAX_DAYS = 1
MAINTENANCE_PROB_URGENT = 0.80       # model-estimated boundary-crossing risk -> urgent
MAINTENANCE_PROB_PLAN = 0.60         # -> plan maintenance
# MAINTENANCE_REMAINING_DAYS_URGENT removed as an independent CRITICAL
# trigger (see maintenance.py) — remaining_days is a bare point-estimate
# extrapolation with no uncertainty accounting, while MAINTENANCE_PROB_URGENT
# uses the same slope estimate plus diffusion estimated from detrended
# condition innovations in a first-passage model (failure_probability.py).
# Confirmed directly for the earlier point-estimate trigger design: on
# data/raw/spindle_train.csv (43,176 rows, 100% health_status=='normal' —
# should never report CRITICAL), the two-trigger version fired CRITICAL on
# 13,880 rows (~32%) — a noisy per-tick slope estimate could floor
# remaining_days at 1 day (always <= the urgent cutoff) even when
# failure_probability correctly stayed low because it accounted for that
# same noise as uncertainty. remaining_days is still computed and reported
# for human context — it's just no longer allowed to escalate on its own.

MAINTENANCE_HEALTH_INSPECT = 30      # health % below this -> inspect regardless
FAILURE_HEALTH_THRESHOLD = 20  # health % at which the asset is considered failed

# ---------------------------------------------------------------------------
# Automatic operating-state gate (no PLC/run-status signal required)
# ---------------------------------------------------------------------------
# The worker learns stationary/running vibration regimes independently for
# each machine. It only enables STOPPED suppression when the two regimes are
# clearly separated; otherwise state stays UNKNOWN and ML remains active.
OPERATING_STATE_HISTORY_ROWS = 1440
OPERATING_STATE_MIN_HISTORY_ROWS = 120
OPERATING_STATE_MIN_CLUSTER_ROWS = 10
OPERATING_STATE_MIN_CLUSTER_FRACTION = 0.005
OPERATING_STATE_MIN_LOG_SEPARATION = 0.45
OPERATING_STATE_MIN_SEPARATION_QUALITY = 2.5
# Months-long commissioning profiles contain far more legitimate within-RUNNING
# speed/load variation than the short live history. A lower robust-MAD quality
# floor is allowed only for that offline profile; the center ratio and
# chronological STOPPED/STARTING confirmation rules still apply unchanged.
OPERATING_STATE_COMMISSIONING_MIN_SEPARATION_QUALITY = 1.5
OPERATING_STATE_REFIT_TICKS = 5
OPERATING_STATE_STOP_CONFIRM_TICKS = 5
OPERATING_STATE_START_CONFIRM_TICKS = 3
# Restart warm-up is derived at runtime from WINDOW_SIZE plus the editable
# KALMAN_INIT_SAMPLES value so the state gate and condition monitor agree.
# The API reports NO_DATA when the newest source row is older than this.
SOURCE_STALE_SECONDS = 3 * 60

# Timestamp-based confirmation applies to direct condition and forecast rules.
# Sustained critical evidence first raises WARN, then advances to CRITICAL if it
# continues. Recovery is longest to prevent state flapping. Durations use
# sensor timestamps, not row counts.
MAINTENANCE_WARN_CONFIRM_MINUTES = 5
MAINTENANCE_CRITICAL_CONFIRM_MINUTES = 10
MAINTENANCE_RECOVERY_MINUTES = 10
