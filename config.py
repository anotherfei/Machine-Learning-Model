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
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "spindle_train.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")
REALTIME_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "realtime_predictions.csv")
VALIDATION_REPORT_PATH = os.path.join(RESULTS_DIR, "validation_report.png")

PREDICT_DATA_PATH = os.path.join(RAW_DATA_DIR, "spindle_given.csv")

# ---------------------------------------------------------------------------
# Raw column names
# ---------------------------------------------------------------------------
# Sensor: ifm VVB001 (IO-Link vibration/temperature) — no analog current
# output, so current_ampere (previously sourced from a separate current
# sensor) is dropped from this pipeline entirely, not just renamed.
COL_TIMESTAMP = "timestamp"
COL_A_RMS = "a_rms_mps2"          # acceleration RMS, m/s^2 — same physical
                                   # quantity the old vibration_mps2 held
COL_V_RMS = "v_rms_mms"           # velocity RMS, mm/s
COL_A_PEAK = "a_peak_mps2"        # peak acceleration, m/s^2
COL_CREST_FACTOR = "crest_factor" # a_peak / a_rms, dimensionless
COL_TEMPERATURE = "temperature_c"

RAW_SENSOR_COLS = [COL_A_RMS, COL_V_RMS, COL_A_PEAK, COL_CREST_FACTOR, COL_TEMPERATURE]

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
SPEC_A_RMS_MAX = 3.0
SPEC_TEMPERATURE_MAX = 50.0
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
WINDOW_SIZE = 10          # rolling window, in rows (1 row = 1 minute in this data)
SAMPLING_RATE_HZ = 1 / 60
MIN_PERIODS = WINDOW_SIZE

FEATURE_CONFIG = {
    "window_size": WINDOW_SIZE,
    "min_periods": MIN_PERIODS,
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "sensor_cols": RAW_SENSOR_COLS,
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

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Anomaly score -> health percentage mapping
# ---------------------------------------------------------------------------
# health = 100 when score is at or above the reference window's mean
# (as normal as the baseline), degrading linearly to 0 at
# HEALTH_SENSITIVITY_STD standard deviations below the baseline mean.
# Tune this if health hits 0% too early/late relative to visible wear.
HEALTH_SENSITIVITY_STD = 4.0

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
# Degradation modeled as a random walk with drift (standard assumption in
# RUL literature): forecast uncertainty grows with sqrt(horizon), using
# the trend fit's residual std as the per-step noise estimate.
#
# Horizons restricted to <=1 day. validate.py's calibration check measured
# actual Brier scores against health_status: 0.5d=0.164, 1d=0.215 (both
# beat the uninformative baseline of 0.25) but 2d=0.311, 3d=0.325 (WORSE
# than guessing 50/50 — not just unproven, actively misleading). An
# earlier version of this list went out to 35 days, copied from an
# illustrative example without checking it against this dataset's own
# ~6.9-day span or measured calibration — don't extend this list past 1
# day without rerunning validate.py and confirming the Brier score still
# beats 0.25 at whatever horizon you add.
FAILURE_PROB_HORIZONS_DAYS = [0.25, 0.5, 0.75, 1]

# ---------------------------------------------------------------------------
# Maintenance recommendation rules
# ---------------------------------------------------------------------------
MAINTENANCE_HORIZON_DAYS = 1          # "how soon" horizon the rules check against — kept within the <=1 day range validated above
MAINTENANCE_PROB_URGENT = 0.80       # failure probability within horizon -> urgent
MAINTENANCE_PROB_PLAN = 0.60         # -> plan maintenance
# MAINTENANCE_REMAINING_DAYS_URGENT removed as an independent CRITICAL
# trigger (see maintenance.py) — remaining_days is a bare point-estimate
# extrapolation with no uncertainty accounting, while MAINTENANCE_PROB_URGENT
# uses the same slope estimate PLUS residual_std through a proper
# random-walk model (failure_probability.py). Confirmed directly: on
# data/raw/spindle_train.csv (43,176 rows, 100% health_status=='normal' —
# should never report CRITICAL), the two-trigger version fired CRITICAL on
# 13,880 rows (~32%) — a noisy per-tick slope estimate could floor
# remaining_days at 1 day (always <= the urgent cutoff) even when
# failure_probability correctly stayed low because it accounted for that
# same noise as uncertainty. remaining_days is still computed and reported
# for human context — it's just no longer allowed to escalate on its own.

MAINTENANCE_HEALTH_INSPECT = 30      # health % below this -> inspect regardless
FAILURE_HEALTH_THRESHOLD = 20  # health % at which the asset is considered failed

# Hysteresis on the trend/failure-probability trigger only — NOT on
# health_percent-based triggers (FAILURE_HEALTH_THRESHOLD,
# MAINTENANCE_HEALTH_INSPECT), which stay instant because they're already
# the Kalman-smoothed signal and don't need a second smoothing pass (see
# maintenance.MaintenanceDebouncer). Escalation needs
# MAINTENANCE_TREND_DEBOUNCE_TICKS consecutive confirming ticks;
# de-escalation needs the longer MAINTENANCE_TREND_RECOVERY_TICKS —
# standard fast-to-alarm, slow-to-clear alarm-management practice, so a
# genuine escalation isn't reported late but a momentary dip doesn't
# report "all clear" prematurely. Starting points, not fit to data beyond
# the spindle_train.csv sanity check above — same caveat as every other
# *_TICKS constant in this file.
MAINTENANCE_TREND_DEBOUNCE_TICKS = 5
MAINTENANCE_TREND_RECOVERY_TICKS = 15
