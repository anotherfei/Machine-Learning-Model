"""
Recalibrates the health%-anchor (baseline_mean/std) for a SPECIFIC
deployment/unit, without refitting the Isolation Forest tree.

Why this exists: the deployed tree (config.ARTIFACTS_DIR/isolation_forest.pkl)
is typically fit once on a broad pooled "normal" corpus — whether that
came from config.RAW_DATA_PATH (spindle_train.csv, the legacy
REFERENCE_SOURCE="csv" path) or from another unit's live commissioning
window (REFERENCE_SOURCE="live", see config.py) — so it generalizes well
across units; confirmed via validate.py's held-out AUC (0.997 on a
genuinely different trajectory, spindle_given.csv). But the health%
SCALE (baseline_mean/std, calibrated once from that same pooled
reference) doesn't transfer as cleanly to a *different* unit than the
one it was fit on: measured directly, spindle_given.csv's own
genuinely-normal rows had a z-score distribution (relative to the pooled
calibration) that OVERLAPPED with that same file's genuinely-bad rows —
no single HEALTH_SENSITIVITY_STD value can separate them, because the
anchor itself doesn't match this specific unit's noise floor. (A model
trained directly on a given unit's own live commissioning data, per
REFERENCE_SOURCE="live", doesn't need this step for THAT unit — this
still matters the moment the same tree gets reused for any other unit.)

Usage — run this ONCE per new deployment, before predict_realtime.py:

    python recalibrate.py --data path/to/this_units_data.csv --name unit_A

Then predict_realtime.py / validate.py can load that unit's calibration
via artifact_utils.load_artifacts(deployment_name="unit_A"). Omit --name
for a single default override (calibration_local.json).

Deliberately unsupervised, same as the rest of this pipeline: selects
this unit's own "normal" rows via preprocessing.select_spec_normal_rows()
— the SAME criterion used to build the pooled training reference — never
health_status. A real new deployment won't have labels; this script
doesn't assume it does, even when run against a file (like
spindle_given.csv) that happens to have them.

Ordering matters here: feature_engineering.create_features() is applied
to the FULL cleaned data first, and select_spec_normal_rows() only picks
which of those already-computed feature rows to keep, by timestamp —
never the other way around. create_features()'s rolling window
(config.WINDOW_SIZE) operates on row POSITION, not elapsed time; filtering
out-of-spec rows before engineering features can leave gaps in the
timeline, and a positional rolling window doesn't know a gap is there —
it will happily average together rows that were never actually adjacent
in time, corrupting the rolling mean/std/kurtosis/skew/crest_factor/
trend_slope/cross-correlations for rows near that gap. Filtering by
timestamp AFTER feature engineering (this script's current order, and
the same order train_isolation_forest.py uses) never has that problem.
An earlier version of this script filtered first — confirmed to
measurably corrupt baseline_mean/baseline_std as a result.
"""

import argparse

import config
import artifact_utils
import preprocessing
import feature_engineering


def recalibrate(data_path: str, deployment_name: str = None, min_rows: int = 200):
    print(f"Recalibrating health% anchor from: {data_path}")
    scorer, feature_cols, metadata = artifact_utils.load_artifacts(use_local_calibration=False)
    old_mean, old_std = scorer.baseline_mean, scorer.baseline_std

    raw_df = preprocessing.clean_data(preprocessing.load_data(data_path))

    # Feature engineering FIRST, over the full contiguous cleaned data —
    # then select which of those feature rows are spec-normal, by
    # timestamp. See module docstring for why the order can't be
    # reversed without corrupting the rolling-window features.
    featured_df = feature_engineering.create_features(raw_df, verbose=False)
    normal_raw = preprocessing.select_spec_normal_rows(raw_df)
    is_normal = featured_df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
    local_features = featured_df[is_normal].reset_index(drop=True)

    if len(local_features) < min_rows:
        raise ValueError(
            f"Only {len(local_features)} spec-passing feature rows found in {data_path} "
            f"(need >= {min_rows}) — not enough to recalibrate confidently. "
            f"Either this deployment's early data is unusually rough, or "
            f"config.SPEC_* bounds don't fit it; check both before proceeding."
        )

    scorer.calibrate(local_features[feature_cols])

    print(f"  Rows used for recalibration: {len(local_features)} "
          f"(of {len(normal_raw)} spec-passing raw rows, {len(raw_df)} total raw rows; "
          f"the {len(normal_raw) - len(local_features)}-row difference is normal — rolling-window "
          f"warm-up at the very start of the trajectory has no feature row at all, spec-normal "
          f"or not)")
    print(f"  baseline_mean: {old_mean:.4f} -> {scorer.baseline_mean:.4f}")
    print(f"  baseline_std:  {old_std:.4f} -> {scorer.baseline_std:.4f}")

    path = artifact_utils.save_local_calibration(scorer, deployment_name=deployment_name)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.PREDICT_DATA_PATH,
                         help=f"Path to this deployment's own data (default: config.PREDICT_DATA_PATH = {config.PREDICT_DATA_PATH})")
    parser.add_argument("--name", default=None,
                         help="Deployment name, for calibration_local_<name>.json. Omit for the single default override.")
    args = parser.parse_args()
    recalibrate(args.data, deployment_name=args.name)
