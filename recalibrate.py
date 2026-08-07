"""
Recalibrates the health%-anchor (baseline_mean/std) for a SPECIFIC
deployment/unit, without refitting the Isolation Forest tree.

Why this exists: config.RAW_DATA_PATH (spindle_train.csv) is a large
pooled "normal" corpus, fit once so the tree generalizes well across
units — confirmed via validate.py's held-out AUC (0.997 on a genuinely
different trajectory, spindle_given.csv). But the health% SCALE
(baseline_mean/std, calibrated once from that same pooled file) doesn't
transfer as cleanly: measured directly, spindle_given.csv's own
genuinely-normal rows had a z-score distribution (relative to the pooled
calibration) that OVERLAPPED with that same file's genuinely-bad rows —
no single HEALTH_SENSITIVITY_STD value can separate them, because the
anchor itself doesn't match this specific unit's noise floor.

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
    normal_df = preprocessing.select_spec_normal_rows(raw_df)
    if len(normal_df) < min_rows:
        raise ValueError(
            f"Only {len(normal_df)} spec-passing rows found in {data_path} "
            f"(need >= {min_rows}) — not enough to recalibrate confidently. "
            f"Either this deployment's early data is unusually rough, or "
            f"config.SPEC_* bounds don't fit it; check both before proceeding."
        )

    local_features = feature_engineering.create_features(normal_df, verbose=False)
    scorer.calibrate(local_features[feature_cols])

    print(f"  Rows used for recalibration: {len(local_features)} "
          f"(of {len(normal_df)} spec-passing, {len(raw_df)} total)")
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
