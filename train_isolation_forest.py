"""
Fit the Isolation Forest on the reference baseline window and save
artifacts. No labels anywhere in this script.

Usage:
    python train_isolation_forest.py
"""

import os

import config
import preprocessing
import feature_engineering
import artifact_utils
from isolation_forest import AnomalyScorer


def get_or_build_features():
    if not os.path.exists(config.FEATURES_DATA_PATH):
        print("[train] No cached features found — running full preprocessing pipeline.")
        preprocessing.run_preprocessing()
        return feature_engineering.run_feature_engineering()

    print(f"[train] Loading cached features from {config.FEATURES_DATA_PATH}")
    import pandas as pd
    return pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])


def main():
    df = get_or_build_features()
    feature_cols = feature_engineering.get_feature_columns(df)

    reference_df, rest_df = preprocessing.split_reference_window(df)
    print(f"[train] Reference window: {len(reference_df)} rows, "
          f"rest of trajectory: {len(rest_df)} rows.")

    scorer = AnomalyScorer()
    scorer.fit(reference_df[feature_cols])
    print(f"[train] Calibration: {scorer.calibration()}")

    # Sanity check — score the full trajectory and report the health range.
    # Not a labeled evaluation (no labels exist in this codebase); just a
    # distributional check that health actually varies across the data
    # rather than sitting flat at 100 or crashing to 0 everywhere.
    all_scores = scorer.score(df[feature_cols])
    all_health = scorer.health_from_score(all_scores)
    print(f"[train] Health over full trajectory — "
          f"min: {all_health.min():.1f}, max: {all_health.max():.1f}, "
          f"mean: {all_health.mean():.1f}")

    artifact_utils.save_artifacts(scorer, feature_cols)


if __name__ == "__main__":
    main()
