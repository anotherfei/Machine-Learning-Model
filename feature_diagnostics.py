"""
Offline diagnostic — not part of the realtime pipeline, run manually
after training (or retraining):

    python feature_diagnostics.py

Answers two questions about how the 3 sensors are actually being weighted
inside the Isolation Forest, since nothing else in this codebase reports
that:

1. Effective per-sensor weight. Isolation Forest doesn't expose feature
   importances directly. This proxies it with permutation importance on
   the raw anomaly score: shuffle one feature at a time across the
   reference set, remeasure the score, and see how much it moves. A
   feature the forest never really splits on barely changes the score
   when shuffled; one it depends on heavily does. Importances are
   aggregated by sensor (vibration / current / temperature / cross-term)
   so you can see whether one sensor's 9 features are collectively
   dominating relative to the other two, rather than that being an
   accident of how many correlated features each sensor happens to have.

2. Near-duplicate features within a sensor's own group (e.g. mean/rms/max
   moving together on a smooth signal), which is the usual reason one
   sensor ends up overweighted without anyone deciding that on purpose —
   9 near-copies of the same signal outvote 1 cross-term.

This never changes the model or the pipeline — it's a report to inform a
human decision (e.g. whether to prune a redundant feature or accept the
imbalance), not something the pipeline acts on automatically.
"""

import os

import numpy as np
import pandas as pd

import config
import artifact_utils
import attribution

N_PERMUTATIONS = 20         # repeats per feature, averaged for stability
CORR_REDUNDANCY_THRESHOLD = 0.95
RANDOM_STATE = 42


def _load_reference_features() -> pd.DataFrame:
    if not os.path.exists(config.FEATURES_DATA_PATH):
        raise FileNotFoundError(
            f"{config.FEATURES_DATA_PATH} not found — run feature_engineering.py "
            f"(or train_isolation_forest.py) first."
        )
    features_df = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])

    ref_timestamps = artifact_utils.load_reference_timestamps()
    if ref_timestamps is None:
        raise RuntimeError(
            "No reference_timestamps.json in artifacts/ — retrain with the current "
            "train_isolation_forest.py so the reference set used to fit the model is "
            "recorded, then rerun this diagnostic against that same set."
        )
    is_reference = features_df[config.COL_TIMESTAMP].isin(ref_timestamps)
    reference_df = features_df[is_reference].reset_index(drop=True)
    if reference_df.empty:
        raise RuntimeError("Reference timestamps didn't match any rows in features.csv.")
    return reference_df


def permutation_importance(scorer, X: pd.DataFrame, feature_cols: list) -> pd.Series:
    """Mean |change in raw anomaly score| when each feature is shuffled,
    averaged over N_PERMUTATIONS repeats. Larger = the model relies on
    that feature more."""
    rng = np.random.RandomState(RANDOM_STATE)
    baseline_scores = scorer.score(X[feature_cols])

    importances = {}
    for col in feature_cols:
        deltas = []
        for _ in range(N_PERMUTATIONS):
            X_shuffled = X[feature_cols].copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            shuffled_scores = scorer.score(X_shuffled)
            deltas.append(np.mean(np.abs(shuffled_scores - baseline_scores)))
        importances[col] = float(np.mean(deltas))
    return pd.Series(importances)


def find_redundant_pairs(X: pd.DataFrame, feature_cols: list, threshold: float) -> list:
    """Within each sensor's own feature group (excluding cross-terms),
    flag pairs whose correlation exceeds `threshold`."""
    groups = {}
    for col in feature_cols:
        if col.endswith("_corr"):
            continue  # cross-terms aren't "one sensor's group"
        sensor = attribution.sensor_label(col)
        groups.setdefault(sensor, []).append(col)

    redundant = []
    for sensor, cols in groups.items():
        if len(cols) < 2:
            continue
        corr = X[cols].corr().abs()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if corr.loc[a, b] >= threshold:
                    redundant.append((sensor, a, b, float(corr.loc[a, b])))
    return redundant


def main():
    print("[feature_diagnostics] Loading model and reference set...")
    scorer, feature_cols, metadata = artifact_utils.load_artifacts()
    reference_df = _load_reference_features()
    print(f"[feature_diagnostics] {len(reference_df)} reference rows, {len(feature_cols)} features.")

    print(f"[feature_diagnostics] Computing permutation importance "
          f"({N_PERMUTATIONS} repeats/feature — this scores the reference set "
          f"{len(feature_cols) * N_PERMUTATIONS + 1} times, may take a moment)...")
    importances = permutation_importance(scorer, reference_df, feature_cols)

    by_sensor = importances.groupby([attribution.sensor_label(c) for c in importances.index]).sum()
    by_sensor_pct = 100 * by_sensor / by_sensor.sum()

    print("\n=== Effective weight by sensor (permutation importance, normalized) ===")
    for sensor, pct in by_sensor_pct.sort_values(ascending=False).items():
        print(f"  {sensor:30s} {pct:5.1f}%")

    print("\n=== Top individual features by importance ===")
    for feat, val in importances.sort_values(ascending=False).head(10).items():
        print(f"  {feat:30s} {val:.5f}")

    print(f"\n=== Near-duplicate features within a sensor's group (|corr| >= {CORR_REDUNDANCY_THRESHOLD}) ===")
    redundant = find_redundant_pairs(reference_df, feature_cols, CORR_REDUNDANCY_THRESHOLD)
    if not redundant:
        print("  None found.")
    else:
        for sensor, a, b, corr_val in sorted(redundant, key=lambda r: -r[3]):
            print(f"  [{sensor}] {a} <-> {b}  (corr={corr_val:.3f})")
        print(
            "\n  These pairs move almost identically across the reference set, so they're "
            "effectively voting together — if the importance-by-sensor split above looks "
            "skewed toward one sensor, this is the first place to look: consider dropping "
            "one feature from each redundant pair rather than changing how sensors are combined."
        )


if __name__ == "__main__":
    main()
