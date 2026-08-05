"""
Save/load helpers for pipeline artifacts: the fitted Isolation Forest,
its baseline calibration (mean/std from the reference window — see
isolation_forest.py), the feature column order, and a hash of the
feature-engineering pipeline so silent drift fails loudly instead of
producing quietly wrong predictions.

That hash covers two things that both need to match between training and
predicting:
  - FEATURE_CONFIG (WINDOW_SIZE, MIN_PERIODS, ...) — a config value change.
  - feature_engineering.py's own source — a code change to create_features()
    itself (added/removed/renamed/redefined a feature), which a config-only
    hash would miss entirely: FEATURE_CONFIG can stay identical while the
    actual columns produced change underneath it. Hashing the source is a
    blunt instrument (a comment-only edit also trips it), but a false-
    positive "please retrain" is a far cheaper mistake than predict_realtime.py
    silently selecting a stale subset of columns (or throwing a confusing
    KeyError) out of a feature set that no longer matches what the model
    was trained on.
"""

import os
import json
import hashlib
import datetime
import joblib
import pandas as pd

import config

FEATURE_ENGINEERING_SOURCE_PATH = os.path.join(config.ROOT_DIR, "feature_engineering.py")


def _feature_engineering_source_hash() -> str:
    with open(FEATURE_ENGINEERING_SOURCE_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def config_hash(feature_config: dict = None) -> str:
    feature_config = feature_config or config.FEATURE_CONFIG
    payload = json.dumps(feature_config, sort_keys=True).encode()
    combined = hashlib.md5(payload).hexdigest() + _feature_engineering_source_hash()
    return hashlib.md5(combined.encode()).hexdigest()


def save_artifacts(scorer, feature_columns: list, reference_timestamps=None):
    """
    scorer: a fitted isolation_forest.AnomalyScorer.
    reference_timestamps: the timestamps of the rows actually used to fit
        `scorer` — whatever selection method produced them (naive window,
        find_stable_reference_window, spec-based filter, ...). Saved so
        validate.py's overfitting check can test the model that was
        actually trained, instead of re-deriving its own idea of what the
        reference set should have been.
    """
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(scorer.model, os.path.join(config.ARTIFACTS_DIR, "isolation_forest.pkl"))

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(os.path.join(config.ARTIFACTS_DIR, "calibration.json"), "w") as f:
        json.dump(scorer.calibration(), f, indent=2)

    if reference_timestamps is not None:
        ts_list = [pd.Timestamp(t).isoformat() for t in reference_timestamps]
        with open(os.path.join(config.ARTIFACTS_DIR, "reference_timestamps.json"), "w") as f:
            json.dump(ts_list, f, indent=2)

    metadata = {
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "n_features": len(feature_columns),
        "pipeline_hash": config_hash(),
        # NOT necessarily what was used to build the reference set (e.g. the
        # spec-based filter ignores this entirely) — this is only meaningful
        # for validate.py's no-reference_timestamps fallback path. Trust
        # n_reference_rows/has_reference_timestamps for what actually happened.
        "reference_window_minutes_config_value": config.REFERENCE_WINDOW_MINUTES,
        "n_reference_rows": len(reference_timestamps) if reference_timestamps is not None else None,
        "has_reference_timestamps": reference_timestamps is not None,
    }
    with open(os.path.join(config.ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    extra = ", reference_timestamps.json" if reference_timestamps is not None else ""
    print(f"[save_artifacts] Saved isolation_forest.pkl, calibration.json, "
          f"feature_columns.json, metadata.json{extra} -> {config.ARTIFACTS_DIR}")


def load_artifacts():
    from isolation_forest import AnomalyScorer  # local import avoids a circular import

    model_path = os.path.join(config.ARTIFACTS_DIR, "isolation_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train_isolation_forest.py first.")

    model = joblib.load(model_path)

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    with open(os.path.join(config.ARTIFACTS_DIR, "calibration.json")) as f:
        calibration = json.load(f)

    with open(os.path.join(config.ARTIFACTS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    stored_hash = metadata.get("pipeline_hash", metadata.get("feature_config_hash"))
    current_hash = config_hash()
    if stored_hash != current_hash:
        raise ValueError(
            f"Feature pipeline drift detected.\n"
            f"Model was trained with pipeline hash {stored_hash}, "
            f"but the current FEATURE_CONFIG + feature_engineering.py source "
            f"hashes to {current_hash}.\n"
            f"Either FEATURE_CONFIG changed (e.g. WINDOW_SIZE) or "
            f"feature_engineering.py's create_features() was edited since this "
            f"model was trained. Retrain the model or revert the change before predicting."
        )

    scorer = AnomalyScorer.from_calibration(model, calibration)
    return scorer, feature_columns, metadata


def load_reference_timestamps():
    """
    Returns the pd.Timestamp index of rows actually used to train the
    currently-saved model — whatever selection method produced them —
    or None if the artifacts predate this being tracked (i.e. no
    reference_timestamps.json exists; callers should fall back loudly,
    not silently, since a stale/reconstructed window is exactly the bug
    this function exists to avoid).
    """
    path = os.path.join(config.ARTIFACTS_DIR, "reference_timestamps.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ts_list = json.load(f)
    return pd.to_datetime(pd.Series(ts_list))
