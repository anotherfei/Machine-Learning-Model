"""
Save/load helpers for pipeline artifacts: the fitted Isolation Forest,
its baseline calibration (mean/std from the reference window — see
isolation_forest.py), the feature column order, and a hash of the
feature-engineering config so silent drift (someone changes WINDOW_SIZE
and forgets predict_realtime.py is still using an old model) fails
loudly instead of producing quietly wrong predictions.
"""

import os
import json
import hashlib
import datetime
import joblib
import pandas as pd

import config


def config_hash(feature_config: dict = None) -> str:
    feature_config = feature_config or config.FEATURE_CONFIG
    payload = json.dumps(feature_config, sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()


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
        "feature_config_hash": config_hash(),
        "reference_window_minutes": config.REFERENCE_WINDOW_MINUTES,
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

    stored_hash = metadata["feature_config_hash"]
    current_hash = config_hash()
    if stored_hash != current_hash:
        raise ValueError(
            f"Feature config drift detected.\n"
            f"Model was trained with config hash {stored_hash}, "
            f"but config.py currently hashes to {current_hash}.\n"
            f"Someone changed FEATURE_CONFIG (e.g. WINDOW_SIZE) since this model was trained. "
            f"Retrain the model or revert the config change before predicting."
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
