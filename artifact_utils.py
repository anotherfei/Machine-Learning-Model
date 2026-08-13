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
import math
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


def to_json_safe(value):
    """Convert numpy/pandas scalars and non-finite values to strict JSON."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _write_optional_json(filename: str, payload) -> bool:
    """Write an optional artifact, removing an older stale copy if absent."""
    path = os.path.join(config.ARTIFACTS_DIR, filename)
    if payload is None:
        if os.path.exists(path):
            os.remove(path)
        return False
    with open(path, "w") as f:
        json.dump(to_json_safe(payload), f, indent=2, allow_nan=False)
    return True


def _write_optional_csv(filename: str, frame) -> bool:
    """Write an optional table artifact, removing an older stale copy if absent."""
    path = os.path.join(config.ARTIFACTS_DIR, filename)
    if frame is None:
        if os.path.exists(path):
            os.remove(path)
        return False
    frame.to_csv(path, index=False)
    return True


def save_artifacts(
    scorer,
    feature_columns: list,
    reference_timestamps=None,
    reference_rows=None,
    reference_features=None,
    machine_calibrations=None,
    metadata_extra: dict | None = None,
):
    """
    scorer: a fitted isolation_forest.AnomalyScorer.
    reference_timestamps: the timestamps of the rows actually used to fit
        `scorer` — whatever selection method produced them (naive window,
        find_stable_reference_window, spec-based filter, ...). Saved so
        validate.py's overfitting check can test the model that was
        actually trained, instead of re-deriving its own idea of what the
        reference set should have been.
    reference_rows: optional machine-aware identities for the selected
        training rows. Each item contains machine_id and timestamp.
    reference_features: optional machine-aware feature table containing the
        exact balanced rows used to fit the shared model. Shadow retraining
        uses this rather than trying to reconstruct live features by timestamp.
    machine_calibrations: optional per-machine health anchors computed
        after fitting the one shared model.
    metadata_extra: training-strategy provenance added to metadata.json.
    """
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(scorer.model, os.path.join(config.ARTIFACTS_DIR, "isolation_forest.pkl"))

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(os.path.join(config.ARTIFACTS_DIR, "calibration.json"), "w") as f:
        json.dump(scorer.calibration(), f, indent=2)

    ts_list = None
    if reference_timestamps is not None:
        ts_list = [pd.Timestamp(t).isoformat() for t in reference_timestamps]
    _write_optional_json("reference_timestamps.json", ts_list)

    normalized_reference_rows = None
    if reference_rows is not None:
        normalized_reference_rows = [
            {
                "machine_id": str(row["machine_id"]),
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            }
            for row in reference_rows
        ]
    wrote_reference_rows = _write_optional_json("reference_rows.json", normalized_reference_rows)
    wrote_reference_features = _write_optional_csv("reference_features.csv", reference_features)
    wrote_machine_calibrations = _write_optional_json("machine_calibrations.json", machine_calibrations)

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
    metadata.update(metadata_extra or {})
    with open(os.path.join(config.ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    extras = []
    if reference_timestamps is not None:
        extras.append("reference_timestamps.json")
    if wrote_reference_rows:
        extras.append("reference_rows.json")
    if wrote_reference_features:
        extras.append("reference_features.csv")
    if wrote_machine_calibrations:
        extras.append("machine_calibrations.json")
    extra = f", {', '.join(extras)}" if extras else ""
    print(f"[save_artifacts] Saved isolation_forest.pkl, calibration.json, "
          f"feature_columns.json, metadata.json{extra} -> {config.ARTIFACTS_DIR}")


def save_local_calibration(scorer, deployment_name: str = None):
    """
    Saves ONLY the health%-anchor calibration (baseline_mean/std, per-
    feature diagnostics) — not the tree, not feature_columns, not
    metadata — to a separate file from the pooled-training default. See
    isolation_forest.AnomalyScorer.calibrate()'s docstring for why this
    needs to exist: the tree (fit once on a broad pooled corpus) and the
    health% anchor (recalibrated per deployment, from that deployment's
    own short known-healthy window) can legitimately need to come from
    different data.

    deployment_name lets multiple units share one trained tree with
    separate calibration files (calibration_local_<name>.json); omit it
    for a single default override (calibration_local.json).
    """
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    fname = f"calibration_local_{deployment_name}.json" if deployment_name else "calibration_local.json"
    path = os.path.join(config.ARTIFACTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(scorer.calibration(), f, indent=2)
    print(f"[save_local_calibration] Saved -> {path}")
    return path


def write_local_calibration(calibration: dict, deployment_name: str = None) -> str:
    """
    Same file convention as save_local_calibration(), but takes an
    already-computed calibration dict directly instead of a fitted
    AnomalyScorer. This remains the file-based helper for standalone
    deployments; the multi-machine web worker uses database assignments.
    """
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    fname = f"calibration_local_{deployment_name}.json" if deployment_name else "calibration_local.json"
    path = os.path.join(config.ARTIFACTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"[write_local_calibration] Saved -> {path}")
    return path


def clear_local_calibration(deployment_name: str = None) -> bool:
    """
    Removes a local calibration override, if one exists, so the next
    load_artifacts() falls back to the pooled default ("normal" in the
    Models page's recalibration UI). Returns whether a file was actually
    removed, so callers can tell "reverted" from "was already normal".
    """
    fname = f"calibration_local_{deployment_name}.json" if deployment_name else "calibration_local.json"
    path = os.path.join(config.ARTIFACTS_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f"[clear_local_calibration] Removed -> {path}")
        return True
    return False


def load_local_calibration(deployment_name: str = None):
    """Returns the local calibration dict, or None if no override has been saved."""
    fname = f"calibration_local_{deployment_name}.json" if deployment_name else "calibration_local.json"
    path = os.path.join(config.ARTIFACTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_artifacts(deployment_name: str = None, use_local_calibration: bool = None):
    """
    use_local_calibration: explicit opt-in/opt-out. None (default) means
    "use a local override if calibration_local(_<name>).json exists, else
    fall back to the pooled default" — and PRINTS which one it picked
    either way, so this is never a silent choice. Pass True to require a
    local override (raises if missing) or False to force the pooled
    default even if a local override exists.
    """
    from isolation_forest import AnomalyScorer  # local import avoids a circular import

    model_path = os.path.join(config.ARTIFACTS_DIR, "isolation_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train_isolation_forest.py first.")

    model = joblib.load(model_path)

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

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

    local_calibration = load_local_calibration(deployment_name)
    if use_local_calibration is True and local_calibration is None:
        raise FileNotFoundError(
            f"use_local_calibration=True but no calibration_local"
            f"{'_' + deployment_name if deployment_name else ''}.json found in "
            f"{config.ARTIFACTS_DIR}. Run recalibrate.py for this deployment first."
        )
    if use_local_calibration is False:
        calibration_source, calibration = "pooled default (forced)", None
    elif local_calibration is not None:
        calibration_source, calibration = "LOCAL override", local_calibration
    else:
        calibration_source, calibration = "pooled default (no local override found)", None

    if calibration is None:
        with open(os.path.join(config.ARTIFACTS_DIR, "calibration.json")) as f:
            calibration = json.load(f)
    print(f"[load_artifacts] Calibration source: {calibration_source}")

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


def load_reference_rows():
    """Return machine-aware training-row identities, or None for old bundles."""
    path = os.path.join(config.ARTIFACTS_DIR, "reference_rows.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_machine_calibrations():
    """Return initial per-machine calibrations saved by pooled training."""
    path = os.path.join(config.ARTIFACTS_DIR, "machine_calibrations.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_reference_features():
    """Load the exact machine-aware feature corpus used to fit the model."""
    path = os.path.join(config.ARTIFACTS_DIR, "reference_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(
        path,
        parse_dates=[config.COL_TIMESTAMP],
        dtype={"machine_id": str},
    )
