"""
Save/load helpers for the fitted Isolation Forest, robust automatic condition
anchors, per-machine feature normalizers, feature order, and a pipeline hash
so silent drift fails loudly instead of producing quietly wrong predictions.

That hash covers two things that both need to match between training and
predicting:
  - FEATURE_CONFIG (WINDOW_SIZE, MIN_PERIODS, ...) — a config value change.
  - feature_engineering.py and machine_normalization.py source — a code change
    to feature construction or its machine-relative transform, which a config-only
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
from pathlib import Path
import joblib
import pandas as pd

import config


# Model-protocol constants, deliberately not runtime policy. Changing these
# alters what evidence is used for fitting and requires recommissioning rather
# than an in-place website setting change.
VALIDATION_HOLDOUT_FRACTION = 0.20
VALIDATION_HOLDOUT_MIN_ROWS = 20
COMMISSIONING_MAX_HOLDOUT_FP_RATE = 0.10

FEATURE_ENGINEERING_SOURCE_PATH = os.path.join(config.ROOT_DIR, "feature_engineering.py")
MACHINE_NORMALIZATION_SOURCE_PATH = os.path.join(config.ROOT_DIR, "machine_normalization.py")


def active_artifacts_dir() -> str:
    """Resolve the immutable bundle selected by ``artifacts/active.json``.

    The root artifact directory remains a narrow pre-registration staging
    fallback so a freshly completed manual training can be registered by the
    API. Once a pointer exists, every production reader uses its version folder.
    """
    import model_registry  # local import keeps registry utilities lightweight

    selected = model_registry.active_bundle_path(required=False)
    return str(selected) if selected is not None else config.ARTIFACTS_DIR


def _pipeline_source_hash() -> str:
    hasher = hashlib.md5()
    for path in (FEATURE_ENGINEERING_SOURCE_PATH, MACHINE_NORMALIZATION_SOURCE_PATH):
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def config_hash(feature_config: dict = None) -> str:
    feature_config = feature_config or config.FEATURE_CONFIG
    payload = json.dumps(feature_config, sort_keys=True).encode()
    combined = hashlib.md5(payload).hexdigest() + _pipeline_source_hash()
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


def finite_float(value):
    """Return one finite native float, or ``None`` for invalid JSON/DB input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_optional_json(directory, filename: str, payload) -> bool:
    """Write an optional artifact, removing an older stale copy if absent."""
    path = Path(directory) / filename
    if payload is None:
        path.unlink(missing_ok=True)
        return False
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, allow_nan=False)
    return True


def _write_optional_csv(directory, filename: str, frame) -> bool:
    """Write an optional table artifact, removing an older stale copy if absent."""
    path = Path(directory) / filename
    if frame is None:
        path.unlink(missing_ok=True)
        return False
    frame.to_csv(path, index=False)
    return True


def _validate_machine_context(machine_calibrations, machine_feature_normalizers) -> None:
    if not machine_calibrations or not machine_feature_normalizers:
        raise ValueError("A model bundle requires machine calibrations and feature normalizers.")
    if set(machine_calibrations) != set(machine_feature_normalizers):
        raise ValueError(
            "Machine condition anchors and feature normalizers must cover "
            "the same commissioned machines."
        )


def write_core_bundle(
    directory,
    scorer,
    feature_columns,
    metadata,
    machine_calibrations,
    machine_feature_normalizers,
) -> None:
    """Write the required files shared by manual training and retraining."""
    _validate_machine_context(machine_calibrations, machine_feature_normalizers)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer.model, directory / "isolation_forest.pkl")
    payloads = {
        "feature_columns.json": feature_columns,
        "calibration.json": scorer.calibration(),
        "metadata.json": metadata,
        "machine_calibrations.json": machine_calibrations,
        "machine_feature_normalizers.json": machine_feature_normalizers,
    }
    for filename, payload in payloads.items():
        (directory / filename).write_text(
            json.dumps(to_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )


def save_artifacts(
    scorer,
    feature_columns: list,
    reference_timestamps=None,
    reference_rows=None,
    reference_features=None,
    validation_features=None,
    machine_calibrations=None,
    machine_feature_normalizers=None,
    metadata_extra: dict | None = None,
    directory=None,
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
    validation_features: optional machine-aware confirmed-normal rows held out
        from fitting and calibration for independent shadow-model validation.
    machine_calibrations: required automatic per-machine condition anchors
        computed after fitting the one shared model.
    machine_feature_normalizers: required per-machine robust feature-space
        transforms used before every fit/score call.
    metadata_extra: training-strategy provenance added to metadata.json.
    directory: immutable bundle destination. Initial training passes a
        directory under artifacts/versions; the artifact root is retained
        only as a backwards-compatible default for older callers.
    """
    _validate_machine_context(machine_calibrations, machine_feature_normalizers)

    directory = Path(directory or config.ARTIFACTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    ts_list = None
    if reference_timestamps is not None:
        ts_list = [pd.Timestamp(t).isoformat() for t in reference_timestamps]
    _write_optional_json(directory, "reference_timestamps.json", ts_list)

    normalized_reference_rows = None
    if reference_rows is not None:
        normalized_reference_rows = [
            {
                "machine_id": str(row["machine_id"]),
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            }
            for row in reference_rows
        ]
    wrote_reference_rows = _write_optional_json(directory, "reference_rows.json", normalized_reference_rows)
    wrote_reference_features = _write_optional_csv(directory, "reference_features.csv", reference_features)
    wrote_validation_features = _write_optional_csv(directory, "validation_features.csv", validation_features)
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
    write_core_bundle(
        directory,
        scorer,
        feature_columns,
        metadata,
        machine_calibrations,
        machine_feature_normalizers,
    )

    extras = []
    if reference_timestamps is not None:
        extras.append("reference_timestamps.json")
    if wrote_reference_rows:
        extras.append("reference_rows.json")
    if wrote_reference_features:
        extras.append("reference_features.csv")
    if wrote_validation_features:
        extras.append("validation_features.csv")
    extras.extend(("machine_calibrations.json", "machine_feature_normalizers.json"))
    extra = f", {', '.join(extras)}" if extras else ""
    print(f"[save_artifacts] Saved isolation_forest.pkl, calibration.json, "
          f"feature_columns.json, metadata.json{extra} -> {directory}")
    return metadata


def load_artifacts(base_dir: str | None = None):
    """Load the shared model and its bundled automatic pooled anchor.

    Production applies the bundled automatic per-machine anchor separately in
    worker.py. Manual/local calibration overrides are deliberately unsupported.
    ``base_dir`` lets a historical simulation pin a versioned bundle without
    installing it over the live model files.
    """
    from isolation_forest import AnomalyScorer  # local import avoids a circular import

    base_dir = base_dir or active_artifacts_dir()
    model_path = os.path.join(base_dir, "isolation_forest.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train_isolation_forest.py first.")

    model = joblib.load(model_path)

    with open(os.path.join(base_dir, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    with open(os.path.join(base_dir, "metadata.json")) as f:
        metadata = json.load(f)

    stored_hash = metadata.get("pipeline_hash", metadata.get("feature_config_hash"))
    current_hash = config_hash()
    if stored_hash != current_hash:
        raise ValueError(
            f"Feature pipeline drift detected.\n"
            f"Model was trained with pipeline hash {stored_hash}, "
            f"but the current FEATURE_CONFIG + feature/normalization source "
            f"hashes to {current_hash}.\n"
            f"Either FEATURE_CONFIG changed (e.g. WINDOW_SIZE) or "
            f"feature_engineering.py or machine_normalization.py was edited since this "
            f"model was trained. Retrain the model or revert the change before predicting."
        )

    with open(os.path.join(base_dir, "calibration.json")) as f:
        calibration = json.load(f)
    print("[load_artifacts] Calibration source: bundled automatic anchor")

    scorer = AnomalyScorer.from_calibration(model, calibration)
    return scorer, feature_columns, metadata


def load_reference_timestamps(base_dir: str | None = None):
    """
    Returns the pd.Timestamp index of rows actually used to train the
    currently-saved model — whatever selection method produced them —
    or None if the artifacts predate this being tracked (i.e. no
    reference_timestamps.json exists; callers should fall back loudly,
    not silently, since a stale/reconstructed window is exactly the bug
    this function exists to avoid).
    """
    path = os.path.join(base_dir or active_artifacts_dir(), "reference_timestamps.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ts_list = json.load(f)
    return pd.to_datetime(pd.Series(ts_list))


def load_reference_rows(base_dir: str | None = None):
    """Return machine-aware training-row identities, or None for old bundles."""
    path = os.path.join(base_dir or active_artifacts_dir(), "reference_rows.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_machine_calibrations(required: bool = True, base_dir: str | None = None):
    """Load automatic per-machine condition anchors bundled with the model."""
    path = os.path.join(base_dir or active_artifacts_dir(), "machine_calibrations.json")
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                "The active model has no machine_calibrations.json. "
                "Retrain it with the current balanced trainer before scoring."
            )
        return None
    with open(path) as f:
        return json.load(f)


def load_machine_feature_normalizers(required: bool = True, base_dir: str | None = None):
    """Load the per-machine feature transforms bundled with the model."""
    path = os.path.join(base_dir or active_artifacts_dir(), "machine_feature_normalizers.json")
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                "The active model has no machine_feature_normalizers.json. "
                "Retrain it with the current balanced trainer before scoring."
            )
        return None
    with open(path) as f:
        return json.load(f)


def load_reference_features(base_dir: str | None = None):
    """Load the exact machine-aware feature corpus used to fit the model."""
    path = os.path.join(base_dir or active_artifacts_dir(), "reference_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(
        path,
        parse_dates=[config.COL_TIMESTAMP],
        dtype={"machine_id": str},
    )


def load_validation_features(base_dir: str | None = None):
    """Load confirmed-normal rows deliberately excluded from model fitting."""
    path = os.path.join(base_dir or active_artifacts_dir(), "validation_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(
        path,
        parse_dates=[config.COL_TIMESTAMP],
        dtype={"machine_id": str},
    )
