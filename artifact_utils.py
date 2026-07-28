"""
Save/load helpers for model artifacts. Everything predict.py needs at
inference time gets persisted here: model, scaler, feature column order,
and a hash of the feature-engineering config so silent drift (someone
changes WINDOW_SIZE and forgets predict.py is still using an old model)
fails loudly instead of producing quietly wrong predictions.
"""

import os
import json
import hashlib
import datetime
import joblib

import config


def config_hash(feature_config: dict = None) -> str:
    feature_config = feature_config or config.FEATURE_CONFIG
    payload = json.dumps(feature_config, sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()


def save_model_artifacts(model_name: str, model, feature_columns: list, scaler=None):
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(config.ARTIFACTS_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(config.ARTIFACTS_DIR, "scaler.pkl"))

    metadata_path = os.path.join(config.ARTIFACTS_DIR, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    metadata[model_name] = {
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "n_features": len(feature_columns),
        "feature_config_hash": config_hash(),
        "uses_scaler": scaler is not None,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[save_model_artifacts] Saved {model_name} -> {model_path}")


def load_model_artifacts(model_name: str):
    model_path = os.path.join(config.ARTIFACTS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py first.")

    model = joblib.load(model_path)

    with open(os.path.join(config.ARTIFACTS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    scaler_path = os.path.join(config.ARTIFACTS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(os.path.join(config.ARTIFACTS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    if model_name not in metadata:
        raise ValueError(f"No metadata entry for '{model_name}' — artifacts may be corrupted or mismatched.")

    stored_hash = metadata[model_name]["feature_config_hash"]
    current_hash = config_hash()
    if stored_hash != current_hash:
        raise ValueError(
            f"Feature config drift detected for '{model_name}'.\n"
            f"Model was trained with config hash {stored_hash}, "
            f"but config.py currently hashes to {current_hash}.\n"
            f"Someone changed FEATURE_CONFIG (e.g. WINDOW_SIZE) since this model was trained. "
            f"Retrain the model or revert the config change before predicting."
        )

    return model, feature_columns, scaler, metadata[model_name]
