"""Robust, machine-relative normalization for shared-model features.

The shared Isolation Forest must compare operating *patterns*, not confuse one
machine's harmless baseline offset with another machine's fault.  Normalizers
are therefore fitted from each machine's confirmed-healthy engineered features
and persisted with the model bundle.  Raw sensor values and engineered feature
artifacts remain in their physical units for auditability; normalization is
applied only at the model boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


METHOD = config.MACHINE_NORMALIZATION_METHOD


def _robust_scale(frame: pd.DataFrame, center: pd.Series) -> pd.Series:
    """Return a positive robust scale for every feature.

    IQR is the primary estimate. MAD and ordinary standard deviation are only
    fallbacks for quantized features whose middle 50% is flat. A completely
    constant training feature receives scale 1.0 and is reported in the
    artifact; because the model saw only zero for that normalized feature it
    cannot silently learn a fabricated amount of variation.
    """
    q25 = frame.quantile(0.25)
    q75 = frame.quantile(0.75)
    iqr_scale = (q75 - q25) / 1.349
    mad_scale = (frame.subtract(center).abs().median()) * 1.4826
    std_scale = frame.std(ddof=0)

    scale = iqr_scale.copy()
    tolerance = np.finfo(float).eps * center.abs().clip(lower=1.0) * 100
    scale = scale.where(scale > tolerance, mad_scale)
    scale = scale.where(scale > tolerance, std_scale)
    scale = scale.where(scale > tolerance, 1.0)
    return scale.astype(float)


def fit_normalizers(machine_frames: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict:
    """Fit one robust feature center/scale from each machine's healthy rows."""
    if not machine_frames:
        raise ValueError("Cannot fit machine normalizers without machine reference data.")

    normalizers = {}
    for machine_id, frame in machine_frames.items():
        missing = [column for column in feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Machine {machine_id!r} is missing normalization features: {missing}")
        if frame.empty:
            raise ValueError(f"Machine {machine_id!r} has no rows for feature normalization.")

        values = frame[feature_cols].astype(float)
        finite = np.isfinite(values.to_numpy())
        if not finite.all():
            bad = [column for index, column in enumerate(feature_cols) if not finite[:, index].all()]
            raise ValueError(f"Machine {machine_id!r} has non-finite normalization features: {bad}")

        center = values.median()
        scale = _robust_scale(values, center)
        constant = [
            column for column in feature_cols
            if values[column].nunique(dropna=False) <= 1
        ]
        normalizers[str(machine_id)] = {
            "method": METHOD,
            "feature_columns": list(feature_cols),
            "center": center.to_dict(),
            "scale": scale.to_dict(),
            "constant_features": constant,
            "source_rows": int(len(values)),
            "clip": float(config.MACHINE_NORMALIZATION_CLIP),
        }
        print(
            f"[normalization] Machine {machine_id!r}: fitted {len(feature_cols)} "
            f"feature baselines from {len(values)} healthy rows; "
            f"{len(constant)} feature(s) were constant."
        )
    return normalizers


def validate_normalizer(normalizer: dict, feature_cols: list[str], machine_id: str) -> None:
    if not normalizer:
        raise ValueError(
            f"No feature normalizer is bundled for machine {machine_id!r}. "
            "Run balanced commissioning training with this machine included."
        )
    if normalizer.get("method") != METHOD:
        raise ValueError(
            f"Machine {machine_id!r} uses unsupported normalizer method "
            f"{normalizer.get('method')!r}; expected {METHOD!r}. Retrain the model."
        )
    stored_columns = normalizer.get("feature_columns")
    if stored_columns != list(feature_cols):
        raise ValueError(
            f"Machine {machine_id!r} normalizer feature order does not match the model. "
            "Retrain the model instead of mixing artifacts."
        )
    for key in ("center", "scale"):
        values = normalizer.get(key) or {}
        if set(values) != set(feature_cols):
            raise ValueError(f"Machine {machine_id!r} normalizer has incomplete {key} values.")
    scale = np.asarray([normalizer["scale"][column] for column in feature_cols], dtype=float)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Machine {machine_id!r} normalizer contains invalid feature scales.")


def transform(frame: pd.DataFrame, feature_cols: list[str], normalizer: dict,
              machine_id: str) -> pd.DataFrame:
    """Return model-space features while preserving any non-feature columns."""
    validate_normalizer(normalizer, feature_cols, machine_id)
    result = frame.copy()
    center = pd.Series(normalizer["center"], index=feature_cols, dtype=float)
    scale = pd.Series(normalizer["scale"], index=feature_cols, dtype=float)
    normalized = (result[feature_cols].astype(float) - center) / scale
    clip = float(normalizer.get("clip", config.MACHINE_NORMALIZATION_CLIP))
    normalized = normalized.clip(lower=-clip, upper=clip)
    finite = np.isfinite(normalized.to_numpy())
    if not finite.all():
        bad = [column for index, column in enumerate(feature_cols) if not finite[:, index].all()]
        raise ValueError(f"Machine {machine_id!r} normalization produced non-finite features: {bad}")
    result.loc[:, feature_cols] = normalized
    return result


def for_machine(normalizers: dict, machine_id: str) -> dict:
    normalizer = (normalizers or {}).get(str(machine_id))
    if normalizer is None:
        raise ValueError(
            f"Machine {machine_id!r} was not commissioned into this model. "
            "Retrain the balanced shared model with that machine included before scoring it."
        )
    return normalizer
