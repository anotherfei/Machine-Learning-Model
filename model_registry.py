"""Versioned model-bundle registry and atomic active-version switching."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import artifact_utils
import config

BUNDLES_DIR = os.path.join(config.ARTIFACTS_DIR, "versions")
BUNDLE_FILES = (
    "isolation_forest.pkl",
    "feature_columns.json",
    "calibration.json",
    "metadata.json",
    "reference_timestamps.json",
    "reference_rows.json",
    "reference_features.csv",
    "validation_features.csv",
    "machine_calibrations.json",
    "machine_feature_normalizers.json",
)

REQUIRED_BUNDLE_FILES = (
    "isolation_forest.pkl",
    "feature_columns.json",
    "calibration.json",
    "metadata.json",
    "machine_calibrations.json",
    "machine_feature_normalizers.json",
)
MODEL_LIFECYCLE_LOCK_ID = 724_913_208


def new_version_id(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    # Retraining is single-flight, but bootstrap registration and an operator
    # action can still land in the same second. Microseconds keep filesystem
    # bundle names and the database primary key collision-free without relying
    # on a retry after partial work has begun.
    return "v" + now.strftime("%Y-%m-%d-%H%M%S-%f")


def bundle_path(version_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", version_id) or ".." in version_id:
        raise ValueError("Invalid model version identifier")
    return os.path.join(BUNDLES_DIR, version_id)


def reference_signature(reference_items) -> str:
    """Hash timestamp-only or machine-aware reference identities."""
    payload = "\n".join(sorted(str(x) for x in reference_items)).encode()
    return hashlib.sha256(payload).hexdigest()


def snapshot_current(version_id: str) -> str:
    target = Path(bundle_path(version_id))
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name in BUNDLE_FILES:
            src = Path(config.ARTIFACTS_DIR) / name
            if src.exists():
                shutil.copy2(src, target / name)
        meta = target / "metadata.json"
        if meta.exists():
            data = json.loads(meta.read_text())
            data["version_id"] = version_id
            meta.write_text(json.dumps(data, indent=2))
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return str(target)


def install_bundle(version_id: str) -> None:
    src_dir = Path(bundle_path(version_id))
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing bundle {src_dir}")
    required = REQUIRED_BUNDLE_FILES
    missing = [name for name in required if not (src_dir / name).exists()]
    if missing:
        raise ValueError(f"Bundle {version_id} missing required files: {missing}")
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    for name in BUNDLE_FILES:
        src = src_dir / name
        destination = Path(config.ARTIFACTS_DIR) / name
        if src.exists():
            tmp = Path(config.ARTIFACTS_DIR) / (name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, destination)
        elif name not in required and destination.exists():
            # Optional provenance files belong to a specific
            # version and must not leak from the previously installed bundle.
            destination.unlink()


def active_version(conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT version_id FROM model_versions WHERE status='active' LIMIT 1")
        row = cur.fetchone()
    return row[0] if row else None


def promote(conn, version_id: str, promoted_by: str) -> None:
    with conn.cursor() as cur:
        # Serialize the database decision and filesystem installation across
        # every API process. Row locks alone do not protect two different
        # version rows from being promoted at the same time.
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (MODEL_LIFECYCLE_LOCK_ID,))
        cur.execute(
            "SELECT status,validation_report FROM model_versions WHERE version_id=%s FOR UPDATE",
            (version_id,),
        )
        target = cur.fetchone()
        if not target or target[0] not in ("shadow", "retired", "active"):
            raise ValueError(f"Unknown or rejected model version: {version_id}")
        report = target[1]
        if isinstance(report, str):
            report = json.loads(report)
        if target[0] == "shadow" and (not isinstance(report, dict) or report.get("passed") is not True):
            raise ValueError(f"Shadow model {version_id} has not passed every validation gate")
        if target[0] == "shadow" and not (Path(bundle_path(version_id)) / "validation_features.csv").is_file():
            raise ValueError(
                f"Shadow model {version_id} is missing its independent validation_features.csv artifact"
            )
        cur.execute("SELECT version_id FROM model_versions WHERE status='active' LIMIT 1")
        previous_row = cur.fetchone()
        previous_version = previous_row[0] if previous_row else None

    # Validate the database state before touching the installed artifact set.
    # Installation can still fail midway at the filesystem boundary, so it is
    # part of the guarded block and restores the prior active bundle on any
    # failure before database activation commits.
    try:
        install_bundle(version_id)
        with conn.cursor() as cur:
            cur.execute("UPDATE model_versions SET status='retired' WHERE status='active' AND version_id<>%s", (version_id,))
            cur.execute("""UPDATE model_versions SET status='active', promoted_at=now(), promoted_by=%s
                           WHERE version_id=%s AND status IN ('shadow','retired','active')""", (promoted_by, version_id))
            if cur.rowcount != 1:
                raise ValueError(f"Model version became unavailable during promotion: {version_id}")
            cur.execute(
                """UPDATE reference_candidates rc SET added_to_reference_at=now()
                   FROM model_version_candidates mvc
                   WHERE mvc.version_id=%s AND mvc.candidate_id=rc.id
                     AND rc.added_to_reference_at IS NULL""",
                (version_id,),
            )
            cur.execute("NOTIFY model_changed, %s", (version_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        if previous_version:
            try:
                install_bundle(previous_version)
            except Exception:
                pass
        raise


def machine_calibration(conn, version_id: str, machine_id: str):
    """Return the automatic robust condition anchor for one machine."""
    with conn.cursor() as cur:
        cur.execute("""SELECT mc.calibration FROM machine_model_calibrations mmc
                       JOIN model_calibrations mc ON mc.id=mmc.calibration_id
                       WHERE mmc.machine_id=%s AND mmc.version_id=%s""",(machine_id,version_id))
        row=cur.fetchone()
    return row[0] if row else None

def delete_version(conn, version_id: str) -> None:
    """
    Deletes a non-active model bundle: the DB row (model_calibrations rows
    for it cascade automatically) plus its artifact directory. The active
    model can never be deleted here — promote a different version first —
    because it is the model currently serving predictions.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (MODEL_LIFECYCLE_LOCK_ID,))
        cur.execute("SELECT status, artifact_path FROM model_versions WHERE version_id=%s FOR UPDATE", (version_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Unknown model version: {version_id}")
        status, artifact_path = row
        if status == "active":
            raise ValueError("Cannot delete the active model version — promote a different version first.")
        resolved_artifact = Path(artifact_path).resolve() if artifact_path else None
        bundles_root = Path(BUNDLES_DIR).resolve()
        if resolved_artifact and (resolved_artifact == bundles_root or not resolved_artifact.is_relative_to(bundles_root)):
            raise ValueError(f"Refusing to delete model artifacts outside the version registry: {artifact_path}")
        cur.execute("DELETE FROM model_versions WHERE version_id=%s", (version_id,))
    conn.commit()
    if resolved_artifact and resolved_artifact.is_dir():
        shutil.rmtree(resolved_artifact, ignore_errors=True)
