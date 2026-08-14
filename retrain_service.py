"""Machine-aware, balanced shadow retraining with pre-deploy gates."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import artifact_utils
import config
import db
import feature_engineering
import machine_normalization
import model_registry
import preprocessing
import runtime_config
from isolation_forest import AnomalyScorer


MACHINE_COL = "machine_id"
CANDIDATE_ID_COL = "__candidate_id"
IS_CANDIDATE_COL = "__is_candidate"
RANDOM_STATE = 42
HOLDOUT_FRACTION = artifact_utils.VALIDATION_HOLDOUT_FRACTION
HOLDOUT_MIN_ROWS = artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS


def _retrain_protocol_hash() -> str:
    """Invalidate unchanged-attempt suppression when gate logic changes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _current_reference_features(feature_cols: list[str]) -> pd.DataFrame:
    """Load the exact balanced corpus used by the active shared model.

    New bundles carry this table directly. The timestamp/features.csv path
    remains only as a single-machine compatibility fallback for old bundles.
    """
    stored = artifact_utils.load_reference_features()
    if stored is not None:
        required = {MACHINE_COL, config.COL_TIMESTAMP, *feature_cols}
        missing = required.difference(stored.columns)
        if missing:
            raise ValueError(f"reference_features.csv is missing columns: {sorted(missing)}")
        stored[MACHINE_COL] = stored[MACHINE_COL].astype(str)
        stored[config.COL_TIMESTAMP] = pd.to_datetime(stored[config.COL_TIMESTAMP], utc=True)
        return (
            stored[[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols]]
            .drop_duplicates(subset=[MACHINE_COL, *feature_cols])
            .reset_index(drop=True)
        )

    if not os.path.exists(config.FEATURES_DATA_PATH):
        return pd.DataFrame(columns=[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols])
    legacy = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    timestamps = artifact_utils.load_reference_timestamps()
    if timestamps is None:
        return pd.DataFrame(columns=[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols])
    timestamp_set = set(pd.to_datetime(timestamps, utc=True))
    legacy_timestamps = pd.to_datetime(legacy[config.COL_TIMESTAMP], utc=True)
    selected = legacy.loc[legacy_timestamps.isin(timestamp_set), [config.COL_TIMESTAMP, *feature_cols]].copy()
    selected.insert(0, MACHINE_COL, db.DEFAULT_MACHINE_ID)
    selected[config.COL_TIMESTAMP] = pd.to_datetime(selected[config.COL_TIMESTAMP], utc=True)
    return selected.reset_index(drop=True)


def _current_validation_frames(feature_cols: list[str]) -> dict[str, pd.DataFrame] | None:
    stored = artifact_utils.load_validation_features()
    if stored is None:
        return None
    required = {MACHINE_COL, config.COL_TIMESTAMP, *feature_cols}
    missing = required.difference(stored.columns)
    if missing:
        raise ValueError(f"validation_features.csv is missing columns: {sorted(missing)}")
    stored[MACHINE_COL] = stored[MACHINE_COL].astype(str)
    stored[config.COL_TIMESTAMP] = pd.to_datetime(stored[config.COL_TIMESTAMP], utc=True)
    frames = {
        str(machine_id): frame[[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols]].reset_index(drop=True)
        for machine_id, frame in stored.groupby(MACHINE_COL, sort=True)
    }
    for machine_id, frame in frames.items():
        if frame[config.COL_TIMESTAMP].duplicated().any():
            raise ValueError(
                f"Persisted validation holdout for {machine_id!r} contains duplicate timestamps"
            )
        if not np.isfinite(frame[feature_cols].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Persisted validation holdout for {machine_id!r} contains non-finite features"
            )
    if any(len(frame) < HOLDOUT_MIN_ROWS for frame in frames.values()):
        raise ValueError(
            "Every persisted machine validation holdout must contain at least "
            f"{HOLDOUT_MIN_ROWS} rows"
        )
    if len({len(frame) for frame in frames.values()}) != 1:
        raise ValueError("Persisted validation holdout must contain equal row counts per machine")
    return frames


def _candidate_rows(conn, feature_cols: list[str]) -> list[dict]:
    months = int(runtime_config.get("REFERENCE_WINDOW_MONTHS", 6))
    with conn.cursor() as cur:
        cur.execute(
            """SELECT rc.id, a.machine_id, rc.tick_timestamp, a.feature_vector
               FROM reference_candidates rc
               JOIN alerts a ON a.id=rc.alert_id
               WHERE rc.added_to_reference_at IS NULL
                 AND rc.tick_timestamp >= now() - (%s || ' months')::interval
                 AND a.status='confirmed_normal'
                  AND a.model_version=(
                     SELECT version_id FROM model_versions
                     WHERE status='active' ORDER BY promoted_at DESC NULLS LAST LIMIT 1
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM model_version_candidates mvc
                      JOIN model_versions staged ON staged.version_id=mvc.version_id
                      WHERE mvc.candidate_id=rc.id AND staged.status='shadow'
                  )
               ORDER BY a.machine_id, rc.tick_timestamp""",
            (months,),
        )
        rows = cur.fetchall()

    result = []
    for candidate_id, machine_id, timestamp, vector in rows:
        if not vector:
            continue
        if isinstance(vector, str):
            vector = json.loads(vector)
        if not all(column in vector for column in feature_cols):
            continue
        values = np.array([float(vector[column]) for column in feature_cols], dtype=float)
        if not np.isfinite(values).all():
            continue
        result.append({
            "id": candidate_id,
            MACHINE_COL: str(machine_id),
            "timestamp": timestamp,
            "values": values,
        })
    return result


def _dedup_within_machine(rows: list[dict], feature_cols: list[str], normalizers: dict) -> list[dict]:
    """Cosine-deduplicate in current machine-relative model space."""
    if not rows:
        return []
    threshold = float(runtime_config.get("REFERENCE_COSINE_SIMILARITY", 0.98))
    hours = float(runtime_config.get("REFERENCE_DEDUP_WINDOW_HOURS", 24))
    accepted_by_machine: dict[str, list[dict]] = {}
    accepted = []
    for item in rows:
        machine_id = item[MACHINE_COL]
        one = pd.DataFrame([item["values"]], columns=feature_cols)
        item_values = machine_normalization.transform(
            one,
            feature_cols,
            machine_normalization.for_machine(normalizers, machine_id),
            machine_id,
        )[feature_cols].to_numpy(dtype=float)[0]
        machine_rows = accepted_by_machine.setdefault(item[MACHINE_COL], [])
        duplicate = False
        for previous in reversed(machine_rows):
            if (item["timestamp"] - previous["timestamp"]).total_seconds() > hours * 3600:
                break
            previous_values = previous["_dedup_values"]
            denominator = np.linalg.norm(item_values) * np.linalg.norm(previous_values)
            similarity = float(np.dot(item_values, previous_values) / denominator) if denominator else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if not duplicate:
            item["_dedup_values"] = item_values
            machine_rows.append(item)
            accepted.append(item)
    return accepted


def _candidate_frame(rows: list[dict], feature_cols: list[str]) -> pd.DataFrame:
    records = []
    for item in rows:
        record = {
            MACHINE_COL: item[MACHINE_COL],
            config.COL_TIMESTAMP: item["timestamp"],
            CANDIDATE_ID_COL: item["id"],
            IS_CANDIDATE_COL: True,
        }
        record.update(dict(zip(feature_cols, item["values"])))
        records.append(record)
    return pd.DataFrame(records)


def _balanced_reference(base: pd.DataFrame, candidates: pd.DataFrame, feature_cols: list[str],
                        reserve_holdout: bool = True):
    base = base.copy()
    base[CANDIDATE_ID_COL] = None
    base[IS_CANDIDATE_COL] = False

    # Candidate rows come first so an exact duplicate of an older baseline
    # row retains the auditable candidate identity and can replace that row.
    combined = pd.concat([candidates, base], ignore_index=True)
    combined = combined.drop_duplicates(subset=[MACHINE_COL, *feature_cols], keep="first")
    if combined.empty:
        raise ValueError("No reference features are available for shadow retraining")

    counts = {str(machine_id): len(frame) for machine_id, frame in combined.groupby(MACHINE_COL)}
    minimum = max(20, len(feature_cols) + 1)
    too_small = {machine_id: count for machine_id, count in counts.items() if count < minimum}
    if too_small:
        raise ValueError(
            f"Reference set is too small for these machines (need at least {minimum} each): {too_small}"
        )

    base_counts = base.groupby(MACHINE_COL).size().to_dict() if not base.empty else {}
    new_machines = sorted(set(counts).difference(str(machine_id) for machine_id in base_counts))
    if new_machines:
        raise ValueError(
            "New machines cannot join through confirmed-normal alert samples because those "
            "are not a complete commissioning baseline. Rerun balanced initial training "
            f"with these machines included: {new_machines}"
        )

    rows_per_machine = min(counts.values())
    holdout_rows = (
        max(HOLDOUT_MIN_ROWS, int(np.ceil(rows_per_machine * HOLDOUT_FRACTION)))
        if reserve_holdout else 0
    )
    minimum_fit = max(20, len(feature_cols) + 1)
    if rows_per_machine < holdout_rows + minimum_fit:
        raise ValueError(
            f"Balanced reference needs at least {holdout_rows + minimum_fit} rows per machine "
            f"to reserve {holdout_rows} independent validation rows and {minimum_fit} fit rows; "
            f"only {rows_per_machine} are available."
        )
    maximum_new_rows = rows_per_machine - holdout_rows
    selected_frames = {}
    selected_candidate_ids = []
    for machine_id, machine_frame in combined.groupby(MACHINE_COL, sort=True):
        machine_frame = machine_frame.reset_index(drop=True)
        new_rows = machine_frame[machine_frame[IS_CANDIDATE_COL]].copy()
        old_rows = machine_frame[~machine_frame[IS_CANDIDATE_COL]].copy()
        if reserve_holdout and len(old_rows) < holdout_rows:
            raise ValueError(
                f"Machine {machine_id!r} has only {len(old_rows)} prior baseline rows; "
                f"{holdout_rows} are required for independent validation."
            )

        chosen_new = (
            new_rows.sample(n=maximum_new_rows, random_state=RANDOM_STATE)
            if len(new_rows) > maximum_new_rows else new_rows
        )
        remaining = rows_per_machine - len(chosen_new)
        ordered_old = old_rows.sort_values(config.COL_TIMESTAMP)
        reserved_holdout = ordered_old.tail(holdout_rows) if holdout_rows else ordered_old.iloc[0:0]
        additional_rows = remaining - holdout_rows
        fit_old_pool = ordered_old.drop(index=reserved_holdout.index)
        chosen_fit_old = (
            fit_old_pool.sample(n=additional_rows, random_state=RANDOM_STATE)
            if len(fit_old_pool) > additional_rows else fit_old_pool
        )
        chosen_old = pd.concat([chosen_fit_old, reserved_holdout], ignore_index=True)
        selected = pd.concat([chosen_new, chosen_old], ignore_index=True)
        selected = selected.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        selected_frames[str(machine_id)] = selected
        selected_candidate_ids.extend(int(value) for value in chosen_new[CANDIDATE_ID_COL].dropna())

    return selected_frames, counts, rows_per_machine, selected_candidate_ids


def _training_holdout_split(selected_frames: dict[str, pd.DataFrame], feature_cols: list[str]):
    """Reserve independent confirmed-normal evidence without dropping new reviews.

    The holdout is taken only from the previously accepted baseline. Every new
    human-confirmed candidate remains in the fit set, so promotion never marks
    fresh evidence consumed without teaching the model from it. The newest
    eligible baseline rows form a forward-looking holdout for each machine.
    """
    fraction = HOLDOUT_FRACTION
    training_frames = {}
    holdout_frames = {}
    minimum_fit = max(20, len(feature_cols) + 1)
    for machine_id, frame in selected_frames.items():
        ordered = frame.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        historical = ordered[~ordered[IS_CANDIDATE_COL]].copy()
        holdout_rows = max(HOLDOUT_MIN_ROWS, int(np.ceil(len(ordered) * fraction)))
        if len(historical) < holdout_rows or len(ordered) - holdout_rows < minimum_fit:
            raise ValueError(
                f"Machine {machine_id!r} needs at least {holdout_rows} prior baseline rows "
                f"for independent validation and {minimum_fit} remaining fit rows; "
                f"available prior/total rows are {len(historical)}/{len(ordered)}."
            )
        holdout_indices = historical.tail(holdout_rows).index
        holdout = ordered.loc[holdout_indices].copy().reset_index(drop=True)
        training = ordered.drop(index=holdout_indices).reset_index(drop=True)
        training_frames[machine_id] = training
        holdout_frames[machine_id] = holdout
    return training_frames, holdout_frames


def _risk(scorer, features):
    if len(features) == 0:
        return np.array([])
    score = scorer.score(features)
    health = scorer.health_from_score(score)
    return np.clip(1.0 - health / 100.0, 0.0, 1.0)


def _raw_feature_window(conn, machine_id, start, end, feature_cols):
    # Fetch the exact preceding row count. An elapsed-time estimate can be too
    # short when the production source has gaps or a lower real sample rate.
    start_ts = pd.Timestamp(start).to_pydatetime()
    rows = db.fetch_rows_before(
        conn,
        db.get_table_name(),
        start_ts,
        config.WINDOW_SIZE,
        machine_id=machine_id,
    )
    rows += db.fetch_rows_between(
        conn,
        db.get_table_name(),
        start_ts,
        pd.Timestamp(end).to_pydatetime(),
        machine_id=machine_id,
    )
    if not rows:
        return pd.DataFrame(columns=feature_cols)

    db_columns = db.get_db_columns()
    raw = preprocessing.clean_data(db.canonical_sensor_frame(rows, db_columns))
    featured = feature_engineering.create_features(raw, verbose=False)
    timestamps = pd.to_datetime(featured[config.COL_TIMESTAMP], utc=True)
    mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
    return featured.loc[mask, [config.COL_TIMESTAMP, *feature_cols]].reset_index(drop=True)


def _regression_feature_windows(conn, feature_cols):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, machine_id, description, lower(timestamp_range), upper(timestamp_range),
                      minimum_anomaly_risk,target_timestamp
               FROM regression_tests WHERE disabled_at IS NULL ORDER BY id"""
        )
        tests = cur.fetchall()
    windows = []
    for regression_id, machine_id, description, start, end, minimum, target_timestamp in tests:
        frame = _raw_feature_window(conn, str(machine_id), start, end, feature_cols)
        evaluation = "window_peak"
        if target_timestamp is not None and not frame.empty:
            timestamps = pd.to_datetime(frame[config.COL_TIMESTAMP], utc=True)
            target = pd.Timestamp(target_timestamp)
            frame = frame.loc[timestamps == target]
            evaluation = "exact_flagged_prediction"
        windows.append((
            regression_id,
            str(machine_id),
            description,
            frame[feature_cols].reset_index(drop=True),
            float(minimum),
            evaluation,
        ))
    return windows


def _active_training_machine_ids():
    path = os.path.join(config.ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        machine_ids = json.load(handle).get("training_machine_ids")
    return {str(machine_id) for machine_id in machine_ids} if machine_ids else None


def attempt_signature(conn) -> str:
    """Fingerprint the exact pending evidence and policy for retry suppression."""
    months = int(runtime_config.get("REFERENCE_WINDOW_MONTHS", 6))
    with conn.cursor() as cur:
        cur.execute(
            """SELECT rc.id
               FROM reference_candidates rc JOIN alerts a ON a.id=rc.alert_id
               WHERE rc.added_to_reference_at IS NULL
                 AND rc.tick_timestamp >= now() - (%s || ' months')::interval
                 AND a.status='confirmed_normal'
                 AND a.feature_vector IS NOT NULL
                 AND a.model_version=(
                     SELECT version_id FROM model_versions
                     WHERE status='active' ORDER BY promoted_at DESC NULLS LAST LIMIT 1
                 )
                 AND NOT EXISTS (
                     SELECT 1 FROM model_version_candidates mvc
                     JOIN model_versions staged ON staged.version_id=mvc.version_id
                     WHERE mvc.candidate_id=rc.id AND staged.status='shadow'
                 )
               ORDER BY rc.id""",
            (months,),
        )
        candidate_ids = [int(row[0]) for row in cur.fetchall()]
        cur.execute(
            """SELECT id,machine_id,lower(timestamp_range),upper(timestamp_range),
                      minimum_anomaly_risk,target_prediction_id,target_timestamp
               FROM regression_tests
               WHERE disabled_at IS NULL
               ORDER BY id"""
        )
        regression_tests = [list(row) for row in cur.fetchall()]
    material_policy_keys = [
        key for key in runtime_config.TRAINING_KEYS
        if key not in {"RETRAIN_CHECK_INTERVAL_MINUTES", "RETRAIN_RETRY_COOLDOWN_HOURS"}
    ]
    payload = {
        "active_version": model_registry.active_version(conn),
        "candidate_ids": candidate_ids,
        "regression_tests": regression_tests,
        "training_policy": {
            key: runtime_config.get(key) for key in material_policy_keys
        },
        "health_cut": runtime_config.get(
            "MAINTENANCE_HEALTH_INSPECT", config.MAINTENANCE_HEALTH_INSPECT
        ),
        "pipeline_hash": artifact_utils.config_hash(),
        "retrain_protocol_hash": _retrain_protocol_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def should_retrain(conn) -> tuple[bool, dict]:
    months = int(runtime_config.get("REFERENCE_WINDOW_MONTHS", 6))
    with conn.cursor() as cur:
        cur.execute(
            """SELECT a.machine_id, count(*), min(rc.created_at)
               FROM reference_candidates rc JOIN alerts a ON a.id=rc.alert_id
               WHERE rc.added_to_reference_at IS NULL
                 AND rc.tick_timestamp >= now() - (%s || ' months')::interval
                 AND a.status='confirmed_normal'
                 AND a.feature_vector IS NOT NULL
                  AND a.model_version=(
                     SELECT version_id FROM model_versions
                     WHERE status='active' ORDER BY promoted_at DESC NULLS LAST LIMIT 1
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM model_version_candidates mvc
                      JOIN model_versions staged ON staged.version_id=mvc.version_id
                      WHERE mvc.candidate_id=rc.id AND staged.status='shadow'
                  )
               GROUP BY a.machine_id ORDER BY a.machine_id""",
            (months,),
        )
        rows = cur.fetchall()
    batch = int(runtime_config.get("RETRAIN_BATCH_SIZE", 50))
    cap = int(runtime_config.get("RETRAIN_TIME_CAP_DAYS", 30))
    now = dt.datetime.now(dt.timezone.utc)
    trained_machine_ids = _active_training_machine_ids()
    by_machine = {}
    due = False
    for machine_id, count, oldest in rows:
        machine_id = str(machine_id)
        age_days = (now - oldest).total_seconds() / 86400 if oldest else 0
        eligible = trained_machine_ids is None or machine_id in trained_machine_ids
        by_machine[machine_id] = {
            "pending": count,
            "oldest_age_days": age_days,
            "eligible": eligible,
            "requires_commissioning": not eligible,
        }
        due = due or (eligible and (count >= batch or (count > 0 and age_days >= cap)))
    for machine_id in sorted(trained_machine_ids or ()):
        by_machine.setdefault(machine_id, {
            "pending": 0,
            "oldest_age_days": 0,
            "eligible": True,
            "requires_commissioning": False,
        })
    total = sum(item["pending"] for item in by_machine.values())
    eligible_total = sum(item["pending"] for item in by_machine.values() if item["eligible"])
    oldest_age_days = max((item["oldest_age_days"] for item in by_machine.values()), default=0)
    return due, {
        "pending": eligible_total,
        "candidate_count": total,
        "eligible_pending": eligible_total,
        "pending_by_machine": by_machine,
        "batch_size_per_machine": batch,
        "batch_size": batch,
        "oldest_age_days": oldest_age_days,
        "time_cap_days": cap,
    }


def _machine_scorer(model, calibration):
    return AnomalyScorer.from_calibration(model, calibration)


def _populate_shadow_bundle(
    target: Path,
    shadow,
    feature_cols,
    metadata,
    selected_frames,
    validation_frames,
    machine_calibrations,
    machine_feature_normalizers,
):
    joblib.dump(shadow.model, target / "isolation_forest.pkl")
    (target / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (target / "calibration.json").write_text(
        json.dumps(artifact_utils.to_json_safe(shadow.calibration()), indent=2, allow_nan=False)
    )
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))

    clean_frames = {
        machine_id: frame[[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols]].copy()
        for machine_id, frame in selected_frames.items()
    }
    reference = pd.concat(clean_frames.values(), ignore_index=True)
    reference.to_csv(target / "reference_features.csv", index=False)
    validation = pd.concat(
        {
            machine_id: frame[[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols]].copy()
            for machine_id, frame in validation_frames.items()
        }.values(),
        ignore_index=True,
    )
    validation.to_csv(target / "validation_features.csv", index=False)
    reference_rows = [
        {MACHINE_COL: machine_id, "timestamp": pd.Timestamp(timestamp).isoformat()}
        for machine_id, frame in clean_frames.items()
        for timestamp in frame[config.COL_TIMESTAMP]
    ]
    (target / "reference_rows.json").write_text(json.dumps(reference_rows, indent=2))
    (target / "reference_timestamps.json").write_text(
        json.dumps([row["timestamp"] for row in reference_rows], indent=2)
    )
    (target / "machine_calibrations.json").write_text(
        json.dumps(artifact_utils.to_json_safe(machine_calibrations), indent=2, allow_nan=False)
    )
    (target / "machine_feature_normalizers.json").write_text(
        json.dumps(
            artifact_utils.to_json_safe(machine_feature_normalizers),
            indent=2,
            allow_nan=False,
        )
    )
    return reference_rows


def _write_shadow_bundle(
    target: Path,
    shadow,
    feature_cols,
    metadata,
    selected_frames,
    validation_frames,
    machine_calibrations,
    machine_feature_normalizers,
):
    """Create a complete shadow directory or remove the partial bundle."""
    target.mkdir(parents=True, exist_ok=False)
    try:
        return _populate_shadow_bundle(
            target,
            shadow,
            feature_cols,
            metadata,
            selected_frames,
            validation_frames,
            machine_calibrations,
            machine_feature_normalizers,
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def run_shadow_retrain(conn, force: bool = False) -> dict:
    due, status = should_retrain(conn)
    if not force and not due:
        return {"started": False, **status}

    active_scorer, feature_cols, active_meta = artifact_utils.load_artifacts()
    active_normalizers = artifact_utils.load_machine_feature_normalizers()
    has_machine_reference = os.path.exists(
        os.path.join(config.ARTIFACTS_DIR, "reference_features.csv")
    )
    if not has_machine_reference and len(active_meta.get("training_machine_ids", [])) > 1:
        raise ValueError(
            "The active pooled model predates machine-aware reference_features.csv. "
            "Run the current initial trainer once before enabling multi-machine retraining."
        )
    base = _current_reference_features(feature_cols)
    persisted_holdout = _current_validation_frames(feature_cols)
    base_machine_ids = {str(machine_id) for machine_id in base[MACHINE_COL].unique()}
    if persisted_holdout is not None and set(persisted_holdout) != base_machine_ids:
        raise ValueError(
            "Persisted validation holdout machines do not match the active fit reference; "
            "rerun commissioning training before retraining."
        )
    if persisted_holdout is not None:
        fit_identities = set(zip(
            base[MACHINE_COL].astype(str),
            pd.to_datetime(base[config.COL_TIMESTAMP], utc=True),
        ))
        holdout_identities = {
            (machine_id, timestamp)
            for machine_id, frame in persisted_holdout.items()
            for timestamp in pd.to_datetime(frame[config.COL_TIMESTAMP], utc=True)
        }
        overlap = fit_identities.intersection(holdout_identities)
        if overlap:
            raise ValueError(
                "The active fit reference overlaps its validation holdout; "
                f"found {len(overlap)} duplicated machine/timestamp identities. "
                "Rerun commissioning training before retraining."
            )
    raw_candidate_rows = _candidate_rows(conn, feature_cols)
    ignored_new_machine_candidates = {}
    for item in raw_candidate_rows:
        if item[MACHINE_COL] not in base_machine_ids:
            ignored_new_machine_candidates[item[MACHINE_COL]] = (
                ignored_new_machine_candidates.get(item[MACHINE_COL], 0) + 1
            )
    all_candidate_rows = [
        item for item in raw_candidate_rows if item[MACHINE_COL] in base_machine_ids
    ]
    deduplicated_rows = _dedup_within_machine(
        all_candidate_rows, feature_cols, active_normalizers
    )
    if not deduplicated_rows:
        detail = (
            f" New-machine candidates require initial commissioning training: "
            f"{ignored_new_machine_candidates}."
            if ignored_new_machine_candidates else ""
        )
        raise ValueError(f"No eligible, non-duplicate confirmed-normal candidates are available.{detail}")
    candidates = _candidate_frame(deduplicated_rows, feature_cols)
    selected_frames, available_counts, rows_per_machine, selected_ids = (
        _balanced_reference(
            base, candidates, feature_cols,
            reserve_holdout=persisted_holdout is None,
        )
    )
    if persisted_holdout is None:
        training_frames, holdout_frames = _training_holdout_split(selected_frames, feature_cols)
        holdout_lineage = "created_from_legacy_active_reference"
    else:
        training_frames, holdout_frames = selected_frames, persisted_holdout
        holdout_lineage = "preserved_from_active_bundle"
    pooled = pd.concat(training_frames.values(), ignore_index=True)

    proposed_normalizers = machine_normalization.fit_normalizers(
        training_frames, feature_cols
    )
    normalized_training_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, proposed_normalizers[machine_id], machine_id
        )
        for machine_id, frame in training_frames.items()
    }
    normalized_holdout_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, proposed_normalizers[machine_id], machine_id
        )
        for machine_id, frame in holdout_frames.items()
    }
    normalized_pool = pd.concat(normalized_training_frames.values(), ignore_index=True)
    shadow = AnomalyScorer().fit(normalized_pool[feature_cols])
    machine_calibrations = {}
    shadow_machine_scorers = {}
    for machine_id, frame in normalized_training_frames.items():
        local = AnomalyScorer()
        local.model = shadow.model
        local.calibrate(frame[feature_cols])
        machine_calibrations[machine_id] = {
            "calibration": local.calibration(),
            "source_rows": len(frame),
        }
        shadow_machine_scorers[machine_id] = local

    # Gate 1: each machine must independently preserve its confirmed-normal
    # false-positive behavior. A good aggregate cannot hide one bad machine.
    health_cut = float(runtime_config.get("MAINTENANCE_HEALTH_INSPECT", config.MAINTENANCE_HEALTH_INSPECT))
    max_fp_increase = float(runtime_config.get("RETRAIN_MAX_FP_RATE_INCREASE", 0.02))
    active_version = model_registry.active_version(conn)
    fp_by_machine = {}
    active_flags = []
    shadow_flags = []
    for machine_id, frame in holdout_frames.items():
        active_calibration = (
            model_registry.machine_calibration(conn, active_version, machine_id)
            if active_version else None
        )
        if active_calibration is None:
            raise ValueError(
                f"Active model {active_version!r} has no automatic condition anchor "
                f"for machine {machine_id!r}; rerun commissioning training."
            )
        active_local = _machine_scorer(active_scorer.model, active_calibration)
        active_features = machine_normalization.transform(
            frame,
            feature_cols,
            machine_normalization.for_machine(active_normalizers, machine_id),
            machine_id,
        )[feature_cols]
        proposed_features = normalized_holdout_frames[machine_id][feature_cols]
        current_flags = active_local.health_from_score(active_local.score(active_features)) <= health_cut
        proposed_flags = shadow_machine_scorers[machine_id].health_from_score(
            shadow_machine_scorers[machine_id].score(proposed_features)
        ) <= health_cut
        active_fp = float(np.mean(current_flags))
        shadow_fp = float(np.mean(proposed_flags))
        passed = shadow_fp <= active_fp + max_fp_increase
        fp_by_machine[machine_id] = {
            "holdout_rows": len(frame),
            "active_reference_fp": active_fp,
            "shadow_reference_fp": shadow_fp,
            "maximum_allowed_increase": max_fp_increase,
            "pass": passed,
        }
        active_flags.extend(current_flags.tolist())
        shadow_flags.extend(proposed_flags.tolist())
    fp_pass = all(item["pass"] for item in fp_by_machine.values())

    # Gate 2: reconstruct each permanent false-negative window from that
    # machine's live raw data and score it with that machine's new calibration.
    regression = []
    regression_pass = True
    for regression_id, machine_id, description, features, minimum, evaluation in _regression_feature_windows(conn, feature_cols):
        if features.empty:
            regression.append({
                "id": regression_id,
                MACHINE_COL: machine_id,
                "description": description,
                "evaluation": evaluation,
                "pass": False,
                "reason": "no matching machine feature rows",
            })
            regression_pass = False
            continue
        scorer = shadow_machine_scorers.get(machine_id)
        if scorer is None:
            regression.append({
                "id": regression_id,
                MACHINE_COL: machine_id,
                "description": description,
                "evaluation": evaluation,
                "pass": False,
                "reason": "machine is absent from the proposed balanced reference",
            })
            regression_pass = False
            continue
        normalized_features = machine_normalization.transform(
            features,
            feature_cols,
            machine_normalization.for_machine(proposed_normalizers, machine_id),
            machine_id,
        )[feature_cols]
        observed_risk = float(np.max(_risk(scorer, normalized_features)))
        passed = observed_risk >= minimum
        regression.append({
            "id": regression_id,
            MACHINE_COL: machine_id,
            "description": description,
            "evaluation": evaluation,
            "observed_anomaly_risk": observed_risk,
            "minimum": minimum,
            "pass": passed,
        })
        regression_pass &= passed

    report = {
        "reference_rows": len(pooled),
        "holdout_rows": sum(len(frame) for frame in holdout_frames.values()),
        "holdout_fraction": HOLDOUT_FRACTION,
        "holdout_lineage": holdout_lineage,
        "machines": sorted(training_frames),
        "available_reference_rows_by_machine": available_counts,
        "balanced_rows_per_machine_before_holdout": rows_per_machine,
        "model_fit_rows_by_machine": {key: len(value) for key, value in training_frames.items()},
        "holdout_rows_by_machine": {key: len(value) for key, value in holdout_frames.items()},
        "processed_candidates": len(all_candidate_rows),
        "ignored_new_machine_candidates": ignored_new_machine_candidates,
        "deduplicated_candidates": len(deduplicated_rows),
        "selected_candidates": len(selected_ids),
        "active_reference_fp": float(np.mean(active_flags)),
        "shadow_reference_fp": float(np.mean(shadow_flags)),
        "false_positive_evaluation": "independent_machine_balanced_holdout",
        "retrain_protocol_hash": _retrain_protocol_hash(),
        "reference_fp_by_machine": fp_by_machine,
        "reference_fp_gate_pass": fp_pass,
        "maximum_fp_rate_increase": max_fp_increase,
        "regression_tests": regression,
        "regression_gate_pass": regression_pass,
    }
    accepted = fp_pass and regression_pass
    report["passed"] = accepted
    version_id = model_registry.new_version_id()
    metadata = dict(active_meta)
    metadata.update({
        "version_id": version_id,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_features": len(feature_cols),
        "pipeline_hash": artifact_utils.config_hash(),
        "retrain_protocol_hash": _retrain_protocol_hash(),
        "human_confirmed_retrain": True,
        "training_mode": "balanced_pooled_retrain",
        "training_machine_ids": sorted(training_frames),
        "available_reference_rows_per_machine": available_counts,
        "model_fit_rows_per_machine": {key: len(value) for key, value in training_frames.items()},
        "validation_rows_per_machine": {key: len(value) for key, value in holdout_frames.items()},
        "validation_holdout_lineage": holdout_lineage,
        "balance_method": "equal_rows_candidate_priority_deterministic_sample",
        "model_feature_space": machine_normalization.METHOD,
    })
    target = Path(model_registry.bundle_path(version_id))
    reference_rows = _write_shadow_bundle(
        target,
        shadow,
        feature_cols,
        metadata,
        training_frames,
        holdout_frames,
        machine_calibrations,
        proposed_normalizers,
    )
    signature_source = [f"{row[MACHINE_COL]}\0{row['timestamp']}" for row in reference_rows]
    signature = model_registry.reference_signature(signature_source)

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_versions(version_id,artifact_path,reference_signature,status,validation_report)
               VALUES(%s,%s,%s,%s,%s::jsonb)""",
            (version_id, str(target), signature, "shadow" if accepted else "rejected", json.dumps(report)),
        )
        if accepted:
            for machine_id, item in machine_calibrations.items():
                cur.execute(
                    """INSERT INTO model_calibrations
                       (version_id,machine_id,calibration,source_rows,source_description,created_by)
                       VALUES(%s,%s,%s::jsonb,%s,%s,'auto-retrain') RETURNING id""",
                    (
                        version_id,
                        machine_id,
                        json.dumps(artifact_utils.to_json_safe(item["calibration"]), allow_nan=False),
                        item["source_rows"],
                        "Automatic robust machine condition anchor from shadow retraining",
                    ),
                )
                calibration_id = cur.fetchone()[0]
                cur.execute(
                    """INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
                       VALUES(%s,%s,%s)""",
                    (machine_id, version_id, calibration_id),
                )
            # Stage every processed candidate against this shadow. Promotion
            # consumes them atomically; deleting the shadow releases them.
            deduplicated_id_set = {int(item["id"]) for item in deduplicated_rows}
            duplicate_ids = [
                int(item["id"]) for item in all_candidate_rows
                if int(item["id"]) not in deduplicated_id_set
            ]
            # Selected candidates were learned directly. Near-duplicates are
            # represented by a selected equivalent and can also be consumed.
            # Diverse candidates omitted only because another machine limited
            # the balanced pool remain pending for a future retrain.
            staged_ids = sorted(set(selected_ids + duplicate_ids))
            for candidate_id in staged_ids:
                cur.execute(
                    """INSERT INTO model_version_candidates(version_id,candidate_id)
                       VALUES(%s,%s) ON CONFLICT DO NOTHING""",
                    (version_id, candidate_id),
                )
    # The persistent job runner commits the model row, staged candidates, and
    # terminal job result together. Keeping this transaction open prevents a
    # briefly visible model version with no matching completed job.
    return {
        "started": True,
        "version_id": version_id,
        "status": "shadow" if accepted else "rejected",
        "validation": report,
    }
