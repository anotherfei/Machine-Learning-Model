"""Machine-aware, balanced shadow retraining with pre-deploy gates."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import artifact_utils
import config
import db
import feature_engineering
import model_registry
import preprocessing
import runtime_config
from isolation_forest import AnomalyScorer


MACHINE_COL = "machine_id"
CANDIDATE_ID_COL = "__candidate_id"
IS_CANDIDATE_COL = "__is_candidate"
RANDOM_STATE = 42


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
        return stored[[MACHINE_COL, config.COL_TIMESTAMP, *feature_cols]].reset_index(drop=True)

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


def _dedup_within_machine(rows: list[dict]) -> list[dict]:
    """Cosine-deduplicate only against recent rows from the same machine."""
    if not rows:
        return []
    threshold = float(runtime_config.get("REFERENCE_COSINE_SIMILARITY", 0.98))
    hours = float(runtime_config.get("REFERENCE_DEDUP_WINDOW_HOURS", 24))
    accepted_by_machine: dict[str, list[dict]] = {}
    accepted = []
    for item in rows:
        machine_rows = accepted_by_machine.setdefault(item[MACHINE_COL], [])
        duplicate = False
        for previous in reversed(machine_rows):
            if (item["timestamp"] - previous["timestamp"]).total_seconds() > hours * 3600:
                break
            denominator = np.linalg.norm(item["values"]) * np.linalg.norm(previous["values"])
            similarity = float(np.dot(item["values"], previous["values"]) / denominator) if denominator else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if not duplicate:
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


def _balanced_reference(base: pd.DataFrame, candidates: pd.DataFrame, feature_cols: list[str]):
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
    selected_frames = {}
    calibration_frames = {}
    selected_candidate_ids = []
    for machine_id, machine_frame in combined.groupby(MACHINE_COL, sort=True):
        machine_frame = machine_frame.reset_index(drop=True)
        calibration_frames[str(machine_id)] = machine_frame
        new_rows = machine_frame[machine_frame[IS_CANDIDATE_COL]].copy()
        old_rows = machine_frame[~machine_frame[IS_CANDIDATE_COL]].copy()

        if len(new_rows) > rows_per_machine:
            chosen_new = new_rows.sample(n=rows_per_machine, random_state=RANDOM_STATE)
            chosen_old = old_rows.iloc[0:0]
        else:
            chosen_new = new_rows
            remaining = rows_per_machine - len(chosen_new)
            chosen_old = (
                old_rows.sample(n=remaining, random_state=RANDOM_STATE)
                if len(old_rows) > remaining else old_rows
            )
        selected = pd.concat([chosen_new, chosen_old], ignore_index=True)
        selected = selected.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        selected_frames[str(machine_id)] = selected
        selected_candidate_ids.extend(int(value) for value in chosen_new[CANDIDATE_ID_COL].dropna())

    pooled = pd.concat(selected_frames.values(), ignore_index=True)
    return pooled, selected_frames, calibration_frames, counts, rows_per_machine, selected_candidate_ids


def _risk(scorer, features):
    if len(features) == 0:
        return np.array([])
    score = scorer.score(features)
    health = scorer.health_from_score(score)
    return np.clip(1.0 - health / 100.0, 0.0, 1.0)


def _raw_feature_window(conn, machine_id, start, end, feature_cols):
    # Pull enough earlier time for the first in-range rolling feature. This
    # uses the configured sampling rate rather than assuming one minute.
    lookback_seconds = (config.WINDOW_SIZE + 1) / config.SAMPLING_RATE_HZ
    fetch_start = pd.Timestamp(start) - pd.Timedelta(seconds=lookback_seconds)
    rows = db.fetch_rows_between(
        conn,
        db.get_table_name(),
        fetch_start.to_pydatetime(),
        pd.Timestamp(end).to_pydatetime(),
        machine_id=machine_id,
    )
    if not rows:
        return pd.DataFrame(columns=feature_cols)

    db_columns = db.get_db_columns()
    rename = {db_columns["timestamp"]: config.COL_TIMESTAMP}
    rename.update({db_name: config_name for config_name, db_name in db_columns["by_config_name"].items()})
    raw = preprocessing.clean_data(pd.DataFrame(rows).rename(columns=rename))
    featured = feature_engineering.create_features(raw, verbose=False)
    timestamps = pd.to_datetime(featured[config.COL_TIMESTAMP], utc=True)
    mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
    return featured.loc[mask, feature_cols].reset_index(drop=True)


def _regression_feature_windows(conn, feature_cols):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, machine_id, lower(timestamp_range), upper(timestamp_range), minimum_anomaly_risk
               FROM regression_tests ORDER BY id"""
        )
        tests = cur.fetchall()
    return [
        (
            regression_id,
            str(machine_id),
            _raw_feature_window(conn, str(machine_id), start, end, feature_cols),
            float(minimum),
        )
        for regression_id, machine_id, start, end, minimum in tests
    ]


def _active_training_machine_ids():
    path = os.path.join(config.ARTIFACTS_DIR, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        machine_ids = json.load(handle).get("training_machine_ids")
    return {str(machine_id) for machine_id in machine_ids} if machine_ids else None


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
    total = sum(item["pending"] for item in by_machine.values())
    oldest_age_days = max((item["oldest_age_days"] for item in by_machine.values()), default=0)
    return due, {
        "pending": total,
        "candidate_count": total,
        "pending_by_machine": by_machine,
        "batch_size_per_machine": batch,
        "batch_size": batch,
        "oldest_age_days": oldest_age_days,
        "time_cap_days": cap,
    }


def _machine_scorer(model, calibration):
    return AnomalyScorer.from_calibration(model, calibration)


def _write_shadow_bundle(
    target: Path,
    shadow,
    feature_cols,
    metadata,
    selected_frames,
    machine_calibrations,
):
    target.mkdir(parents=True, exist_ok=False)
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
    return reference_rows


def run_shadow_retrain(conn, force: bool = False) -> dict:
    due, status = should_retrain(conn)
    if not force and not due:
        return {"started": False, **status}

    active_scorer, feature_cols, active_meta = artifact_utils.load_artifacts(use_local_calibration=False)
    has_machine_reference = os.path.exists(
        os.path.join(config.ARTIFACTS_DIR, "reference_features.csv")
    )
    if not has_machine_reference and len(active_meta.get("training_machine_ids", [])) > 1:
        raise ValueError(
            "The active pooled model predates machine-aware reference_features.csv. "
            "Run the current initial trainer once before enabling multi-machine retraining."
        )
    base = _current_reference_features(feature_cols)
    base_machine_ids = {str(machine_id) for machine_id in base[MACHINE_COL].unique()}
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
    deduplicated_rows = _dedup_within_machine(all_candidate_rows)
    if not deduplicated_rows:
        detail = (
            f" New-machine candidates require initial commissioning training: "
            f"{ignored_new_machine_candidates}."
            if ignored_new_machine_candidates else ""
        )
        raise ValueError(f"No eligible, non-duplicate confirmed-normal candidates are available.{detail}")
    candidates = _candidate_frame(deduplicated_rows, feature_cols)
    pooled, selected_frames, calibration_frames, available_counts, rows_per_machine, selected_ids = (
        _balanced_reference(base, candidates, feature_cols)
    )

    shadow = AnomalyScorer().fit(pooled[feature_cols])
    machine_calibrations = {}
    shadow_machine_scorers = {}
    for machine_id, frame in calibration_frames.items():
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
    active_version = model_registry.active_version(conn)
    fp_by_machine = {}
    active_flags = []
    shadow_flags = []
    for machine_id, frame in selected_frames.items():
        active_calibration = (
            model_registry.machine_calibration(conn, active_version, machine_id)
            if active_version else None
        )
        active_local = (
            _machine_scorer(active_scorer.model, active_calibration)
            if active_calibration else active_scorer
        )
        features = frame[feature_cols]
        current_flags = active_local.health_from_score(active_local.score(features)) <= health_cut
        proposed_flags = shadow_machine_scorers[machine_id].health_from_score(
            shadow_machine_scorers[machine_id].score(features)
        ) <= health_cut
        active_fp = float(np.mean(current_flags))
        shadow_fp = float(np.mean(proposed_flags))
        passed = shadow_fp <= active_fp + 0.02
        fp_by_machine[machine_id] = {
            "reference_rows": len(frame),
            "active_reference_fp": active_fp,
            "shadow_reference_fp": shadow_fp,
            "pass": passed,
        }
        active_flags.extend(current_flags.tolist())
        shadow_flags.extend(proposed_flags.tolist())
    fp_pass = all(item["pass"] for item in fp_by_machine.values())

    # Gate 2: reconstruct each permanent false-negative window from that
    # machine's live raw data and score it with that machine's new calibration.
    regression = []
    regression_pass = True
    for regression_id, machine_id, features, minimum in _regression_feature_windows(conn, feature_cols):
        if features.empty:
            regression.append({
                "id": regression_id,
                MACHINE_COL: machine_id,
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
                "pass": False,
                "reason": "machine is absent from the proposed balanced reference",
            })
            regression_pass = False
            continue
        max_risk = float(np.max(_risk(scorer, features)))
        passed = max_risk >= minimum
        regression.append({
            "id": regression_id,
            MACHINE_COL: machine_id,
            "max_anomaly_risk": max_risk,
            "minimum": minimum,
            "pass": passed,
        })
        regression_pass &= passed

    report = {
        "reference_rows": len(pooled),
        "machines": sorted(selected_frames),
        "available_reference_rows_by_machine": available_counts,
        "balanced_rows_per_machine": rows_per_machine,
        "processed_candidates": len(all_candidate_rows),
        "ignored_new_machine_candidates": ignored_new_machine_candidates,
        "deduplicated_candidates": len(deduplicated_rows),
        "selected_candidates": len(selected_ids),
        "active_reference_fp": float(np.mean(active_flags)),
        "shadow_reference_fp": float(np.mean(shadow_flags)),
        "reference_fp_by_machine": fp_by_machine,
        "reference_fp_gate_pass": fp_pass,
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
        "human_confirmed_retrain": True,
        "training_mode": "balanced_pooled_retrain",
        "training_machine_ids": sorted(selected_frames),
        "available_reference_rows_per_machine": available_counts,
        "model_fit_rows_per_machine": rows_per_machine,
        "balance_method": "equal_rows_candidate_priority_deterministic_sample",
    })
    target = Path(model_registry.bundle_path(version_id))
    reference_rows = _write_shadow_bundle(
        target,
        shadow,
        feature_cols,
        metadata,
        selected_frames,
        machine_calibrations,
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
                        "Machine-specific anchor from balanced shadow retraining",
                    ),
                )
                calibration_id = cur.fetchone()[0]
                cur.execute(
                    """INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
                       VALUES(%s,%s,%s)""",
                    (machine_id, version_id, calibration_id),
                )

            # Every eligible candidate was processed by this accepted cycle.
            # Mark deduplicated/sampled-out rows too so they do not retrigger
            # identical shadow runs forever.
            processed_ids = [item["id"] for item in all_candidate_rows]
            if processed_ids:
                cur.execute(
                    "UPDATE reference_candidates SET added_to_reference_at=now() WHERE id=ANY(%s)",
                    (processed_ids,),
                )
    conn.commit()
    return {
        "started": True,
        "version_id": version_id,
        "status": "shadow" if accepted else "rejected",
        "validation": report,
    }
