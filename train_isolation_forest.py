"""
Fit the Isolation Forest on the reference/commissioning baseline window
and save artifacts. No labels anywhere in this script.

Where the reference window comes from is controlled by
config.REFERENCE_SOURCE:

  "live" (default) - pulls directly from the same Postgres table +
      connection worker.py/db.py use in production, restricted to
      [REFERENCE_WINDOW_START, REFERENCE_WINDOW_END). Training and
      production then share one source of truth instead of two schemas
      that can drift apart. Requires those two timestamps to be set
      (in config.py, or via --start/--end below) — this script will not
      guess a window from "whatever the table currently holds", because
      that would make every run pull a different, unreproducible
      reference set.
  "csv" - config.RAW_DATA_PATH, the original offline-file behavior. Kept
      for offline experimentation / CI fixtures that shouldn't need a
      reachable database.

Usage:
    python train_isolation_forest.py
    python train_isolation_forest.py --full
    python train_isolation_forest.py --source live --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source live --all-machines --machine-workers 2 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source live --machine-id MACHINE-001 --machine-id MACHINE-002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source csv
"""

import argparse
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import db
import db_schema
import preprocessing
import feature_engineering
import artifact_utils
import operating_state
import machine_normalization
import streaming_training
import model_registry
import runtime_config
import simulation_jobs
from isolation_forest import AnomalyScorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Treat every confirmed-RUNNING row in the reference window as "
             "normal instead of "
             "spec-filtering it (preprocessing.select_spec_normal_rows()). "
             "Only use this when you're sure the running portions are "
             "genuinely healthy. STOPPED and STARTING rows remain excluded "
             "unless --include-non-running is also supplied."
    )
    parser.add_argument(
        "--include-non-running", action="store_true",
        help="Explicit unsafe override that disables the sensor-derived "
             "RUNNING gate for live training. Use only when the selected "
             "window is independently confirmed to contain running data only."
    )
    parser.add_argument(
        "--source", choices=["live", "csv"], default=None,
        help="Override config.REFERENCE_SOURCE for this run without "
             "editing config.py."
    )
    parser.add_argument(
        "--start", default=None,
        help="Override config.REFERENCE_WINDOW_START for this run "
             "(only used when the resolved source is 'live'). ISO 8601, "
             "e.g. 2026-01-05T00:00:00Z."
    )
    parser.add_argument(
        "--end", default=None,
        help="Override config.REFERENCE_WINDOW_END for this run "
             "(only used when the resolved source is 'live')."
    )
    machine_group = parser.add_mutually_exclusive_group()
    machine_group.add_argument(
        "--machine-id", action="append", default=None,
        help="Machine to include in live training. Repeat this option (or use "
             "a comma-separated value) to build one balanced shared model from "
             "multiple machines. Defaults to DEFAULT_MACHINE_ID."
    )
    machine_group.add_argument(
        "--all-machines", action="store_true",
        help="Discover every machine ID in the live source and include all of "
             "them in the balanced shared model."
    )
    parser.add_argument(
        "--max-rows-per-machine", type=int, default=None,
        help="Override the automatic bounded model/validation reservoir per "
             "machine (default: config.TRAINING_MAX_ROWS_PER_MACHINE). Every "
             "source row is still scanned and eligible; this controls only "
             "how many engineered rows are retained in memory."
    )
    parser.add_argument(
        "--machine-workers", type=int, default=2,
        help="Number of machines to scan concurrently in separate processes "
             "for live training (default: 2). Use 2 first; higher values add "
             "PostgreSQL, CPU, and memory pressure."
    )
    parser.add_argument(
        "--preview-reference", action="store_true",
        help="Load, normalize, feature-engineer, operating-state filter, and "
             "balance the live reference selection, then stop before fitting "
             "or writing model artifacts."
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Optional operator-facing name shown in the website. The stable "
             "technical version ID and artifact directory are generated "
             "independently and do not change when this label is renamed."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared: fit + save, given a feature table and a reference-row selection
# ---------------------------------------------------------------------------
def _fit_and_save(
    reference_df,
    feature_cols,
    health_check_df,
    health_check_label,
    reference_rows=None,
    machine_reference_frames=None,
    machine_health_frames=None,
    machine_validation_frames=None,
    machine_feature_normalizers=None,
    reference_features=None,
    validation_features=None,
    metadata_extra=None,
    model_name=None,
):
    scorer = AnomalyScorer()
    scorer.fit(reference_df[feature_cols])
    print(f"[train] Calibration: {scorer.calibration()}")

    # Sanity check — not a labeled evaluation (no labels exist in this
    # codebase); just a distributional check that health looks reasonable
    # rather than sitting flat at 100 or crashing to 0 everywhere.
    scores = scorer.score(health_check_df[feature_cols])
    health = scorer.health_from_score(scores)
    print(f"[train] Condition score over {health_check_label} — "
          f"min: {health.min():.1f}, max: {health.max():.1f}, "
          f"mean: {health.mean():.1f}")

    machine_calibrations = None
    if machine_reference_frames:
        machine_calibrations = {}
        for machine_id, machine_reference in machine_reference_frames.items():
            local_scorer = AnomalyScorer()
            local_scorer.model = scorer.model
            local_scorer.calibrate(machine_reference[feature_cols])
            machine_calibrations[machine_id] = {
                "calibration": local_scorer.calibration(),
                "source_rows": len(machine_reference),
            }

            machine_health = (machine_health_frames or {}).get(machine_id, machine_reference)
            local_scores = local_scorer.score(machine_health[feature_cols])
            local_health = local_scorer.health_from_score(local_scores)
            print(f"[train] Machine {machine_id!r} automatic condition anchor — "
                  f"min: {local_health.min():.1f}, max: {local_health.max():.1f}, "
                  f"mean: {local_health.mean():.1f}")

    if not machine_validation_frames:
        raise ValueError("Initial training requires independent per-machine validation holdouts.")
    health_cut = float(runtime_config.get("MAINTENANCE_HEALTH_INSPECT"))
    maximum_fp = artifact_utils.COMMISSIONING_MAX_HOLDOUT_FP_RATE
    fp_by_machine = {}
    all_passed = True
    for machine_id, holdout in machine_validation_frames.items():
        normalized_holdout = machine_normalization.transform(
            holdout,
            feature_cols,
            machine_feature_normalizers[machine_id],
            machine_id,
        )
        local_scorer = AnomalyScorer.from_calibration(
            scorer.model, machine_calibrations[machine_id]["calibration"]
        )
        holdout_scores = local_scorer.score(normalized_holdout[feature_cols])
        holdout_health = local_scorer.health_from_score(holdout_scores)
        finite = np.isfinite(holdout_scores) & np.isfinite(holdout_health)
        coverage = float(np.mean(finite)) if len(finite) else 0.0
        fp_rate = (
            float(np.mean(holdout_health[finite] <= health_cut))
            if finite.any() else 1.0
        )
        passed = coverage == 1.0 and fp_rate <= maximum_fp
        all_passed = all_passed and passed
        fp_by_machine[machine_id] = {
            "holdout_rows": len(holdout),
            "finite_score_coverage": coverage,
            "holdout_false_positive_rate": fp_rate,
            "maximum_allowed_false_positive_rate": maximum_fp,
            "condition_alert_threshold": health_cut,
            "pass": passed,
        }

    validation_report = {
        "validation_type": "initial_commissioning_forward_holdout",
        "passed": all_passed,
        "holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
        "holdout_lineage": "commissioning_forward_holdout",
        "holdout_rows": sum(len(frame) for frame in machine_validation_frames.values()),
        "machines": sorted(machine_validation_frames),
        "false_positive_evaluation": "independent_per_machine_absolute_gate",
        "maximum_holdout_false_positive_rate": maximum_fp,
        "reference_fp_by_machine": fp_by_machine,
    }
    version_id = model_registry.new_version_id()
    target = Path(model_registry.bundle_path(version_id))
    bundle_metadata = dict(metadata_extra or {})
    bundle_metadata.update({
        "version_id": version_id,
        "validation_report": validation_report,
    })
    try:
        artifact_utils.save_artifacts(
            scorer,
            feature_cols,
            reference_timestamps=reference_df[config.COL_TIMESTAMP],
            reference_rows=reference_rows,
            reference_features=reference_features,
            validation_features=validation_features,
            machine_calibrations=machine_calibrations,
            machine_feature_normalizers=machine_feature_normalizers,
            metadata_extra=bundle_metadata,
            directory=target,
        )
        connection = db.get_connection()
        try:
            db_schema.migrate(connection)
            simulation_gate = simulation_jobs.evaluate_validation_suite(
                connection, version_id
            )
            validation_report["labelled_simulation_evaluation"] = simulation_gate
            validation_report["labelled_simulation_is_advisory"] = True
            validation_report["passed"] = all_passed
            metadata_path = target / "metadata.json"
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored_metadata["validation_report"] = validation_report
            metadata_path.write_text(
                json.dumps(
                    artifact_utils.to_json_safe(stored_metadata),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            registered = model_registry.register_initial_bundle(
                connection,
                version_id,
                validation_report,
                display_name=model_name,
            )
        finally:
            connection.close()
    except Exception:
        # Remove only an incomplete write. A complete immutable bundle is
        # deliberately preserved when the later database registration fails;
        # multi-hour training output must not be destroyed by a transient
        # control-plane connection error.
        if any(
            not (target / filename).is_file()
            for filename in model_registry.REQUIRED_BUNDLE_FILES
        ):
            shutil.rmtree(target, ignore_errors=True)
        raise
    print(
        f"[train] Initial model {version_id} registered as {registered['status']} "
        f"at {target}. Validation passed={validation_report['passed']}."
    )
    return registered


# ---------------------------------------------------------------------------
# Live-source path: reference window = an explicit, pinned range pulled
# from the production Postgres table itself.
# ---------------------------------------------------------------------------
def _resolve_live_window(args):
    start = args.start or config.REFERENCE_WINDOW_START
    end = args.end or config.REFERENCE_WINDOW_END
    if not start or not end:
        raise ValueError(
            "REFERENCE_SOURCE is 'live' but no commissioning window is set. "
            "Set config.REFERENCE_WINDOW_START and config.REFERENCE_WINDOW_END "
            "to a confirmed-healthy stretch of the live table (or pass "
            "--start/--end on the command line) before training. See "
            "config.py's REFERENCE_WINDOW_START comment for why this isn't "
            "inferred automatically."
        )
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"REFERENCE_WINDOW_END ({end}) must be after REFERENCE_WINDOW_START ({start}).")
    return start, end


def _clean_stream_chunk(rows, dbcols, previous_source_timestamp):
    """Canonicalize one ordered DB chunk and deduplicate across boundaries."""
    frame = db.canonical_sensor_frame(rows, dbcols)
    frame[config.COL_TIMESTAMP] = pd.to_datetime(
        frame[config.COL_TIMESTAMP], errors="coerce"
    )
    frame = frame.dropna(subset=[config.COL_TIMESTAMP]).sort_values(
        config.COL_TIMESTAMP
    )
    if previous_source_timestamp is not None:
        frame = frame[frame[config.COL_TIMESTAMP] > previous_source_timestamp]
    next_source_timestamp = previous_source_timestamp
    if not frame.empty:
        # Advance on source timestamps before sensor-value cleaning. This
        # exactly prevents a valid duplicate in the next DB chunk replacing an
        # invalid first row that full-frame clean_data() would have kept then
        # rejected.
        next_source_timestamp = frame[config.COL_TIMESTAMP].max()
    return preprocessing.clean_data(frame, verbose=False), next_source_timestamp


def _activity_frame(clean_rows):
    motion = clean_rows[list(operating_state.MOTION_COLS)].to_numpy(dtype=float)
    scores = np.median(np.log10(np.maximum(np.abs(motion), 1e-12)), axis=1)
    return pd.DataFrame({"activity_score": scores})


def _progress(machine_id, pass_label, scanned, clean, started, force=False):
    if force or scanned % 500_000 < config.TRAINING_DB_CHUNK_ROWS:
        print(
            f"[train] Machine {machine_id!r} {pass_label}: "
            f"scanned={scanned:,}, clean={clean:,}, "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )


def _learn_streaming_operating_state(start, end, machine_id):
    """First bounded pass: every clean row can enter the motion profile."""
    print(
        f"[train] Machine {machine_id!r}: pass 1/2 learning motion regimes "
        f"from the complete selected range...",
        flush=True,
    )
    profile = streaming_training.PriorityReservoir(
        config.TRAINING_STATE_PROFILE_ROWS,
        streaming_training.machine_seed(
            config.TRAINING_RESERVOIR_SEED, machine_id, "operating-state"
        ),
    )
    scanned = clean = 0
    previous_timestamp = None
    started = time.perf_counter()
    dbcols = db.get_db_columns()
    for rows in db.iter_row_chunks_between(
        db.get_table_name(),
        start.to_pydatetime(),
        end.to_pydatetime(),
        machine_id=machine_id,
    ):
        scanned += len(rows)
        clean_rows, previous_timestamp = _clean_stream_chunk(
            rows, dbcols, previous_timestamp
        )
        clean += len(clean_rows)
        profile.offer(_activity_frame(clean_rows))
        _progress(machine_id, "motion profile", scanned, clean, started)

    profile_frame = profile.result()
    if profile_frame.empty:
        raise ValueError(
            f"No clean rows found for machine {machine_id!r} between {start} and {end}."
        )
    detector = operating_state.OperatingStateDetector(
        history_rows=len(profile_frame)
    )
    for score in profile_frame["activity_score"].to_numpy(dtype=float):
        detector.observe_activity_score(score)
    if not detector.fit_history(require_sustained_low=False):
        print(
            f"[train] Machine {machine_id!r}: motion-regime diagnostics: "
            f"{detector.last_fit_diagnostics}",
            flush=True,
        )
        raise ValueError(
            f"Machine {machine_id!r} has no clearly separable stationary and "
            "rotating vibration regimes in the selected window. Choose a "
            "window containing both regimes, or use --include-non-running "
            "only when the entire range is independently confirmed running. "
            "The diagnostic line above identifies the rejected cluster rule."
        )
    _progress(machine_id, "motion profile complete", scanned, clean, started, force=True)
    print(
        f"[train] Machine {machine_id!r}: learned activity thresholds "
        f"stop={detector.stop_threshold:.6g}, run={detector.run_threshold:.6g} "
        f"from {len(profile_frame):,}/{clean:,} profiled clean rows.",
        flush=True,
    )
    return detector, {
        "motion_profile_source_rows": scanned,
        "motion_profile_clean_rows": clean,
        "motion_profile_retained_rows": len(profile_frame),
        "motion_profile_diagnostics": detector.last_fit_diagnostics,
    }


def _resolve_live_machine_ids(args):
    if args.all_machines:
        conn = db.get_connection()
        try:
            machine_ids = db.fetch_machine_ids(conn, db.get_table_name())
        finally:
            conn.close()
    elif args.machine_id:
        machine_ids = []
        for value in args.machine_id:
            machine_ids.extend(part.strip() for part in value.split(","))
    else:
        machine_ids = [db.DEFAULT_MACHINE_ID]

    # Preserve command/source order while removing blanks and duplicates.
    machine_ids = list(dict.fromkeys(machine_id for machine_id in machine_ids if machine_id))
    if not machine_ids:
        raise ValueError("No machine IDs were selected for live training.")
    print(f"[train] Training machines ({len(machine_ids)}): {', '.join(machine_ids)}")
    return machine_ids


def _stream_live_machine(
    start,
    end,
    machine_id,
    *,
    detector,
    include_non_running,
    use_full,
    retained_capacity,
):
    """Second bounded pass: exact rolling features and automatic eligibility."""
    print(
        f"[train] Machine {machine_id!r}: pass 2/2 building rolling features; "
        f"automatic retained capacity={retained_capacity:,}...",
        flush=True,
    )
    sampler = streaming_training.ForwardHoldoutReservoir(
        capacity=retained_capacity,
        holdout_fraction=artifact_utils.VALIDATION_HOLDOUT_FRACTION,
        minimum_holdout=artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS,
        seed=streaming_training.machine_seed(
            config.TRAINING_RESERVOIR_SEED, machine_id, "reference-features"
        ),
    )
    state_counts = {state: 0 for state in operating_state.VALID_STATES}
    if include_non_running:
        state_counts["BYPASSED"] = 0
    stats = {
        "source_rows_scanned": 0,
        "clean_rows": 0,
        "complete_feature_rows": 0,
        "applicable_rows": 0,
        "reference_rows_considered": 0,
    }
    previous_timestamp = None
    carry_rows = None
    feature_cols = None
    dbcols = db.get_db_columns()
    started = time.perf_counter()

    for rows in db.iter_row_chunks_between(
        db.get_table_name(),
        start.to_pydatetime(),
        end.to_pydatetime(),
        machine_id=machine_id,
    ):
        stats["source_rows_scanned"] += len(rows)
        clean_rows, previous_timestamp = _clean_stream_chunk(
            rows, dbcols, previous_timestamp
        )
        stats["clean_rows"] += len(clean_rows)
        if clean_rows.empty:
            continue

        featured, carry_rows = feature_engineering.create_features_chunk(
            clean_rows, carry_rows
        )
        if featured.empty:
            continue
        current_feature_cols = feature_engineering.get_feature_columns(featured)
        if feature_cols is None:
            feature_cols = current_feature_cols
        elif feature_cols != current_feature_cols:
            raise ValueError(f"Feature columns changed while streaming machine {machine_id!r}.")
        stats["complete_feature_rows"] += len(featured)

        if include_non_running:
            applicable = np.ones(len(clean_rows), dtype=bool)
            state_counts["BYPASSED"] += len(clean_rows)
        else:
            applicable_values = []
            for reading in clean_rows[config.RAW_SENSOR_COLS].to_dict("records"):
                state = detector.update(reading)
                state_counts[state.state] = state_counts.get(state.state, 0) + 1
                applicable_values.append(
                    state.state == "RUNNING" and not state.low_motion
                )
            applicable = np.asarray(applicable_values, dtype=bool)

        if use_full:
            within_spec = np.ones(len(clean_rows), dtype=bool)
        else:
            within_spec = np.ones(len(clean_rows), dtype=bool)
            for column, bound in config.SPEC_MAX.items():
                if bound is not None:
                    within_spec &= clean_rows[column].to_numpy(dtype=float) <= float(bound)

        eligibility = clean_rows[[config.COL_TIMESTAMP]].copy()
        eligibility["__applicable"] = applicable
        eligibility["__reference"] = applicable & within_spec
        selected = featured.merge(
            eligibility, on=config.COL_TIMESTAMP, how="inner", validate="one_to_one"
        )
        stats["applicable_rows"] += int(selected["__applicable"].sum())
        reference_chunk = selected[selected["__reference"]][
            [config.COL_TIMESTAMP, *feature_cols]
        ].reset_index(drop=True)
        stats["reference_rows_considered"] += len(reference_chunk)
        sampler.offer(reference_chunk)
        _progress(
            machine_id,
            "feature scan",
            stats["source_rows_scanned"],
            stats["clean_rows"],
            started,
        )

    retained = sampler.result(config.COL_TIMESTAMP)
    if retained.empty or feature_cols is None:
        raise ValueError(
            f"Machine {machine_id!r} produced no usable healthy feature rows. "
            "Verify the selected range and operating-state/spec assumptions."
        )
    stats["retained_rows_before_fleet_balance"] = len(retained)
    stats["state_counts"] = state_counts
    stats["operating_state_stop_threshold"] = (
        None if detector is None else detector.stop_threshold
    )
    stats["operating_state_run_threshold"] = (
        None if detector is None else detector.run_threshold
    )
    _progress(
        machine_id,
        "feature scan complete",
        stats["source_rows_scanned"],
        stats["clean_rows"],
        started,
        force=True,
    )
    filter_description = "spec filter skipped (--full)" if use_full else "spec filter applied"
    print(
        f"[train] Machine {machine_id!r}: considered "
        f"{stats['reference_rows_considered']:,} eligible feature rows "
        f"({filter_description}); retained {len(retained):,} for balanced "
        f"fit/validation; states={state_counts}.",
        flush=True,
    )
    return retained, feature_cols, stats


def _prepare_live_machine_task(task):
    """Top-level, spawn-safe worker for one machine's complete two-pass scan."""
    machine_id = task["machine_id"]
    start = pd.Timestamp(task["start"])
    end = pd.Timestamp(task["end"])
    include_non_running = bool(task["include_non_running"])
    if include_non_running:
        detector = None
        profile_stats = {
            "motion_profile_source_rows": 0,
            "motion_profile_clean_rows": 0,
            "motion_profile_retained_rows": 0,
            "motion_profile_bypassed": True,
        }
    else:
        detector, profile_stats = _learn_streaming_operating_state(
            start, end, machine_id
        )
    retained, feature_cols, machine_stats = _stream_live_machine(
        start,
        end,
        machine_id,
        detector=detector,
        include_non_running=include_non_running,
        use_full=bool(task["use_full"]),
        retained_capacity=int(task["retained_capacity"]),
    )
    return machine_id, retained, feature_cols, {**profile_stats, **machine_stats}


def _balanced_reference_pool(machine_reference_frames, max_rows_per_machine=None):
    counts = {machine_id: len(frame) for machine_id, frame in machine_reference_frames.items()}
    rows_per_machine = min(counts.values())
    if max_rows_per_machine is not None:
        if max_rows_per_machine < 1:
            raise ValueError("--max-rows-per-machine must be at least 1.")
        rows_per_machine = min(rows_per_machine, max_rows_per_machine)

    balanced = {}
    for machine_id, frame in machine_reference_frames.items():
        ordered = frame.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        if len(frame) > rows_per_machine:
            holdout_rows = min(
                rows_per_machine,
                max(
                    artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS,
                    int(np.ceil(rows_per_machine * artifact_utils.VALIDATION_HOLDOUT_FRACTION)),
                ),
            )
            newest = ordered.tail(holdout_rows)
            fit_rows = rows_per_machine - holdout_rows
            fit_pool = ordered.iloc[:-holdout_rows]
            sampled_fit = fit_pool.sample(n=fit_rows, random_state=42)
            selected = pd.concat([sampled_fit, newest], ignore_index=True)
            selected = selected.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        else:
            selected = ordered
        balanced[machine_id] = selected

    print(f"[train] Balanced pool before validation holdout: {rows_per_machine} "
          f"healthy feature rows per machine, {rows_per_machine * len(balanced)} "
          f"total. Available before balancing: {counts}.")
    return balanced, counts, rows_per_machine


def _split_validation_holdout(balanced_frames, feature_cols):
    """Create equal, forward-looking per-machine commissioning holdouts."""
    fraction = artifact_utils.VALIDATION_HOLDOUT_FRACTION
    minimum_holdout = artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS
    minimum_fit = max(20, len(feature_cols) + 1)
    training_frames = {}
    validation_frames = {}
    for machine_id, frame in balanced_frames.items():
        ordered = frame.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        holdout_rows = max(minimum_holdout, int(np.ceil(len(ordered) * fraction)))
        if len(ordered) < holdout_rows + minimum_fit:
            raise ValueError(
                f"Machine {machine_id!r} needs at least {holdout_rows + minimum_fit} "
                f"balanced healthy feature rows to keep {holdout_rows} for validation "
                f"and {minimum_fit} for model fitting; only {len(ordered)} are available."
            )
        validation_frames[machine_id] = ordered.tail(holdout_rows).reset_index(drop=True)
        training_frames[machine_id] = ordered.iloc[:-holdout_rows].reset_index(drop=True)
    return training_frames, validation_frames


def _train_from_live(args):
    start, end = _resolve_live_window(args)
    machine_ids = _resolve_live_machine_ids(args)
    machine_reference_frames = {}
    streaming_stats = {}
    feature_cols = None
    retained_capacity = (
        args.max_rows_per_machine
        if args.max_rows_per_machine is not None
        else config.TRAINING_MAX_ROWS_PER_MACHINE
    )
    if retained_capacity < 100:
        raise ValueError(
            "--max-rows-per-machine must be at least 100 so model fitting and "
            "forward validation both retain enough evidence."
        )

    requested_workers = int(args.machine_workers)
    if requested_workers < 1:
        raise ValueError("--machine-workers must be at least 1.")
    machine_workers = min(requested_workers, len(machine_ids))
    tasks = [
        {
            "machine_id": machine_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "include_non_running": args.include_non_running,
            "use_full": args.full,
            "retained_capacity": retained_capacity,
        }
        for machine_id in machine_ids
    ]

    # Each task owns one machine's two passes, DB connections, operating state,
    # and reservoir. Separate spawned processes provide actual CPU concurrency
    # on Windows. Results are assembled in source-machine order afterward so
    # parallel completion order cannot change the deterministic shared model.
    results_by_machine = {}
    if machine_workers == 1:
        print("[train] Machine scan concurrency: 1 (sequential).", flush=True)
        for task in tasks:
            result = _prepare_live_machine_task(task)
            results_by_machine[result[0]] = result
    else:
        print(
            f"[train] Machine scan concurrency: {machine_workers} processes. "
            "Progress lines will be interleaved by machine.",
            flush=True,
        )
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=machine_workers)
        try:
            for result in pool.imap_unordered(_prepare_live_machine_task, tasks):
                results_by_machine[result[0]] = result
            pool.close()
        except BaseException:
            # Public Pool termination stops active child scans as well as queued
            # tasks, avoiding orphaned PostgreSQL reads after Ctrl+C or failure.
            pool.terminate()
            raise
        finally:
            pool.join()

    for machine_id in machine_ids:
        _, retained, current_feature_cols, machine_stats = results_by_machine[machine_id]
        if feature_cols is None:
            feature_cols = current_feature_cols
        elif current_feature_cols != feature_cols:
            raise ValueError(f"Feature columns differ for machine {machine_id!r}.")
        machine_reference_frames[machine_id] = retained
        streaming_stats[machine_id] = machine_stats

    balanced_frames, retained_counts, rows_per_machine = _balanced_reference_pool(
        machine_reference_frames, retained_capacity
    )
    training_frames, validation_frames = _split_validation_holdout(
        balanced_frames, feature_cols
    )
    machine_feature_normalizers = machine_normalization.fit_normalizers(
        training_frames, feature_cols
    )
    normalized_training_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, machine_feature_normalizers[machine_id], machine_id
        )
        for machine_id, frame in training_frames.items()
    }
    normalized_health_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, machine_feature_normalizers[machine_id], machine_id
        )
        for machine_id, frame in balanced_frames.items()
    }
    reference_df = pd.concat(normalized_training_frames.values(), ignore_index=True)
    health_check_df = pd.concat(normalized_health_frames.values(), ignore_index=True)
    reference_rows = [
        {"machine_id": machine_id, "timestamp": timestamp}
        for machine_id, frame in training_frames.items()
        for timestamp in frame[config.COL_TIMESTAMP]
    ]
    reference_features = pd.concat(
        [frame.assign(machine_id=machine_id) for machine_id, frame in training_frames.items()],
        ignore_index=True,
    )[["machine_id", config.COL_TIMESTAMP, *feature_cols]]
    validation_features = pd.concat(
        [frame.assign(machine_id=machine_id) for machine_id, frame in validation_frames.items()],
        ignore_index=True,
    )[["machine_id", config.COL_TIMESTAMP, *feature_cols]]

    if args.preview_reference:
        print(
            "[train] Reference preview complete; no model was fitted and no "
            "artifacts were written."
        )
        return

    _fit_and_save(
        reference_df,
        feature_cols,
        health_check_df=health_check_df,
        health_check_label="all selected commissioning windows",
        reference_rows=reference_rows,
        reference_features=reference_features,
        validation_features=validation_features,
        machine_reference_frames=normalized_training_frames,
        machine_health_frames=normalized_health_frames,
        machine_validation_frames=validation_frames,
        machine_feature_normalizers=machine_feature_normalizers,
        model_name=args.model_name,
        metadata_extra={
            "training_mode": "balanced_pooled" if len(machine_ids) > 1 else "single_machine",
            "training_machine_ids": machine_ids,
            "available_reference_rows_per_machine": {
                machine_id: values["reference_rows_considered"]
                for machine_id, values in streaming_stats.items()
            },
            "streaming_retained_rows_per_machine": retained_counts,
            "balanced_rows_per_machine_before_holdout": rows_per_machine,
            "model_fit_rows_per_machine": {
                machine_id: len(frame) for machine_id, frame in training_frames.items()
            },
            "validation_rows_per_machine": {
                machine_id: len(frame) for machine_id, frame in validation_frames.items()
            },
            "validation_holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
            "validation_holdout_lineage": "commissioning_forward_holdout",
            "balance_method": "equal_rows_deterministic_priority_reservoir_with_forward_holdout",
            "streaming_source_scan": streaming_stats,
            "training_db_chunk_rows": config.TRAINING_DB_CHUNK_ROWS,
            "training_db_slice_hours": config.TRAINING_DB_SLICE_HOURS,
            "training_reservoir_capacity_per_machine": retained_capacity,
            "training_reservoir_seed": config.TRAINING_RESERVOIR_SEED,
            "machine_scan_workers": machine_workers,
            "isolation_forest_params": config.ISOLATION_FOREST_PARAMS,
            "model_feature_space": machine_normalization.METHOD,
            "reference_window_start": start.isoformat(),
            "reference_window_end": end.isoformat(),
            "postgres_sensor_scales": config.POSTGRES_SENSOR_SCALES,
            "operating_state_filter": (
                "disabled_explicitly" if args.include_non_running
                else "confirmed_running_per_machine"
            ),
        },
    )


# ---------------------------------------------------------------------------
# CSV-source path: original offline-file behavior.
# ---------------------------------------------------------------------------
def _get_or_build_csv_features():
    if preprocessing.cached_features_are_stale():
        print(f"[train] Cached features missing or stale for {config.RAW_DATA_PATH} — "
              f"rebuilding from raw data (see preprocessing.cached_features_are_stale()).")
        preprocessing.run_preprocessing()
        return feature_engineering.run_feature_engineering()

    print(f"[train] Loading cached features from {config.FEATURES_DATA_PATH}")
    return pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])


def _train_from_csv(args):
    df = _get_or_build_csv_features()
    feature_cols = feature_engineering.get_feature_columns(df)

    if args.full:
        # Trust the whole file as normal — no spec filter, no reference-
        # window cutoff. Every row in config.RAW_DATA_PATH becomes the
        # reference set the Isolation Forest is fit on.
        reference_df = df.reset_index(drop=True)
        print(f"[train] --full: using all {len(reference_df)} rows as the "
              f"reference set (spec filter skipped).")
    else:
        # Spec-based row selection needs the raw sensor values, which don't
        # survive feature_engineering.create_features() — so reload/re-clean
        # the raw CSV here rather than reading them off the feature table.
        # Cheap: just load_data() + clean_data(), no feature engineering rerun.
        raw_df = preprocessing.clean_data(preprocessing.load_data())
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)

        is_reference = df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reference_df = df[is_reference].reset_index(drop=True)
        rest_df = df[~is_reference].reset_index(drop=True)
        print(f"[train] Spec-based reference set: {len(reference_df)} rows, "
              f"rest of trajectory: {len(rest_df)} rows.")

    machine_id = db.DEFAULT_MACHINE_ID
    training_frames, validation_frames = _split_validation_holdout(
        {machine_id: reference_df}, feature_cols
    )
    training_reference = training_frames[machine_id]
    validation_reference = validation_frames[machine_id]
    machine_feature_normalizers = machine_normalization.fit_normalizers(
        training_frames, feature_cols
    )
    normalizer = machine_feature_normalizers[machine_id]
    normalized_reference = machine_normalization.transform(
        training_reference, feature_cols, normalizer, machine_id
    )
    normalized_full = machine_normalization.transform(
        df, feature_cols, normalizer, machine_id
    )
    reference_features = training_reference.assign(machine_id=machine_id)[
        ["machine_id", config.COL_TIMESTAMP, *feature_cols]
    ]
    validation_features = validation_reference.assign(machine_id=machine_id)[
        ["machine_id", config.COL_TIMESTAMP, *feature_cols]
    ]
    reference_rows = [
        {"machine_id": machine_id, "timestamp": timestamp}
        for timestamp in training_reference[config.COL_TIMESTAMP]
    ]

    _fit_and_save(
        normalized_reference, feature_cols,
        health_check_df=normalized_full,
        health_check_label="full trajectory",
        reference_rows=reference_rows,
        reference_features=reference_features,
        validation_features=validation_features,
        machine_reference_frames={machine_id: normalized_reference},
        machine_health_frames={machine_id: normalized_full},
        machine_validation_frames={machine_id: validation_reference},
        machine_feature_normalizers=machine_feature_normalizers,
        model_name=args.model_name,
        metadata_extra={
            "training_mode": "single_machine_csv",
            "training_machine_ids": [machine_id],
            "model_feature_space": machine_normalization.METHOD,
            "model_fit_rows_per_machine": {machine_id: len(training_reference)},
            "validation_rows_per_machine": {machine_id: len(validation_reference)},
            "validation_holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
            "validation_holdout_lineage": "commissioning_forward_holdout",
        },
    )


def main():
    args = parse_args()
    source = args.source or config.REFERENCE_SOURCE
    print(f"[train] REFERENCE_SOURCE = {source!r}"
          f"{' (overridden via --source)' if args.source else ' (from config.py)'}")

    if source != "live" and (args.include_non_running or args.preview_reference):
        raise ValueError(
            "--include-non-running and --preview-reference are live-source options."
        )
    if source != "live" and args.machine_workers != 1:
        raise ValueError("--machine-workers is available only for live-source training.")

    # Load the same persisted policy used by the worker before calculating
    # commissioning holdout outcomes. Registration later reuses this schema;
    # doing it up front prevents a web override from being silently replaced
    # by config.py defaults during initial validation.
    policy_connection = db.get_connection()
    try:
        db_schema.migrate(policy_connection)
    finally:
        policy_connection.close()

    if source == "live":
        _train_from_live(args)
    elif source == "csv":
        _train_from_csv(args)
    else:
        raise ValueError(f"Unknown REFERENCE_SOURCE {source!r} — must be 'live' or 'csv'.")


if __name__ == "__main__":
    main()
