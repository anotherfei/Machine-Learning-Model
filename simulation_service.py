"""Read-only historical replay and human-labelled maintenance evaluation.

Each labelled event range gets a fresh production monitor. It is warmed before
a seven-day causal lead window, then receives the same operating-state gate,
feature pipeline, Isolation Forest, condition smoothing, trend forecast, and
maintenance rules as worker.py through the end of the event.
Run coordination and report persistence live in :mod:`simulation_jobs`.
This module only reads raw PostgreSQL history; production predictions, alerts,
and source measurements are never changed.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import time

import artifact_utils
import config
import db
import model_registry
import operating_state
import predict_realtime
import runtime_config


EXPECTED_STATUSES = {
    "OK", "WARN", "CRITICAL", "STOPPED", "STARTING", "SENSOR_FAULT",
    "NO_PREDICTION", "NO_DATA",
}
MAX_CASES = 50
LEAD_WINDOW_DAYS = 7
MAX_EVENT_WINDOW_DAYS = 7
RANGE_MATCH_MIN_COVERAGE = 0.50
# Seven days at the deployment's observed one-row-per-second machine cadence,
# a maximum seven-day event, and production warm-up fit below this guardrail.
MAX_PROCESSED_ROWS_PER_EVENT = 1_500_000
MAX_TRACE_POINTS = 320

POLICY_KEYS = (
    "MAINTENANCE_PROB_URGENT",
    "MAINTENANCE_PROB_PLAN",
    "FAILURE_HEALTH_THRESHOLD",
    "MAINTENANCE_HEALTH_INSPECT",
    "MAINTENANCE_HORIZON_DAYS",
    "MAINTENANCE_URGENT_HORIZON_DAYS",
    "TREND_MIN_POINTS",
    "TREND_SLOPE_Z_THRESHOLD",
    "TREND_SETTLE_TICKS",
    "MAINTENANCE_WARN_CONFIRM_MINUTES",
    "MAINTENANCE_CRITICAL_CONFIRM_MINUTES",
    "MAINTENANCE_RECOVERY_MINUTES",
    "OPERATING_STATE_STOP_CONFIRM_TICKS",
    "OPERATING_STATE_START_CONFIRM_TICKS",
    "SOURCE_STALE_SECONDS",
    "HEALTH_SENSITIVITY_STD",
    "KALMAN_INIT_SAMPLES",
    "TREND_LOOKBACK_MINUTES",
)

_MAINTENANCE_SEVERITY = {"OK": 0, "WARN": 1, "CRITICAL": 2}
_RANGE_STATUS_PRIORITY = {
    "OK": 0,
    "NO_DATA": 1,
    "NO_PREDICTION": 2,
    "STARTING": 3,
    "STOPPED": 4,
    "SENSOR_FAULT": 5,
    "WARN": 6,
    "CRITICAL": 7,
}
_LOCAL_TIMEZONE = dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def utc_datetime(value) -> dt.datetime:
    """Normalize UI, JSON, and PostgreSQL timestamps to aware UTC.

    The production source currently stores local wall-clock values in a
    timestamp-without-time-zone column, while browser-run simulations arrive
    as offset-aware ISO values. Naive values therefore use the host's local
    production timezone before conversion; aware values preserve their instant.
    """
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, dt.datetime):
        raise ValueError(f"Unsupported timestamp value: {value!r}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=_LOCAL_TIMEZONE)
    return value.astimezone(dt.timezone.utc)


def policy_snapshot() -> dict:
    """Capture every value that can change a replay decision."""
    return {key: runtime_config.get(key) for key in POLICY_KEYS}


def load_bundle(version_id: str) -> predict_realtime.ModelBundle:
    versioned = Path(model_registry.bundle_path(version_id))
    if not versioned.is_dir():
        raise FileNotFoundError(f"Model bundle {version_id!r} is not available for replay")
    return predict_realtime.ModelBundle(str(versioned), expected_version=version_id)


class _TraceSampler:
    """Bounded chronological trace that progressively thins long ranges."""
    def __init__(self, maximum: int = MAX_TRACE_POINTS):
        self.maximum = maximum
        self.stride = 1
        self.seen = 0
        self.points: list[dict] = []

    def add(self, point: dict) -> None:
        self.seen += 1
        if (self.seen - 1) % self.stride == 0:
            self.points.append(point)
        if len(self.points) > self.maximum * 2:
            self.points = self.points[::2]
            self.stride *= 2

    def finish(self) -> list[dict]:
        if len(self.points) <= self.maximum:
            return self.points
        step = max(1, math.ceil(len(self.points) / self.maximum))
        result = self.points[::step]
        if self.points and result[-1] is not self.points[-1]:
            result[-1] = self.points[-1]
        return result[: self.maximum]


def _result_point(timestamp, state: dict, result: dict | None) -> dict:
    point = {
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        "operating_state": state["state"],
        "maintenance_status": result["maintenance"]["level"] if result else None,
        "condition": result.get("health_state") if result else None,
        "condition_risk": (
            max(0.0, min(1.0, 1.0 - float(result["health_state"]) / 100.0))
            if result else None
        ),
        "anomaly_score": result.get("anomaly_score") if result else None,
    }
    return point


def _representative_tick(timestamp, state: dict, result: dict | None) -> dict:
    if result is None:
        return {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "operating_state": state["state"],
            "operating_reason": state["reason"],
            "activity_score": state.get("activity_score"),
            "stop_threshold": state.get("stop_threshold"),
            "run_threshold": state.get("run_threshold"),
        }
    health = float(result["health_state"])
    return {
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        "operating_state": state["state"],
        "operating_reason": state["reason"],
        "maintenance_status": result["maintenance"]["level"],
        "maintenance_reason": result["maintenance"]["reason"],
        "maintenance_trigger": result["maintenance"].get("trigger", "none"),
        "condition": health,
        "condition_risk": max(0.0, min(1.0, 1.0 - health / 100.0)),
        "anomaly_score": artifact_utils.finite_float(result.get("anomaly_score")),
        "trend_slope_per_day": artifact_utils.finite_float(result.get("trend_slope_per_day")),
        "remaining_days": artifact_utils.finite_float(result.get("remaining_days")),
        "failure_probability": result.get("failure_probability") or {},
        "top_contributors": artifact_utils.to_json_safe(result.get("top_contributors") or []),
        "sensor_snapshot": {
            column: artifact_utils.finite_float(result.get(column)) for column in config.RAW_SENSOR_COLS
        },
    }


def _lead_hours(target: dt.datetime, timestamp: dt.datetime | None) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (target - timestamp).total_seconds() / 3600.0)


def _timing_assessment(
    expected: str,
    event_start: dt.datetime,
    event_end: dt.datetime,
    warn_times: list[dt.datetime],
    critical_times: list[dt.datetime],
) -> dict:
    """Evaluate lead time against event onset, independently of range accuracy."""
    plan_days = float(runtime_config.get(
        "MAINTENANCE_HORIZON_DAYS", config.MAINTENANCE_HORIZON_DAYS
    ))
    urgent_days = float(runtime_config.get(
        "MAINTENANCE_URGENT_HORIZON_DAYS", config.MAINTENANCE_URGENT_HORIZON_DAYS
    ))
    planning_start = event_start - dt.timedelta(days=plan_days)
    urgent_start = event_start - dt.timedelta(days=urgent_days)
    planning_warn_times = [
        timestamp for timestamp in warn_times
        if planning_start <= timestamp < urgent_start
    ]
    urgent_critical_times = [
        timestamp for timestamp in critical_times
        if urgent_start <= timestamp <= event_end
    ]
    premature_critical_times = [timestamp for timestamp in critical_times if timestamp < urgent_start]
    first_warn = min(warn_times, default=None)
    first_critical = min(critical_times, default=None)
    planning_warn = min(planning_warn_times, default=None)
    urgent_critical = min(urgent_critical_times, default=None)
    maintenance_alert_seen = bool(warn_times or critical_times)

    timing_evaluable = expected in _MAINTENANCE_SEVERITY
    planning_warning_pass = None
    urgent_critical_pass = None
    false_alert = False
    if expected == "CRITICAL":
        planning_warning_pass = planning_warn is not None
        urgent_critical_pass = urgent_critical is not None and not premature_critical_times
        timing_pass = planning_warning_pass and urgent_critical_pass
    elif expected == "WARN":
        planning_warning_pass = any(planning_start <= timestamp <= event_end for timestamp in warn_times)
        urgent_critical_pass = not critical_times
        timing_pass = planning_warning_pass and urgent_critical_pass
    elif expected == "OK":
        false_alert = maintenance_alert_seen
        timing_pass = not false_alert
    else:
        timing_pass = None

    return {
        "timing_evaluable": timing_evaluable,
        "timing_pass": timing_pass,
        "planning_horizon_days": plan_days,
        "urgent_horizon_days": urgent_days,
        "planning_warning_pass": planning_warning_pass,
        "urgent_critical_pass": urgent_critical_pass,
        "premature_critical": bool(premature_critical_times),
        "false_alert": false_alert,
        "first_warn_timestamp": first_warn.isoformat() if first_warn else None,
        "first_warn_lead_hours": _lead_hours(event_start, first_warn),
        "first_critical_timestamp": first_critical.isoformat() if first_critical else None,
        "first_critical_lead_hours": _lead_hours(event_start, first_critical),
        "planning_warning_timestamp": planning_warn.isoformat() if planning_warn else None,
        "urgent_critical_timestamp": urgent_critical.isoformat() if urgent_critical else None,
    }


def _replay_range(
    conn, row: dict, factory: predict_realtime.ModelBundle, progress_callback=None,
) -> dict:
    machine_id = str(row["machine_id"])
    event_start = utc_datetime(row["event_start"])
    event_end = utc_datetime(row["event_end"])
    if event_end < event_start:
        raise ValueError("Event end must be at or after event start")
    event_duration = event_end - event_start
    if event_duration > dt.timedelta(days=MAX_EVENT_WINDOW_DAYS):
        raise ValueError(f"Event range cannot exceed {MAX_EVENT_WINDOW_DAYS} days")
    window_start = event_start - dt.timedelta(days=LEAD_WINDOW_DAYS)
    planning_start = event_start - dt.timedelta(days=float(runtime_config.get(
        "MAINTENANCE_HORIZON_DAYS", config.MAINTENANCE_HORIZON_DAYS
    )))
    urgent_start = event_start - dt.timedelta(days=float(runtime_config.get(
        "MAINTENANCE_URGENT_HORIZON_DAYS", config.MAINTENANCE_URGENT_HORIZON_DAYS
    )))
    expected = str(row["expected_status"]).upper()
    if expected not in EXPECTED_STATUSES:
        raise ValueError(f"Unsupported expected status {expected!r}")
    if not factory.commissioned(machine_id):
        raise ValueError(f"Machine {machine_id!r} is not commissioned in this model")

    table = db.get_table_name()
    columns = db.get_db_columns()
    context_minutes = max(
        5,
        int(runtime_config.get("TREND_LOOKBACK_MINUTES", config.TREND_LOOKBACK_MINUTES)),
    )
    context_start = window_start - dt.timedelta(minutes=context_minutes)
    replay_started = time.monotonic()
    replay_span_seconds = max((event_end - context_start).total_seconds(), 1.0)
    processed_rows = 0
    emitted_predictions = 0

    def report_progress(phase: str, message: str, timestamp=None):
        if progress_callback is None:
            return
        timeline_fraction = 0.0
        if timestamp is not None:
            timeline_fraction = max(0.0, min(
                1.0,
                (utc_datetime(timestamp) - context_start).total_seconds() / replay_span_seconds,
            ))
        event_progress = min(0.99, 0.02 + 0.96 * timeline_fraction)
        elapsed_seconds = max(0.0, time.monotonic() - replay_started)
        rows_per_second = processed_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
        eta_seconds = None
        if 0.02 < event_progress < 0.99 and elapsed_seconds > 0:
            eta_seconds = elapsed_seconds * (1.0 - event_progress) / event_progress
        progress_callback({
            "phase": phase,
            "message": message,
            "event_progress": event_progress,
            "processed_rows": processed_rows,
            "emitted_predictions": emitted_predictions,
            "rows_per_second": rows_per_second,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "current_timestamp": utc_datetime(timestamp).isoformat() if timestamp is not None else None,
            "range_start": context_start.isoformat(),
            "range_end": event_end.isoformat(),
        })

    detector = operating_state.OperatingStateDetector()
    report_progress("motion_profile", "Learning this machine's motion regimes.")
    profile_rows = db.fetch_rows_before(
        conn,
        table,
        context_start,
        config.OPERATING_STATE_HISTORY_ROWS,
        machine_id=machine_id,
    )
    for source_row in profile_rows:
        detector.observe_history(db.canonical_sensor_reading(source_row, columns))
    detector.fit_history()
    report_progress("source_replay", "Motion profile ready; reading historical source rows.")

    monitor = None
    replay_rows = 0
    event_rows = 0
    event_predictions = 0
    maintenance_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    operating_counts: Counter = Counter()
    event_maintenance_counts: Counter = Counter()
    event_outcome_counts: Counter = Counter()
    event_operating_counts: Counter = Counter()
    event_status_seconds: Counter = Counter()
    warn_times: list[dt.datetime] = []
    critical_times: list[dt.datetime] = []
    representative_by_status: dict[str, dict] = {}
    first_event_timestamp = None
    previous_event_timestamp = None
    previous_event_status = None
    max_event_gap_seconds = 0.0
    last_source_timestamp = None
    sampler = _TraceSampler()

    for chunk in db.iter_row_chunks_between(
        table,
        context_start,
        event_end,
        machine_id=machine_id,
        chunk_rows=predict_realtime.INFERENCE_BATCH_ROWS,
        slice_hours=6,
    ):
        evaluated_chunk = []
        pending = []

        def score_pending():
            if not pending:
                return
            report_progress(
                "feature_scoring",
                f"Scoring a batch of {len(pending):,} chronological readings.",
                pending[-1][2],
            )
            results = monitor.update_many(
                [item[1] for item in pending],
                [item[2] for item in pending],
            )
            for (position, _reading, _timestamp, state_name), result in zip(pending, results):
                if state_name != "STARTING":
                    evaluated_chunk[position][2] = result
            pending.clear()

        for source_row in chunk:
            processed_rows += 1
            if processed_rows > MAX_PROCESSED_ROWS_PER_EVENT:
                raise ValueError(
                    f"Event replay requires more than {MAX_PROCESSED_ROWS_PER_EVENT:,} rows. "
                    "Shorten the labelled event range or reduce source density before retrying."
                )
            timestamp = utc_datetime(source_row[columns["timestamp"]])
            reading = db.canonical_sensor_reading(source_row, columns)
            state = detector.update(reading).to_dict()
            position = len(evaluated_chunk)
            evaluated_chunk.append([timestamp, state, None])
            if state["state"] == "STOPPED":
                if state["changed"]:
                    score_pending()
                    monitor = None
            elif state["state"] != "SENSOR_FAULT" and not state["low_motion"]:
                if monitor is None:
                    monitor = factory.create_monitor(machine_id)
                numeric = {column: float(reading[column]) for column in config.RAW_SENSOR_COLS}
                pending.append((position, numeric, timestamp, state["state"]))

        score_pending()

        for timestamp, state, result in evaluated_chunk:
            if timestamp < window_start:
                continue
            replay_rows += 1
            last_source_timestamp = timestamp
            operating_counts[state["state"]] += 1
            in_event = timestamp >= event_start
            if in_event:
                event_rows += 1
                first_event_timestamp = first_event_timestamp or timestamp
                if previous_event_timestamp is not None:
                    max_event_gap_seconds = max(
                        max_event_gap_seconds,
                        (timestamp - previous_event_timestamp).total_seconds(),
                    )
                event_operating_counts[state["state"]] += 1
            if result is not None:
                emitted_predictions += 1
                level = str(result["maintenance"]["level"]).upper()
                maintenance_counts[level] += 1
                outcome_counts[level] += 1
                effective = level
                if level == "WARN":
                    if not warn_times:
                        warn_times.append(timestamp)
                    if (
                        planning_start <= timestamp < urgent_start
                        and not any(planning_start <= item < urgent_start for item in warn_times)
                    ):
                        warn_times.append(timestamp)
                    if (
                        urgent_start <= timestamp <= event_end
                        and not any(urgent_start <= item <= event_end for item in warn_times)
                    ):
                        warn_times.append(timestamp)
                elif level == "CRITICAL":
                    if not critical_times:
                        critical_times.append(timestamp)
                    if timestamp < urgent_start and not any(item < urgent_start for item in critical_times):
                        critical_times.append(timestamp)
                    if (
                        urgent_start <= timestamp <= event_end
                        and not any(urgent_start <= item <= event_end for item in critical_times)
                    ):
                        critical_times.append(timestamp)
                candidate = _representative_tick(timestamp, state, result)
            else:
                effective = (
                    state["state"]
                    if state["state"] in ("STOPPED", "STARTING", "SENSOR_FAULT")
                    else "NO_PREDICTION"
                )
                outcome_counts[effective] += 1
                candidate = _representative_tick(timestamp, state, None)
            if in_event:
                event_outcome_counts[effective] += 1
                representative_by_status[effective] = candidate
                if previous_event_timestamp is None:
                    event_status_seconds[effective] += max(
                        0.0, (timestamp - event_start).total_seconds()
                    )
                else:
                    event_status_seconds[previous_event_status] += max(
                        0.0, (timestamp - previous_event_timestamp).total_seconds()
                    )
                previous_event_timestamp = timestamp
                previous_event_status = effective
                if result is not None:
                    event_predictions += 1
                    event_maintenance_counts[effective] += 1
            sampler.add(_result_point(timestamp, state, result))

        if evaluated_chunk:
            report_progress(
                "source_replay",
                "Historical replay is advancing through the selected timeline.",
                evaluated_chunk[-1][0],
            )

    report_progress("finalizing", "Building event metrics and timing evidence.", event_end)
    if previous_event_timestamp is not None and previous_event_status is not None:
        event_status_seconds[previous_event_status] += max(
            0.0, (event_end - previous_event_timestamp).total_seconds()
        )

    predicted_status = "NO_DATA"
    representative_tick = None
    coverage_weights = (
        event_status_seconds
        if sum(event_status_seconds.values()) > 0
        else event_outcome_counts
    )
    coverage_total = float(sum(coverage_weights.values()))
    if coverage_weights:
        predicted_status = max(
            coverage_weights,
            key=lambda status: (
                coverage_weights[status],
                _RANGE_STATUS_PRIORITY.get(status, -1),
            ),
        )
        representative_tick = representative_by_status[predicted_status]

    stale_seconds = float(runtime_config.get("SOURCE_STALE_SECONDS", config.SOURCE_STALE_SECONDS))
    source_delay_seconds = (
        max(0.0, (first_event_timestamp - event_start).total_seconds())
        if first_event_timestamp is not None else None
    )
    source_age_seconds = (
        max(0.0, (event_end - last_source_timestamp).total_seconds())
        if last_source_timestamp is not None else None
    )
    missing_reason = None
    if event_rows == 0:
        missing_reason = "No source rows exist inside the labelled event range."
    elif source_delay_seconds is not None and source_delay_seconds > stale_seconds:
        missing_reason = (
            f"The first source row is {source_delay_seconds:.0f} seconds after event start; "
            f"the configured freshness limit is {stale_seconds:.0f} seconds."
        )
    elif source_age_seconds is not None and source_age_seconds > stale_seconds:
        missing_reason = (
            f"The newest source row is {source_age_seconds:.0f} seconds before event end; "
            f"the configured freshness limit is {stale_seconds:.0f} seconds."
        )
    elif max_event_gap_seconds > stale_seconds:
        missing_reason = (
            f"The event contains a {max_event_gap_seconds:.0f}-second source gap; "
            f"the configured freshness limit is {stale_seconds:.0f} seconds."
        )

    if missing_reason:
        predicted_status = "NO_DATA"
        representative_tick = {
            "timestamp": event_end.isoformat(),
            "operating_state": "NO_DATA",
            "operating_reason": missing_reason,
        }

    if missing_reason:
        dominant_coverage = 1.0
        expected_coverage = 1.0 if expected == "NO_DATA" else 0.0
        explanation = missing_reason
    else:
        dominant_coverage = (
            coverage_weights[predicted_status] / coverage_total if coverage_total else 0.0
        )
        expected_coverage = (
            coverage_weights[expected] / coverage_total if coverage_total else 0.0
        )
        if predicted_status == expected and expected_coverage >= RANGE_MATCH_MIN_COVERAGE:
            explanation = (
                f"{expected} was the dominant production state across the event range "
                f"({expected_coverage:.1%} of evaluated event time)."
            )
        else:
            explanation = (
                f"The labelled {expected} state covered {expected_coverage:.1%} of evaluated event time; "
                f"the dominant production state was {predicted_status} at {dominant_coverage:.1%}."
            )

    evaluable = True
    match = bool(
        predicted_status == expected
        and expected_coverage >= RANGE_MATCH_MIN_COVERAGE
    )
    timing = _timing_assessment(
        expected,
        event_start,
        event_end,
        warn_times,
        critical_times,
    )

    return artifact_utils.to_json_safe({
        "expected_status": expected,
        "predicted_status": predicted_status,
        "match": match,
        "evaluable": evaluable,
        "aggregation": (
            "dominant causal production state across the labelled event range; "
            f"at least {RANGE_MATCH_MIN_COVERAGE:.0%} time-weighted expected-state coverage is required"
        ),
        "explanation": explanation,
        "window_start": window_start.isoformat(),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        # Legacy aliases keep completed exact-target reports viewable.
        "target_timestamp": event_end.isoformat(),
        "target_rows": event_rows,
        "evaluation_rows": event_rows,
        "processed_rows": processed_rows,
        "replay_rows": replay_rows,
        "context_rows": max(0, processed_rows - replay_rows),
        "emitted_predictions": event_predictions,
        "replay_emitted_predictions": emitted_predictions,
        "coverage": (event_predictions / event_rows) if event_rows else None,
        "expected_status_coverage": expected_coverage,
        "dominant_status_coverage": dominant_coverage,
        "match_minimum_coverage": RANGE_MATCH_MIN_COVERAGE,
        "first_event_source_timestamp": (
            first_event_timestamp.isoformat() if first_event_timestamp else None
        ),
        "source_delay_seconds_at_event_start": source_delay_seconds,
        "source_age_seconds_at_target": source_age_seconds,
        "source_age_seconds_at_event_end": source_age_seconds,
        "maximum_source_gap_seconds": max_event_gap_seconds if event_rows else None,
        "event_status_distribution": dict(event_outcome_counts),
        "event_status_duration_seconds": dict(event_status_seconds),
        "event_maintenance_distribution": dict(event_maintenance_counts),
        "event_operating_state_distribution": dict(event_operating_counts),
        "maintenance_distribution": dict(event_maintenance_counts),
        "lead_window_maintenance_distribution": dict(maintenance_counts),
        "pipeline_outcome_distribution": dict(event_outcome_counts),
        "lead_window_outcome_distribution": dict(outcome_counts),
        "operating_state_distribution": dict(event_operating_counts),
        "lead_window_operating_state_distribution": dict(operating_counts),
        "target_tick": representative_tick,
        "representative_tick": representative_tick,
        "timing": timing,
        "trace": sampler.finish(),
    })


def summarize(range_rows: list[dict]) -> dict:
    """Build event accuracy, within-range coverage, and lead-time evidence."""
    evaluated = []
    errors = 0
    for row in range_rows:
        result = row.get("result") or {}
        if result.get("error"):
            errors += 1
        elif result.get("evaluable"):
            evaluated.append((row, result))

    correct = sum(1 for _, result in evaluated if result.get("match") is True)
    status_coverages = [
        float(result["expected_status_coverage"])
        for _, result in evaluated
        if result.get("expected_status_coverage") is not None
    ]
    timing_evaluated = [
        (row, result) for row, result in evaluated
        if (result.get("timing") or {}).get("timing_evaluable")
    ]
    timing_correct = sum(
        1 for _, result in timing_evaluated
        if (result.get("timing") or {}).get("timing_pass") is True
    )
    confusion: dict[str, dict[str, int]] = {}
    per_machine: dict[str, dict[str, int | float | None]] = {}
    for row, result in evaluated:
        expected = str(result.get("expected_status"))
        predicted = str(result.get("predicted_status"))
        confusion.setdefault(expected, {})[predicted] = confusion.setdefault(expected, {}).get(predicted, 0) + 1
        machine = str(row["machine_id"])
        item = per_machine.setdefault(machine, {
            "evaluated": 0, "correct": 0, "accuracy": None,
            "status_coverage_total": 0.0, "mean_status_coverage": None,
            "timing_evaluated": 0, "timing_correct": 0, "timing_compliance": None,
        })
        item["evaluated"] += 1
        item["correct"] += int(bool(result.get("match")))
        item["status_coverage_total"] += float(result.get("expected_status_coverage") or 0.0)
        timing = result.get("timing") or {}
        if timing.get("timing_evaluable"):
            item["timing_evaluated"] += 1
            item["timing_correct"] += int(timing.get("timing_pass") is True)
    for item in per_machine.values():
        item["accuracy"] = item["correct"] / item["evaluated"] if item["evaluated"] else None
        item["mean_status_coverage"] = (
            item["status_coverage_total"] / item["evaluated"]
            if item["evaluated"] else None
        )
        item["timing_compliance"] = (
            item["timing_correct"] / item["timing_evaluated"]
            if item["timing_evaluated"] else None
        )
        item.pop("status_coverage_total", None)

    return {
        "accuracy": correct / len(evaluated) if evaluated else None,
        "target_accuracy": correct / len(evaluated) if evaluated else None,
        "event_accuracy": correct / len(evaluated) if evaluated else None,
        "mean_status_coverage": (
            sum(status_coverages) / len(status_coverages) if status_coverages else None
        ),
        "timing_compliance": timing_correct / len(timing_evaluated) if timing_evaluated else None,
        "correct_ranges": correct,
        "evaluated_ranges": len(evaluated),
        "correct_targets": correct,
        "evaluated_targets": len(evaluated),
        "timing_correct": timing_correct,
        "timing_evaluated": len(timing_evaluated),
        "premature_critical_cases": sum(
            int(bool((result.get("timing") or {}).get("premature_critical")))
            for _, result in evaluated
        ),
        "false_alert_cases": sum(
            int(bool((result.get("timing") or {}).get("false_alert")))
            for _, result in evaluated
        ),
        "total_ranges": len(range_rows),
        "total_targets": len(range_rows),
        "unscored_ranges": len(range_rows) - len(evaluated) - errors,
        "unscored_targets": len(range_rows) - len(evaluated) - errors,
        "error_ranges": errors,
        "error_targets": errors,
        "accuracy_definition": "expected status versus the time-weighted dominant production state in each labelled event range",
        "range_prediction_rule": "causal seven-day lead-in; require at least 50% labelled-state duration coverage inside the event range",
        "confusion_matrix": confusion,
        "per_machine": per_machine,
    }
