"""Per-machine motion-state gate for feeds without a PLC run signal.

This module does not claim to detect electrical power.  It distinguishes a
stationary spindle from a rotating spindle using that machine's own vibration
history.  If the history does not contain two clearly separated regimes, the
state remains UNKNOWN and inference is allowed to continue conservatively.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np

import config
import runtime_config


MOTION_COLS = (config.COL_A_RMS, config.COL_V_RMS, config.COL_A_PEAK)
VALID_STATES = ("UNKNOWN", "RUNNING", "STOPPED", "STARTING", "SENSOR_FAULT")
OPERATOR_STATES = ("RUNNING", "STOPPED")
_LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo or timezone.utc


def _timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_LOCAL_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def override_is_active(document: dict | None, at=None, *, require_started: bool = False) -> bool:
    """Return whether a bounded operator confirmation applies at ``at``."""
    if not document or document.get("state") not in OPERATOR_STATES:
        return False
    moment = _timestamp(at) or datetime.now(timezone.utc)
    expires_at = _timestamp(document.get("expires_at"))
    set_at = _timestamp(document.get("set_at"))
    if expires_at is None or moment >= expires_at:
        return False
    return not (require_started and set_at is not None and moment < set_at)


def load_override(conn, machine_id: str, at=None) -> dict | None:
    """Load one current operator confirmation from the consolidated state table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT value,updated_at,updated_by FROM state "
            "WHERE namespace='operating_override' AND key=%s",
            (machine_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    document = dict(row[0] or {})
    document.setdefault("set_at", row[1])
    document.setdefault("set_by", row[2])
    return document if override_is_active(document, at) else None


def load_active_overrides(conn, at=None) -> dict[str, dict]:
    """Load all non-expired confirmations once for the realtime worker."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT key,value,updated_at,updated_by FROM state "
            "WHERE namespace='operating_override'"
        )
        rows = cur.fetchall()
    active = {}
    for machine_id, value, updated_at, updated_by in rows:
        document = dict(value or {})
        document.setdefault("set_at", updated_at)
        document.setdefault("set_by", updated_by)
        if override_is_active(document, at):
            active[str(machine_id)] = document
    return active


def save_override(conn, machine_id: str, state: str, expires_minutes: int, username: str, note: str = "") -> dict:
    """Persist a time-bounded, attributable operator motion confirmation."""
    state = str(state).strip().upper()
    if state not in OPERATOR_STATES:
        raise ValueError("state must be RUNNING or STOPPED")
    if not 5 <= int(expires_minutes) <= 1440:
        raise ValueError("expires_minutes must be between 5 and 1440")
    note = str(note or "").strip()
    if len(note) > 240:
        raise ValueError("note must contain at most 240 characters")
    now = datetime.now(timezone.utc)
    document = {
        "state": state,
        "set_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=int(expires_minutes))).isoformat(),
        "set_by": username,
        "note": note,
    }
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO state(namespace,key,value,updated_at,updated_by)
               VALUES('operating_override',%s,%s::jsonb,now(),%s)
               ON CONFLICT(namespace,key) DO UPDATE SET
                 value=EXCLUDED.value,updated_at=now(),updated_by=EXCLUDED.updated_by""",
            (machine_id, json.dumps(document), username),
        )
        cur.execute("NOTIFY operating_override_changed, %s", (machine_id,))
    return document


def clear_override(conn, machine_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM state WHERE namespace='operating_override' AND key=%s RETURNING key",
            (machine_id,),
        )
        removed = cur.fetchone() is not None
        cur.execute("NOTIFY operating_override_changed, %s", (machine_id,))
    return removed


def apply_operator_override(detected: dict, document: dict | None, at=None) -> dict:
    """Return an effective state while retaining the automatic evidence."""
    result = dict(detected)
    result.update({
        "state_source": "automatic",
        "detected_state": detected.get("state", "UNKNOWN"),
        "detected_reason": detected.get("reason"),
        "detected_confidence": detected.get("confidence"),
        "override_set_by": None,
        "override_set_at": None,
        "override_expires_at": None,
        "override_note": None,
    })
    # A human can confirm motion, but cannot make invalid sensor channels valid.
    if (
        detected.get("state") == "SENSOR_FAULT"
        or not override_is_active(document)
        or not override_is_active(document, at, require_started=True)
    ):
        return result
    state = document["state"]
    result.update({
        "state": state,
        "reason": (
            f"Operator {document.get('set_by') or 'unknown'} confirmed this machine "
            f"{state.lower()} until {_timestamp(document.get('expires_at')).isoformat()}."
        ),
        "confidence": 1.0,
        "low_motion": state == "STOPPED",
        "state_source": "operator",
        "override_set_by": document.get("set_by"),
        "override_set_at": document.get("set_at"),
        "override_expires_at": document.get("expires_at"),
        "override_note": document.get("note") or None,
    })
    return result


@dataclass(frozen=True)
class OperatingStateResult:
    state: str
    reason: str
    confidence: float
    changed: bool
    activity_score: float | None
    low_motion: bool
    stop_threshold: float | None
    run_threshold: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class OperatingStateDetector:
    """Learn two motion regimes and apply debounced state transitions.

    Activity is the median log-magnitude of the three vibration channels.  A
    log score makes the detector insensitive to a machine's absolute scale,
    and the median prevents a single noisy channel from deciding the state.
    """

    def __init__(self, history_rows: int | None = None):
        self.state = "UNKNOWN"
        self.reason = "Learning this machine's vibration regimes."
        self.confidence = 0.0
        self.stop_threshold = None
        self.run_threshold = None
        self._scores = deque(maxlen=history_rows or config.OPERATING_STATE_HISTORY_ROWS)
        self._low_ticks = 0
        self._high_ticks = 0
        self._starting_ticks = 0
        self._valid_since_fit_attempt = 0
        self.last_fit_diagnostics = {}

    @staticmethod
    def _validate(reading: dict) -> str | None:
        for column in config.RAW_SENSOR_COLS:
            value = reading.get(column)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return f"Required sensor channel {column} is missing or invalid."
            if not math.isfinite(value):
                return f"Required sensor channel {column} is not finite."
            if column in MOTION_COLS and value < 0:
                return f"Motion channel {column} cannot be negative."
        return None

    @staticmethod
    def activity_score(reading: dict) -> float:
        logs = [math.log10(max(abs(float(reading[column])), 1e-12)) for column in MOTION_COLS]
        return float(np.median(logs))

    def observe_history(self, reading: dict) -> None:
        """Add a historical sample without changing the live state."""
        if self._validate(reading) is None:
            self._scores.append(self.activity_score(reading))

    def observe_activity_score(self, score: float) -> None:
        """Add an already validated activity score to an offline profile."""
        score = float(score)
        if math.isfinite(score):
            self._scores.append(score)

    def fit_history(self, *, require_sustained_low: bool = True) -> bool:
        """Fit conservative low/high regimes; return whether they are usable."""
        self._valid_since_fit_attempt = 0
        values = np.asarray(self._scores, dtype=float)
        self.last_fit_diagnostics = {
            "sample_rows": int(len(values)),
            "require_sustained_low": bool(require_sustained_low),
            "candidates": [],
        }
        if len(values) < config.OPERATING_STATE_MIN_HISTORY_ROWS:
            self.last_fit_diagnostics["rejection"] = "insufficient_history"
            return False

        if not np.isfinite(values).all():
            self.last_fit_diagnostics["rejection"] = "non_finite_activity_scores"
            return False

        # Live fitting retains the original broad 20/80 initialization and
        # fractional cluster requirement. Offline commissioning profiles are a
        # deterministic uniform reservoir, where a genuine shutdown may be a
        # very small fraction of months of running data. Try progressively
        # rarer low-tail initializations there; the chronological second pass
        # still requires consecutive low ticks before declaring STOPPED.
        seed_quantiles = (
            [(0.20, 0.80)] if require_sustained_low else
            [(0.20, 0.80), (0.05, 0.80), (0.01, 0.65), (0.001, 0.50)]
        )
        stop_confirm = int(runtime_config.get(
            "OPERATING_STATE_STOP_CONFIRM_TICKS",
            config.OPERATING_STATE_STOP_CONFIRM_TICKS,
        ))
        minimum_cluster = (
            max(
                config.OPERATING_STATE_MIN_CLUSTER_ROWS,
                int(len(values) * config.OPERATING_STATE_MIN_CLUSTER_FRACTION),
            )
            if require_sustained_low else
            max(config.OPERATING_STATE_MIN_CLUSTER_ROWS, stop_confirm * 5)
        )
        selected = None
        for low_quantile, high_quantile in seed_quantiles:
            low, high = np.quantile(values, [low_quantile, high_quantile])
            if not np.isfinite(low + high) or low == high:
                self.last_fit_diagnostics["candidates"].append({
                    "seed_quantiles": [low_quantile, high_quantile],
                    "accepted": False,
                    "rejection": "identical_or_invalid_initial_centers",
                })
                continue

            # Deterministic one-dimensional two-means; no model artifact or
            # shared cross-machine scale is involved.
            for _ in range(50):
                low_mask = np.abs(values - low) <= np.abs(values - high)
                low_group = values[low_mask]
                high_group = values[~low_mask]
                if not len(low_group) or not len(high_group):
                    break
                new_low, new_high = float(low_group.mean()), float(high_group.mean())
                if abs(new_low - low) + abs(new_high - high) < 1e-7:
                    low, high = new_low, new_high
                    break
                low, high = new_low, new_high
            if low > high:
                low, high = high, low
            low_mask = np.abs(values - low) <= np.abs(values - high)
            low_group = values[low_mask]
            high_group = values[~low_mask]
            if not len(low_group) or not len(high_group):
                self.last_fit_diagnostics["candidates"].append({
                    "seed_quantiles": [low_quantile, high_quantile],
                    "accepted": False,
                    "rejection": "empty_cluster",
                })
                continue

            longest_low_run = 0
            current_low_run = 0
            for is_low in low_mask:
                current_low_run = current_low_run + 1 if is_low else 0
                longest_low_run = max(longest_low_run, current_low_run)
            separation = float(high - low)
            ordinary_pooled_spread = max(
                float(np.std(low_group) + np.std(high_group)), 1e-6
            )
            robust_pooled_spread = max(
                float(
                    np.median(np.abs(low_group - np.median(low_group))) * 1.4826
                    + np.median(np.abs(high_group - np.median(high_group))) * 1.4826
                ),
                1e-6,
            )
            # A months-long commissioning profile can contain legitimate
            # speed/load variation within RUNNING. Robust within-cluster spread
            # prevents a few high-load readings from hiding a tightly
            # quantized idle cluster. The shorter live history remains on the
            # more conservative ordinary-spread rule.
            pooled_spread = (
                ordinary_pooled_spread if require_sustained_low
                else robust_pooled_spread
            )
            separation_quality = float(separation / pooled_spread)
            minimum_separation_quality = (
                config.OPERATING_STATE_MIN_SEPARATION_QUALITY
                if require_sustained_low else
                config.OPERATING_STATE_COMMISSIONING_MIN_SEPARATION_QUALITY
            )
            rejection = None
            if len(low_group) < minimum_cluster or len(high_group) < minimum_cluster:
                rejection = "cluster_too_small"
            elif require_sustained_low and longest_low_run < stop_confirm:
                rejection = "no_sustained_low_run"
            elif separation < config.OPERATING_STATE_MIN_LOG_SEPARATION:
                rejection = "insufficient_log_separation"
            elif separation_quality < minimum_separation_quality:
                rejection = "insufficient_separation_quality"
            candidate = {
                "seed_quantiles": [low_quantile, high_quantile],
                "low_center": float(low),
                "high_center": float(high),
                "low_rows": int(len(low_group)),
                "high_rows": int(len(high_group)),
                "minimum_cluster_rows": int(minimum_cluster),
                "longest_low_run": int(longest_low_run),
                "log_separation": separation,
                "ordinary_pooled_spread": ordinary_pooled_spread,
                "robust_pooled_spread": robust_pooled_spread,
                "separation_quality_method": (
                    "ordinary_std" if require_sustained_low else "robust_mad"
                ),
                "separation_quality": separation_quality,
                "minimum_separation_quality": float(minimum_separation_quality),
                "accepted": rejection is None,
                "rejection": rejection,
            }
            self.last_fit_diagnostics["candidates"].append(candidate)
            if rejection is None:
                selected = candidate
                break

        if selected is None:
            self.last_fit_diagnostics["rejection"] = "no_usable_two_regime_candidate"
            return False

        low = selected["low_center"]
        high = selected["high_center"]
        separation = selected["log_separation"]
        separation_quality = selected["separation_quality"]
        self.stop_threshold = low + separation * 0.40
        self.run_threshold = low + separation * 0.65
        self.confidence = round(min(1.0, separation_quality / 8.0), 3)
        self.last_fit_diagnostics["selected"] = selected
        self.reason = "Distinct stationary and rotating vibration regimes were learned for this machine."
        return True

    def _result(self, changed: bool, score: float | None) -> OperatingStateResult:
        return OperatingStateResult(
            state=self.state,
            reason=self.reason,
            confidence=self.confidence,
            changed=changed,
            activity_score=score,
            low_motion=(score is not None and self.stop_threshold is not None and score <= self.stop_threshold),
            stop_threshold=self.stop_threshold,
            run_threshold=self.run_threshold,
        )

    def _set_state(self, state: str, reason: str) -> bool:
        changed = state != self.state
        self.state = state
        self.reason = reason
        return changed

    def update(self, reading: dict) -> OperatingStateResult:
        invalid_reason = self._validate(reading)
        if invalid_reason:
            changed = self._set_state("SENSOR_FAULT", invalid_reason)
            self._low_ticks = self._high_ticks = self._starting_ticks = 0
            return self._result(changed, None)

        score = self.activity_score(reading)
        self._scores.append(score)
        self._valid_since_fit_attempt += 1

        # Retry learning as live data accumulates, but freeze a good threshold
        # pair for the rest of the process so it cannot drift toward a fault.
        if self.stop_threshold is None and (
            len(self._scores) == config.OPERATING_STATE_MIN_HISTORY_ROWS or
            self._valid_since_fit_attempt >= config.OPERATING_STATE_REFIT_TICKS
        ):
            self.fit_history()

        if self.stop_threshold is None or self.run_threshold is None:
            changed = self._set_state(
                "UNKNOWN",
                "No clearly separated stationary/running regimes yet; ML monitoring remains enabled.",
            )
            return self._result(changed, score)

        low = score <= self.stop_threshold
        high = score >= self.run_threshold
        self._low_ticks = self._low_ticks + 1 if low else 0
        self._high_ticks = self._high_ticks + 1 if high else 0

        previous = self.state
        if self.state == "STOPPED":
            if self._high_ticks >= runtime_config.get(
                "OPERATING_STATE_START_CONFIRM_TICKS",
                config.OPERATING_STATE_START_CONFIRM_TICKS,
            ):
                self._starting_ticks = 1
                self._set_state("STARTING", "Rotation returned; rebuilding the monitoring window.")
            elif high:
                self.reason = "Possible restart detected; waiting for sustained rotating vibration."
            else:
                self.reason = "Vibration remains in this machine's learned stationary regime."
        elif self.state == "STARTING":
            if low:
                self._starting_ticks = 0
                self._set_state("STOPPED", "Restart did not persist; vibration returned to the stationary regime.")
            else:
                self._starting_ticks += 1
                startup_warmup = config.WINDOW_SIZE + int(runtime_config.get(
                    "KALMAN_INIT_SAMPLES", config.KALMAN_INIT_SAMPLES
                ))
                if self._starting_ticks >= startup_warmup:
                    self._set_state("RUNNING", "Rotation is stable and the monitoring pipeline is warm.")
                else:
                    self.reason = "Rotation detected; rebuilding rolling features and health history."
        elif self._low_ticks >= runtime_config.get(
            "OPERATING_STATE_STOP_CONFIRM_TICKS",
            config.OPERATING_STATE_STOP_CONFIRM_TICKS,
        ):
            self._set_state("STOPPED", "Sustained low vibration confirms the spindle is stationary.")
        elif self._high_ticks >= runtime_config.get(
            "OPERATING_STATE_START_CONFIRM_TICKS",
            config.OPERATING_STATE_START_CONFIRM_TICKS,
        ):
            self._set_state("RUNNING", "Vibration is in this machine's learned rotating regime.")
        elif self.state == "SENSOR_FAULT":
            self._set_state("UNKNOWN", "Valid readings resumed; confirming the operating regime.")
        else:
            self.reason = (
                "Possible shutdown detected; suppressing ML while low vibration is confirmed."
                if low else
                "Vibration is between learned state thresholds; retaining the current state."
            )

        return self._result(self.state != previous, score)
