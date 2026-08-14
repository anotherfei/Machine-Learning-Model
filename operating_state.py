"""Per-machine motion-state gate for feeds without a PLC run signal.

This module does not claim to detect electrical power.  It distinguishes a
stationary spindle from a rotating spindle using that machine's own vibration
history.  If the history does not contain two clearly separated regimes, the
state remains UNKNOWN and inference is allowed to continue conservatively.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

import config
import runtime_config


MOTION_COLS = (config.COL_A_RMS, config.COL_V_RMS, config.COL_A_PEAK)
VALID_STATES = ("UNKNOWN", "RUNNING", "STOPPED", "STARTING", "SENSOR_FAULT")


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
        if len(values) < config.OPERATING_STATE_MIN_HISTORY_ROWS:
            return False

        low, high = np.quantile(values, [0.2, 0.8])
        if not np.isfinite(low + high) or high - low < config.OPERATING_STATE_MIN_LOG_SEPARATION:
            return False

        # Deterministic one-dimensional two-means; no model artifact or shared
        # cross-machine scale is involved.
        for _ in range(50):
            low_group = values[np.abs(values - low) <= np.abs(values - high)]
            high_group = values[np.abs(values - low) > np.abs(values - high)]
            if not len(low_group) or not len(high_group):
                return False
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

        longest_low_run = 0
        current_low_run = 0
        for is_low in low_mask:
            current_low_run = current_low_run + 1 if is_low else 0
            longest_low_run = max(longest_low_run, current_low_run)
        minimum_cluster = max(
            config.OPERATING_STATE_MIN_CLUSTER_ROWS,
            int(len(values) * config.OPERATING_STATE_MIN_CLUSTER_FRACTION),
        )
        separation = high - low
        if (len(low_group) < minimum_cluster or len(high_group) < minimum_cluster or
                (require_sustained_low and longest_low_run < runtime_config.get(
                    "OPERATING_STATE_STOP_CONFIRM_TICKS",
                    config.OPERATING_STATE_STOP_CONFIRM_TICKS,
                )) or
                separation < config.OPERATING_STATE_MIN_LOG_SEPARATION):
            return False

        pooled_spread = max(float(np.std(low_group) + np.std(high_group)), 1e-6)
        separation_quality = separation / pooled_spread
        if separation_quality < config.OPERATING_STATE_MIN_SEPARATION_QUALITY:
            return False

        self.stop_threshold = low + separation * 0.40
        self.run_threshold = low + separation * 0.65
        self.confidence = round(min(1.0, separation_quality / 8.0), 3)
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
