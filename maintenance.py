"""Maintenance policy and timestamp-based status stabilization."""

import config
import runtime_config

_SEVERITY = {"OK": 0, "WARN": 1, "CRITICAL": 2}


def is_alert_escalation(previous_level, current_level):
    """Return true only when maintenance moves into a higher alert severity."""
    if current_level not in ("WARN", "CRITICAL"):
        return False
    return _SEVERITY[current_level] > _SEVERITY.get(previous_level, 0)


def _risk_at(failure_prob_table: dict, horizon_days: float) -> float:
    """Return the closest available horizon, accepting JSON string keys."""
    probabilities = {float(key): float(value) for key, value in failure_prob_table.items()}
    if not probabilities:
        return 0.0
    closest = min(probabilities, key=lambda value: abs(value - horizon_days))
    return probabilities[closest]


def recommend(health_percent: float, remaining_days: int, failure_prob_table: dict,
              trend_trusted: bool = True) -> dict:
    """Return the raw OK/WARN/CRITICAL decision before persistence filtering.

    Direct condition rules use the Kalman-smoothed condition score. Forecast
    rules are considered only after the upstream trend confidence gate passes.
    CRITICAL uses a short urgent horizon; WARN can use a broader planning
    horizon. ``remaining_days`` remains explanatory and never fires a rule by
    itself because it is a point estimate without uncertainty accounting.
    """
    plan_horizon = float(runtime_config.get(
        "MAINTENANCE_HORIZON_DAYS", config.MAINTENANCE_HORIZON_DAYS
    ))
    urgent_horizon = float(runtime_config.get(
        "MAINTENANCE_URGENT_HORIZON_DAYS", config.MAINTENANCE_URGENT_HORIZON_DAYS
    ))
    plan_probability = _risk_at(failure_prob_table, plan_horizon)
    urgent_probability = _risk_at(failure_prob_table, urgent_horizon)

    if health_percent <= runtime_config.get(
        "FAILURE_HEALTH_THRESHOLD", config.FAILURE_HEALTH_THRESHOLD
    ):
        return {
            "level": "CRITICAL",
            "reason": "Condition at or below the critical threshold now.",
            "trigger": "health_threshold",
        }

    if trend_trusted and urgent_probability >= runtime_config.get(
        "MAINTENANCE_PROB_URGENT", config.MAINTENANCE_PROB_URGENT
    ):
        return {
            "level": "CRITICAL",
            "reason": (
                f"Forecast boundary-crossing risk within {urgent_horizon:g}d is "
                f"{urgent_probability:.0%} (est. {remaining_days}d remaining at current trend)."
            ),
            "trigger": "trend_probability",
        }

    if trend_trusted and plan_probability >= runtime_config.get(
        "MAINTENANCE_PROB_PLAN", config.MAINTENANCE_PROB_PLAN
    ):
        return {
            "level": "WARN",
            "reason": (
                f"Forecast boundary-crossing risk within {plan_horizon:g}d is "
                f"{plan_probability:.0%} (est. {remaining_days}d remaining at current trend) — "
                "schedule maintenance."
            ),
            "trigger": "trend_probability",
        }

    if health_percent <= runtime_config.get(
        "MAINTENANCE_HEALTH_INSPECT", config.MAINTENANCE_HEALTH_INSPECT
    ):
        return {
            "level": "WARN",
            "reason": f"Condition ({health_percent:.1f}%) below inspection threshold.",
            "trigger": "health_inspect",
        }

    return {"level": "OK", "reason": "No action needed.", "trigger": "none"}


class MaintenanceDebouncer:
    """Stabilize maintenance status using elapsed source time.

    WARN and CRITICAL require sustained evidence before promotion. A CRITICAL
    candidate that begins from OK passes through WARN once the warning duration
    is reached, then becomes CRITICAL only if it survives the critical duration.
    Any move to a lower severity requires the recovery duration. Using source
    timestamps keeps every duration correct when cadence or polling changes.
    """

    def __init__(self):
        self._reported = {
            "level": "OK",
            "reason": "No action needed.",
            "trigger": "none",
        }
        self._candidate_level = None
        self._candidate_since = None
        self._fallback_elapsed_minutes = -1.0
        self._last_elapsed_minutes = None

    def evaluate(self, health_percent: float, remaining_days: int,
                 failure_prob_table: dict, trend_trusted: bool = True,
                 elapsed_minutes: float | None = None) -> dict:
        raw = recommend(
            health_percent,
            remaining_days,
            failure_prob_table,
            trend_trusted=trend_trusted,
        )
        if elapsed_minutes is None:
            self._fallback_elapsed_minutes += 1.0
            now = self._fallback_elapsed_minutes
        else:
            now = float(elapsed_minutes)

        stale_gap_minutes = float(runtime_config.get(
            "SOURCE_STALE_SECONDS", config.SOURCE_STALE_SECONDS
        )) / 60.0
        discontinuous = (
            self._last_elapsed_minutes is not None
            and (
                now < self._last_elapsed_minutes
                or now - self._last_elapsed_minutes > stale_gap_minutes
            )
        )
        self._last_elapsed_minutes = now
        if discontinuous:
            # Missing source time is not confirming evidence. Begin any
            # candidate again from the first fresh reading after the gap.
            self._candidate_level = None
            self._candidate_since = None

        level = raw["level"]
        reported_level = self._reported["level"]
        if level == reported_level:
            self._candidate_level = None
            self._candidate_since = None
            self._reported = raw
            return raw

        if (
            self._candidate_level != level
            or self._candidate_since is None
            or now < self._candidate_since
        ):
            self._candidate_level = level
            self._candidate_since = now

        persisted = max(0.0, now - self._candidate_since)
        escalating = _SEVERITY[level] > _SEVERITY[reported_level]
        if escalating and level == "CRITICAL":
            critical_required = float(runtime_config.get(
                "MAINTENANCE_CRITICAL_CONFIRM_MINUTES",
                config.MAINTENANCE_CRITICAL_CONFIRM_MINUTES,
            ))
            warn_required = float(runtime_config.get(
                "MAINTENANCE_WARN_CONFIRM_MINUTES",
                config.MAINTENANCE_WARN_CONFIRM_MINUTES,
            ))
            if reported_level == "OK" and persisted < critical_required:
                if persisted >= warn_required:
                    self._reported = {
                        "level": "WARN",
                        "reason": (
                            "Sustained critical evidence has reached the warning "
                            f"stage; CRITICAL requires {critical_required:g} minutes. "
                            f"Candidate evidence: {raw['reason']}"
                        ),
                        "trigger": raw["trigger"],
                    }
                    return {
                        **self._reported,
                        "stabilizing": True,
                        "candidate_level": "CRITICAL",
                        "candidate_elapsed_minutes": round(persisted, 3),
                        "candidate_required_minutes": critical_required,
                    }
                required = warn_required
            else:
                required = critical_required
        elif escalating:
            required = float(runtime_config.get(
                "MAINTENANCE_WARN_CONFIRM_MINUTES",
                config.MAINTENANCE_WARN_CONFIRM_MINUTES,
            ))
        else:
            required = float(runtime_config.get(
                "MAINTENANCE_RECOVERY_MINUTES",
                config.MAINTENANCE_RECOVERY_MINUTES,
            ))

        if persisted >= required:
            self._reported = raw
            self._candidate_level = None
            self._candidate_since = None
            return raw

        action = "promote" if escalating else "clear"
        return {
            **self._reported,
            "reason": (
                f"Holding at {reported_level}: candidate {level} has persisted for "
                f"{persisted:.1f}/{required:g} minutes; waiting to {action}. "
                f"Candidate evidence: {raw['reason']}"
            ),
            "stabilizing": True,
            "candidate_level": level,
            "candidate_elapsed_minutes": round(persisted, 3),
            "candidate_required_minutes": required,
        }
