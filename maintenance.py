"""
Maintenance recommendation — the business-rule layer. Everything upstream
(Isolation Forest -> Kalman -> trend forecast -> failure probability)
estimates the machine's state and likely future; this module just decides
when to act on it. Deliberately simple and easy to override per-deployment
— these thresholds are operational policy, not something a model learns.

Three levels: OK / WARN / CRITICAL. An earlier version had a 4th level
(PLAN, between INSPECT and URGENT) but it added a rung of granularity
without a distinct action attached to it, so INSPECT and PLAN were merged
into WARN.
"""

import config

_SEVERITY = {"OK": 0, "WARN": 1, "CRITICAL": 2}


def recommend(health_percent: float, remaining_days: int, failure_prob_table: dict,
              trend_trusted: bool = True) -> dict:
    """
    trend_trusted: when False, failure_prob_table is not used to trigger
    WARN/CRITICAL — only health_percent's own two thresholds are checked.
    See config.TREND_SETTLE_TICKS for why this exists: right after the
    trend fit first has enough points, its slope estimate is still
    unreliable, and this is the gate that keeps that unreliable estimate
    from driving an escalation on its own. health_percent-based detection
    is unaffected either way — that's the one signal validate.py's
    ROC/PR/Brier numbers are actually about.

    remaining_days no longer independently triggers WARN/CRITICAL — only
    failure_prob_table does. remaining_days is a bare point-estimate
    extrapolation (see trend_forecast.remaining_days()) with no
    uncertainty accounting; failure_prob_table uses the SAME slope
    estimate plus the fit's residual_std through a proper random-walk
    model (failure_probability.py), so it's strictly more informative —
    a noisy tick can floor remaining_days at 1 day while
    failure_prob_table correctly stays low because it accounts for that
    same noise as uncertainty. Keeping both as independent triggers meant
    a noisy tick could bypass the uncertainty accounting entirely through
    the remaining_days branch. remaining_days is still computed upstream
    and reported here for human context (see "reason" text) — it just
    can't escalate anything on its own anymore.

    Returns {"level", "reason", "trigger"}. trigger identifies which rule
    actually fired: "health_threshold" or "health_inspect" (both instant,
    already the Kalman-smoothed signal), "trend_probability" (needs
    debounce — see MaintenanceDebouncer), or "none" (OK).
    """
    prob_at_horizon = failure_prob_table.get(config.MAINTENANCE_HORIZON_DAYS)
    if prob_at_horizon is None:
        # horizon not in the table — use the closest available one
        closest = min(failure_prob_table, key=lambda h: abs(h - config.MAINTENANCE_HORIZON_DAYS))
        prob_at_horizon = failure_prob_table[closest]

    if health_percent <= config.FAILURE_HEALTH_THRESHOLD:
        return {"level": "CRITICAL", "reason": "Health at or below failure threshold now.",
                "trigger": "health_threshold"}

    if trend_trusted and prob_at_horizon >= config.MAINTENANCE_PROB_URGENT:
        return {"level": "CRITICAL", "reason": (
            f"Failure probability within {config.MAINTENANCE_HORIZON_DAYS}d is "
            f"{prob_at_horizon:.0%} (est. {remaining_days}d remaining at current trend)."
        ), "trigger": "trend_probability"}

    if trend_trusted and prob_at_horizon >= config.MAINTENANCE_PROB_PLAN:
        return {"level": "WARN", "reason": (
            f"Failure probability within {config.MAINTENANCE_HORIZON_DAYS}d is "
            f"{prob_at_horizon:.0%} (est. {remaining_days}d remaining at current trend) — "
            f"schedule maintenance."
        ), "trigger": "trend_probability"}

    if health_percent <= config.MAINTENANCE_HEALTH_INSPECT:
        return {"level": "WARN", "reason": f"Health ({health_percent:.1f}%) below inspection threshold.",
                "trigger": "health_inspect"}

    return {"level": "OK", "reason": "No action needed.", "trigger": "none"}


class MaintenanceDebouncer:
    """
    Wraps recommend() with hysteresis on the trend/failure-probability
    trigger ONLY. health_threshold and health_inspect triggers pass
    through instantly, unchanged — they're already the Kalman-smoothed
    signal (see kalman.py) and don't need a second smoothing pass on top;
    debouncing an already-smoothed signal just adds reporting lag to a
    real emergency for no benefit.

    Why this exists, confirmed not assumed: even after collapsing
    remaining_days/failure_probability into one trigger (see recommend()),
    replaying data/raw/spindle_train.csv (43,176 rows, 100%
    health_status=='normal') through this debouncer instead of raw
    recommend() cut false CRITICAL ticks further — see validate.py's
    output for the current before/after count; a single-trigger fix alone
    reduces noise-driven escalations, hysteresis reduces the remainder
    that still come from short streaks of noisy-but-still-significant
    trend fits.

    One instance per monitored unit, holding state across ticks —
    predict_realtime.py's SpindleMonitor and validate.py's run_backtest()
    must each create exactly one and call .evaluate() every tick, the
    same way both already do for HealthKalmanFilter, or their reported
    maintenance_level will diverge from each other.
    """

    def __init__(self):
        self._candidate_level = "OK"
        self._candidate_count = 0
        self._reported_trend_level = "OK"

    def evaluate(self, health_percent: float, remaining_days: int, failure_prob_table: dict,
                 trend_trusted: bool = True) -> dict:
        raw = recommend(health_percent, remaining_days, failure_prob_table, trend_trusted=trend_trusted)

        if raw["trigger"] in ("health_threshold", "health_inspect"):
            # Instant bypass. Also resets trend-candidate tracking so a
            # health-driven tick doesn't silently count toward an
            # unrelated trend streak once health recovers.
            self._candidate_level = raw["level"]
            self._candidate_count = 1
            return raw

        # raw["trigger"] is "trend_probability" or "none" (OK) — both go
        # through the same hysteresis state machine. "none" competing
        # against a currently-reported WARN/CRITICAL is exactly the
        # de-escalation case this exists to slow down; it isn't a bypass.
        level = raw["level"]
        if level == self._candidate_level:
            self._candidate_count += 1
        else:
            self._candidate_level = level
            self._candidate_count = 1

        escalating = _SEVERITY[level] > _SEVERITY[self._reported_trend_level]
        required = (config.MAINTENANCE_TREND_DEBOUNCE_TICKS if escalating
                    else config.MAINTENANCE_TREND_RECOVERY_TICKS)
        if self._candidate_count >= required:
            self._reported_trend_level = level

        if self._reported_trend_level == "OK":
            return {"level": "OK", "reason": "No action needed.", "trigger": "none"}
        if self._reported_trend_level == level:
            return raw
        remaining_ticks = required - self._candidate_count
        return {
            "level": self._reported_trend_level,
            "reason": (f"Holding at {self._reported_trend_level} (trend-based) — latest tick reads "
                       f"{level}, needs {remaining_ticks} more consecutive tick(s) to "
                       f"{'confirm' if escalating else 'clear'}."),
            "trigger": "trend_probability",
        }