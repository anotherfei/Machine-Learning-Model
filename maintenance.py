"""
Maintenance recommendation — the business-rule layer. Everything upstream
(Isolation Forest -> Kalman -> trend forecast -> failure probability)
estimates the machine's state and likely future; this module just decides
when to act on it. Deliberately simple and easy to override per-deployment
— these thresholds are operational policy, not something a model learns.

Three levels: OK / WARN / CRITICAL. An earlier version had a 4th level
(PLAN, between INSPECT and URGENT) but it added a rung of granularity
without a distinct action attached to it, so INSPECT and PLAN were merged
into WARN. The underlying trigger conditions (health threshold, remaining
days, failure probability) are unchanged — only the reported category
changed, so this doesn't affect the pipeline's detection behavior.
"""

import config


def recommend(health_percent: float, remaining_days: int, failure_prob_table: dict,
              trend_trusted: bool = True) -> dict:
    """
    trend_trusted: when False, remaining_days/failure_prob_table are not
    used to trigger WARN/CRITICAL — only health_percent's own two
    thresholds are checked. See config.TREND_SETTLE_TICKS for why this
    exists: right after the trend fit first has enough points, its slope
    estimate is still unreliable, and this is the gate that keeps that
    unreliable estimate from driving an escalation on its own. health_
    percent-based detection is unaffected either way — that's the one
    signal validate.py's ROC/PR/Brier numbers are actually about.
    """
    prob_at_horizon = failure_prob_table.get(config.MAINTENANCE_HORIZON_DAYS)
    if prob_at_horizon is None:
        # horizon not in the table — use the closest available one
        closest = min(failure_prob_table, key=lambda h: abs(h - config.MAINTENANCE_HORIZON_DAYS))
        prob_at_horizon = failure_prob_table[closest]

    if health_percent <= config.FAILURE_HEALTH_THRESHOLD:
        level, reason = "CRITICAL", "Health at or below failure threshold now."
    elif trend_trusted and remaining_days <= config.MAINTENANCE_REMAINING_DAYS_URGENT:
        level, reason = "CRITICAL", f"Estimated remaining life ({remaining_days}d) is critically short."
    elif trend_trusted and prob_at_horizon >= config.MAINTENANCE_PROB_URGENT:
        level, reason = "CRITICAL", (
            f"Failure probability within {config.MAINTENANCE_HORIZON_DAYS}d is "
            f"{prob_at_horizon:.0%}."
        )
    elif trend_trusted and prob_at_horizon >= config.MAINTENANCE_PROB_PLAN:
        level, reason = "WARN", (
            f"Failure probability within {config.MAINTENANCE_HORIZON_DAYS}d is "
            f"{prob_at_horizon:.0%} — schedule maintenance."
        )
    elif health_percent <= config.MAINTENANCE_HEALTH_INSPECT:
        level, reason = "WARN", f"Health ({health_percent:.1f}%) below inspection threshold."
    else:
        level, reason = "OK", "No action needed."

    return {"level": level, "reason": reason}