"""
Diagnosis-only helpers: turn the per-feature z-scores from
AnomalyScorer.feature_z_scores() into a human-readable "which sensor(s)
drove this" summary. Never used for the status decision — maintenance.py
still decides OK/WARN/CRITICAL from the single combined health score.
This module only explains a reading after the fact.
"""

SENSOR_PREFIXES = ("vibration", "current", "temperature")


def sensor_label(feature_name: str) -> str:
    """
    'current_rms' -> 'current'; a cross-term like
    'vibration_temperature_corr' -> 'vibration & temperature'.
    """
    if feature_name.endswith("_corr"):
        parts = [p for p in feature_name[: -len("_corr")].split("_") if p in SENSOR_PREFIXES]
        if parts:
            return " & ".join(parts)
    for prefix in SENSOR_PREFIXES:
        if feature_name.startswith(prefix):
            return prefix
    return feature_name


def top_contributors(z_scores, top_k: int = 3) -> list:
    """
    z_scores: pd.Series indexed by feature name (from
    AnomalyScorer.feature_z_scores()).

    Returns up to top_k (feature_name, z, sensor_label) tuples, ranked by
    |z|, most deviant first.
    """
    ranked = z_scores.abs().sort_values(ascending=False)
    out = []
    for feature_name in ranked.index[:top_k]:
        z = float(z_scores[feature_name])
        out.append((feature_name, z, sensor_label(feature_name)))
    return out


def summarize(z_scores, top_k: int = 3) -> str:
    """
    Short string for logs/UI, e.g.:
    "driven mainly by current_rms (z=3.1), current_trend_slope (z=2.7)"
    """
    contributors = top_contributors(z_scores, top_k=top_k)
    if not contributors:
        return ""
    parts = [f"{name} (z={z:+.1f})" for name, z, _ in contributors]
    return "driven mainly by " + ", ".join(parts)
