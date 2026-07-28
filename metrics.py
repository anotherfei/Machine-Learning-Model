"""
Shared evaluation metrics — every model must be scored with this exact
function so comparisons across models/*.py are apples-to-apples.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def phm08_score(y_true, y_pred, alpha_early: int = None, alpha_late: int = None) -> float:
    """
    PHM08 asymmetric scoring function. Penalizes late predictions
    (pred > true, i.e. model thinks there's more life left than there is)
    more heavily than early predictions — because an overly optimistic RUL
    estimate is the operationally dangerous failure mode.

    Lower is better. Not bounded the same way RMSE is — use it for relative
    model comparison, not as a standalone interpretable number.
    """
    alpha_early = alpha_early or config.PHM08_ALPHA_EARLY
    alpha_late = alpha_late or config.PHM08_ALPHA_LATE

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true  # positive = late (overestimate), negative = early (underestimate)

    scores = np.where(
        d < 0,
        np.exp(-d / alpha_early) - 1,
        np.exp(d / alpha_late) - 1,
    )
    return float(np.sum(scores))


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "PHM08": phm08_score(y_true, y_pred),
    }


def print_scores(model_name: str, scores: dict):
    line = " | ".join(f"{k}: {v:.4f}" for k, v in scores.items())
    print(f"[{model_name}] {line}")
