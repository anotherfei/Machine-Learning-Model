"""
Stacking ensemble (RF + SVR + LightGBM) — DO NOT wire this into
compare_models.py until:

  1. You've run RF / SVR / LightGBM individually via compare_models.py
  2. You know how many independent unit trajectories you have — if it's a
     small number (roughly under 10), a stacking meta-learner will overfit
     to those few trajectories even with many rows, because the rows within
     a trajectory are highly correlated, not independent samples.
  3. LightGBM alone isn't already within noise of your best-case score.

If you do build this out, use nested cross-validation: outer GroupKFold
by unit_id for the final evaluation, inner GroupKFold (also by unit_id)
to fit the meta-learner on out-of-fold base-model predictions. Fitting the
meta-learner on in-sample base predictions will leak and look far better
than it is.
"""

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

import config
from models.base import BaseModel
from models.random_forest import RandomForestModel
from models.svr import SVRModel
from models.lightgbm_model import LightGBMModel


class StackingModel(BaseModel):
    name = "stacking"
    needs_scaling = False  # base estimators handle their own scaling needs internally if wrapped in a Pipeline

    def build(self):
        estimators = [
            ("rf", RandomForestModel().build()),
            ("svr", SVRModel().build()),
            ("lightgbm", LightGBMModel().build()),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=config.N_SPLITS,  # inner CV for generating out-of-fold predictions
            n_jobs=-1,
        )
