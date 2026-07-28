from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.svr import SVRModel
from models.lightgbm_model import LightGBMModel
# Stacking is intentionally excluded from the default registry — see
# models/stacking.py docstring for why. Import it explicitly if/when ready.

MODEL_REGISTRY = {
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
    "svr": SVRModel,
    "lightgbm": LightGBMModel,
}
