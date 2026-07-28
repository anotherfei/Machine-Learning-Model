from sklearn.linear_model import LinearRegression as _LinearRegression

import config
from models.base import BaseModel


class LinearRegressionModel(BaseModel):
    name = "linear_regression"
    needs_scaling = True  # linear models benefit from scaled inputs

    def build(self):
        return _LinearRegression(**config.LINEAR_REGRESSION_PARAMS)
