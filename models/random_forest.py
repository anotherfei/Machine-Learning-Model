from sklearn.ensemble import RandomForestRegressor

import config
from models.base import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"
    needs_scaling = False  # tree models are scale-invariant

    def build(self):
        return RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
