from lightgbm import LGBMRegressor

import config
from models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"
    needs_scaling = False  # tree models are scale-invariant

    def build(self):
        return LGBMRegressor(**config.LIGHTGBM_PARAMS)
