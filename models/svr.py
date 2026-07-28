from sklearn.svm import SVR as _SVR

import config
from models.base import BaseModel


class SVRModel(BaseModel):
    name = "svr"
    needs_scaling = True  # SVR is sensitive to feature scale — never skip this

    def build(self):
        return _SVR(**config.SVR_PARAMS)
