"""
Common interface every model wrapper implements, so train.py and
compare_models.py can treat all models identically.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str
    needs_scaling: bool = False

    @abstractmethod
    def build(self):
        """Return an unfitted sklearn-compatible estimator."""
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.model = self.build()
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
