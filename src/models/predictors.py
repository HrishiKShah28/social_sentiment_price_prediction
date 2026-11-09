"""Machine learning predictive models for financial return forecasting."""

from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.models.baselines import BaseForecaster


class RidgePredictor(BaseForecaster):
    """L2 Regularized Ridge Regressor with internal feature standardization."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgePredictor":
        if len(X) < 2:
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(shape=(len(X),), dtype=float)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class GradientBoostingPredictor(BaseForecaster):
    """Gradient Boosted Trees regressor capturing non-linear sentiment-price interactions."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 3, learning_rate: float = 0.05, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingPredictor":
        if len(X) < 3:
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(shape=(len(X),), dtype=float)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
