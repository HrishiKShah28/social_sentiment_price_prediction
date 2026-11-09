"""Benchmark baseline models for rigorous financial forecasting comparison."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.linear_model import Ridge


class BaseForecaster(ABC):
    """Abstract base class for all return and price predictors."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseForecaster":
        """Fit model on historical feature matrix X and target returns y."""
        pass

    @abstractmethod
    def predict_return(self, X: np.ndarray) -> np.ndarray:
        """Predict next-period percentage return R_{t+1}."""
        pass

    def predict_price(self, X: np.ndarray, base_prices: np.ndarray) -> np.ndarray:
        """Derive predicted next-period price P_{t+1} = P_t * (1 + R_{t+1})."""
        pred_returns = self.predict_return(X)
        return base_prices * (1.0 + pred_returns)


class RandomWalkBaseline(BaseForecaster):
    """Naive persistence model: assumes price follows a martingale (E[R_{t+1}] = 0)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomWalkBaseline":
        return self

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        # Returns zero predicted return (price unchanged)
        return np.zeros(shape=(len(X),), dtype=float)


class TechnicalOnlyBaseline(BaseForecaster):
    """Baseline model trained exclusively on classical market technical indicators."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TechnicalOnlyBaseline":
        # Uses columns corresponding to returns, RSI, MACD, volatility
        self.model.fit(X, y)
        return self

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class SentimentOnlyBaseline(BaseForecaster):
    """Baseline model trained exclusively on social and news sentiment features."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SentimentOnlyBaseline":
        self.model.fit(X, y)
        return self

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
