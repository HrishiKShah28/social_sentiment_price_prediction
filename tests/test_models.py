"""Unit tests for predictive models, baselines, and walk-forward validation."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from src.models.baselines import RandomWalkBaseline, TechnicalOnlyBaseline, SentimentOnlyBaseline
from src.models.predictors import RidgePredictor, GradientBoostingPredictor
from src.models.validation import WalkForwardValidator


def test_random_walk_baseline():
    """Verify that Random Walk persistence predicts 0 return and equal price."""
    rw = RandomWalkBaseline()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    base_px = np.array([100.0, 150.0])

    preds_ret = rw.predict_return(X)
    assert np.all(preds_ret == 0.0)

    preds_px = rw.predict_price(X, base_px)
    assert np.all(preds_px == base_px)


def test_ridge_and_gb_predictors():
    """Verify ML predictors fit and produce numerical return forecasts."""
    np.random.seed(42)
    X = np.random.randn(20, 15)
    y = 0.05 * X[:, 0] - 0.03 * X[:, 7] + 0.01 * np.random.randn(20)

    ridge = RidgePredictor(alpha=1.0).fit(X[:15], y[:15])
    preds_ridge = ridge.predict_return(X[15:])
    assert len(preds_ridge) == 5
    assert not np.isnan(preds_ridge).any()

    gb = GradientBoostingPredictor(n_estimators=10, max_depth=2).fit(X[:15], y[:15])
    preds_gb = gb.predict_return(X[15:])
    assert len(preds_gb) == 5
    assert not np.isnan(preds_gb).any()


def test_walk_forward_validator():
    """Test full chronological expanding-window walk-forward validation."""
    base_time = datetime(2026, 11, 1)
    feature_rows = []

    # Generate 15 days of synthetic feature rows with non-trivial trend
    for i in range(15):
        current_px = 100.0 + i * 1.5
        ret = 0.015 if i % 2 == 0 else -0.01
        next_px = current_px * (1.0 + ret)

        row = {
            "timestamp_utc": base_time + timedelta(days=i),
            "current_price": current_px,
            "social_mentions_count": 10 + i,
            "social_sentiment_mean": 0.2,
            "social_sentiment_weighted": 0.25,
            "social_sentiment_momentum": 0.05,
            "news_articles_count": 3,
            "news_sentiment_mean": 0.1,
            "composite_sentiment": 0.2,
            "return_lag1": 0.01,
            "return_lag5": 0.04,
            "rsi_14": 55.0,
            "macd_hist": 0.12,
            "atr_14": 0.02,
            "hist_vol_10": 0.18,
            "volume_ratio": 1.1,
            "sentiment_price_divergence": 0.1,
            "target_next_return": ret if i < 14 else None,
            "target_next_price": next_px if i < 14 else None,
        }
        feature_rows.append(row)

    results = WalkForwardValidator.evaluate(feature_rows, min_train_size=6)

    assert "Random Walk (Baseline)" in results
    assert "Technical Only (Baseline)" in results
    assert "Sentiment Only (Baseline)" in results
    assert "Ridge Regression" in results
    assert "Gradient Boosting" in results

    rw_metrics, rw_preds = results["Random Walk (Baseline)"]
    assert rw_metrics.total_predictions == (14 - 6)
    assert rw_metrics.rmse >= 0.0
    assert 0.0 <= rw_metrics.directional_accuracy <= 100.0
    assert len(rw_preds) == (14 - 6)

