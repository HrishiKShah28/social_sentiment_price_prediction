"""Chronological walk-forward validation harness for out-of-sample evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from src.models.baselines import BaseForecaster, RandomWalkBaseline, TechnicalOnlyBaseline, SentimentOnlyBaseline
from src.models.predictors import RidgePredictor, GradientBoostingPredictor


@dataclass
class ValidationMetrics:
    """Out-of-sample evaluation metrics."""

    model_name: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float  # Percentage of correct directional sign predictions
    total_predictions: int


class WalkForwardValidator:
    """Executes chronological expanding-window walk-forward validation with zero lookahead bias."""

    # Standard feature column ordering
    FEATURE_COLS = [
        "social_mentions_count",
        "social_sentiment_mean",
        "social_sentiment_weighted",
        "social_sentiment_momentum",
        "news_articles_count",
        "news_sentiment_mean",
        "composite_sentiment",
        "return_lag1",
        "return_lag5",
        "rsi_14",
        "macd_hist",
        "atr_14",
        "hist_vol_10",
        "volume_ratio",
        "sentiment_price_divergence",
    ]

    TECH_COLS = ["return_lag1", "return_lag5", "rsi_14", "macd_hist", "atr_14", "hist_vol_10", "volume_ratio"]
    SENT_COLS = ["social_mentions_count", "social_sentiment_mean", "social_sentiment_weighted", "social_sentiment_momentum", "news_articles_count", "news_sentiment_mean", "composite_sentiment", "sentiment_price_divergence"]

    @classmethod
    def evaluate(
        cls,
        feature_rows: List[Dict],
        min_train_size: int = 5,
    ) -> Dict[str, Tuple[ValidationMetrics, List[Dict]]]:
        """Perform walk-forward validation across all models and baseline benchmarks."""
        # Filter to rows that have realized targets
        valid_rows = [r for r in feature_rows if r.get("target_next_return") is not None and r.get("target_next_price") is not None]

        if len(valid_rows) <= min_train_size:
            logger.warning(
                f"Insufficient historical bars for walk-forward validation: {len(valid_rows)} bars, {min_train_size} min required."
            )
            return {}

        df = pd.DataFrame(valid_rows)
        X_all = df[cls.FEATURE_COLS].values
        X_tech = df[cls.TECH_COLS].values
        X_sent = df[cls.SENT_COLS].values
        y_returns = df["target_next_return"].values
        actual_prices = df["target_next_price"].values
        current_prices = df["current_price"].values

        model_factories = {
            "Random Walk (Baseline)": (RandomWalkBaseline, X_all),
            "Technical Only (Baseline)": (TechnicalOnlyBaseline, X_tech),
            "Sentiment Only (Baseline)": (SentimentOnlyBaseline, X_sent),
            "Ridge Regression": (RidgePredictor, X_all),
            "Gradient Boosting": (GradientBoostingPredictor, X_all),
        }

        results: Dict[str, Tuple[ValidationMetrics, List[Dict]]] = {}

        for model_name, (factory, X_feat) in model_factories.items():
            preds_records: List[Dict] = []
            errors_abs: List[float] = []
            errors_sq: List[float] = []
            errors_pct: List[float] = []
            directional_hits: List[bool] = []

            # Expanding window walk-forward loop
            for test_idx in range(min_train_size, len(valid_rows)):
                # Training history strictly precedes test observation
                X_train = X_feat[:test_idx]
                y_train = y_returns[:test_idx]

                X_test = X_feat[test_idx : test_idx + 1]
                actual_ret = y_returns[test_idx]
                actual_px = actual_prices[test_idx]
                base_px = current_prices[test_idx]

                # Train fresh instance
                model: BaseForecaster = factory()
                model.fit(X_train, y_train)

                pred_ret = float(model.predict_return(X_test)[0])
                pred_px = float(base_px * (1.0 + pred_ret))

                # Compute errors
                err_abs = abs(pred_px - actual_px)
                err_sq = (pred_px - actual_px) ** 2
                err_pct = (err_abs / actual_px) * 100.0 if actual_px > 0 else 0.0

                # Directional check (did we get the sign of return right?)
                dir_correct = bool(np.sign(pred_ret) == np.sign(actual_ret)) if actual_ret != 0 else True

                errors_abs.append(err_abs)
                errors_sq.append(err_sq)
                errors_pct.append(err_pct)
                directional_hits.append(dir_correct)

                preds_records.append(
                    {
                        "timestamp_utc": df["timestamp_utc"].iloc[test_idx],
                        "base_price": round(base_px, 2),
                        "predicted_return": round(pred_ret, 6),
                        "actual_return": round(actual_ret, 6),
                        "predicted_price": round(pred_px, 2),
                        "actual_price": round(actual_px, 2),
                        "error_abs": round(err_abs, 2),
                        "directional_correct": dir_correct,
                    }
                )

            rmse = float(np.sqrt(np.mean(errors_sq))) if errors_sq else 0.0
            mae = float(np.mean(errors_abs)) if errors_abs else 0.0
            mape = float(np.mean(errors_pct)) if errors_pct else 0.0
            hit_rate = float(np.mean(directional_hits) * 100.0) if directional_hits else 50.0

            metrics = ValidationMetrics(
                model_name=model_name,
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                mape=round(mape, 2),
                directional_accuracy=round(hit_rate, 2),
                total_predictions=len(preds_records),
            )

            results[model_name] = (metrics, preds_records)

        return results

