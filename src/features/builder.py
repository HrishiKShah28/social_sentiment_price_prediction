"""Feature engineering and point-in-time dataset generator."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
from src.features.aligner import TimeSeriesAligner
from src.nlp.sentiment import SentimentScorer
from src.storage.models import PriceBar, SocialPost, NewsArticle


class FeatureBuilder:
    """Computes technical indicators, sentiment signals, and constructs the modeling dataset."""

    def __init__(self, sentiment_scorer: Optional[SentimentScorer] = None):
        self.scorer = sentiment_scorer or SentimentScorer()

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Default neutral RSI if insufficient history
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(float(100.0 - (100.0 / (1.0 + rs))), 2)

    @staticmethod
    def calculate_macd(prices: np.ndarray) -> float:
        """Calculate MACD histogram (12, 26, 9)."""
        if len(prices) < 26:
            return 0.0
        s = pd.Series(prices)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return round(float(macd_hist.iloc[-1]), 4)

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate normalized Average True Range (ATR / Close)."""
        if len(closes) < 2:
            return 0.0
        n = min(period, len(closes) - 1)
        tr = np.maximum(
            highs[-n:] - lows[-n:],
            np.maximum(
                np.abs(highs[-n:] - closes[-n - 1 : -1]),
                np.abs(lows[-n:] - closes[-n - 1 : -1]),
            ),
        )
        atr = float(np.mean(tr))
        current_close = float(closes[-1])
        return round(float(atr / current_close), 4) if current_close > 0 else 0.0

    @staticmethod
    def calculate_hist_vol(closes: np.ndarray, period: int = 10) -> float:
        """Calculate annualized historical volatility of returns."""
        if len(closes) < 3:
            return 0.0
        n = min(period, len(closes) - 1)
        returns = np.diff(np.log(closes[-n - 1 :]))
        if len(returns) < 2:
            return 0.0
        vol = float(np.std(returns, ddof=1) * np.sqrt(252))
        return round(vol, 4)

    def build_features(
        self,
        bars: List[PriceBar],
        social_posts: List[SocialPost],
        news_articles: List[NewsArticle],
    ) -> List[Dict]:
        """Construct the complete point-in-time feature matrix for an asset."""
        if not bars:
            return []

        sorted_bars = sorted(bars, key=lambda b: b.timestamp_utc)
        aligned_windows = TimeSeriesAligner.align_data_to_bars(sorted_bars, social_posts, news_articles)

        closes = np.array([float(b.close) for b in sorted_bars])
        highs = np.array([float(b.high) for b in sorted_bars])
        lows = np.array([float(b.low) for b in sorted_bars])
        volumes = np.array([float(b.volume) for b in sorted_bars])

        features_list: List[Dict] = []
        prev_social_score = 0.0

        for i, window in enumerate(aligned_windows):
            bar = window["bar"]
            posts = window["posts"]
            news = window["news"]

            # 1. Social Sentiment Processing
            social_scores: List[float] = []
            weighted_social_sum = 0.0
            total_weight = 0.0
            platform_scores: Dict[str, List[float]] = {}

            for p in posts:
                res = self.scorer.score_text(p.post_text)
                w = self.scorer.compute_engagement_weight(
                    likes=p.likes_count,
                    reposts=p.reposts_count,
                    views=p.views_count,
                    followers=p.author_followers,
                )
                social_scores.append(res.raw_score)
                weighted_social_sum += res.raw_score * w
                total_weight += w

                platform_scores.setdefault(p.platform, []).append(res.raw_score)

            social_count = len(posts)
            social_mean = float(np.mean(social_scores)) if social_scores else 0.0
            social_weighted = float(weighted_social_sum / total_weight) if total_weight > 0 else 0.0
            social_momentum = float(social_weighted - prev_social_score)
            prev_social_score = social_weighted

            # 2. News Sentiment Processing
            news_scores = [self.scorer.score_text(n.title + " " + (n.snippet or "")).raw_score for n in news]
            news_count = len(news)
            news_mean = float(np.mean(news_scores)) if news_scores else 0.0

            # 3. Composite Sentiment Blending
            avg_platform_scores = {plat: float(np.mean(scs)) for plat, scs in platform_scores.items()}
            composite = self.scorer.compute_composite_sentiment(
                social_scores_by_platform=avg_platform_scores,
                news_sentiment=news_mean,
            )

            # 4. Technical Indicators using history up to bar i
            hist_closes = closes[: i + 1]
            hist_highs = highs[: i + 1]
            hist_lows = lows[: i + 1]
            hist_vols = volumes[: i + 1]

            return_lag1 = (
                float((hist_closes[-1] - hist_closes[-2]) / hist_closes[-2]) if i >= 1 and hist_closes[-2] > 0 else 0.0
            )
            return_lag5 = (
                float((hist_closes[-1] - hist_closes[-6]) / hist_closes[-6]) if i >= 5 and hist_closes[-6] > 0 else 0.0
            )

            rsi_14 = self.calculate_rsi(hist_closes, period=14)
            macd_hist = self.calculate_macd(hist_closes)
            atr_14 = self.calculate_atr(hist_highs, hist_lows, hist_closes, period=14)
            hist_vol_10 = self.calculate_hist_vol(hist_closes, period=10)

            # Volume ratio (today volume vs 5-day mean)
            vol_mean_5 = float(np.mean(hist_vols[-5:])) if len(hist_vols) >= 5 else float(hist_vols[-1])
            volume_ratio = float(hist_vols[-1] / vol_mean_5) if vol_mean_5 > 0 else 1.0

            # Sentiment-Price Divergence: normalized sentiment vs return
            sentiment_price_div = composite - (return_lag1 * 10.0)

            features_list.append(
                {
                    "asset_id": bar.asset_id,
                    "timestamp_utc": bar.timestamp_utc,
                    "timeframe": bar.timeframe,
                    "current_price": float(bar.close),
                    # Sentiment features
                    "social_mentions_count": social_count,
                    "social_sentiment_mean": round(social_mean, 4),
                    "social_sentiment_weighted": round(social_weighted, 4),
                    "social_sentiment_momentum": round(social_momentum, 4),
                    "news_articles_count": news_count,
                    "news_sentiment_mean": round(news_mean, 4),
                    "composite_sentiment": round(composite, 4),
                    # Price technical features
                    "return_lag1": round(return_lag1, 6),
                    "return_lag5": round(return_lag5, 6),
                    "rsi_14": round(rsi_14, 2),
                    "macd_hist": round(macd_hist, 4),
                    "atr_14": round(atr_14, 4),
                    "hist_vol_10": round(hist_vol_10, 4),
                    "volume_ratio": round(volume_ratio, 4),
                    "sentiment_price_divergence": round(sentiment_price_div, 4),
                    # Target labels
                    "target_next_return": window["target_return"],
                    "target_next_price": window["target_price"],
                }
            )

        logger.info(f"Built {len(features_list)} feature rows for asset ID {bars[0].asset_id}")
        return features_list
