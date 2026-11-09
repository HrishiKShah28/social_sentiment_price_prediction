"""Multilingual financial sentiment scoring and engagement weighting engine."""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class SentimentResult:
    """Structured sentiment scoring output."""

    raw_score: float  # [-1.0, +1.0]
    confidence: float  # [0.0, 1.0]
    label: str  # 'positive', 'neutral', 'negative'


class SentimentScorer:
    """Multilingual sentiment analyzer supporting English, Chinese (Cantonese), and Korean."""

    # Curated financial keyword lexicons for fast/lightweight fallback and token validation
    FINANCIAL_TERMS = {
        "bullish": [
            "breakout", "surge", "rally", "buy", "target price", "outperform", "upgrade",
            "revenue beat", "earnings beat", "growth", "all-time high", "calls", "long",
            "升", "爆升", "突破", "買入", "睇好", "升浪", "入市", "支持", "好消息", "目標價",
            "상승", "급등", "매수", "호재", "돌파", "신고가", "목표가", "성장", "반등", "상한가"
        ],
        "bearish": [
            "crash", "plunge", "dump", "sell", "downgrade", "earnings miss", "revenue miss",
            "drop", "all-time low", "puts", "short", "bearish", "loss", "warning",
            "跌", "暴跌", "沽", "看淡", "破位", "止蝕", "熊市", "壞消息", "減持",
            "하락", "폭락", "매도", "악재", "신저가", "하방", "손절", "약세", "하한가", "적자"
        ],
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "finbert-multilingual"
        self._hf_pipeline = None

    def _init_hf_pipeline(self):
        """Optionally load Hugging Face pipeline if torch and transformers are available."""
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline
                self._hf_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    truncation=True,
                    max_length=512,
                )
                logger.info("Loaded Hugging Face XLM-RoBERTa sentiment pipeline")
            except Exception as e:
                logger.info(f"Using rule-augmented financial sentiment scoring (HF fallback: {e})")
                self._hf_pipeline = False

    def score_text(self, text: str) -> SentimentResult:
        """Calculate normalized sentiment score in [-1.0, +1.0] with confidence."""
        if not text or not text.strip():
            return SentimentResult(raw_score=0.0, confidence=1.0, label="neutral")

        clean_text = text.lower().strip()

        # Check for deep learning pipeline if initialized
        if self._hf_pipeline:
            try:
                hf_out = self._hf_pipeline(clean_text[:512])[0]
                label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                lbl = hf_out["label"].lower()
                conf = float(hf_out["score"])
                raw = label_map.get(lbl, 0.0) * conf
                assigned_label = "positive" if raw > 0.15 else ("negative" if raw < -0.15 else "neutral")
                return SentimentResult(raw_score=round(raw, 4), confidence=round(conf, 4), label=assigned_label)
            except Exception as e:
                logger.debug(f"HF inference error, falling back to lexicon: {e}")

        # Lexicon scoring for financial colloquialisms (Chinese / Korean / English)
        pos_hits = sum(1 for kw in self.FINANCIAL_TERMS["bullish"] if kw in clean_text)
        neg_hits = sum(1 for kw in self.FINANCIAL_TERMS["bearish"] if kw in clean_text)

        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            return SentimentResult(raw_score=0.0, confidence=0.5, label="neutral")

        polarity = (pos_hits - neg_hits) / total_hits
        confidence = min(0.5 + 0.1 * total_hits, 0.95)

        if polarity > 0.15:
            label = "positive"
        elif polarity < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            raw_score=round(polarity, 4),
            confidence=round(confidence, 4),
            label=label,
        )

    @staticmethod
    def compute_engagement_weight(
        likes: int = 0,
        reposts: int = 0,
        views: int = 0,
        followers: int = 0,
    ) -> float:
        """Compute non-linear engagement multiplier."""
        interaction_term = math.log1p(max(0, likes) + 2.0 * max(0, reposts) + 0.1 * max(0, views))
        reach_term = math.log1p(max(0, followers) + 10.0)
        # Bounded weight multiplier >= 1.0
        return round(max(1.0, interaction_term * reach_term), 4)

    def aggregate_source_sentiment(self, posts_with_scores: List[Dict]) -> float:
        """Compute engagement-weighted mean sentiment for a single source channel."""
        if not posts_with_scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for item in posts_with_scores:
            score = item.get("raw_score", 0.0)
            weight = item.get("engagement_weight", 1.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0

        return round(weighted_sum / total_weight, 4)

    def compute_composite_sentiment(
        self,
        social_scores_by_platform: Dict[str, float],
        news_sentiment: float,
        social_weight: float = 0.65,
    ) -> float:
        """Synthesize separate source-level sentiments into an asset composite score."""
        if not social_scores_by_platform:
            return round(news_sentiment, 4)

        # Equal weighting across available distinct social sources
        avg_social = sum(social_scores_by_platform.values()) / len(social_scores_by_platform)

        if math.isclose(news_sentiment, 0.0):
            return round(avg_social, 4)

        composite = social_weight * avg_social + (1.0 - social_weight) * news_sentiment
        return round(composite, 4)
