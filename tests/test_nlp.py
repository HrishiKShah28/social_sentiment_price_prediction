"""Unit tests for sentiment analysis and technical level extraction."""

import pytest
from datetime import datetime, timedelta
from src.nlp.sentiment import SentimentScorer
from src.nlp.levels import TechnicalLevelExtractor
from src.storage.models import PriceBar


def test_sentiment_scoring_english():
    """Test sentiment scoring on English financial statements."""
    scorer = SentimentScorer()

    bullish_res = scorer.score_text("Tencent reported massive earnings beat and strong revenue growth!")
    assert bullish_res.raw_score > 0.0
    assert bullish_res.label == "positive"

    bearish_res = scorer.score_text("Company warns of earnings miss and major revenue crash.")
    assert bearish_res.raw_score < 0.0
    assert bearish_res.label == "negative"

    neutral_res = scorer.score_text("The board will hold its annual meeting on Tuesday.")
    assert neutral_res.label == "neutral"


def test_sentiment_scoring_multilingual():
    """Test sentiment scoring on Chinese (Cantonese) and Korean text."""
    scorer = SentimentScorer()

    # Cantonese bullish
    hk_bull = scorer.score_text("0700 騰訊業績大好，突破目標價，強烈買入！")
    assert hk_bull.raw_score > 0.0
    assert hk_bull.label == "positive"

    # Cantonese bearish
    hk_bear = scorer.score_text("大市破位暴跌，睇淡後市，快啲止蝕沽出！")
    assert hk_bear.raw_score < 0.0
    assert hk_bear.label == "negative"

    # Korean bullish
    kr_bull = scorer.score_text("삼성전자 HBM 반도체 공급 계약 호재, 실적 급등 기대!")
    assert kr_bull.raw_score > 0.0
    assert kr_bull.label == "positive"

    # Korean bearish
    kr_bear = scorer.score_text("외국인 대규모 매도 폭락, 적자 전환 악재 발생.")
    assert kr_bear.raw_score < 0.0
    assert kr_bear.label == "negative"


def test_engagement_weight_calculation():
    """Test engagement weighting math."""
    w_low = SentimentScorer.compute_engagement_weight(likes=0, reposts=0, views=10, followers=5)
    w_high = SentimentScorer.compute_engagement_weight(likes=150, reposts=50, views=10000, followers=50000)

    assert w_high > w_low
    assert w_low >= 1.0


def test_composite_sentiment_blending():
    """Test multi-source isolation and composite blending."""
    scorer = SentimentScorer()

    # Source 1 (LIHKG): +0.6
    # Source 2 (Telegram): +0.4
    # News (Exa): +0.2
    social_sources = {"LIHKG": 0.6, "Telegram": 0.4}
    composite = scorer.compute_composite_sentiment(social_sources, news_sentiment=0.2, social_weight=0.6)

    # Expected: 0.6 * 0.5 + 0.4 * 0.2 = 0.30 + 0.08 = 0.38
    assert composite > 0.3
    assert composite < 0.5


def test_technical_level_extractor():
    """Test algorithmic support/resistance levels and position classification."""
    base_time = datetime(2026, 11, 1)
    bars = []
    # Create 10 ascending price bars
    for i in range(10):
        bars.append(
            PriceBar(
                asset_id=1,
                timeframe="1d",
                timestamp_utc=base_time + timedelta(days=i),
                open=100.0 + i * 2,
                high=105.0 + i * 2,
                low=98.0 + i * 2,
                close=103.0 + i * 2,
                adjusted_close=103.0 + i * 2,
                volume=100000,
            )
        )

    signal = TechnicalLevelExtractor.extract_levels(bars, composite_sentiment=0.4, lookback_window=10)

    assert signal is not None
    assert signal.current_price == 103.0 + 9 * 2  # 121.0
    assert signal.support_1 < signal.resistance_1
    assert signal.trend_direction == "Bullish"
    assert signal.suggested_position in ["Long", "Flat"]
