"""Unit tests for feature engineering, indicators, and point-in-time leakage prevention."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from src.features.aligner import TimeSeriesAligner
from src.features.builder import FeatureBuilder
from src.storage.models import PriceBar, SocialPost, NewsArticle


def test_strict_point_in_time_alignment():
    """Verify that social posts published after a bar's cutoff are never leaked into that bar."""
    bar1_time = datetime(2026, 11, 1, 16, 0)
    bar2_time = datetime(2026, 11, 2, 16, 0)
    bar3_time = datetime(2026, 11, 3, 16, 0)

    bars = [
        PriceBar(asset_id=1, timeframe="1d", timestamp_utc=bar1_time, open=100, high=105, low=99, close=102, volume=1000),
        PriceBar(asset_id=1, timeframe="1d", timestamp_utc=bar2_time, open=102, high=108, low=101, close=106, volume=1200),
        PriceBar(asset_id=1, timeframe="1d", timestamp_utc=bar3_time, open=106, high=110, low=105, close=109, volume=1100),
    ]

    posts = [
        # Post 1: During Bar 1 (Nov 1 12:00) -> should be in Bar 1
        SocialPost(asset_id=1, platform="LIHKG", actor_id="a1", raw_post_id="p1", post_text="Good morning Tencent", timestamp_utc=datetime(2026, 11, 1, 12, 0), raw_payload={}),
        # Post 2: After Bar 1 close (Nov 1 17:30) -> MUST be in Bar 2, NOT Bar 1!
        SocialPost(asset_id=1, platform="LIHKG", actor_id="a1", raw_post_id="p2", post_text="Post market breakout news", timestamp_utc=datetime(2026, 11, 1, 17, 30), raw_payload={}),
        # Post 3: During Bar 2 (Nov 2 10:00) -> should be in Bar 2
        SocialPost(asset_id=1, platform="LIHKG", actor_id="a1", raw_post_id="p3", post_text="Intraday rally continue", timestamp_utc=datetime(2026, 11, 2, 10, 0), raw_payload={}),
    ]

    news = [
        # News 1: Published after Bar 2 close -> should be in Bar 3
        NewsArticle(asset_id=1, exa_id="n1", title="Earnings Report", url="http://exa.ai/1", published_date_utc=datetime(2026, 11, 2, 18, 0), raw_payload={})
    ]

    aligned = TimeSeriesAligner.align_data_to_bars(bars, posts, news)

    assert len(aligned) == 3

    # Bar 1 checks:
    # Only Post 1 allowed! Post 2 must NOT be present
    bar1_post_ids = [p.raw_post_id for p in aligned[0]["posts"]]
    assert bar1_post_ids == ["p1"]
    assert "p2" not in bar1_post_ids
    assert len(aligned[0]["news"]) == 0
    assert aligned[0]["target_price"] == 106.0  # Bar 2 close
    assert aligned[0]["target_return"] == pytest.approx((106.0 - 102.0) / 102.0, rel=1e-4)

    # Bar 2 checks:
    # Post 2 and Post 3 must be here
    bar2_post_ids = [p.raw_post_id for p in aligned[1]["posts"]]
    assert "p2" in bar2_post_ids
    assert "p3" in bar2_post_ids
    assert len(aligned[1]["news"]) == 0

    # Bar 3 checks:
    # News 1 must be in Bar 3
    assert len(aligned[2]["news"]) == 1
    assert aligned[2]["news"][0].exa_id == "n1"
    # Target for the final bar MUST be None (no future lookahead!)
    assert aligned[2]["target_price"] is None
    assert aligned[2]["target_return"] is None


def test_feature_builder_computation():
    """Test indicator calculations and complete feature dictionary generation."""
    builder = FeatureBuilder()
    base_time = datetime(2026, 11, 1)

    # Create 20 days of synthetic bars
    bars = [
        PriceBar(
            asset_id=1,
            timeframe="1d",
            timestamp_utc=base_time + timedelta(days=i),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            adjusted_close=101.0 + i,
            volume=500000 + i * 1000,
        )
        for i in range(20)
    ]

    features = builder.build_features(bars, social_posts=[], news_articles=[])

    assert len(features) == 20
    last_row = features[-1]

    assert "rsi_14" in last_row
    assert 0 <= last_row["rsi_14"] <= 100
    assert "macd_hist" in last_row
    assert "atr_14" in last_row
    assert "hist_vol_10" in last_row
    assert "sentiment_price_divergence" in last_row
    assert last_row["target_next_return"] is None  # Most recent bar has no future target

