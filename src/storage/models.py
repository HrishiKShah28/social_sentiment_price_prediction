"""SQLAlchemy models for all relational database entities."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from src.storage.database import Base


class Exchange(Base):
    """Exchange entity (e.g. HKEX, KRX)."""

    __tablename__ = "exchanges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(16), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    timezone = Column(String(64), nullable=False)
    currency = Column(String(8), nullable=False)

    assets = relationship("Asset", back_populates="exchange", cascade="all, delete-orphan")


class Asset(Base):
    """Target tradeable asset (e.g. 0700.HK, 005930.KS)."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False, index=True)
    ticker = Column(String(16), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    sector = Column(String(64), nullable=False)
    currency = Column(String(8), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    exchange = relationship("Exchange", back_populates="assets")
    social_posts = relationship("SocialPost", back_populates="asset", cascade="all, delete-orphan")
    news_articles = relationship("NewsArticle", back_populates="asset", cascade="all, delete-orphan")
    price_bars = relationship("PriceBar", back_populates="asset", cascade="all, delete-orphan")
    technical_levels = relationship("TechnicalLevel", back_populates="asset", cascade="all, delete-orphan")
    features = relationship("ModelFeature", back_populates="asset", cascade="all, delete-orphan")
    predictions = relationship("ModelPrediction", back_populates="asset", cascade="all, delete-orphan")


class SocialPost(Base):
    """Scraped social post from Apify (LIHKG, Naver, Telegram, AAStocks)."""

    __tablename__ = "social_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    platform = Column(String(32), nullable=False, index=True)
    actor_id = Column(String(128), nullable=False)
    raw_post_id = Column(String(128), nullable=False)
    author_id = Column(String(128), nullable=True)
    author_followers = Column(Integer, default=0, nullable=False)
    post_text = Column(Text, nullable=False)
    post_url = Column(String(512), nullable=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    likes_count = Column(Integer, default=0, nullable=False)
    reposts_count = Column(Integer, default=0, nullable=False)
    views_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    raw_payload = Column(JSON, nullable=False)
    ingested_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)

    asset = relationship("Asset", back_populates="social_posts")

    __table_args__ = (
        UniqueConstraint("platform", "raw_post_id", name="uq_social_platform_post_id"),
        Index("ix_social_asset_time", "asset_id", "timestamp_utc"),
    )


class NewsArticle(Base):
    """Financial news article retrieved from Exa.ai."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    exa_id = Column(String(128), unique=True, nullable=False, index=True)
    title = Column(String(512), nullable=False)
    url = Column(String(1024), nullable=False)
    snippet = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    published_date_utc = Column(DateTime, nullable=False, index=True)
    exa_score = Column(Float, nullable=True)
    domain = Column(String(128), nullable=True, index=True)
    raw_payload = Column(JSON, nullable=False)
    ingested_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)

    asset = relationship("Asset", back_populates="news_articles")

    __table_args__ = (
        Index("ix_news_asset_time", "asset_id", "published_date_utc"),
    )


class PriceBar(Base):
    """Historical OHLCV market price bar."""

    __tablename__ = "price_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    timeframe = Column(String(8), default="1d", nullable=False)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    asset = relationship("Asset", back_populates="price_bars")

    __table_args__ = (
        UniqueConstraint("asset_id", "timeframe", "timestamp_utc", name="uq_price_asset_timeframe_time"),
        Index("ix_price_asset_time", "asset_id", "timestamp_utc"),
    )


class SentimentScore(Base):
    """NLP sentiment inference output for a social post or news article."""

    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(16), nullable=False)  # 'social_post' or 'news_article'
    source_id = Column(Integer, nullable=False)
    model_name = Column(String(64), nullable=False)
    raw_score = Column(Float, nullable=False)  # [-1.0, +1.0]
    confidence = Column(Float, nullable=False)  # [0.0, 1.0]
    label = Column(String(16), nullable=False)  # positive, neutral, negative
    engagement_weight = Column(Float, default=1.0, nullable=False)
    computed_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_sentiment_source_ref", "source_type", "source_id"),
    )


class TechnicalLevel(Base):
    """Derived support/resistance levels, trend direction, and suggested position."""

    __tablename__ = "technical_levels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    current_price = Column(Float, nullable=False)
    support_1 = Column(Float, nullable=False)
    support_2 = Column(Float, nullable=False)
    resistance_1 = Column(Float, nullable=False)
    resistance_2 = Column(Float, nullable=False)
    trend_direction = Column(String(16), nullable=False)  # Bullish, Bearish, Neutral
    suggested_position = Column(String(16), nullable=False)  # Long, Short, Flat
    calculation_method = Column(String(64), nullable=False)
    computed_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)

    asset = relationship("Asset", back_populates="technical_levels")

    __table_args__ = (
        Index("ix_tech_levels_asset_time", "asset_id", "timestamp_utc"),
    )


class ModelFeature(Base):
    """Point-in-time aligned feature vector per asset and timestamp."""

    __tablename__ = "model_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(8), default="1d", nullable=False)

    # Sentiment Features
    social_mentions_count = Column(Integer, default=0, nullable=False)
    social_sentiment_mean = Column(Float, default=0.0, nullable=False)
    social_sentiment_weighted = Column(Float, default=0.0, nullable=False)
    social_sentiment_momentum = Column(Float, default=0.0, nullable=False)
    news_articles_count = Column(Integer, default=0, nullable=False)
    news_sentiment_mean = Column(Float, default=0.0, nullable=False)
    composite_sentiment = Column(Float, default=0.0, nullable=False)

    # Price Technicals
    return_lag1 = Column(Float, nullable=False)
    return_lag5 = Column(Float, nullable=False)
    rsi_14 = Column(Float, nullable=False)
    macd_hist = Column(Float, nullable=False)
    atr_14 = Column(Float, nullable=False)
    hist_vol_10 = Column(Float, nullable=False)
    volume_ratio = Column(Float, nullable=False)
    sentiment_price_divergence = Column(Float, nullable=False)

    # Prediction Targets
    target_next_return = Column(Float, nullable=True)  # R_{t+1}
    target_next_price = Column(Float, nullable=True)  # P_{t+1}

    asset = relationship("Asset", back_populates="features")

    __table_args__ = (
        UniqueConstraint("asset_id", "timestamp_utc", name="uq_features_asset_time"),
    )


class ModelPrediction(Base):
    """Out-of-sample forecast for price and return at n+1."""

    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    model_name = Column(String(64), nullable=False, index=True)
    model_version = Column(String(32), default="1.0.0", nullable=False)
    feature_cutoff_utc = Column(DateTime, nullable=False)
    target_timestamp_utc = Column(DateTime, nullable=False, index=True)

    base_price = Column(Float, nullable=False)
    predicted_return = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    predicted_direction = Column(String(16), nullable=False)

    actual_price = Column(Float, nullable=True)
    actual_direction = Column(String(16), nullable=True)
    error_abs = Column(Float, nullable=True)
    error_pct = Column(Float, nullable=True)
    directional_correct = Column(Boolean, nullable=True)

    generated_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)

    asset = relationship("Asset", back_populates="predictions")
    pipeline_run = relationship("PipelineRun", back_populates="predictions")


class PipelineRun(Base):
    """Audit log of pipeline executions and evaluation metrics."""

    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    pipeline_stage = Column(String(32), nullable=False)  # ingest, features, train, all
    status = Column(String(16), default="running", nullable=False)
    started_at_utc = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at_utc = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    run_metrics = Column(JSON, nullable=True)

    predictions = relationship("ModelPrediction", back_populates="pipeline_run")
