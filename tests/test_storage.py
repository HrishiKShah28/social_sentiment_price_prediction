"""Unit tests for database storage, models, and repository operations."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.storage.database import Base
from src.storage.models import Asset, Exchange, PriceBar, SocialPost, NewsArticle, PipelineRun
from src.storage.repository import DataRepository


@pytest.fixture
def in_memory_db():
    """Create a fresh in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


def test_seed_exchanges_and_assets(in_memory_db):
    """Test seeding exchanges and the 10 target assets."""
    repo = DataRepository(in_memory_db)
    repo.seed_exchanges_and_assets()

    # Check exchanges
    hkex = repo.session.query(Exchange).filter_by(code="HKEX").first()
    krx = repo.session.query(Exchange).filter_by(code="KRX").first()
    assert hkex is not None
    assert krx is not None
    assert hkex.currency == "HKD"
    assert krx.currency == "KRW"

    # Check assets
    hk_assets = repo.get_active_assets(exchange_code="HKEX")
    kr_assets = repo.get_active_assets(exchange_code="KRX")
    assert len(hk_assets) == 5
    assert len(kr_assets) == 5

    # Check specific tickers
    hk_tickers = {a.ticker for a in hk_assets}
    kr_tickers = {a.ticker for a in kr_assets}
    assert "0700.HK" in hk_tickers
    assert "9988.HK" in hk_tickers
    assert "005930.KS" in kr_tickers
    assert "000660.KS" in kr_tickers

    # Idempotency check: running seed again should not duplicate
    repo.seed_exchanges_and_assets()
    assert len(repo.get_active_assets()) == 10


def test_social_post_insertion_and_deduplication(in_memory_db):
    """Test inserting social posts and avoiding duplicates."""
    repo = DataRepository(in_memory_db)
    repo.seed_exchanges_and_assets()

    tencent = repo.get_asset_by_ticker("0700.HK")
    assert tencent is not None

    now = datetime.utcnow()
    posts = [
        {
            "asset_id": tencent.id,
            "platform": "LIHKG",
            "actor_id": "test_actor",
            "raw_post_id": "post_1001",
            "author_id": "trader88",
            "author_followers": 250,
            "post_text": "Tencent earnings breakout today! 0700 to the moon.",
            "post_url": "https://lihkg.com/thread/1001",
            "timestamp_utc": now,
            "likes_count": 45,
            "reposts_count": 10,
            "views_count": 1200,
            "comments_count": 15,
            "raw_payload": {"test": "data"},
        }
    ]

    inserted = repo.add_social_posts(posts)
    assert inserted == 1

    # Inserting the same post again should be skipped
    inserted_second = repo.add_social_posts(posts)
    assert inserted_second == 0


def test_price_bar_insertion_and_query(in_memory_db):
    """Test inserting price bars and chronological querying."""
    repo = DataRepository(in_memory_db)
    repo.seed_exchanges_and_assets()

    samsung = repo.get_asset_by_ticker("005930.KS")
    assert samsung is not None

    base_time = datetime(2026, 11, 1, 6, 30)
    bars = [
        {
            "asset_id": samsung.id,
            "timeframe": "1d",
            "timestamp_utc": base_time + timedelta(days=i),
            "open": 70000.0 + i * 500,
            "high": 71000.0 + i * 500,
            "low": 69500.0 + i * 500,
            "close": 70500.0 + i * 500,
            "adjusted_close": 70500.0 + i * 500,
            "volume": 1000000 + i * 50000,
        }
        for i in range(5)
    ]

    inserted = repo.add_price_bars(bars)
    assert inserted == 5

    retrieved = repo.get_price_bars(samsung.id)
    assert len(retrieved) == 5
    assert retrieved[0].timestamp_utc < retrieved[1].timestamp_utc


def test_pipeline_run_lifecycle(in_memory_db):
    """Test pipeline run tracking."""
    repo = DataRepository(in_memory_db)
    run = repo.start_pipeline_run(stage="ingest")
    assert run.status == "running"
    assert run.run_id is not None

    repo.finish_pipeline_run(run.run_id, status="completed", metrics={"items_scraped": 42})
    updated = in_memory_db.query(PipelineRun).filter_by(run_id=run.run_id).first()
    assert updated.status == "completed"
    assert updated.run_metrics["items_scraped"] == 42
