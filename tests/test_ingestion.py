"""Unit tests for price, news, and social ingestion clients."""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.storage.database import Base
from src.storage.models import Asset, Exchange, PriceBar, NewsArticle, SocialPost
from src.storage.repository import DataRepository
from src.ingestion.price_client import PriceDataService
from src.ingestion.exa_client import ExaNewsService
from src.ingestion.apify_client import ApifySocialService


@pytest.fixture
def mock_repo():
    """Create test repository with seeded data."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    repo = DataRepository(session)
    repo.seed_exchanges_and_assets()
    yield repo
    session.close()


def test_price_client_processing(mock_repo):
    """Test PriceDataService parses and inserts OHLCV bars properly."""
    service = PriceDataService(mock_repo)
    asset = mock_repo.get_asset_by_ticker("0700.HK")
    assert asset is not None

    mock_df = pd.DataFrame(
        {
            "Open": [380.0, 385.0],
            "High": [390.0, 392.0],
            "Low": [378.0, 383.0],
            "Close": [388.0, 391.0],
            "Adj Close": [388.0, 391.0],
            "Volume": [15000000, 18000000],
        },
        index=[
            pd.Timestamp("2026-11-01 08:00:00+0000", tz="UTC"),
            pd.Timestamp("2026-11-02 08:00:00+0000", tz="UTC"),
        ],
    )

    mock_yf = MagicMock()
    mock_instance = MagicMock()
    mock_instance.history.return_value = mock_df
    mock_yf.Ticker.return_value = mock_instance

    with patch.dict(sys.modules, {"yfinance": mock_yf}):
        inserted = service.fetch_and_store_bars(asset, timeframe="1d", lookback_days=2)
        assert inserted == 2

    # Verify bars in DB
    bars = mock_repo.get_price_bars(asset.id)
    assert len(bars) == 2
    assert bars[0].close == 388.0
    assert bars[1].close == 391.0
    assert bars[0].adjusted_close == 388.0


def test_exa_news_client_mocked(mock_repo):
    """Test ExaNewsService processes API search results."""
    service = ExaNewsService(mock_repo, api_key="test_exa_key")
    asset = mock_repo.get_asset_by_ticker("005930.KS")
    assert asset is not None

    query = service.build_query(asset)
    assert "005930.KS" in query
    assert "Samsung Electronics" in query

    mock_article = MagicMock()
    mock_article.id = "exa_doc_123"
    mock_article.title = "Samsung Electronics expands AI chip manufacturing"
    mock_article.url = "https://bloomberg.com/news/samsung-ai-chip"
    mock_article.highlights = ["Samsung announced massive investment in HBM production."]
    mock_article.text = "Full text of the news article covering Samsung Electronics..."
    mock_article.published_date = "2026-11-05T04:30:00.000Z"
    mock_article.score = 0.94

    mock_response = MagicMock()
    mock_response.results = [mock_article]

    mock_client = MagicMock()
    mock_client.search_and_contents.return_value = mock_response
    service._client = mock_client

    inserted = service.fetch_and_store_news(asset, lookback_days=5)
    assert inserted == 1

    articles = mock_repo.session.query(NewsArticle).filter_by(asset_id=asset.id).all()
    assert len(articles) == 1
    assert articles[0].exa_id == "exa_doc_123"
    assert "bloomberg.com" in articles[0].domain
    assert "Samsung announced" in articles[0].snippet


def test_exa_news_client_missing_key(mock_repo):
    """Test graceful handling when EXA_API_KEY is not configured."""
    service = ExaNewsService(mock_repo, api_key=None)
    asset = mock_repo.get_asset_by_ticker("0700.HK")
    inserted = service.fetch_and_store_news(asset)
    assert inserted == 0


def test_apify_social_client_dataset_parsing(mock_repo):
    """Test ApifySocialService parsing dataset records into social posts."""
    service = ApifySocialService(mock_repo, api_token="test_apify_token")
    asset = mock_repo.get_asset_by_ticker("0700.HK")
    assert asset is not None

    dataset_items = [
        {
            "id": "lihkg_thread_991",
            "title": "0700 Tencent target price 500!",
            "content": "Bullish momentum on Tencent with gaming recovery.",
            "author": "hk_trader_king",
            "followers": 1500,
            "likes": 88,
            "reposts": 12,
            "views": 4200,
            "replies": 35,
            "date": "2026-11-06T03:15:00Z",
            "url": "https://lihkg.com/thread/991",
        },
        {
            "msg_id": "tg_msg_8842",
            "text": "Tencent options volume spiking ahead of earnings. Heavy call buying.",
            "author": "HKStockAlerts",
            "subscribers": 12000,
            "forwards": 45,
            "read_count": 8900,
            "date": 1730872800,  # UNIX timestamp
            "url": "https://t.me/hkstock/8842",
        },
    ]

    inserted = service.ingest_from_dataset(
        asset=asset,
        platform="LIHKG",
        actor_id="claude_code_reviewer/lihkg-scraper",
        dataset_items=dataset_items,
    )
    assert inserted == 2

    posts = mock_repo.session.query(SocialPost).filter_by(asset_id=asset.id).all()
    assert len(posts) == 2
    followers_set = {p.author_followers for p in posts}
    views_set = {p.views_count for p in posts}
    assert 1500 in followers_set
    assert 12000 in followers_set
    assert 4200 in views_set
    assert 8900 in views_set
