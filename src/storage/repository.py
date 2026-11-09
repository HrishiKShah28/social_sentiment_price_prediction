"""Data access repository for database operations."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.config import AssetConfig
from src.storage.models import (
    Asset,
    Exchange,
    ModelFeature,
    ModelPrediction,
    NewsArticle,
    PipelineRun,
    PriceBar,
    SentimentScore,
    SocialPost,
    TechnicalLevel,
)


class DataRepository:
    """Repository providing typed query and persistence methods."""

    def __init__(self, session: Session):
        self.session = session

    def seed_exchanges_and_assets(self) -> None:
        """Seed default exchanges and target assets if not present."""
        exchanges_data = [
            {"code": "HKEX", "name": "Hong Kong Stock Exchange", "timezone": "Asia/Hong_Kong", "currency": "HKD"},
            {"code": "KRX", "name": "Korea Exchange", "timezone": "Asia/Seoul", "currency": "KRW"},
        ]

        for ex_data in exchanges_data:
            existing = self.session.execute(select(Exchange).where(Exchange.code == ex_data["code"])).scalar_one_or_none()
            if not existing:
                exchange = Exchange(**ex_data)
                self.session.add(exchange)
                self.session.flush()
                logger.info(f"Seeded exchange: {exchange.code}")
            else:
                exchange = existing

            # Seed assets for this exchange
            assets_to_seed = AssetConfig.HKEX_ASSETS if exchange.code == "HKEX" else AssetConfig.KRX_ASSETS
            for asset_data in assets_to_seed:
                existing_asset = self.session.execute(
                    select(Asset).where(Asset.ticker == asset_data["ticker"])
                ).scalar_one_or_none()

                if not existing_asset:
                    asset = Asset(
                        exchange_id=exchange.id,
                        ticker=asset_data["ticker"],
                        name=asset_data["name"],
                        sector=asset_data["sector"],
                        currency=asset_data["currency"],
                        is_active=True,
                    )
                    self.session.add(asset)
                    logger.info(f"Seeded asset: {asset.ticker} ({asset.name})")

        self.session.commit()

    def get_active_assets(self, exchange_code: Optional[str] = None) -> List[Asset]:
        """Retrieve all active assets, optionally filtered by exchange."""
        query = select(Asset).where(Asset.is_active.is_(True))
        if exchange_code:
            query = query.join(Asset.exchange).where(Exchange.code == exchange_code)
        return list(self.session.execute(query).scalars().all())

    def get_asset_by_ticker(self, ticker: str) -> Optional[Asset]:
        """Retrieve asset by its normalized ticker."""
        return self.session.execute(select(Asset).where(Asset.ticker == ticker)).scalar_one_or_none()

    def add_social_posts(self, posts: List[Dict]) -> int:
        """Insert social posts, skipping existing posts."""
        inserted = 0
        for p in posts:
            existing = self.session.execute(
                select(SocialPost).where(
                    SocialPost.platform == p["platform"],
                    SocialPost.raw_post_id == str(p["raw_post_id"]),
                )
            ).scalar_one_or_none()
            if not existing:
                post = SocialPost(
                    asset_id=p["asset_id"],
                    platform=p["platform"],
                    actor_id=p["actor_id"],
                    raw_post_id=str(p["raw_post_id"]),
                    author_id=p.get("author_id"),
                    author_followers=p.get("author_followers", 0),
                    post_text=p["post_text"],
                    post_url=p.get("post_url"),
                    timestamp_utc=p["timestamp_utc"],
                    likes_count=p.get("likes_count", 0),
                    reposts_count=p.get("reposts_count", 0),
                    views_count=p.get("views_count", 0),
                    comments_count=p.get("comments_count", 0),
                    raw_payload=p["raw_payload"],
                )
                self.session.add(post)
                inserted += 1
        self.session.commit()
        return inserted

    def add_news_articles(self, articles: List[Dict]) -> int:
        """Insert news articles, skipping existing by exa_id."""
        inserted = 0
        for a in articles:
            existing = self.session.execute(
                select(NewsArticle).where(NewsArticle.exa_id == a["exa_id"])
            ).scalar_one_or_none()
            if not existing:
                article = NewsArticle(
                    asset_id=a["asset_id"],
                    exa_id=a["exa_id"],
                    title=a["title"],
                    url=a["url"],
                    snippet=a.get("snippet"),
                    full_text=a.get("full_text"),
                    published_date_utc=a["published_date_utc"],
                    exa_score=a.get("exa_score"),
                    domain=a.get("domain"),
                    raw_payload=a["raw_payload"],
                )
                self.session.add(article)
                inserted += 1
        self.session.commit()
        return inserted

    def add_price_bars(self, bars: List[Dict]) -> int:
        """Insert price bars, skipping duplicates for the same asset/timeframe/time."""
        inserted = 0
        for b in bars:
            existing = self.session.execute(
                select(PriceBar).where(
                    PriceBar.asset_id == b["asset_id"],
                    PriceBar.timeframe == b.get("timeframe", "1d"),
                    PriceBar.timestamp_utc == b["timestamp_utc"],
                )
            ).scalar_one_or_none()
            if not existing:
                bar = PriceBar(
                    asset_id=b["asset_id"],
                    timeframe=b.get("timeframe", "1d"),
                    timestamp_utc=b["timestamp_utc"],
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    adjusted_close=b.get("adjusted_close", b["close"]),
                    volume=b["volume"],
                )
                self.session.add(bar)
                inserted += 1
        self.session.commit()
        return inserted

    def get_price_bars(self, asset_id: int, start_time: Optional[datetime] = None) -> List[PriceBar]:
        """Fetch chronological price bars for an asset."""
        query = select(PriceBar).where(PriceBar.asset_id == asset_id)
        if start_time:
            query = query.where(PriceBar.timestamp_utc >= start_time)
        query = query.order_by(PriceBar.timestamp_utc.asc())
        return list(self.session.execute(query).scalars().all())

    def start_pipeline_run(self, stage: str) -> PipelineRun:
        """Log the start of a pipeline run."""
        run = PipelineRun(
            run_id=str(uuid.uuid4()),
            pipeline_stage=stage,
            status="running",
            started_at_utc=datetime.utcnow(),
        )
        self.session.add(run)
        self.session.commit()
        return run

    def finish_pipeline_run(
        self, run_id: str, status: str = "completed", metrics: Optional[Dict] = None, error: Optional[str] = None
    ) -> None:
        """Mark pipeline run completed or failed."""
        run = self.session.execute(select(PipelineRun).where(PipelineRun.run_id == run_id)).scalar_one_or_none()
        if run:
            run.status = status
            run.finished_at_utc = datetime.utcnow()
            run.run_metrics = metrics
            run.error_message = error
            self.session.commit()

