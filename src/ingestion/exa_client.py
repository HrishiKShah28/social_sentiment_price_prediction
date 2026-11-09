"""Financial news ingestion client using Exa.ai."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from loguru import logger
from src.config import settings
from src.storage.models import Asset
from src.storage.repository import DataRepository


class ExaNewsService:
    """Service to discover and ingest financial news via Exa.ai semantic search."""

    def __init__(self, repository: DataRepository, api_key: Optional[str] = None):
        self.repo = repository
        self.api_key = api_key or settings.EXA_API_KEY
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Exa client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("EXA_API_KEY is not configured. Please set it in .env")
            from exa_py import Exa
            self._client = Exa(api_key=self.api_key)
        return self._client

    def build_query(self, asset: Asset) -> str:
        """Construct domain-focused semantic search query for asset news."""
        clean_name = asset.name.replace(" Ltd", "").replace(" Inc.", "").replace(" Co.", "")
        if "HK" in asset.ticker:
            return f"{asset.ticker} {clean_name} stock earnings market Hong Kong"
        else:
            return f"{asset.ticker} {clean_name} stock earnings semiconductor market Korea"

    def fetch_and_store_news(
        self,
        asset: Asset,
        lookback_days: Optional[int] = None,
        num_results: int = 15,
    ) -> int:
        """Execute Exa semantic search and persist news articles."""
        if not self.api_key:
            logger.warning(
                f"Skipping Exa news ingestion for {asset.ticker}: EXA_API_KEY is not configured."
            )
            return 0

        days = lookback_days or settings.LOOKBACK_DAYS
        start_published_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        query = self.build_query(asset)

        logger.info(f"Querying Exa for {asset.ticker}: '{query}' (published >= {start_published_date})")

        try:
            response = self.client.search_and_contents(
                query=query,
                type="auto",
                num_results=num_results,
                start_published_date=start_published_date,
                highlights=True,
                text=True,
            )
        except Exception as e:
            logger.error(f"Exa search failed for {asset.ticker}: {e}")
            return 0

        if not response or not hasattr(response, "results") or not response.results:
            logger.warning(f"No news results returned by Exa for {asset.ticker}")
            return 0

        articles: List[Dict] = []
        for item in response.results:
            # Parse publication timestamp
            pub_date = getattr(item, "published_date", None)
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
                except Exception:
                    dt_utc = datetime.utcnow()
            else:
                dt_utc = datetime.utcnow()

            # Highlights or text snippet
            highlights = getattr(item, "highlights", [])
            snippet = " ".join(highlights) if highlights else (getattr(item, "text", "")[:300] if getattr(item, "text", None) else "")

            # Domain extraction
            url = getattr(item, "url", "")
            domain = ""
            if "://" in url:
                domain = url.split("://")[1].split("/")[0]

            article_dict = {
                "asset_id": asset.id,
                "exa_id": str(getattr(item, "id", f"exa_{hash(url)}")),
                "title": getattr(item, "title", "Untitled News Article"),
                "url": url,
                "snippet": snippet,
                "full_text": getattr(item, "text", None),
                "published_date_utc": dt_utc,
                "exa_score": getattr(item, "score", None),
                "domain": domain,
                "raw_payload": {
                    "id": getattr(item, "id", None),
                    "title": getattr(item, "title", None),
                    "url": getattr(item, "url", None),
                    "score": getattr(item, "score", None),
                    "published_date": pub_date,
                },
            }
            articles.append(article_dict)

        inserted_count = self.repo.add_news_articles(articles)
        logger.info(f"Persisted {len(articles)} Exa news articles for {asset.ticker} ({inserted_count} new)")
        return inserted_count

