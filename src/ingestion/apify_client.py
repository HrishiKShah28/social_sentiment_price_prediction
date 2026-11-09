"""Social media data ingestion client using Apify Actors."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
from src.config import settings
from src.storage.models import Asset
from src.storage.repository import DataRepository


class ApifySocialService:
    """Service to execute Apify actors and ingest multi-platform social sentiment posts."""

    def __init__(self, repository: DataRepository, api_token: Optional[str] = None):
        self.repo = repository
        self.api_token = api_token or settings.APIFY_API_TOKEN
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Apify Client."""
        if self._client is None:
            if not self.api_token:
                raise ValueError("APIFY_API_TOKEN is not configured. Please set it in .env")
            from apify_client import ApifyClient
            self._client = ApifyClient(token=self.api_token)
        return self._client

    def ingest_from_dataset(
        self,
        asset: Asset,
        platform: str,
        actor_id: str,
        dataset_items: List[Dict],
    ) -> int:
        """Parse raw dataset items from an Apify Actor run and persist normalized social posts."""
        posts: List[Dict] = []
        for item in dataset_items:
            # Platform-specific field mapping with robust fallbacks
            raw_id = (
                item.get("id")
                or item.get("post_id")
                or item.get("msg_id")
                or item.get("thread_id")
                or str(hash(str(item)))
            )
            text = (
                item.get("text")
                or item.get("msg")
                or item.get("content")
                or item.get("title", "") + " " + item.get("body", "")
            )
            if not text or not text.strip():
                continue

            # Timestamp parsing
            raw_ts = item.get("date") or item.get("datetime") or item.get("created_at") or item.get("timestamp")
            if raw_ts:
                try:
                    if isinstance(raw_ts, (int, float)):
                        dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                    dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
                except Exception:
                    dt_utc = datetime.utcnow()
            else:
                dt_utc = datetime.utcnow()

            author = item.get("author") or item.get("user") or item.get("username")
            followers = int(item.get("followers") or item.get("subscribers") or item.get("author_followers") or 0)
            likes = int(item.get("likes") or item.get("upvotes") or item.get("agree_count") or 0)
            reposts = int(item.get("reposts") or item.get("shares") or item.get("forwards") or 0)
            views = int(item.get("views") or item.get("read_count") or 0)
            comments = int(item.get("comments") or item.get("replies") or 0)
            url = item.get("url") or item.get("link")

            posts.append(
                {
                    "asset_id": asset.id,
                    "platform": platform,
                    "actor_id": actor_id,
                    "raw_post_id": str(raw_id),
                    "author_id": str(author) if author else None,
                    "author_followers": followers,
                    "post_text": text.strip(),
                    "post_url": url,
                    "timestamp_utc": dt_utc,
                    "likes_count": likes,
                    "reposts_count": reposts,
                    "views_count": views,
                    "comments_count": comments,
                    "raw_payload": item,
                }
            )

        inserted_count = self.repo.add_social_posts(posts)
        logger.info(
            f"Stored {len(posts)} {platform} posts for {asset.ticker} via Actor '{actor_id}' ({inserted_count} new)"
        )
        return inserted_count

    def run_actor_for_asset(
        self,
        asset: Asset,
        platform: str,
        actor_id: str,
        run_input: Dict,
    ) -> int:
        """Execute an Apify actor synchronously and store output items."""
        if not self.api_token:
            logger.warning(
                f"Skipping live Apify scraping for {asset.ticker} ({platform}): APIFY_API_TOKEN is not configured."
            )
            return 0

        logger.info(f"Triggering Apify Actor '{actor_id}' for {asset.ticker} ({platform})")
        try:
            actor_client = self.client.actor(actor_id)
            run = actor_client.call(run_input=run_input)
            dataset = self.client.dataset(run["defaultDatasetId"])
            items = list(dataset.iterate_items())
            return self.ingest_from_dataset(asset, platform, actor_id, items)
        except Exception as e:
            logger.error(f"Apify Actor '{actor_id}' execution failed for {asset.ticker}: {e}")
            return 0

