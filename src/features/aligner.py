"""Point-in-time alignment engine enforcing strict time-series leakage prevention."""

from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger
from src.storage.models import PriceBar, SocialPost, NewsArticle


class TimeSeriesAligner:
    """Enforces strict point-in-time cutoff rules between alternative data and price bars."""

    @staticmethod
    def align_data_to_bars(
        bars: List[PriceBar],
        social_posts: List[SocialPost],
        news_articles: List[NewsArticle],
    ) -> List[Dict]:
        """Group social posts and news articles strictly into the preceding interval (t-1, t]."""
        if not bars:
            return []

        # Ensure bars are sorted strictly ascending in time
        sorted_bars = sorted(bars, key=lambda b: b.timestamp_utc)
        sorted_posts = sorted(social_posts, key=lambda p: p.timestamp_utc)
        sorted_news = sorted(news_articles, key=lambda n: n.published_date_utc)

        aligned_windows: List[Dict] = []

        for i, bar in enumerate(sorted_bars):
            # Window start is previous bar timestamp (or beginning of history for first bar)
            window_start = sorted_bars[i - 1].timestamp_utc if i > 0 else datetime.min
            window_end = bar.timestamp_utc  # Strict information cutoff t

            # Collect posts published strictly in (window_start, window_end]
            matched_posts = [
                p for p in sorted_posts
                if window_start < p.timestamp_utc <= window_end
            ]

            # Collect news published strictly in (window_start, window_end]
            matched_news = [
                n for n in sorted_news
                if window_start < n.published_date_utc <= window_end
            ]

            # Target is the NEXT bar t+1 (if available)
            target_return = None
            target_price = None
            if i + 1 < len(sorted_bars):
                next_bar = sorted_bars[i + 1]
                target_price = float(next_bar.close)
                base_close = float(bar.close)
                if base_close > 0:
                    target_return = round((target_price - base_close) / base_close, 6)

            aligned_windows.append(
                {
                    "bar": bar,
                    "cutoff_utc": window_end,
                    "posts": matched_posts,
                    "news": matched_news,
                    "target_return": target_return,
                    "target_price": target_price,
                }
            )

        logger.debug(
            f"Aligned {len(sorted_bars)} bars with {len(sorted_posts)} posts and {len(sorted_news)} news articles."
        )
        return aligned_windows

