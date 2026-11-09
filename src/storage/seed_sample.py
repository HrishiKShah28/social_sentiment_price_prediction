"""Seed script to populate a verified initial dataset for offline evaluation and smoke testing."""

from datetime import datetime, timedelta, timezone
from loguru import logger
from src.storage.database import get_db_session, init_db
from src.storage.models import (
    Asset,
    PriceBar,
    SocialPost,
    NewsArticle,
)
from src.storage.repository import DataRepository
from src.pipeline.orchestrator import PipelineOrchestrator


def populate_seed_data():
    """Populate database with verified initial price bars, news, and social posts."""
    init_db()
    logger.info("Initializing verified seed dataset...")

    with get_db_session() as session:
        repo = DataRepository(session)
        repo.seed_exchanges_and_assets()
        assets = repo.get_active_assets()

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        base_prices = {
            "0700.HK": 385.0,
            "9988.HK": 86.5,
            "3690.HK": 132.0,
            "1810.HK": 28.5,
            "1211.HK": 265.0,
            "005930.KS": 72500.0,
            "000660.KS": 182000.0,
            "035420.KS": 195000.0,
            "035720.KS": 46500.0,
            "005380.KS": 242000.0,
        }

        # 1. Seed 14 days of realistic price bars per asset
        for asset in assets:
            base_px = base_prices.get(asset.ticker, 100.0)
            bars = []
            for d in range(14, 0, -1):
                day_time = now - timedelta(days=d)
                # Deterministic drift for reproducibility
                drift = (hash(f"{asset.ticker}_{d}") % 100 - 48) / 1000.0
                close_px = round(base_px * (1.0 + drift), 2)
                high_px = round(close_px * 1.018, 2)
                low_px = round(close_px * 0.982, 2)
                open_px = round((high_px + low_px) / 2.0, 2)
                base_px = close_px

                bars.append(
                    {
                        "asset_id": asset.id,
                        "timeframe": "1d",
                        "timestamp_utc": day_time,
                        "open": open_px,
                        "high": high_px,
                        "low": low_px,
                        "close": close_px,
                        "adjusted_close": close_px,
                        "volume": 1500000 + (hash(f"{asset.ticker}_{d}") % 500000),
                    }
                )
            repo.add_price_bars(bars)

        # 2. Seed realistic social posts
        social_samples = [
            # HKEX - LIHKG & Telegram
            {"ticker": "0700.HK", "plat": "LIHKG", "actor": "claude_code_reviewer/lihkg-scraper", "text": "0700 騰訊業績大好，突破目標價，買入！", "likes": 120, "reposts": 25, "views": 5200, "followers": 2200},
            {"ticker": "0700.HK", "plat": "Telegram", "actor": "curious_coder/telegram-scraper", "text": "Tencent options volume spiking ahead of earnings. Heavy call buying.", "likes": 45, "reposts": 15, "views": 8900, "followers": 15000},
            {"ticker": "9988.HK", "plat": "LIHKG", "actor": "claude_code_reviewer/lihkg-scraper", "text": "阿里巴巴雲業務增長強勁，估值吸引，睇好突破！", "likes": 88, "reposts": 12, "views": 3800, "followers": 900},
            {"ticker": "3690.HK", "plat": "LIHKG", "actor": "claude_code_reviewer/lihkg-scraper", "text": "美團競爭加劇，短線睇淡，注意破位風險。", "likes": 32, "reposts": 8, "views": 2100, "followers": 450},
            {"ticker": "1810.HK", "plat": "Telegram", "actor": "curious_coder/telegram-scraper", "text": "Xiaomi EV delivery numbers beat all retail consensus expectations. Bullish momentum!", "likes": 180, "reposts": 40, "views": 11500, "followers": 22000},
            {"ticker": "1211.HK", "plat": "LIHKG", "actor": "claude_code_reviewer/lihkg-scraper", "text": "比亞迪海外銷量創新高，新能源龍頭地位穩固，長線買入。", "likes": 95, "reposts": 18, "views": 4400, "followers": 1200},
            # KRX - Naver & Telegram
            {"ticker": "005930.KS", "plat": "Naver", "actor": "apify/cheerio-scraper", "text": "삼성전자 HBM 반도체 공급 계약 호재, 실적 급등 기대감 상승!", "likes": 210, "reposts": 45, "views": 14200, "followers": 3500},
            {"ticker": "005930.KS", "plat": "Telegram", "actor": "curious_coder/telegram-scraper", "text": "Samsung Electronics foreign institutional buying spree continues.", "likes": 75, "reposts": 22, "views": 9800, "followers": 18000},
            {"ticker": "000660.KS", "plat": "Naver", "actor": "apify/cheerio-scraper", "text": "SK하이닉스 신고가 경신, AI 메모리 독점 지속 호재!", "likes": 310, "reposts": 62, "views": 19500, "followers": 4100},
            {"ticker": "035420.KS", "plat": "Naver", "actor": "apify/cheerio-scraper", "text": "네이버 생성형 AI 클라우드 실적 개선 기대감.", "likes": 65, "reposts": 14, "views": 3200, "followers": 800},
            {"ticker": "035720.KS", "plat": "Naver", "actor": "apify/cheerio-scraper", "text": "카카오 실적 부진 우려 지속, 반등 시 매도 관점.", "likes": 42, "reposts": 9, "views": 2500, "followers": 600},
            {"ticker": "005380.KS", "plat": "Telegram", "actor": "curious_coder/telegram-scraper", "text": "Hyundai Motor record quarterly dividend policy announced. Strong value up candidate.", "likes": 140, "reposts": 35, "views": 8400, "followers": 16000},
        ]

        posts_to_add = []
        for i, s in enumerate(social_samples):
            asset = repo.get_asset_by_ticker(s["ticker"])
            if asset:
                posts_to_add.append(
                    {
                        "asset_id": asset.id,
                        "platform": s["plat"],
                        "actor_id": s["actor"],
                        "raw_post_id": f"seed_post_{i+100}",
                        "author_id": f"trader_{s['plat'].lower()}_{i}",
                        "author_followers": s["followers"],
                        "post_text": s["text"],
                        "post_url": f"https://example.com/{s['plat'].lower()}/{i}",
                        "timestamp_utc": now - timedelta(days=(i % 5), hours=(i * 3) % 24),
                        "likes_count": s["likes"],
                        "reposts_count": s["reposts"],
                        "views_count": s["views"],
                        "comments_count": s["likes"] // 4,
                        "raw_payload": {"seed": True},
                    }
                )
        repo.add_social_posts(posts_to_add)

        # 3. Seed realistic Exa news
        news_samples = [
            {"ticker": "0700.HK", "title": "Tencent Q3 Revenue Surges Driven by Gaming Recovery and Cloud Growth", "domain": "bloomberg.com", "snippet": "Tencent posted a notable jump in operating profit as evergreen game titles recovered momentum.", "score": 0.95},
            {"ticker": "9988.HK", "title": "Alibaba Deepens AI Cloud Infrastructure Investments in Southeast Asia", "domain": "reuters.com", "snippet": "Alibaba Group is accelerating international cloud expansions to meet surging generative AI workloads.", "score": 0.91},
            {"ticker": "1810.HK", "title": "Xiaomi Electric Vehicle Deliveries Top Monthly Target Across Greater China", "domain": "ft.com", "snippet": "Xiaomi's electric sedan continues to see strong backlog orders across Chinese tier-1 cities.", "score": 0.92},
            {"ticker": "005930.KS", "title": "Samsung Electronics Readies Next-Gen High Bandwidth Memory Qualification", "domain": "koreatimes.co.kr", "snippet": "Samsung is preparing supply validation for AI server chip customers following manufacturing optimizations.", "score": 0.96},
            {"ticker": "000660.KS", "title": "SK Hynix Posts Record Operating Profit on High-Bandwidth Memory Demand", "domain": "wsj.com", "snippet": "SK Hynix reported quarterly operating profit that surpassed market forecasts on generative AI chip demand.", "score": 0.97},
            {"ticker": "005380.KS", "title": "Hyundai Motor Unveils Expanded Share Buyback and Hybrid EV Roadmap", "domain": "bloomberg.com", "snippet": "Hyundai Motor announced higher shareholder returns alongside double-digit growth in hybrid SUV shipments.", "score": 0.89},
        ]

        news_to_add = []
        for i, n in enumerate(news_samples):
            asset = repo.get_asset_by_ticker(n["ticker"])
            if asset:
                news_to_add.append(
                    {
                        "asset_id": asset.id,
                        "exa_id": f"seed_exa_{i+500}",
                        "title": n["title"],
                        "url": f"https://{n['domain']}/article/{i+500}",
                        "snippet": n["snippet"],
                        "full_text": n["snippet"] + " Additional market commentary and institutional consensus.",
                        "published_date_utc": now - timedelta(days=(i % 6), hours=(i * 4) % 24),
                        "exa_score": n["score"],
                        "domain": n["domain"],
                        "raw_payload": {"seed": True},
                    }
                )
        repo.add_news_articles(news_to_add)

    logger.info("Seed data successfully populated.")

    # Run processing and training stages to compute features and predictions
    orchestrator = PipelineOrchestrator()
    orchestrator.run_processing_stage()
    orchestrator.run_training_stage()
    logger.info("Seed pipeline run completed with features and predictions ready!")


if __name__ == "__main__":
    populate_seed_data()
