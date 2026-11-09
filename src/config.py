"""Application configuration and environment settings."""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class AssetConfig:
    """Pre-configured target assets per exchange."""

    HKEX_ASSETS: List[Dict[str, str]] = [
        {"ticker": "0700.HK", "name": "Tencent Holdings Ltd", "sector": "Technology", "currency": "HKD"},
        {"ticker": "9988.HK", "name": "Alibaba Group Holding", "sector": "E-Commerce", "currency": "HKD"},
        {"ticker": "3690.HK", "name": "Meituan", "sector": "Technology / Consumer", "currency": "HKD"},
        {"ticker": "1810.HK", "name": "Xiaomi Corporation", "sector": "Hardware / EV", "currency": "HKD"},
        {"ticker": "1211.HK", "name": "BYD Company Ltd", "sector": "Automotive / EV", "currency": "HKD"},
    ]

    KRX_ASSETS: List[Dict[str, str]] = [
        {"ticker": "005930.KS", "name": "Samsung Electronics Co.", "sector": "Semiconductors", "currency": "KRW"},
        {"ticker": "000660.KS", "name": "SK Hynix Inc.", "sector": "Semiconductors / AI", "currency": "KRW"},
        {"ticker": "035420.KS", "name": "NAVER Corporation", "sector": "Internet / AI", "currency": "KRW"},
        {"ticker": "035720.KS", "name": "Kakao Corporation", "sector": "Fintech / Internet", "currency": "KRW"},
        {"ticker": "005380.KS", "name": "Hyundai Motor Company", "sector": "Automotive", "currency": "KRW"},
    ]


class Settings(BaseSettings):
    """Core application settings validated via environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Credentials
    APIFY_API_TOKEN: Optional[str] = Field(default=None, description="Apify API Token")
    EXA_API_KEY: Optional[str] = Field(default=None, description="Exa.ai API Key")

    # Database
    DATABASE_URL: str = Field(
        default=f"sqlite:///{DATA_DIR / 'sentiment_agent.db'}",
        description="SQLAlchemy Database URL",
    )

    # Ingestion & Time-series settings
    LOOKBACK_DAYS: int = Field(default=10, description="Lookback window in days for sentiment and news")
    WARMUP_DAYS: int = Field(default=60, description="Historical warm-up days for technical indicators (RSI, MACD)")
    DEFAULT_TIMEFRAME: str = Field(default="1d", description="Default price bar resolution")

    # Apify Actor Defaults
    APIFY_LIHKG_ACTOR_ID: str = Field(default="claude_code_reviewer/lihkg-scraper", description="Apify LIHKG scraper actor")
    APIFY_TELEGRAM_ACTOR_ID: str = Field(default="curious_coder/telegram-scraper", description="Apify Telegram scraper actor")
    APIFY_NAVER_ACTOR_ID: str = Field(default="apify/cheerio-scraper", description="Apify Naver scraper actor")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
