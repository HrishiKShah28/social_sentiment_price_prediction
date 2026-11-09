"""Market price data client fetching OHLCV bars via yfinance."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd
from src.config import settings
from src.storage.models import Asset
from src.storage.repository import DataRepository


class PriceDataService:
    """Service to fetch, clean, and persist market OHLCV bars."""

    def __init__(self, repository: DataRepository):
        self.repo = repository

    def fetch_and_store_bars(
        self,
        asset: Asset,
        timeframe: str = "1d",
        lookback_days: Optional[int] = None,
    ) -> int:
        """Fetch historical price bars for an asset and save them to the database."""
        import yfinance as yf

        days = lookback_days or (settings.LOOKBACK_DAYS + settings.WARMUP_DAYS)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        logger.info(
            f"Fetching {timeframe} price bars for {asset.ticker} from {start_dt.date()} to {end_dt.date()}"
        )

        try:
            ticker_obj = yf.Ticker(asset.ticker)
            df: pd.DataFrame = ticker_obj.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=timeframe,
                auto_adjust=False,
            )
        except Exception as e:
            logger.error(f"Failed to fetch market data for {asset.ticker}: {e}")
            return 0

        if df.empty:
            logger.warning(f"No price bars returned by yfinance for {asset.ticker}")
            return 0

        bars: List[Dict] = []
        for index, row in df.iterrows():
            # Normalize timestamp to UTC datetime
            if hasattr(index, "to_pydatetime"):
                ts = index.to_pydatetime()
            else:
                ts = pd.to_datetime(index).to_pydatetime()

            if ts.tzinfo is not None:
                ts_utc = ts.astimezone(timezone.utc).replace(tzinfo=None)
            else:
                ts_utc = ts

            open_val = float(row.get("Open", 0.0))
            high_val = float(row.get("High", 0.0))
            low_val = float(row.get("Low", 0.0))
            close_val = float(row.get("Close", 0.0))
            adj_close_val = float(row.get("Adj Close", close_val))
            volume_val = int(row.get("Volume", 0))

            # Quality validation
            if low_val > high_val or open_val <= 0 or close_val <= 0:
                logger.warning(f"Invalid bar detected for {asset.ticker} at {ts_utc}: skipping.")
                continue

            bars.append(
                {
                    "asset_id": asset.id,
                    "timeframe": timeframe,
                    "timestamp_utc": ts_utc,
                    "open": round(open_val, 4),
                    "high": round(high_val, 4),
                    "low": round(low_val, 4),
                    "close": round(close_val, 4),
                    "adjusted_close": round(adj_close_val, 4),
                    "volume": volume_val,
                }
            )

        inserted_count = self.repo.add_price_bars(bars)
        logger.info(f"Successfully processed {len(bars)} bars for {asset.ticker} ({inserted_count} new inserted)")
        return inserted_count

    def fetch_all_active_assets(self, timeframe: str = "1d") -> Dict[str, int]:
        """Fetch and store bars for all registered assets across exchanges."""
        assets = self.repo.get_active_assets()
        results: Dict[str, int] = {}
        for asset in assets:
            count = self.fetch_and_store_bars(asset, timeframe=timeframe)
            results[asset.ticker] = count
        return results
