"""Algorithmic support/resistance level calculation, trend direction, and position extraction."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from loguru import logger
from src.storage.models import PriceBar


@dataclass
class TechnicalSignal:
    """Structured technical signal output."""

    current_price: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    poc_price: float  # Point of Control (volume-weighted price cluster)
    trend_direction: str  # 'Bullish', 'Bearish', 'Neutral'
    suggested_position: str  # 'Long', 'Short', 'Flat'
    calculation_method: str


class TechnicalLevelExtractor:
    """Computes objective support/resistance levels and rule-based trade postures."""

    @staticmethod
    def extract_levels(
        bars: List[PriceBar],
        composite_sentiment: float = 0.0,
        lookback_window: int = 10,
    ) -> Optional[TechnicalSignal]:
        """Compute support/resistance pivots, volume profile mode, direction, and position."""
        if not bars or len(bars) < 3:
            logger.warning("Insufficient price bars to compute technical levels (minimum 3 required).")
            return None

        # Sort bars chronologically
        sorted_bars = sorted(bars, key=lambda b: b.timestamp_utc)
        recent_bars = sorted_bars[-lookback_window:]

        current_price = float(sorted_bars[-1].close)
        highs = np.array([b.high for b in recent_bars])
        lows = np.array([b.low for b in recent_bars])
        closes = np.array([b.close for b in recent_bars])
        volumes = np.array([b.volume for b in recent_bars])

        # Classical Standard Pivot Points based on latest bar
        last_h = highs[-1]
        last_l = lows[-1]
        last_c = closes[-1]
        pivot = (last_h + last_l + last_c) / 3.0

        r1 = (2.0 * pivot) - last_l
        s1 = (2.0 * pivot) - last_h
        r2 = pivot + (last_h - last_l)
        s2 = pivot - (last_h - last_l)

        # Refine with local rolling extrema if available
        rolling_max = float(np.max(highs))
        rolling_min = float(np.min(lows))
        r1 = max(r1, rolling_max)
        s1 = min(s1, rolling_min)

        # Volume Profile POC (Point of Control) estimation
        num_bins = min(10, len(recent_bars))
        hist, bin_edges = np.histogram(closes, bins=num_bins, weights=volumes)
        max_bin_idx = int(np.argmax(hist))
        poc_price = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)

        # Moving Average for trend direction
        sma = float(np.mean(closes))

        # Direction synthesis
        if composite_sentiment > 0.15 and current_price >= sma:
            direction = "Bullish"
        elif composite_sentiment < -0.15 and current_price <= sma:
            direction = "Bearish"
        else:
            direction = "Neutral"

        # Position Recommendation based on proximity to support/resistance
        dist_to_s1 = abs(current_price - s1) / current_price if current_price > 0 else 1.0
        dist_to_r1 = abs(r1 - current_price) / current_price if current_price > 0 else 1.0

        if direction == "Bullish" and dist_to_s1 <= 0.035:
            # Favorable long entry near support
            position = "Long"
        elif direction == "Bearish" and dist_to_r1 <= 0.035:
            # Favorable short entry near resistance
            position = "Short"
        elif direction == "Bullish" and current_price > sma:
            position = "Long"
        elif direction == "Bearish" and current_price < sma:
            position = "Short"
        else:
            position = "Flat"

        return TechnicalSignal(
            current_price=round(current_price, 2),
            support_1=round(s1, 2),
            support_2=round(s2, 2),
            resistance_1=round(r1, 2),
            resistance_2=round(r2, 2),
            poc_price=round(poc_price, 2),
            trend_direction=direction,
            suggested_position=position,
            calculation_method="Rolling Swing Pivot + Volume Profile POC",
        )

