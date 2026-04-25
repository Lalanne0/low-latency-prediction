"""
Stock data ingestion service — fetches real market data via yfinance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone

import yfinance as yf
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class StockDataService:
    """
    Fetches historical data on startup, then polls yfinance for live ticks.
    Maintains a rolling window of recent prices for feature engineering.
    """

    def __init__(self, ticker: str = settings.DEFAULT_TICKER) -> None:
        self.ticker_symbol = ticker.upper()
        self._ticker = yf.Ticker(self.ticker_symbol)
        self.prices: deque[dict] = deque(maxlen=settings.ROLLING_WINDOW)
        self._running = False

    # ── Public API ────────────────────────────

    def load_history(self) -> list[dict]:
        """Download historical daily data for model training."""
        logger.info("Downloading %d days of history for %s …", settings.HISTORY_DAYS, self.ticker_symbol)
        df = self._ticker.history(period=f"{settings.HISTORY_DAYS}d", interval="1d")
        if df.empty:
            logger.warning("No historical data returned for %s", self.ticker_symbol)
            return []
        records = []
        for ts, row in df.iterrows():
            records.append({
                "timestamp": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        logger.info("Loaded %d historical records", len(records))
        return records

    async def start_polling(self, callback) -> None:
        """
        Begin polling for live price updates.
        *callback* is an async callable receiving a tick dict.
        """
        self._running = True
        logger.info("Live polling started for %s (interval=%.1fs)", self.ticker_symbol, settings.POLL_INTERVAL_SEC)

        while self._running:
            try:
                tick = await asyncio.to_thread(self._fetch_tick)
                if tick is not None:
                    self.prices.append(tick)
                    await callback(tick)
            except Exception:
                logger.exception("Polling error")
            await asyncio.sleep(settings.POLL_INTERVAL_SEC)

    def stop(self) -> None:
        self._running = False

    def change_ticker(self, new_ticker: str) -> None:
        self.ticker_symbol = new_ticker.upper()
        self._ticker = yf.Ticker(self.ticker_symbol)
        self.prices.clear()
        logger.info("Switched ticker to %s", self.ticker_symbol)

    def get_recent_closes(self) -> np.ndarray:
        """Return numpy array of recent close prices."""
        if not self.prices:
            return np.array([])
        return np.array([p["close"] for p in self.prices], dtype=np.float64)

    # ── Private ───────────────────────────────

    def _fetch_tick(self) -> dict | None:
        """Synchronous tick fetch — runs in a thread."""
        try:
            info = self._ticker.fast_info
            price = float(info.last_price)
            now = datetime.now(timezone.utc)
            return {
                "timestamp": now.isoformat(),
                "time": int(now.timestamp()),
                "close": price,
                "open": price,
                "high": price,
                "low": price,
                "volume": 0,
            }
        except Exception:
            logger.exception("Failed to fetch tick for %s", self.ticker_symbol)
            return None
