"""
Synthetic data generator for demo mode — geometric Brownian motion
with configurable crash simulation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates realistic-looking stock price data using geometric Brownian motion.
    Supports a triggered crash simulation for demo purposes.
    """

    def __init__(
        self,
        initial_price: float = settings.DEMO_INITIAL_PRICE,
        tick_interval: float = settings.DEMO_TICK_INTERVAL_SEC,
    ) -> None:
        self.price = initial_price
        self.initial_price = initial_price
        self.tick_interval = tick_interval
        self.prices: deque[dict] = deque(maxlen=settings.ROLLING_WINDOW)

        # GBM parameters
        self._mu = 0.0001       # slight upward drift
        self._sigma = 0.002     # normal volatility
        self._dt = tick_interval

        # Crash state
        self._crash_active = False
        self._crash_ticks_remaining = 0
        self._crash_drop_per_tick = 0.0

        self._running = False

    # ── Public API ────────────────────────────

    async def start_streaming(self, callback) -> None:
        """
        Begin generating synthetic ticks.
        *callback* is an async callable receiving a tick dict.
        """
        self._running = True
        logger.info("Synthetic data stream started (interval=%.2fs)", self.tick_interval)

        while self._running:
            tick = self._generate_tick()
            self.prices.append(tick)
            await callback(tick)
            await asyncio.sleep(self.tick_interval)

    def stop(self) -> None:
        self._running = False

    def simulate_crash(self) -> None:
        """Trigger a rapid price crash over N ticks."""
        self._crash_active = True
        self._crash_ticks_remaining = settings.DEMO_CRASH_TICKS
        total_drop = self.price * settings.DEMO_CRASH_DROP_PCT
        self._crash_drop_per_tick = total_drop / settings.DEMO_CRASH_TICKS
        logger.info(
            "💥 Crash triggered: %.2f%% drop over %d ticks",
            settings.DEMO_CRASH_DROP_PCT * 100,
            settings.DEMO_CRASH_TICKS,
        )

    def reset(self) -> None:
        """Reset generator to initial state."""
        self.price = self.initial_price
        self._crash_active = False
        self._crash_ticks_remaining = 0
        self.prices.clear()

    def get_recent_closes(self) -> np.ndarray:
        if not self.prices:
            return np.array([])
        return np.array([p["close"] for p in self.prices], dtype=np.float64)

    # ── Private ───────────────────────────────

    def _generate_tick(self) -> dict:
        if self._crash_active and self._crash_ticks_remaining > 0:
            # Crash mode — steep decline with some noise
            noise = random.gauss(0, self._crash_drop_per_tick * 0.2)
            self.price -= self._crash_drop_per_tick + noise
            self._crash_ticks_remaining -= 1
            if self._crash_ticks_remaining == 0:
                self._crash_active = False
                logger.info("Crash sequence complete. Price: %.2f", self.price)
        else:
            # Normal GBM step
            drift = (self._mu - 0.5 * self._sigma ** 2) * self._dt
            diffusion = self._sigma * math.sqrt(self._dt) * random.gauss(0, 1)
            self.price *= math.exp(drift + diffusion)

        self.price = max(self.price, 0.01)  # floor at 1 cent

        now = datetime.now(timezone.utc)
        return {
            "timestamp": now.isoformat(),
            "time": int(now.timestamp()),
            "close": round(self.price, 4),
            "open": round(self.price, 4),
            "high": round(self.price * (1 + abs(random.gauss(0, 0.0005))), 4),
            "low": round(self.price * (1 - abs(random.gauss(0, 0.0005))), 4),
            "volume": random.randint(100_000, 500_000),
        }
