"""
Feature engineering pipeline — all features computed in pure NumPy
for maximum speed.
"""

from __future__ import annotations

import numpy as np

from src.config import settings


class FeatureEngine:
    """
    Computes technical-indicator features from a 1-D array of close prices.
    Designed for sub-millisecond execution on rolling windows.
    """

    FEATURE_NAMES: list[str] = [
        "ret_5", "ret_10", "ret_20",
        "vol_10", "vol_20",
        "rsi_14",
        "sma_cross",
        "roc",
        "bb_position",
    ]

    # ── Public API ────────────────────────────

    @staticmethod
    def compute(closes: np.ndarray) -> np.ndarray | None:
        """
        Compute feature vector from close prices.

        Returns a 1-D numpy array of shape (9,) or *None* if the window
        is too short.
        """
        min_required = max(settings.FEATURE_WINDOWS[-1], settings.RSI_PERIOD, settings.BOLLINGER_PERIOD) + 1
        if len(closes) < min_required:
            return None

        features = np.empty(9, dtype=np.float64)

        # ── Momentum returns ──────────────────
        features[0] = (closes[-1] / closes[-5] - 1.0) if closes[-5] != 0 else 0.0
        features[1] = (closes[-1] / closes[-10] - 1.0) if closes[-10] != 0 else 0.0
        features[2] = (closes[-1] / closes[-20] - 1.0) if closes[-20] != 0 else 0.0

        # ── Rolling volatility ────────────────
        features[3] = np.std(closes[-10:])
        features[4] = np.std(closes[-20:])

        # ── RSI (14-period) ───────────────────
        features[5] = FeatureEngine._rsi(closes, settings.RSI_PERIOD)

        # ── SMA crossover ─────────────────────
        sma5 = np.mean(closes[-5:])
        sma20 = np.mean(closes[-20:])
        features[6] = (sma5 - sma20) / sma20 if sma20 != 0 else 0.0

        # ── Rate of change (first derivative) ─
        features[7] = closes[-1] - closes[-2] if len(closes) >= 2 else 0.0

        # ── Bollinger Band position ───────────
        features[8] = FeatureEngine._bb_position(
            closes, settings.BOLLINGER_PERIOD, settings.BOLLINGER_STD
        )

        return features

    @staticmethod
    def compute_batch(closes: np.ndarray, window: int = 21) -> np.ndarray | None:
        """
        Compute features for every valid position in a price series.
        Used for training. Returns (N, 9) array.
        """
        min_required = max(settings.FEATURE_WINDOWS[-1], settings.RSI_PERIOD, settings.BOLLINGER_PERIOD) + 1
        if len(closes) < min_required:
            return None

        rows = []
        for i in range(min_required, len(closes) + 1):
            feat = FeatureEngine.compute(closes[:i])
            if feat is not None:
                rows.append(feat)
        if not rows:
            return None
        return np.vstack(rows)

    # ── Private helpers ───────────────────────

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> float:
        """Relative Strength Index."""
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _bb_position(closes: np.ndarray, period: int, num_std: float) -> float:
        """Where current price sits relative to Bollinger Bands (0–1 range)."""
        window = closes[-period:]
        mid = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.5
        upper = mid + num_std * std
        lower = mid - num_std * std
        band_width = upper - lower
        if band_width == 0:
            return 0.5
        return (closes[-1] - lower) / band_width
