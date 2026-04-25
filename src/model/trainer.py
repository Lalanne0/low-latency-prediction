"""
Model trainer — downloads historical data, engineers features,
fits a LogisticRegression, and persists the artifacts.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.data.ingestion import StockDataService
from src.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains a lightweight LogisticRegression on historical stock data.
    Labels: price drops > threshold within *horizon* candles → SELL (1), else KEEP (0).
    """

    def __init__(self, ticker: str = settings.DEFAULT_TICKER) -> None:
        self.ticker = ticker

    def train(self) -> tuple[LogisticRegression, StandardScaler]:
        """
        Full training pipeline: fetch → features → label → fit → save.
        Returns (model, scaler).
        """
        # ── 1. Fetch historical data ─────────
        service = StockDataService(self.ticker)
        history = service.load_history()
        if len(history) < 50:
            logger.warning("Insufficient history (%d records). Using fallback model.", len(history))
            return self._fallback_model()

        closes = np.array([r["close"] for r in history], dtype=np.float64)

        # ── 2. Compute features ──────────────
        X = FeatureEngine.compute_batch(closes)
        if X is None or len(X) < 30:
            logger.warning("Insufficient features. Using fallback model.")
            return self._fallback_model()

        # ── 3. Generate labels ───────────────
        # Offset: features start at position min_required, so align labels
        min_req = max(settings.FEATURE_WINDOWS[-1], settings.RSI_PERIOD, settings.BOLLINGER_PERIOD) + 1
        label_closes = closes[min_req:]

        y = self._generate_labels(label_closes, settings.PREDICTION_HORIZON, settings.SELL_THRESHOLD)

        # Trim to matching length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        if len(X) < 30:
            logger.warning("Not enough labelled samples (%d). Fallback.", len(X))
            return self._fallback_model()

        # ── 4. Scale features ────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── 5. Train / evaluate ──────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False,
        )

        model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Training complete — accuracy: %.2f%%", acc * 100)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["KEEP", "SELL"], zero_division=0))

        # ── 6. Persist ───────────────────────
        self._save(model, scaler)

        return model, scaler

    # ── Private ───────────────────────────────

    @staticmethod
    def _generate_labels(closes: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
        """
        Label each point: SELL (1) if price drops > threshold within *horizon* steps,
        else KEEP (0).
        """
        n = len(closes)
        labels = np.zeros(n, dtype=np.int64)
        for i in range(n - horizon):
            future_min = np.min(closes[i + 1: i + 1 + horizon])
            pct_change = (future_min - closes[i]) / closes[i]
            if pct_change < -threshold:
                labels[i] = 1
        return labels

    @staticmethod
    def _save(model: LogisticRegression, scaler: StandardScaler) -> None:
        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
        joblib.dump(model, settings.MODEL_PATH)
        joblib.dump(scaler, settings.SCALER_PATH)
        logger.info("Model saved to %s", settings.MODEL_PATH)

    @staticmethod
    def _fallback_model() -> tuple[LogisticRegression, StandardScaler]:
        """Create a dummy model when no data is available."""
        logger.warning("Creating fallback model with random data")
        rng = np.random.RandomState(42)
        X_dummy = rng.randn(100, 9)
        y_dummy = rng.randint(0, 2, 100)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
        model.fit(X_scaled, y_dummy)
        ModelTrainer._save(model, scaler)
        return model, scaler
