"""
Application configuration — environment-driven with sensible defaults.
"""

import os


class Settings:
    """Centralised configuration for the low-latency prediction app."""

    # ── Ticker ────────────────────────────────
    DEFAULT_TICKER: str = os.getenv("DEFAULT_TICKER", "AAPL")

    # ── Prediction ────────────────────────────
    PREDICTION_HORIZON: int = int(os.getenv("PREDICTION_HORIZON", "5"))
    SELL_THRESHOLD: float = float(os.getenv("SELL_THRESHOLD", "0.005"))  # 0.5 %

    # ── Data ingestion ────────────────────────
    POLL_INTERVAL_SEC: float = float(os.getenv("POLL_INTERVAL_SEC", "1.0"))
    HISTORY_DAYS: int = int(os.getenv("HISTORY_DAYS", "180"))  # 6 months
    ROLLING_WINDOW: int = int(os.getenv("ROLLING_WINDOW", "500"))

    # ── Demo mode ─────────────────────────────
    DEMO_TICK_INTERVAL_SEC: float = float(os.getenv("DEMO_TICK_INTERVAL_SEC", "0.2"))
    DEMO_INITIAL_PRICE: float = float(os.getenv("DEMO_INITIAL_PRICE", "185.0"))
    DEMO_CRASH_DROP_PCT: float = float(os.getenv("DEMO_CRASH_DROP_PCT", "0.20"))  # 20 %
    DEMO_CRASH_TICKS: int = int(os.getenv("DEMO_CRASH_TICKS", "15"))

    # ── Feature engineering ───────────────────
    FEATURE_WINDOWS: list[int] = [5, 10, 20]
    RSI_PERIOD: int = 14
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # ── Server ────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # ── Model persistence ─────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model_artifacts/model.joblib")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "model_artifacts/scaler.joblib")


settings = Settings()
