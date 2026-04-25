# ⚡ Low-Latency Stock Prediction

> **A real-time stock prediction app focused on inference-speed optimization.**
> Uses lightweight ML models to deliver sub-millisecond buy/sell predictions streamed live to a premium trading-style UI.

> ⚠️ **Disclaimer**: This is a **technical demonstration** of inference-speed optimization techniques. It is **not financial advice** and should not be used for real trading decisions. The predictions have no proven accuracy for actual market conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  Data Layer   │──▶│   Feature    │──▶│  ML Model    │ │
│  │  (yfinance)   │   │   Engine     │   │  (LogReg)    │ │
│  └──────────────┘   └──────────────┘   └──────┬───────┘ │
│                                                │         │
│  ┌──────────────┐                    ┌─────────▼───────┐ │
│  │  Synthetic   │───────────────────▶│   WebSocket     │ │
│  │  Generator   │                    │   Manager       │ │
│  └──────────────┘                    └─────────┬───────┘ │
│                                                │         │
└────────────────────────────────────────────────┼─────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │   Browser UI     │
                                        │  (LW Charts)     │
                                        └──────────────────┘
```

## Features

- **Live Mode** — Real market data via yfinance with continuous KEEP/SELL predictions
- **Demo Mode** — Synthetic data stream with a "Simulate Stock Crash" button
- **Sub-millisecond Inference** — Logistic Regression on 9 NumPy-computed features
- **Real-time UI** — TradingView Lightweight Charts with WebSocket streaming
- **Latency Monitoring** — Nanosecond-precision timing on every prediction
- **Alert System** — Animated warning when signal changes to SELL

## Quick Start

### Docker (recommended)

```bash
# Build and run
make run

# Or with docker compose directly
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000)

### Local Development

```bash
# Install dependencies with UV
uv sync

# Run dev server with auto-reload
make dev
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| ML Model | scikit-learn (Logistic Regression) |
| Features | NumPy (9 technical indicators) |
| Frontend | TradingView Lightweight Charts |
| Transport | WebSocket |
| Data | yfinance |
| Package Manager | UV |
| Container | Docker (multi-stage) |

## Model Details

The model uses **Logistic Regression** — chosen specifically for its sub-millisecond inference speed. Features include:

- Price momentum (5, 10, 20-tick returns)
- Rolling volatility (10, 20-tick standard deviation)
- RSI (14-period Relative Strength Index)
- SMA crossover (5 vs 20-period moving average)
- Rate of change (first derivative)
- Bollinger Band position

The model is automatically trained on 6 months of historical data on first startup.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TICKER` | `AAPL` | Stock ticker symbol |
| `PREDICTION_HORIZON` | `5` | Candles to look ahead |
| `SELL_THRESHOLD` | `0.005` | Drop % to trigger SELL |
| `POLL_INTERVAL_SEC` | `1.0` | Live data fetch interval |
| `DEMO_TICK_INTERVAL_SEC` | `0.2` | Demo data generation speed |

## License

MIT