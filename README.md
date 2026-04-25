# Low-Latency Stock Prediction

> **A real-time stock prediction app focused on inference-speed optimization.**
> Uses lightweight ML models with multiple optimized backends to deliver **sub-10μs** buy/sell predictions streamed live to a premium trading-style UI.

> **Disclaimer**: This is a **demonstration** (and much of an experiment too) of inference-speed optimization techniques. It is **not financial advice** and should not be used for real trading decisions. The predictions have no accuracy for actual market conditions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Data Layer  │──>│   Feature    │──>│  ML Model    │  │
│  │  (yfinance)  │   │   Engine     │   │  (LogReg)    │  │
│  └──────────────┘   └──────────────┘   └───────┬──────┘  │
│                                                │         │
│  ┌──────────────┐                    ┌─────────v───────┐ │
│  │  Synthetic   │───────────────────>│   WebSocket     │ │
│  │  Generator   │                    │   Manager       │ │
│  └──────────────┘                    └─────────┬───────┘ │
│                                                │         │
└────────────────────────────────────────────────┼─────────┘
                                                 │
                                        ┌────────v─────────┐
                                        │   Browser UI     │
                                        │  (LW Charts)     │
                                        └──────────────────┘
```

## Features

- **Live Mode** — Market data via yfinance with continuous KEEP/SELL predictions
- **Demo Mode** — Synthetic data stream with a "Simulate Stock Crash" button
- **4 Inference Backends** — Switch in real-time to compare latency (see below)
- **Real-time UI** — TradingView Lightweight Charts with WebSocket streaming
- **Latency Monitoring** — Nanosecond-precision timing on every prediction

## Inference Backends

The app ships with **4 switchable inference backends**, selectable live from the UI. All produce identical predictions — only the speed changes.

| Backend | Latency | Speedup | Technique |
|---------|---------|---------|-----------|
| **sklearn** | ~100–400 μs | 1× (baseline) | `predict_proba()` with full input validation |
| **NumPy** | ~10–30 μs | ~10× | Raw `np.dot()` + sigmoid, bypasses sklearn |
| **Fused** (f32) | ~10-20 μs | ~15× | Scaler baked into weights, float32 arithmetic |
| **Numba** (JIT) | ~9-15 μs | ~25× | LLVM-compiled native code via `@njit` |

### How it works

**sklearn (baseline):** Standard scikit-learn `model.predict_proba()`. Includes input validation, array reshaping, and type-checking overhead.

**NumPy raw:** Logistic Regression is just `σ(w·x + b)`. We extract the weights at startup and compute the dot product directly:
```python
logit = np.dot(weights, features_scaled) + bias
prob_sell = 1.0 / (1.0 + np.exp(-logit))
```

**Fused (float32):** The StandardScaler transform is algebraically fused into the model weights at startup, eliminating the `scaler.transform()` call entirely:
```python
fused_weights = weights / scaler.scale_
fused_bias = bias - np.dot(fused_weights, scaler.mean_)
# At inference: just one dot product on raw features
```

**Numba JIT:** The fused inference is compiled to native machine code via LLVM:
```python
@njit(cache=True, fastmath=True)
def _numba_predict(weights, bias, features):
    logit = 0.0
    for i in range(features.shape[0]):
        logit += weights[i] * features[i]
    logit += bias
    return 1.0 / (1.0 + np.exp(-logit))
```

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
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| ML Model | scikit-learn (Logistic Regression) |
| Inference | NumPy / Numba JIT (4 switchable backends) |
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

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/api/config` | GET | Current config + available inference modes |
| `/api/health` | GET | Health check |
| `/api/mode` | POST | Switch inference backend `{"mode": "numba"}` |
| `/api/ticker` | POST | Change tracked ticker `{"ticker": "MSFT"}` |
| `/api/demo/crash` | POST | Trigger 20% crash simulation |
| `/api/demo/reset` | POST | Reset demo data |
| `/ws/live` | WS | Live prediction stream |
| `/ws/demo` | WS | Demo prediction stream |

## License

MIT