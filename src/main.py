"""
FastAPI application — entry point for the low-latency prediction service.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.data.ingestion import StockDataService
from src.data.synthetic import SyntheticDataGenerator
from src.features.engine import FeatureEngine
from src.model.predictor import LowLatencyPredictor
from src.model.trainer import ModelTrainer
from src.ws.manager import ConnectionManager

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────
manager = ConnectionManager()
predictor = LowLatencyPredictor()
stock_service = StockDataService(settings.DEFAULT_TICKER)
demo_generator = SyntheticDataGenerator()

_live_task: asyncio.Task | None = None
_demo_task: asyncio.Task | None = None
_previous_signal: str = "KEEP"
_demo_previous_signal: str = "KEEP"

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifespan ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _live_task
    # Train or load model
    if not predictor.load():
        logger.info("Training model for %s …", settings.DEFAULT_TICKER)
        model, scaler = await asyncio.to_thread(ModelTrainer(settings.DEFAULT_TICKER).train)
        predictor.set_model(model, scaler)
    # Start live polling
    _live_task = asyncio.create_task(stock_service.start_polling(_on_live_tick))
    logger.info("🚀 Low-latency predictor ready — %s", settings.DEFAULT_TICKER)
    yield
    # Shutdown
    stock_service.stop()
    demo_generator.stop()
    if _live_task:
        _live_task.cancel()


app = FastAPI(
    title="Low-Latency Prediction",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Tick callbacks ────────────────────────────

async def _on_live_tick(tick: dict) -> None:
    global _previous_signal
    closes = stock_service.get_recent_closes()
    features = FeatureEngine.compute(closes)
    prediction = predictor.predict(features) if features is not None else {
        "signal": "KEEP", "confidence": 0.0, "inference_us": 0.0
    }

    alert = None
    if _previous_signal == "KEEP" and prediction["signal"] == "SELL":
        alert = "⚠️ SIGNAL CHANGE: KEEP → SELL"
    _previous_signal = prediction["signal"]

    await manager.broadcast_live({
        "type": "update",
        "tick": tick,
        "prediction": prediction,
        "alert": alert,
    })


async def _on_demo_tick(tick: dict) -> None:
    global _demo_previous_signal
    closes = demo_generator.get_recent_closes()
    features = FeatureEngine.compute(closes)
    prediction = predictor.predict(features) if features is not None else {
        "signal": "KEEP", "confidence": 0.0, "inference_us": 0.0
    }

    alert = None
    if _demo_previous_signal == "KEEP" and prediction["signal"] == "SELL":
        alert = "⚠️ SIGNAL CHANGE: KEEP → SELL"
    _demo_previous_signal = prediction["signal"]

    await manager.broadcast_demo({
        "type": "update",
        "tick": tick,
        "prediction": prediction,
        "alert": alert,
    })


# ── REST endpoints ────────────────────────────

@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_ready": predictor.is_ready}


@app.get("/api/config")
async def get_config():
    return {
        "ticker": stock_service.ticker_symbol,
        "model_ready": predictor.is_ready,
        "prediction_horizon": settings.PREDICTION_HORIZON,
        "sell_threshold": settings.SELL_THRESHOLD,
        "inference_mode": predictor.mode.value,
        "inference_modes": predictor.get_available_modes(),
    }


@app.post("/api/mode")
async def change_mode(payload: dict):
    mode = payload.get("mode", "")
    if not mode:
        return JSONResponse({"error": "mode is required"}, status_code=400)
    active = predictor.set_mode(mode)
    return {"status": "ok", "mode": active, "modes": predictor.get_available_modes()}


@app.post("/api/ticker")
async def change_ticker(payload: dict):
    global _live_task, _previous_signal
    new_ticker = payload.get("ticker", "").upper()
    if not new_ticker:
        return JSONResponse({"error": "ticker is required"}, status_code=400)

    # Stop current polling
    stock_service.stop()
    if _live_task:
        _live_task.cancel()

    # Retrain model
    stock_service.change_ticker(new_ticker)
    logger.info("Retraining model for %s …", new_ticker)
    model, scaler = await asyncio.to_thread(ModelTrainer(new_ticker).train)
    predictor.set_model(model, scaler)
    _previous_signal = "KEEP"

    # Restart polling
    _live_task = asyncio.create_task(stock_service.start_polling(_on_live_tick))

    return {"status": "ok", "ticker": new_ticker}


@app.post("/api/demo/crash")
async def trigger_crash():
    demo_generator.simulate_crash()
    return {"status": "crash_triggered"}


@app.post("/api/demo/reset")
async def reset_demo():
    global _demo_previous_signal
    demo_generator.reset()
    _demo_previous_signal = "KEEP"
    return {"status": "reset"}


# ── WebSocket endpoints ───────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await manager.connect_live(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect_live(ws)


@app.websocket("/ws/demo")
async def ws_demo(ws: WebSocket):
    global _demo_task
    await manager.connect_demo(ws)

    # Start demo generator if not already running
    if _demo_task is None or _demo_task.done():
        demo_generator.reset()
        _demo_task = asyncio.create_task(demo_generator.start_streaming(_on_demo_tick))

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_demo(ws)
