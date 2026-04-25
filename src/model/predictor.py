"""
Low-latency ML predictor — multiple inference backends with
nanosecond-precision timing instrumentation.

Supported modes:
  • sklearn     — standard scikit-learn predict_proba (baseline)
  • numpy       — raw dot-product + sigmoid in NumPy
  • numpy_fused — scaler fused into weights, float32, zero-overhead
  • numba       — Numba JIT-compiled dot-product + sigmoid
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)

# ── Try importing numba (optional) ────────────
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _numba_predict(weights, bias, features):
        """JIT-compiled logistic regression inference."""
        logit = 0.0
        for i in range(features.shape[0]):
            logit += weights[i] * features[i]
        logit += bias
        prob_sell = 1.0 / (1.0 + np.exp(-logit))
        return prob_sell

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class InferenceMode(str, Enum):
    SKLEARN = "sklearn"
    NUMPY = "numpy"
    NUMPY_FUSED = "numpy_fused"
    NUMBA = "numba"


# Labels for the UI
MODE_LABELS = {
    InferenceMode.SKLEARN: "scikit-learn (baseline)",
    InferenceMode.NUMPY: "Raw NumPy",
    InferenceMode.NUMPY_FUSED: "NumPy fused (float32)",
    InferenceMode.NUMBA: "Numba JIT (float32)",
}


class LowLatencyPredictor:
    """
    Multi-backend predictor for KEEP / SELL signals.
    Every call to *predict* is timed at nanosecond resolution.
    Switch backends at runtime via *set_mode* to compare latency.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self._ready = False

        # Active inference mode
        self._mode: InferenceMode = InferenceMode.NUMPY_FUSED

        # Pre-extracted weights (populated in _precompute)
        self._weights_f64: np.ndarray | None = None
        self._bias_f64: float = 0.0
        self._weights_f32: np.ndarray | None = None
        self._bias_f32: float = 0.0

        # Fused weights (scaler baked in)
        self._fused_weights_f64: np.ndarray | None = None
        self._fused_bias_f64: float = 0.0
        self._fused_weights_f32: np.ndarray | None = None
        self._fused_bias_f32: float = 0.0

    # ── Lifecycle ─────────────────────────────

    def load(self) -> bool:
        """Attempt to load a previously saved model from disk."""
        if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.SCALER_PATH):
            self.model = joblib.load(settings.MODEL_PATH)
            self.scaler = joblib.load(settings.SCALER_PATH)
            self._ready = True
            self._precompute()
            logger.info("Model loaded from %s", settings.MODEL_PATH)
            return True
        logger.info("No saved model found — training required")
        return False

    def set_model(self, model: LogisticRegression, scaler: StandardScaler) -> None:
        """Install a freshly trained model."""
        self.model = model
        self.scaler = scaler
        self._ready = True
        self._precompute()

    def _precompute(self) -> None:
        """
        Pre-extract model weights and fuse the scaler for zero-overhead
        inference in optimised modes.
        """
        if self.model is None or self.scaler is None:
            return

        # Raw weights (float64)
        self._weights_f64 = self.model.coef_[0].copy()
        self._bias_f64 = float(self.model.intercept_[0])

        # Raw weights (float32)
        self._weights_f32 = self._weights_f64.astype(np.float32)
        self._bias_f32 = np.float32(self._bias_f64)

        # Fused weights: fold scaler transform into the linear model
        # StandardScaler does: x_scaled = (x - mean) / scale
        # LogReg does:         logit = w · x_scaled + b
        # Combined:            logit = (w / scale) · x + (b - w · mean / scale)
        scale = self.scaler.scale_
        mean = self.scaler.mean_
        self._fused_weights_f64 = self._weights_f64 / scale
        self._fused_bias_f64 = self._bias_f64 - float(np.dot(self._weights_f64 / scale, mean))

        # Fused weights (float32)
        self._fused_weights_f32 = self._fused_weights_f64.astype(np.float32)
        self._fused_bias_f32 = np.float32(self._fused_bias_f64)

        # Warm up numba JIT if available
        if NUMBA_AVAILABLE:
            dummy = np.zeros(len(self._fused_weights_f32), dtype=np.float32)
            _numba_predict(self._fused_weights_f32, self._fused_bias_f32, dummy)
            logger.info("Numba JIT warmed up")

        logger.info("Pre-computed weights for all inference modes")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    @property
    def mode_label(self) -> str:
        return MODE_LABELS.get(self._mode, str(self._mode))

    def set_mode(self, mode: str) -> str:
        """Switch inference mode. Returns the active mode name."""
        try:
            new_mode = InferenceMode(mode)
        except ValueError:
            return self._mode.value

        if new_mode == InferenceMode.NUMBA and not NUMBA_AVAILABLE:
            logger.warning("Numba not installed — falling back to numpy_fused")
            new_mode = InferenceMode.NUMPY_FUSED

        self._mode = new_mode
        logger.info("Inference mode set to: %s", self._mode.value)
        return self._mode.value

    def get_available_modes(self) -> list[dict]:
        """Return list of available modes for the UI."""
        modes = []
        for m in InferenceMode:
            available = True
            if m == InferenceMode.NUMBA and not NUMBA_AVAILABLE:
                available = False
            modes.append({
                "value": m.value,
                "label": MODE_LABELS[m],
                "available": available,
                "active": m == self._mode,
            })
        return modes

    # ── Prediction ────────────────────────────

    def predict(self, features: np.ndarray) -> dict:
        """
        Run inference using the active backend and return prediction + timing.

        The timing covers ONLY the inference step (not feature scaling),
        except in sklearn mode where scaling is part of the pipeline.
        """
        if not self._ready:
            return {"signal": "KEEP", "confidence": 0.0, "inference_us": 0.0, "mode": self._mode.value}

        if self._mode == InferenceMode.SKLEARN:
            return self._predict_sklearn(features)
        elif self._mode == InferenceMode.NUMPY:
            return self._predict_numpy(features)
        elif self._mode == InferenceMode.NUMPY_FUSED:
            return self._predict_numpy_fused(features)
        elif self._mode == InferenceMode.NUMBA:
            return self._predict_numba(features)
        else:
            return self._predict_sklearn(features)

    def _predict_sklearn(self, features: np.ndarray) -> dict:
        """Baseline: scikit-learn predict_proba."""
        feat_scaled = self.scaler.transform(features.reshape(1, -1))

        t0 = time.perf_counter_ns()
        proba = self.model.predict_proba(feat_scaled)[0]
        t1 = time.perf_counter_ns()

        return self._format_result(proba, t1 - t0)

    def _predict_numpy(self, features: np.ndarray) -> dict:
        """Raw NumPy: manual dot product + sigmoid, still uses scaler.transform."""
        feat_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        t0 = time.perf_counter_ns()
        logit = np.dot(self._weights_f64, feat_scaled) + self._bias_f64
        prob_sell = 1.0 / (1.0 + np.exp(-logit))
        t1 = time.perf_counter_ns()

        prob_keep = 1.0 - prob_sell
        proba = np.array([prob_keep, prob_sell])
        return self._format_result(proba, t1 - t0)

    def _predict_numpy_fused(self, features: np.ndarray) -> dict:
        """
        Fused NumPy: scaler baked into weights, float32.
        No scaler.transform call. Minimal overhead.
        """
        feat_f32 = features.astype(np.float32)

        t0 = time.perf_counter_ns()
        logit = np.dot(self._fused_weights_f32, feat_f32) + self._fused_bias_f32
        prob_sell = 1.0 / (1.0 + np.exp(-float(logit)))
        t1 = time.perf_counter_ns()

        prob_keep = 1.0 - prob_sell
        proba = np.array([prob_keep, prob_sell])
        return self._format_result(proba, t1 - t0)

    def _predict_numba(self, features: np.ndarray) -> dict:
        """Numba JIT: compiled dot-product + sigmoid, float32."""
        if not NUMBA_AVAILABLE:
            return self._predict_numpy_fused(features)

        feat_f32 = features.astype(np.float32)

        t0 = time.perf_counter_ns()
        prob_sell = float(_numba_predict(self._fused_weights_f32, self._fused_bias_f32, feat_f32))
        t1 = time.perf_counter_ns()

        prob_keep = 1.0 - prob_sell
        proba = np.array([prob_keep, prob_sell])
        return self._format_result(proba, t1 - t0)

    def _format_result(self, proba: np.ndarray, inference_ns: int) -> dict:
        """Format prediction result with timing."""
        inference_us = inference_ns / 1_000.0
        predicted_class = int(np.argmax(proba))
        signal = "SELL" if predicted_class == 1 else "KEEP"
        confidence = float(proba[predicted_class])

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "inference_us": round(inference_us, 2),
            "mode": self._mode.value,
        }
