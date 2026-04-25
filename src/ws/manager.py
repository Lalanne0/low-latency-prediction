"""
WebSocket connection manager — broadcasts price ticks and predictions
to all connected clients.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._live_connections: list[WebSocket] = []
        self._demo_connections: list[WebSocket] = []

    # ── Connection lifecycle ──────────────────

    async def connect_live(self, ws: WebSocket) -> None:
        await ws.accept()
        self._live_connections.append(ws)
        logger.info("Live client connected (%d total)", len(self._live_connections))

    async def connect_demo(self, ws: WebSocket) -> None:
        await ws.accept()
        self._demo_connections.append(ws)
        logger.info("Demo client connected (%d total)", len(self._demo_connections))

    def disconnect_live(self, ws: WebSocket) -> None:
        if ws in self._live_connections:
            self._live_connections.remove(ws)
        logger.info("Live client disconnected (%d remaining)", len(self._live_connections))

    def disconnect_demo(self, ws: WebSocket) -> None:
        if ws in self._demo_connections:
            self._demo_connections.remove(ws)
        logger.info("Demo client disconnected (%d remaining)", len(self._demo_connections))

    # ── Broadcasting ──────────────────────────

    async def broadcast_live(self, message: dict) -> None:
        await self._broadcast(self._live_connections, message)

    async def broadcast_demo(self, message: dict) -> None:
        await self._broadcast(self._demo_connections, message)

    async def _broadcast(self, connections: list[WebSocket], message: dict) -> None:
        dead: list[WebSocket] = []
        payload = json.dumps(message)
        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in connections:
                connections.remove(ws)
