# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────
# Low-Latency Prediction — Multi-stage Docker build
# ──────────────────────────────────────────────

# ── Builder stage ────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev 2>/dev/null || \
    uv sync --no-install-project --no-dev

# Copy app code and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev 2>/dev/null || \
    uv sync --no-dev

# ── Runtime stage ────────────────────────────
FROM python:3.13-slim

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

# Copy virtual env and app code from builder
COPY --from=builder --chown=app:app /app /app

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
