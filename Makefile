.PHONY: build run dev stop logs clean

# ── Docker ───────────────────────────────────
build:
	docker compose build

run: build
	docker compose up -d
	@echo "🚀 App running at http://localhost:8000"

stop:
	docker compose down

logs:
	docker compose logs -f predictor

# ── Local development ────────────────────────
dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

install:
	uv sync

# ── Utilities ────────────────────────────────
clean:
	docker compose down --rmi local --volumes
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache

test:
	uv run pytest tests/ -v
