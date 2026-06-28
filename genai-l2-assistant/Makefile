.PHONY: dev test lint format clean docker-up docker-down migrate worker

# ============================================================================
# Development
# ============================================================================

dev:  ## Start FastAPI dev server with hot reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:  ## Start Celery worker
	celery -A app.workers.celery_app worker --loglevel=info

beat:  ## Start Celery beat scheduler
	celery -A app.workers.celery_app beat --loglevel=info

# ============================================================================
# Docker
# ============================================================================

docker-up:  ## Start all services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop all Docker Compose services
	docker-compose down

docker-build:  ## Build Docker images
	docker-compose build

docker-logs:  ## Follow Docker Compose logs
	docker-compose logs -f

# ============================================================================
# Database
# ============================================================================

migrate:  ## Run Alembic migrations
	alembic upgrade head

migrate-new:  ## Create a new Alembic migration (usage: make migrate-new MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down:  ## Rollback last migration
	alembic downgrade -1

# ============================================================================
# Testing
# ============================================================================

test:  ## Run all unit tests
	pytest tests/unit/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/unit/ -v --cov=app --cov-report=html --cov-report=term

test-integration:  ## Run integration tests (requires running services)
	pytest tests/integration/ -v --tb=short

eval-retrieval:  ## Run retrieval evaluation
	python scripts/eval_run.py --eval-type retrieval --output results.json

eval-generation:  ## Run generation evaluation
	python scripts/eval_run.py --eval-type generation --output results.json

# ============================================================================
# Code Quality
# ============================================================================

lint:  ## Run linter
	ruff check app/ tests/

format:  ## Format code
	ruff format app/ tests/

typecheck:  ## Run type checker
	mypy app/

# ============================================================================
# Data Pipeline
# ============================================================================

bootstrap:  ## Run initial historical ticket indexing
	python scripts/bootstrap_index.py

seed:  ## Seed test data for local development
	python scripts/seed_test_data.py

# ============================================================================
# Utilities
# ============================================================================

clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
