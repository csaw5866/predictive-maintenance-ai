"""
Makefile for common development tasks
"""

.PHONY: help install test lint format clean docker-build docker-up docker-down run-api run-dashboard run-train

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Lint code"
	@echo "  make format           - Format code with black"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start Docker containers"
	@echo "  make docker-down      - Stop Docker containers"
	@echo "  make run-api          - Run API locally"
	@echo "  make run-dashboard    - Run dashboard locally"
	@echo "  make run-train        - Run training pipeline"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=pma

lint:
	flake8 pma/ api/ dashboard/ pipelines/ tests/
	mypy pma/ --ignore-missing-imports

format:
	black pma/ api/ dashboard/ pipelines/ tests/
	isort pma/ api/ dashboard/ pipelines/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .mypy_cache build dist *.egg-info

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	docker compose logs -f

docker-down:
	docker compose down

run-api:
	python -m uvicorn api.main:app --reload --port 8000

run-dashboard:
	streamlit run dashboard/app.py

run-train:
	python -m pipelines.train
