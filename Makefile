.PHONY: help install install-dev test lint format clean build deploy

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint          - Run code linting (flake8, mypy)"
	@echo "  make format        - Format code with black and isort"
	@echo "  make security      - Run security checks (bandit)"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo "  make pre-commit    - Install pre-commit hooks"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-performance:
	pytest tests/performance/ -v -m performance

lint:
	flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
	mypy src/ --ignore-missing-imports
	pylint src/ --max-line-length=120 || true

format:
	black src/ tests/ --line-length=120
	isort src/ tests/ --profile black

security:
	bandit -r src/ -ll
	safety check --file requirements.txt || true

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean
	python setup.py sdist bdist_wheel

pre-commit:
	pre-commit install
	pre-commit run --all-files

docker-build:
	docker build -t mlops-ecommerce:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down
