.PHONY: help install install-dev test run clean format lint type-check docs

help:
	@echo "LLM Business Insight Lab - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black"
	@echo "  make lint             Lint code with ruff"
	@echo "  make type-check       Type check with mypy"
	@echo "  make test             Run pytest suite"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-symcot       Run SymCoT experiment"
	@echo "  make run-nocot        Run NoCoT experiment"
	@echo "  make run-demo         Run demo with small sample"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Clean cache/build files"
	@echo "  make logs             Tail latest logs"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src

format:
	black src/ tests/ runner.py

lint:
	ruff check src/ tests/ runner.py

type-check:
	mypy src/ --ignore-missing-imports

run-symcot:
	python runner.py --model gpt-4 --strategy symcot --n_samples 80

run-nocot:
	python runner.py --model gpt-4 --strategy nocot --n_samples 80

run-demo:
	python runner.py --model gpt-4 --strategy symcot --n_samples 5

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

logs:
	tail -f logs/experiment_*.log 2>/dev/null || echo "No logs found yet. Run an experiment first."
