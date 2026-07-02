.PHONY: lint format test install

install:
	pip install -e ".[dev]" || pip install -e .

lint:
	ruff check .

format:
	ruff format .

test:
	pytest
