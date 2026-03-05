.PHONY: help install dev test lint format build clean publish

help:
	@echo "Targets:"
	@echo "  install  - instala em modo editável"
	@echo "  dev      - instala com deps de dev"
	@echo "  test     - roda pytest"
	@echo "  lint     - roda ruff + black --check"
	@echo "  format   - aplica black + ruff --fix"
	@echo "  build    - gera dist/"
	@echo "  clean    - remove builds"
	@echo "  publish  - publica no PyPI (precisa de credenciais)"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check . --fix

build:
	python -m build
	twine check dist/*

clean:
	rm -rf dist build .pytest_cache .ruff_cache **/*.egg-info

publish: build
	twine upload dist/*