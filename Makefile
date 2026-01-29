.PHONY: lint typecheck test demo clean install

install:
	poetry install

lint:
	ruff check .

typecheck:
	mypy core data features execution risk

test:
	pytest tests/ -v

demo:
	python scripts/run_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
