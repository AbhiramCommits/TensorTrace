PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip
PYTHON_BIN ?= python3.11

.PHONY: setup lint test profile report clean

setup:
	$(PYTHON_BIN) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q --cov=tensortrace --cov-report=term-missing

profile:
	$(PYTHON) scripts/bench_sweep.py

report:
	@for f in reports/run-*.json; do $(PYTHON) -m tensortrace.cli report "$$f"; done

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .onnx-cache build dist *.egg-info
