.PHONY: test demo compile lint

PYTHON ?= python3

test:
	PYTHONPATH=src pytest

demo:
	PYTHONPATH=src $(PYTHON) -m inverse_shape.cli demo --out artifacts/demo --samples 160

compile:
	$(PYTHON) -m compileall -q src tests examples/reconstruction

lint:
	ruff check src tests examples/reconstruction
