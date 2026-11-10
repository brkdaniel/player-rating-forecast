PYTHON ?= python
DATA := data/raw/fifa_world_cup_2026_player_performance.csv
TRIALS ?= 30

.PHONY: setup test test-all lint format pipeline pipeline-fast figures notebooks all

setup:  ## CPU-only PyTorch first, then the pinned dependencies and this package
	$(PYTHON) -m pip install -r requirements-torch-cpu.txt
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

test:  ## fast tests that need no data (the same set CI runs)
	$(PYTHON) -m pytest -m "not data"

test-all: $(DATA)  ## all tests, including leakage checks on the real data
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

pipeline: $(DATA)  ## full run: features, ceiling, tuning, final models, intervals, figures
	$(PYTHON) -m player_rating.pipeline --trials $(TRIALS)

pipeline-fast: $(DATA)  ## quick smoke run with a tiny tuning budget
	$(PYTHON) -m player_rating.pipeline --trials 3 --seeds 1 --n-boot 200

figures:  ## redraw the report figures from reports/metrics (no retraining)
	$(PYTHON) -m player_rating.plots

notebooks: $(DATA)
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

all: test-all pipeline notebooks

$(DATA):
	@echo "Missing $(DATA). See data/README.md for how to download it." && exit 1
