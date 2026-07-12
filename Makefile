# HybriDock-Pep — one-command entry points. `make help` lists targets.
# Judge-facing path: make install && make verify.

.DEFAULT_GOAL := help
PY ?= python
OMP ?= OMP_NUM_THREADS=1

.PHONY: help install install-dev test verify reproduce demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Editable install of the scoring package (GPU sampling env: see INSTALL.md)
	$(PY) -m pip install -e .

install-dev: ## Install with dev extras (pytest, ruff, mypy)
	$(PY) -m pip install -e ".[dev]"

test: ## Full unit-test suite
	$(OMP) $(PY) -m pytest -q

verify: ## Fast offline proof the core math holds (no external data / GPU needed)
	$(OMP) $(PY) -m pytest -q \
	  tests/test_double_difference.py \
	  tests/test_anchoring.py \
	  tests/test_selectivity.py

reproduce: ## Print how to regenerate headline benchmarks (needs PDBbind/PPIKB — see RESULTS.md)
	@echo "Headline numbers + exact commands: RESULTS.md"
	@echo "Large inputs (PDBbind v2020, PPIKB) are gitignored — see INSTALL.md."
	@echo "Then, from the experiments/ directory:"
	@echo "  cd experiments && $(OMP) $(PY) e331_ours_vs_ppiclone_clustered.py  # 1.35 vs 1.46 (n=865)"
	@echo "  cd experiments && $(OMP) $(PY) e330_ours_pdbbind.py               # full-set MAE 1.40"
	@echo "  cd experiments && $(OMP) $(PY) e366_identity_threshold_trend.py    # full identity sweep"

demo: ## End-to-end dock on a shipped example receptor (MDM2/p53)
	hybridock-pep dock --peptide ETFSDLWKLLPE \
	  --receptor data/pdbs/1YCR_mdm2.pdb --site 25.20 -25.61 -7.97 --box 30 \
	  --n-samples 20 --output-dir runs/demo

clean: ## Remove run outputs and caches
	rm -rf runs/* .pytest_cache **/__pycache__ 2>/dev/null || true
