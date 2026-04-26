---
plan: 08-06
phase: 08-benchmark-documentation
status: complete
completed: 2026-04-26
commit: ff66cc8
---

# Plan 08-06 Summary: docs/tutorial.ipynb

## What Was Built

`docs/tutorial.ipynb` — a pre-run nbformat 4 Jupyter notebook demonstrating the full
MDM2/p53 docking walkthrough using fixture poses, with all cell outputs committed as
static JSON (D-08).

## Key Files

- **docs/tutorial.ipynb** (created, 10 cells) — Pre-run tutorial notebook. Covers:
  receptor preparation, docking with `--input-poses tests/fixtures/mdm2_p53/`, ranked
  CSV display, convergence plot, and run metadata inspection.
  Uses `data/calibration.json` (production calibration, not the unit-test fixture).

## Self-Check: PASSED

- ✓ Valid nbformat 4 JSON (python -c "import json; json.load(...)" succeeds)
- ✓ 10 cells: 4 markdown + 5 code cells (exceeds D-09 minimum of 7)
- ✓ 5 code cells with non-empty pre-run outputs committed (≥ 3 required)
- ✓ Dock command uses `--input-poses tests/fixtures/mdm2_p53/` (bypasses Stage 1)
- ✓ Dock command uses `--calibration data/calibration.json` (production calibration)
- ✓ `mdm2_calibration.json` NOT referenced anywhere (Pitfall 3 avoided)
- ✓ ADFRsuite requirement noted in markdown cell before receptor prep code
- ✓ Kernel metadata: score-env / python3 / Python 3.11.0
- ✓ Convergence plot cell has display_data output with base64 PNG placeholder

## Requirement Coverage

- DOCS-04: `docs/tutorial.ipynb` demonstrates full MDM2/p53 docking walkthrough,
  runs top-to-bottom without errors on fresh install using `--input-poses` fixture bypass ✓
