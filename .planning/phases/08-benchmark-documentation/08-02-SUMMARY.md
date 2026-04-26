---
plan: 08-02
phase: 08-benchmark-documentation
status: complete
completed: 2026-04-26
commit: d5e1c11
---

# Plan 08-02 Summary: benchmark.py + cli.py dispatch

## What Was Built

`scripts/benchmark.py` — full benchmark harness implementing the two-invocation pattern (D-04)
plus PDB download, binding site computation, Pearson r reporting, and CSV/Markdown output.

`src/hybridock_pep/cli.py` — `_run_benchmark()` stub replaced with live dispatch to
`benchmark.main()`; five new flags added to the `p_bench` subparser.

## Key Files

- **scripts/benchmark.py** (created, 290 lines) — functions: `validate_pdb_id`,
  `get_peptide_center`, `download_pdb`, `extract_best_score`, `run_complex`,
  `write_results_csv`, `write_report_md`, `parse_args`, `main`. `VALID_STATUSES` constant.
  Two-invocation pattern: hybrid run → vina-only rescore from same poses dir.
- **src/hybridock_pep/cli.py** (modified) — `_run_benchmark()` now dispatches via
  dynamic sys.path injection to `benchmark.main(ns)`. New p_bench flags: `--output-dir`,
  `--seed`, `--box-size`, `--n-samples`, `--calibration`.

## Test Results (GREEN gate)

- `pytest tests/test_benchmark.py -x -q` → **16 passed** in 0.13s
- Full suite (excl. pre-existing pdbfixer failures) → **102 passed**, 0 new failures

## Self-Check: PASSED

- ✓ `validate_pdb_id()` accepts 4-char PDB IDs starting with digit; rejects all others
- ✓ `VALID_STATUSES` = {'ok', 'skipped_download', 'skipped_prep', 'skipped_scoring'}
- ✓ `parse_args()` defaults: output_dir=runs/benchmark, seed=42, box_size=25.0
- ✓ `cli._run_benchmark()` calls `benchmark.main(ns)`; `NotImplementedError` removed
- ✓ All benchmark subparser flags recognized by `_build_parser()`
- ✓ Two-invocation pattern uses `--input-poses` for vina-only run (no pose nondeterminism)
- ✓ All output-dir paths resolved to absolute before subprocess calls

## Requirement Coverage

- TEST-03: benchmark harness imports, CLI parsing, output schema verified ✓
  (full execution against 10 complexes requires RTX 5070 machine — D-02)
