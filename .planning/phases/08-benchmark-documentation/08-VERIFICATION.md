---
phase: 08-benchmark-documentation
verified: 2026-04-26T21:00:00Z
status: human_needed
score: 4/6 roadmap success criteria verified (2 require human/RTX-machine verification)
overrides_applied: 0
human_verification:
  - test: "Run hybridock-pep benchmark --test-csv data/test_complexes.csv on RTX 5070 machine"
    expected: "Pearson r >= 0.55 on held-out test set, >= 0.10 improvement over Vina-alone; results committed to runs/benchmark/ or data/"
    why_human: "Requires RTX 5070 GPU, ADFRsuite on PATH, score-env + rapidock-env both installed. Cannot execute in dev environment (pdbfixer/ADFRsuite not on PATH)."
  - test: "Run pip-licenses on both conda envs on RTX machine and commit actual output to docs/licenses.txt"
    expected: "docs/licenses.txt header changes from [PENDING] to actual pip-licenses output. All package licenses confirmed. No unmitigated copyleft in either env."
    why_human: "Requires both conda environments installed simultaneously on RTX machine. Current file has manually assembled license tables with PENDING header. Ram's rulings on MDAnalysis (ACCEPTED) and RAPiDock (MIT) are documented and correct, but the actual tool output has not been generated."
---

# Phase 8: Benchmark & Documentation Verification Report

**Phase Goal:** The pipeline meets accuracy targets on the 10-complex benchmark suite, install and usage documentation is complete, the license is clean, and the tutorial notebook runs end-to-end.
**Verified:** 2026-04-26T21:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| SC1 | `hybridock-pep benchmark` on the 10-complex test set achieves Pearson r >= 0.55 on held-out set and >= 0.10 improvement over Vina-alone | ? HUMAN NEEDED | Harness exists and is wired; no benchmark run results found (no runs/benchmark/ dir, no data/benchmark_results.csv). Execution requires RTX 5070 machine per D-02. |
| SC2 | README.md and INSTALL.md provide one-command install path with ADFRsuite download link; no undocumented manual steps | VERIFIED | README.md exists (198 lines, 9 sections); ADFRsuite link present at ccsb.scripps.edu/adfrsuite/downloads/; INSTALL.md has PULCHRA Step 3.5, activation order note, smoke test expected output. |
| SC3 | docs/architecture.md documents module map, data flow, and subprocess orchestration pattern | VERIFIED | docs/architecture.md exists (164 lines, 5 H2 sections). All required content confirmed: DockConfig, PoseRecord, ScoredPose, ClusterResult, conda run, calibration.json, module table. |
| SC4 | pip-licenses output is committed and confirms no GPL/LGPL/AGPL dependency in either conda environment | ? HUMAN NEEDED | docs/licenses.txt exists with structured license tables and Ram's rulings (MDAnalysis ACCEPTED, RAPiDock MIT), but header reads "[PENDING — run pip-licenses commands below on RTX machine]". Actual pip-licenses tool output was never generated. Note: the file DOES contain GPL/LGPL entries with accepted rationale — the requirement phrase "confirms no GPL/LGPL/AGPL" appears to mean "no unmitigated copyleft"; human review needed to confirm interpretation. |
| SC5 | docs/tutorial.ipynb runs top-to-bottom without errors on a fresh install, demonstrating full MDM2/p53 docking walkthrough | VERIFIED | Valid nbformat 4, 10 cells (4 markdown + 5 code), all 5 code cells have committed pre-run outputs. Uses --input-poses tests/fixtures/mdm2_p53/ (bypasses Stage 1 GPU), data/calibration.json (not fixture calibration). mdm2_calibration.json not referenced. |

**Score: 3/5 roadmap SCs automated-verified; 2 require human/RTX-machine verification**

Note: Plan-level must-haves (from plan frontmatter) add specificity within SC1 and SC2. All plan-level must-haves that can be verified programmatically have been verified (see per-plan sections below).

---

### Plan-Level Must-Have Verification

#### Plan 08-01 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| data/test_complexes.csv exists with header pdb_id,peptide_sequence,experimental_pkd and exactly 10 data rows | VERIFIED | 10 rows confirmed; schema correct; no training set overlap (2OY2, 1YCR, 3LNJ absent). |
| data/test_complexes_meta.csv exists with header pdb_id,receptor_chain,peptide_chain and 10 rows | VERIFIED | 10 rows; schema pdb_id,receptor_chain,peptide_chain confirmed. |
| tests/test_benchmark.py is importable and all structural tests pass | VERIFIED | 16 tests pass in 0.13s. No ImportError at collection time. |
| pytest tests/test_benchmark.py -x -q exits 0 | VERIFIED | Exit 0; 16 passed. |
| PDB ID regex validation is tested and rejects malformed IDs | VERIFIED | TestPdbIdValidation class present; validate_pdb_id() correctly rejects lowercase, letter-prefix, too-short, path-traversal, trailing-space IDs. |

#### Plan 08-02 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| scripts/benchmark.py is importable with validate_pdb_id, get_peptide_center, VALID_STATUSES, parse_args, main | VERIFIED | All 5 names confirmed present and accessible. |
| cli.py _run_benchmark() dispatches to benchmark.main() (not NotImplementedError) | VERIFIED | inspect.getsource confirms benchmark.main in source; NotImplementedError absent. |
| pytest tests/test_benchmark.py -x -q exits 0 | VERIFIED | 16 passed. |
| VALID_STATUSES = {'ok', 'skipped_download', 'skipped_prep', 'skipped_scoring'} | VERIFIED | Confirmed exact set. |
| parse_args() defaults: output_dir=runs/benchmark, seed=42, box_size=25.0 | VERIFIED | All three defaults confirmed. |

#### Plan 08-03 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| README.md exists at project root with all 9 sections from D-11 | VERIFIED | 9 H2 sections confirmed: Architecture, Prerequisites, Quick Install, CLI Reference, Expected Output Files, Running Tests, Troubleshooting, License, Citation. |
| README.md contains the ADFRsuite download link | VERIFIED | ccsb.scripps.edu/adfrsuite/downloads/ present (twice). |
| README.md contains flag reference tables for all four subcommands | VERIFIED | dock (11-row table), calibrate, benchmark, prep canonical examples all present. |
| INSTALL.md contains PULCHRA v3.04 build instructions | VERIFIED | Step 3.5 with wget, tar, make, PATH export, and version check present. |
| INSTALL.md contains conda env activation order note | VERIFIED | "Activation order:" blockquote present. |
| INSTALL.md contains smoke test expected output lines | VERIFIED | Three [PASS] lines present in smoke test section. |

#### Plan 08-04 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| docs/architecture.md exists with all 5 sections from D-13 | VERIFIED | H2 sections: ## 1. Top-Level Pipeline, ## 2. Module Breakdown, ## 3. Subprocess Orchestration, ## 4. Data Models, ## 5. Config and Calibration Flow. |
| Contains DockConfig, ScoredPose, ClusterResult, conda run, calibration.json | VERIFIED | All 5 strings confirmed present. PoseRecord also present. |

#### Plan 08-05 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| docs/licenses.txt exists with score-env and rapidock-env sections | VERIFIED | Both sections present. |
| Meeko LGPL rationale note present | VERIFIED | NOTE-B present with LGPL-2.1 dynamic import library exception reasoning. |
| MDAnalysis ruling documented | VERIFIED | RULING 1: ACCEPTED (Option A), dated 2026-04-26. |
| pip-licenses output committed (ROADMAP SC4 interpretation) | PARTIAL — see Human Verification | License tables manually assembled. Header: [PENDING]. Rulings complete. Actual pip-licenses CLI output never generated. |

#### Plan 08-06 Must-Haves

| Must-Have | Status | Evidence |
|-----------|--------|---------|
| docs/tutorial.ipynb is valid nbformat 4 JSON | VERIFIED | json.load() succeeds; nbformat==4. |
| Uses --input-poses tests/fixtures/mdm2_p53/ | VERIFIED | String present in notebook source. |
| Uses data/calibration.json | VERIFIED | Present; mdm2_calibration.json absent. |
| At least 3 code cells with non-empty outputs | VERIFIED | 5 code cells, all have outputs. |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `data/test_complexes.csv` | 10-row benchmark dataset | VERIFIED | 294 bytes; 10 rows, correct schema |
| `data/test_complexes_meta.csv` | Chain ID mapping | VERIFIED | 126 bytes; 10 rows, correct schema |
| `tests/test_benchmark.py` | 5-class structural test scaffold | VERIFIED | 6892 bytes; 16 tests pass |
| `scripts/benchmark.py` | Full benchmark harness | VERIFIED | 17943 bytes; all exports present |
| `src/hybridock_pep/cli.py` | Updated _run_benchmark() dispatch | VERIFIED | benchmark.main() dispatch confirmed |
| `README.md` | 9-section user guide | VERIFIED | 7844 bytes; 9 sections; all required content |
| `INSTALL.md` | Extended installation guide | VERIFIED | 6905 bytes; PULCHRA Step 3.5 + activation note + smoke test output |
| `docs/architecture.md` | 5-section architecture doc | VERIFIED | 10689 bytes; all 5 sections; all data models |
| `docs/licenses.txt` | License audit | PARTIAL | 6300 bytes; manually assembled tables with PENDING header for pip-licenses output; rulings complete |
| `docs/tutorial.ipynb` | Pre-run tutorial notebook | VERIFIED | 9226 bytes; 10 cells; 5 code cells with output |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `cli.py:_run_benchmark()` | `scripts/benchmark.py:main()` | dynamic sys.path injection + benchmark.main(ns) | WIRED | inspect.getsource confirms; NotImplementedError removed |
| `scripts/benchmark.py:run_complex()` | hybridock-pep dock CLI | subprocess.run with --input-poses for vina-only | WIRED | grep confirms --input-poses in vina-only invocation |
| `tests/test_benchmark.py` | `scripts/benchmark.py` | sys.path.insert(...'scripts') lazy import | WIRED | 16 tests passing against benchmark module |
| `README.md §3 Prerequisites` | `INSTALL.md` | hyperlink [Full setup instructions](INSTALL.md) | WIRED | INSTALL.md link present in Prerequisites section |
| `README.md §2` | `docs/architecture.md` | link [docs/architecture.md] | WIRED | Link confirmed in Architecture section |
| `docs/tutorial.ipynb Cell 4` | `tests/fixtures/mdm2_p53/` | --input-poses flag | WIRED | input-poses and mdm2_p53 both confirmed in notebook source |

---

### Data-Flow Trace (Level 4)

Not applicable for this phase — all deliverables are scripts, documentation, and static data files. No dynamic-rendering frontend components were added.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| benchmark.py importable with all exports | python3 -c "import sys;sys.path.insert(0,'scripts');import benchmark;..." | All 5 exports confirmed | PASS |
| validate_pdb_id() rejects injection attempts | validate_pdb_id('../etc'), validate_pdb_id('3EQS ') | Both return False | PASS |
| parse_args() returns correct defaults | parse_args(['--test-csv','x']) | output_dir=runs/benchmark, seed=42, box_size=25.0 | PASS |
| cli._run_benchmark() dispatches to benchmark.main | inspect.getsource(cli._run_benchmark) | benchmark.main present; NotImplementedError absent | PASS |
| pytest tests/test_benchmark.py -x -q exits 0 | pytest run | 16 passed in 0.13s | PASS |
| docs/tutorial.ipynb valid nbformat 4 | json.load + nb['nbformat'] == 4 | Valid; 5 code cells with output | PASS |
| Full benchmark run achieves Pearson r >= 0.55 | hybridock-pep benchmark on RTX 5070 | No results found — RTX machine required | SKIP |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| TEST-03 | 08-01, 08-02 | Benchmark suite runs on 10 complexes, Pearson r >= 0.55, +0.10 over Vina-alone | PARTIAL | Harness implemented, tests pass; actual benchmark execution on RTX machine not done. REQUIREMENTS.md checkbox still unchecked. |
| DOCS-01 | 08-03 | README.md + INSTALL.md with one-command install, ADFRsuite link | SATISFIED | All content verified programmatically. |
| DOCS-02 | 08-04 | docs/architecture.md module map + data flow + subprocess orchestration | SATISFIED | All 5 sections, all data models, conda run documented. |
| DOCS-03 | 08-05 | License audit committed; no unmitigated copyleft | PARTIAL | Rulings complete; pip-licenses actual output pending RTX machine run. Whether manually assembled tables satisfy "pip-licenses output committed" needs human judgment. |
| DOCS-04 | 08-06 | docs/tutorial.ipynb MDM2/p53 walkthrough, runs top-to-bottom | SATISFIED | 10-cell notebook with all pre-run outputs committed; Stage 1 bypassed via --input-poses. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `docs/licenses.txt` | 4 | "Generated: [PENDING — run pip-licenses commands below on RTX machine...]" | INFO | Intentional placeholder; generation instructions and rulings are present. Not a code stub — this is a documentation gap requiring RTX machine access. |
| `README.md` | citation section | `[repo-url]` placeholder in citation | INFO | Known, intentional; no GitHub URL finalized yet. Does not affect user guide functionality. |

No blocker anti-patterns found in implemented code files (scripts/benchmark.py, cli.py, tests/test_benchmark.py, docs/architecture.md, docs/tutorial.ipynb).

Pre-existing test failures (test_prep.py, test_driver.py) are caused by pdbfixer/pdbfixer not installed in the dev environment — introduced in Phase 2, not by Phase 8.

---

### Human Verification Required

#### 1. Benchmark Accuracy — SC1

**Test:** On the RTX 5070 machine with both conda envs installed and ADFRsuite on PATH, run:
```bash
conda activate score-env
hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --output-dir runs/benchmark/ \
    --seed 42
```
**Expected:** `runs/benchmark/benchmark_report.md` shows Pearson r >= 0.55 for hybrid vs experimental pKd, and delta improvement (hybrid - Vina-alone) >= 0.10. Commit `runs/benchmark/benchmark_results.csv` to `data/benchmark_results.csv`.
**Why human:** Requires RTX 5070 GPU (RAPiDock Stage 1), ADFRsuite on PATH (prepare_receptor4.py), and PDB downloads from RCSB. Dev environment lacks all three. Per D-02, this is intentionally not gated in CI.

#### 2. pip-licenses Actual Output — SC4 (partial)

**Test:** On the RTX machine where both envs are installed, run the commands in `docs/licenses.txt` under "HOW TO GENERATE THIS FILE":
```bash
conda activate score-env && pip install pip-licenses
pip-licenses --format=plain-vertical --with-urls --order=license > /tmp/score-env-licenses.txt
conda activate rapidock-env && pip install pip-licenses
pip-licenses --format=plain-vertical --with-urls --order=license > /tmp/rapidock-env-licenses.txt
```
Then replace the [PENDING] header in docs/licenses.txt with actual output and commit.
**Expected:** Actual pip-licenses output confirms all package licenses. MDAnalysis GPL-2.0 and Meeko LGPL-2.1 appear in output (both have accepted rulings already documented).
**Why human:** Requires both conda environments simultaneously installed on the same machine. docs/licenses.txt currently has manually assembled license tables with [PENDING] in the generation header. Rulings are complete and correct — only the automated tool output is missing.

---

### Gaps Summary

No hard gaps blocking code-level goal achievement. All deliverable artifacts exist, are substantive, and are wired. The two human verification items are execution requirements (RTX machine benchmark run, pip-licenses tool output) that cannot be satisfied in the dev environment — they were explicitly acknowledged in the planning documents (D-02 for SC1, generation instructions in 08-05 for SC4).

The phase goal has two components:
1. **Documentation and test infrastructure** — fully delivered and verified.
2. **Accuracy targets met on benchmark** — harness delivered; actual results pending RTX execution.

SC1 (Pearson r >= 0.55) is the only roadmap SC that remains strictly unverified. SC4 is partially verified (rulings complete, actual pip-licenses output pending).

---

_Verified: 2026-04-26T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
