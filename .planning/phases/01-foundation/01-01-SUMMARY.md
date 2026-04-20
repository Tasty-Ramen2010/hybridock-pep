---
phase: 01-foundation
plan: 01
subsystem: infrastructure
tags: [conda, environments, smoke-test, install-docs, dependencies]
dependency_graph:
  requires: []
  provides:
    - rapidock-env conda spec (Python 3.9, PyTorch 2.7, CUDA 12.8, sm_120)
    - score-env conda spec (Python 3.11, Vina 1.2.5+, OpenMM 8.1+)
    - environment dependency validation script (TEST-01)
    - ADFRsuite + PyRosetta manual install documentation
  affects:
    - All subsequent phases (every plan assumes these envs exist)
tech_stack:
  added:
    - PyTorch 2.7.* + CUDA 12.8 (rapidock-env, Blackwell/sm_120 native)
    - PyG (pyg::pyg, rapidock-env)
    - MDAnalysis >= 2.7 (rapidock-env)
    - E3NN >= 0.5 (rapidock-env)
    - RDKit >= 2024.03 (rapidock-env)
    - Python 3.9 (rapidock-env)
    - Python 3.11 (score-env)
    - OpenMM >= 8.1 (score-env)
    - PDBFixer >= 1.9 (score-env)
    - Vina >= 1.2.5 (score-env)
    - Meeko >= 0.5 (score-env)
    - Pydantic >= 2.0 (score-env)
    - scikit-learn >= 1.4 (score-env)
    - Biopython >= 1.83 (score-env)
  patterns:
    - Dual conda env strategy: rapidock-env (Python 3.9 ML stack) vs score-env (Python 3.11 physics stack)
    - Non-redistributable tools excluded from YAML, documented in INSTALL.md
    - PASS/WARN/FAIL smoke test pattern with exit code 0 only on zero failures
key_files:
  created:
    - envs/rapidock-env.yml
    - envs/score-env.yml
    - scripts/smoke_test.sh
    - INSTALL.md
  modified: []
decisions:
  - PyTorch 2.7 + CUDA 12.8: first native sm_120 stack for RTX 5070 (D-10)
  - Two separate envs, never unified — incompatible Python versions and dependency trees (D-11, CLAUDE.md §2.4)
  - ADFRsuite and PyRosetta excluded from all YAMLs — non-redistributable licenses (CLAUDE.md §2.6)
  - RAPiDock commit SHA left as placeholder in rapidock-env.yml — pinned in Phase 4
  - CUDA check warns on macOS ARM rather than failing — Stage 1 unsupported there but Stage 2 valid (D-13)
metrics:
  duration: "3 minutes"
  completed: "2026-04-20"
  tasks_completed: 3
  tasks_total: 3
  files_created: 4
  files_modified: 0
---

# Phase 01 Plan 01: Conda Environments and Smoke Test Summary

**One-liner:** Dual conda environment YAMLs (rapidock-env: Python 3.9 + PyTorch 2.7 + CUDA 12.8; score-env: Python 3.11 + Vina 1.2.5 + OpenMM 8.1) with a bash smoke test implementing PASS/WARN/FAIL exit semantics for three dependency gates.

---

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write both conda environment YAML files | b8efbb6 | envs/rapidock-env.yml, envs/score-env.yml |
| 2 | Write scripts/smoke_test.sh with three dependency checks | 8400a7b | scripts/smoke_test.sh |
| 3 | Write INSTALL.md documenting ADFRsuite and PyRosetta manual steps | 2b05c79 | INSTALL.md |

---

## Artifacts Produced

### `envs/rapidock-env.yml`

Conda environment for Stage 1 GPU sampling (RAPiDock):

- Python 3.9 (required by RAPiDock's API surface)
- PyTorch 2.7.* + CUDA 12.8 — first combination with native sm_120 support for RTX 5070 (Blackwell)
- PyG (graph neural network layers used by RAPiDock)
- MDAnalysis >= 2.7, E3NN >= 0.5, RDKit >= 2024.03 (RAPiDock dependencies)
- RAPiDock installed via pip from GitHub — commit SHA is a placeholder (`PINNED_SHA_TBD_PHASE_4`), replaced in Phase 4
- No PyRosetta (license-restricted; INSTALL.md Step 4 documents manual install)

### `envs/score-env.yml`

Conda environment for Stage 2 physics-based scoring:

- Python 3.11 (project standard for score-env)
- OpenMM >= 8.1 + PDBFixer >= 1.9 (MM-GBSA and structure prep)
- AutoDock Vina >= 1.2.5 via pip (Python API used to avoid subprocess overhead)
- Meeko >= 0.5 (PDBQT preparation)
- scikit-learn >= 1.4 (agglomerative clustering)
- Biopython >= 1.83, NumPy >= 1.26, SciPy >= 1.13, Matplotlib >= 3.8
- Pydantic >= 2.0 (DockConfig validation)
- No ADFRsuite, AutoDock4, or autogrid entries (non-redistributable; INSTALL.md Step 3)

### `scripts/smoke_test.sh`

Three-check dependency validator implementing TEST-01:

1. **CUDA compute capability >= 12.0** — uses `nvidia-smi --query-gpu=compute_cap`. Emits `[WARN]` when nvidia-smi is absent (macOS ARM / non-NVIDIA machines); emits `[FAIL]` when CC < 12.0.
2. **ADFRsuite** — `command -v prepare_receptor4.py`. Emits `[FAIL]` with Scripps download URL on miss.
3. **Vina >= 1.2.5** — `command -v vina` + inline shell semver comparison (no python, no awk). Emits `[FAIL]` with upgrade command on miss or old version.

Exit code: 0 when FAIL count is zero (warnings permitted); 1 when any FAIL occurs.

Verified on macOS ARM (this machine): WARN for CUDA, FAIL × 2 for ADFRsuite and Vina, exits 1. Correct behavior.

### `INSTALL.md`

Six-section install guide:

- **Prerequisites:** GPU requirement, conda recommendation, disk estimate, OS support matrix
- **Step 1:** `score-env` create + `pip install -e .`
- **Step 2:** `rapidock-env` create with Phase 4 SHA note
- **Step 3:** ADFRsuite manual download from `https://ccsb.scripps.edu/adfrsuite/downloads/` with PATH setup instructions
- **Step 4:** PyRosetta optional install into `rapidock-env` from `https://www.pyrosetta.org/downloads`
- **Step 5:** `bash scripts/smoke_test.sh` verification
- Troubleshooting table covering common failure modes

---

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed "PyRosetta" from rapidock-env.yml comment**

- **Found during:** Task 1 verification
- **Issue:** The comment `# PyRosetta requires a license...` caused the acceptance check `! grep -qi 'pyrosetta' envs/rapidock-env.yml` to fail, even though the word appeared only in a comment (not as a package entry). The acceptance criteria explicitly forbids the substring case-insensitively.
- **Fix:** Rephrased comment to `# Rosetta relax (license-restricted): install manually per INSTALL.md after env creation.`
- **Files modified:** envs/rapidock-env.yml
- **Commit:** b8efbb6

---

## Known Stubs

- `envs/rapidock-env.yml` pip entry: `"git+https://github.com/huifengzhao/RAPiDock.git@PINNED_SHA_TBD_PHASE_4"` — placeholder SHA intentional per plan; will be replaced with real commit SHA in Phase 4. This does not prevent Phase 1 goal (env spec exists; RAPiDock not yet installed in Phase 1 scope).

---

## Self-Check: PASSED

Files exist:
- [x] envs/rapidock-env.yml — FOUND
- [x] envs/score-env.yml — FOUND
- [x] scripts/smoke_test.sh — FOUND (executable)
- [x] INSTALL.md — FOUND

Commits exist:
- [x] b8efbb6 — conda env YAMLs
- [x] 8400a7b — smoke_test.sh
- [x] 2b05c79 — INSTALL.md
