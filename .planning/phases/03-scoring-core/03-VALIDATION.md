---
phase: 3
slug: scoring-core
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_scoring.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_scoring.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SCORE-01 | T-03-01 | Vina score_only via Python API, no subprocess | unit | `python -m pytest tests/test_scoring.py::TestVinaScorer -x -q` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 2 | SCORE-02 | T-03-02 | AD4 load_maps (not set_receptor), anomaly flag | unit | `python -m pytest tests/test_scoring.py::TestAD4Scorer -x -q` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | SCORE-03 | T-03-03 | α/β range validation, abort outside bounds | unit | `python -m pytest tests/test_scoring.py::TestEntropy -x -q` | ❌ W0 | ⬜ pending |
| 03-04-01 | 04 | 3 | SCORE-01/02/03 | — | All scoring tests pass, coverage ≥ 70% | integration | `python -m pytest tests/test_scoring.py -x -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_scoring.py` — stubs for SCORE-01, SCORE-02, SCORE-03
- [ ] `data/calibration.json` — default calibration file (α≈0.65, β≈0.22)

*Existing infrastructure (pytest, fixtures) covers the rest.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vina Python API returns correct kcal/mol on real PDBQT | SCORE-01 | Requires ADFRsuite + real receptor/ligand in score-env | Run `hybridock-pep prep` on 1CZB then score one pose |
| AD4 load_maps produces non-None score | SCORE-02 | Requires autogrid4 maps in score-env | Run prep + grids, then score one pose via AD4 path |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
