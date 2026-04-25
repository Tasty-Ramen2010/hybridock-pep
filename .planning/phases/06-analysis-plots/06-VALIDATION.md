---
phase: 6
slug: analysis-plots
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-25
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/test_clustering.py -x` |
| **Full suite command** | `pytest --cov=hybridock_pep` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_clustering.py -x`
- **After every plan wave:** Run `pytest --cov=hybridock_pep`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 6-W0-01 | 01 | 0 | ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05 | — | N/A | unit (RED gate) | `pytest tests/test_clustering.py -x` | ❌ W0 | ⬜ pending |
| 6-01-01 | 01 | 1 | ANAL-01 | — | N/A | unit | `pytest tests/test_clustering.py::test_contact_zone_indices -x` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | ANAL-01 | — | N/A | unit | `pytest tests/test_clustering.py::test_contact_zone_fallback -x` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | ANAL-01 | — | N/A | unit | `pytest tests/test_clustering.py::test_rmsd_matrix_symmetry -x` | ❌ W0 | ⬜ pending |
| 6-01-04 | 01 | 1 | ANAL-01 | — | N/A | unit | `pytest tests/test_clustering.py::test_cluster_poses_assigns_ids -x` | ❌ W0 | ⬜ pending |
| 6-01-05 | 01 | 1 | ANAL-01 | — | N/A | unit | `pytest tests/test_clustering.py::test_silhouette_k_selection -x` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | ANAL-02 | — | N/A | unit | `pytest tests/test_clustering.py::test_cluster_summary_csv -x` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 | 1 | ANAL-02 | — | N/A | unit | `pytest tests/test_clustering.py::test_ci95 -x` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 1 | ANAL-03, OUT-04 | — | N/A | unit | `pytest tests/test_clustering.py::test_convergence_plot_written -x` | ❌ W0 | ⬜ pending |
| 6-03-02 | 03 | 1 | OUT-05 | — | N/A | unit | `pytest tests/test_clustering.py::test_silhouette_plot_written -x` | ❌ W0 | ⬜ pending |
| 6-04-01 | 04 | 2 | ANAL-01..03, OUT-04, OUT-05 | — | N/A | integration | `pytest tests/test_clustering.py -x && pytest --cov=hybridock_pep` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_clustering.py` — RED-gate stubs for ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05
- [ ] Fixture: 10×10 precomputed RMSD distance matrix with two clear clusters (within=0.1Å, between=5.0Å)
- [ ] Fixture: list of 10 `ScoredPose` objects with `ca_coords` (shape [5,3]) and `hybrid_score` populated

*Existing pytest infrastructure (pyproject.toml, conftest.py) confirmed from prior phases.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| convergence_plot.png visually shows running mean ± σ stabilizing | ANAL-03 | Visual correctness of plot content | Open output PNG; verify mean line and shaded σ band are present and trend toward stability |
| silhouette_plot.png shows bar chart with k_optimal annotated | OUT-05 | Visual correctness of plot content | Open output PNG; verify bars across k range and vertical marker at selected k |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
