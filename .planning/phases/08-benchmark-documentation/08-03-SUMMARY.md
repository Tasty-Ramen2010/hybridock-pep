---
phase: 08-benchmark-documentation
plan: "03"
subsystem: documentation
tags: [readme, install, docs, user-guide, cli-reference]
dependency_graph:
  requires: []
  provides: [README.md, INSTALL.md-D12-additions]
  affects: [user-onboarding, iGEM-wiki-rubric]
tech_stack:
  added: []
  patterns: [INSTALL.md-style headings, blockquote callouts, flag-reference tables]
key_files:
  created:
    - README.md
  modified:
    - INSTALL.md
decisions:
  - "README.md uses INSTALL.md style conventions (# Title — Subtitle H1, --- rules, > **Note:** blockquotes)"
  - "PULCHRA Step 3.5 inserted between Step 3 (ADFRsuite) and Step 4 (PyRosetta) to preserve logical install flow"
  - "Activation order note added after existing rapidock-env subprocess explanation in Step 2 (not as a new step)"
  - "Smoke test expected output placed inline after the [PASS] lines reference that already existed in Step 5"
metrics:
  duration: "2 min 25 sec"
  completed: "2026-04-26"
  tasks_completed: 2
  files_changed: 2
---

# Phase 8 Plan 03: README + INSTALL Documentation Summary

README.md (9-section user guide, D-11) created from scratch; INSTALL.md extended with PULCHRA v3.04 build step, conda activation order note, and smoke test expected output (D-12).

---

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write README.md (9-section comprehensive user guide) | dff57bb | README.md (created, 198 lines) |
| 2 | Extend INSTALL.md with three targeted additions (D-12) | 1f3d4f7 | INSTALL.md (+42 lines) |

---

## What Was Built

**README.md** — new file at project root, 198 lines, 9 sections:
1. Project overview (HybriDock-Pep description, iGEM 2026 context, target application)
2. Architecture (two-stage pipeline summary, subprocess boundary, link to docs/architecture.md)
3. Prerequisites (GPU/conda/ADFRsuite/PULCHRA v3.04, disk space, link to INSTALL.md)
4. Quick Install (three-step conda setup, macOS ARM note)
5. CLI Reference (dock flag table with 11 rows, calibrate/benchmark/prep canonical examples)
6. Expected Output Files (8-file table: ranked_poses.csv, best_pose.pdb, cluster_summary.csv, etc.)
7. Running Tests (pytest, -m slow, --cov, MDM2/p53 integration baseline note)
8. Troubleshooting (6-row table: pdbfixer, CUDA 12.0, ADFRsuite PATH, PULCHRA version, Vina import, macOS)
9. License + Citation (MIT, Meeko/Vina/ADFRsuite notices, RAPiDock DOI)

**INSTALL.md** — three targeted additions, no existing content modified:
- Step 3.5: PULCHRA v3.04 build-from-source with version check callout (CLAUDE.md §2.3 reference)
- Step 2: activation order blockquote (score-env for normal use; rapidock-env auto-invoked by driver)
- Step 5: three expected `[PASS]` lines inline with existing smoke test section

---

## Deviations from Plan

None — plan executed exactly as written.

---

## Known Stubs

None — README.md and INSTALL.md contain no placeholder text that blocks their purpose. The
`[repo-url]` in the citation section is a known placeholder that will be filled when the
GitHub repo URL is finalised; it does not block the user guide goal.

---

## Threat Flags

None — documentation-only changes, no new executable surface.

---

## Self-Check: PASSED

Files exist:
- README.md: FOUND
- INSTALL.md: FOUND (modified)
- .planning/phases/08-benchmark-documentation/08-03-SUMMARY.md: FOUND

Commits exist:
- dff57bb: FOUND (README.md)
- 1f3d4f7: FOUND (INSTALL.md)
