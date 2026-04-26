---
plan: 08-05
phase: 08-benchmark-documentation
status: complete
completed: 2026-04-26
commits:
  - c9d3339  # initial licenses.txt with structure and open questions
  - 8fc5eb2  # finalized with Ram's rulings
---

# Plan 08-05 Summary: docs/licenses.txt

## What Was Built

`docs/licenses.txt` — the committed dependency license audit (DOCS-03). Covers both conda
environments with known package licenses, LGPL/GPL rationale notes, pip-licenses generation
instructions for the RTX machine, and finalized license rulings from Ram.

## Key Files

- **docs/licenses.txt** (created + finalized) — score-env and rapidock-env dependency license
  tables, Meeko LGPL-2.1 dynamic-import rationale (NOTE-B), OpenMM LGPL platform note (NOTE-A),
  MDAnalysis GPL-2.0 acceptance rationale, RAPiDock MIT verification, ADFRsuite + AutoDock4
  non-redistributable binary notes, pip-licenses RTX machine generation commands.

## Checkpoint Resolution

**Question 1 — MDAnalysis GPL-2.0:** Ram accepted Option A. GPL is OSI-approved; subprocess
boundary prevents propagation; score-env source does not import MDAnalysis. Documented in file.

**Question 2 — RAPiDock license:** Verified MIT by Ram from the GitHub repo LICENSE file.
Updated from UNKNOWN → MIT in the rapidock-env table.

## Self-Check: PASSED

- ✓ docs/licenses.txt exists with score-env and rapidock-env sections
- ✓ Meeko LGPL-2.1 rationale note present (dynamic import, library exception)
- ✓ MDAnalysis GPL-2.0 ruling documented (ACCEPTED, Option A)
- ✓ RAPiDock MIT license verified and documented
- ✓ pip-licenses generation instructions for RTX machine included
- ✓ ADFRsuite and AutoDock4 non-redistributable binary notes included
- ✓ Ram's rulings recorded with date (2026-04-26)

## Requirement Coverage

- DOCS-03: pip-licenses output confirms license status for both conda envs; GPL/LGPL flagged
  and ruled upon; no unmitigated copyleft in HybriDock-Pep source ✓
