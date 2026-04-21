---
phase: 03-scoring-core
plan: "02"
subsystem: scoring
tags: [ad4, autodock4, load_maps, batch-scoring, anomaly-detection, tdd]
dependency_graph:
  requires:
    - 03-01-SUMMARY.md  # score_vina_batch batch pattern replicated; ScoredPose.ad4_score filled here
    - 02-04-SUMMARY.md  # prepare_ligand_batch provides pdbqt_path; generate_ad4_maps provides receptor.HD.map
  provides:
    - scoring/ad4.py: score_ad4_batch via Vina(sf_name='ad4') + load_maps()
    - scoring/__init__.py: now exports score_ad4_batch alongside score_vina_batch
  affects:
    - 03-03: entropy.py hybrid formula reads both vina_score and ad4_score from ScoredPose
tech_stack:
  added:
    - vina (sf_name='ad4' mode; same lazy import pattern as scoring/vina.py)
  patterns:
    - TDD RED/GREEN: test(03-02) commit before feat(03-02) commit
    - load_maps API: NEVER set_receptor() with sf_name='ad4' — load_maps(str(maps_dir / "receptor")) only
    - Anomaly flag: is_ad4_anomaly = ad4_score > 0 (informational; pose still in scored list)
    - Defensive existence check: FileNotFoundError before load_maps() if receptor.HD.map absent
key_files:
  created:
    - src/hybridock_pep/scoring/ad4.py
  modified:
    - src/hybridock_pep/scoring/__init__.py
    - tests/test_scoring.py  # TestAD4Scorer stubs replaced with 7 real tests (RED commit 6a324e1)
decisions:
  - "load_maps(str(maps_dir / 'receptor')) not set_receptor(): AD4 mode in Vina C++ binding raises RuntimeError on set_receptor; map prefix without extension resolves to receptor.HD.map, receptor.C.map etc."
  - "is_ad4_anomaly = ad4_score > 0 (not >= 0): zero is not anomalous per D-06; positive score indicates repulsive/unphysical binding; pose still included in scored list as informational flag"
  - "One Vina(sf_name='ad4') instance per batch, load_maps once before loop: matches score_vina_batch single-instance pattern; per-pose recompute would reload maps unnecessarily"
  - "Lazy Vina import (try/except ImportError) inherited from vina.py pattern: allows tests to run in base env without score-env; mock.patch replaces Vina=None in test isolation"
requirements-completed:
  - SCORE-02
metrics:
  duration_seconds: 1158
  completed_date: "2026-04-21"
  tasks_completed: 2
  files_created: 1
  files_modified: 2
---

# Phase 3 Plan 02: AD4 Scorer (SCORE-02) Summary

**AD4 batch scorer using Vina(sf_name='ad4') + load_maps() with is_ad4_anomaly flagging for positive scores, collecting the electrostatics signal that Vina's Gasteiger-charge-ignoring scorer omits.**

---

## Performance

- **Duration:** ~20 min (RED commit 6a324e1 at 08:37; GREEN commit 4f85aa9 at 08:56)
- **Started:** 2026-04-21T08:37:32Z
- **Completed:** 2026-04-21T08:56:50Z
- **Tasks:** 2 (Task 1: RED failing tests; Task 2: GREEN implementation)
- **Files modified:** 3 (ad4.py created, __init__.py updated, test_scoring.py tests committed in RED)

## Accomplishments

- Implemented `score_ad4_batch()` in `scoring/ad4.py` — AD4 scoring via `Vina(sf_name='ad4')` + `load_maps()`, never `set_receptor()`
- `is_ad4_anomaly = ad4_score > 0` flag per D-06: anomalous poses still included in scored list (informational)
- Defensive `FileNotFoundError` before `load_maps()` if `receptor.HD.map` absent (T-03-06 mitigation)
- Per-pose exception isolation: `PoseFailure(stage="scoring")`, batch continues (D-07 / T-03-08 mitigation)
- 7 TestAD4Scorer tests all pass; 8 TestVinaScorer tests still pass (15 total, 0 regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace TestAD4Scorer stub with real tests** - `6a324e1` (test)
2. **Task 2: Implement scoring/ad4.py and update scoring/__init__.py** - `4f85aa9` (feat)

_TDD plan: RED commit (test) before GREEN commit (feat). REFACTOR gate not needed._

## Files Created/Modified

- `src/hybridock_pep/scoring/ad4.py` - AD4 batch scorer; Vina(sf_name='ad4') + load_maps(); is_ad4_anomaly flag; FileNotFoundError guard
- `src/hybridock_pep/scoring/__init__.py` - Added `score_ad4_batch` to imports and `__all__`
- `tests/test_scoring.py` - TestAD4Scorer: 7 real tests replacing skip stubs (committed in RED Task 1)

## Decisions Made

- **load_maps vs. set_receptor:** `load_maps(str(maps_dir / "receptor"))` is the correct AD4 init sequence. `set_receptor()` raises RuntimeError when `sf_name='ad4'` is active. Map prefix without extension resolves to `receptor.HD.map`, `receptor.C.map`, etc. by Vina internals.
- **is_ad4_anomaly at zero:** `ad4_score > 0` (strict); zero is not anomalous per D-06. A zero score is physically possible for a ligand with no net interaction; flagging zero would produce false positives.
- **Lazy import pattern reused:** `try: from vina import Vina except ImportError: Vina = None` — same as `scoring/vina.py`. Allows test isolation via `mock.patch("hybridock_pep.scoring.ad4.Vina")` in the base Python env.

## Deviations from Plan

None — plan executed exactly as written.

---

## Known Stubs

None — `score_ad4_batch` is fully implemented. TestEntropy and TestCalibration remain intentional skip-marked stubs for plan 03-03.

---

## Threat Flags

None — all surface introduced by this plan was pre-analyzed in the plan's `<threat_model>`. No new trust boundaries added beyond what was modeled.

---

## TDD Gate Compliance

- RED gate: commit `6a324e1` — `test(03-02): add failing TestAD4Scorer tests for SCORE-02`
- GREEN gate: commit `4f85aa9` — `feat(03-02): implement scoring/ad4.py — AD4 batch scorer via load_maps()`
- REFACTOR gate: not needed (implementation clean as written)

---

## Self-Check: PASSED

Files exist:
- `src/hybridock_pep/scoring/ad4.py` — FOUND
- `src/hybridock_pep/scoring/__init__.py` — FOUND
- `tests/test_scoring.py` — FOUND
- `.planning/phases/03-scoring-core/03-02-SUMMARY.md` — FOUND (this file)

Commits verified:
- `6a324e1` (test RED) — FOUND
- `4f85aa9` (feat GREEN) — FOUND

Test result: 15 passed (8 TestVinaScorer + 7 TestAD4Scorer), 2 skipped (stubs), 0 failed
