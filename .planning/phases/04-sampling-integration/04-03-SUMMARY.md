---
plan: "04-03"
phase: "04-sampling-integration"
status: complete
completed_at: "2026-04-23"
tasks_completed: 1
files_modified:
  - src/hybridock_pep/sampling/pose_io.py
tests_added: 0
tests_passed: 5
commit: 7603bc1
---

# Plan 04-03 Summary — pose_io.py

## What was built

`parse_poses(poses_dir)` — collect-all-failures Biopython PDB parser implementing SAMP-01.

- Iterates `pose_*.pdb` files, catches all per-pose exceptions as `PoseFailure(stage="parsing")`
- Extracts Cα coordinates via `Bio.PDB.PDBParser(QUIET=True)` → `np.ndarray` shape `(n_res, 3)` dtype `float64`
- D-14 SEQRES-first sequence extraction with ATOM fallback (`_extract_sequence_seqres_first`)
- `len(records) + len(failures) == n_files` invariant always holds

## Fix applied

`three_to_one` was removed from `Bio.PDB.Polypeptide` in Biopython 1.80+. Replaced with inline lookup dict built from `Bio.Data.IUPACData.protein_letters_3to1` with uppercase keys to match PDB residue names.

## Tests

5/5 TestPoseIO passed: valid parse, malformed PDB, batch invariant, SEQRES-preferred, ATOM fallback.
