# Phase 2: Preparation Pipeline - Discussion Log

**Date:** 2026-04-20
**Outcome:** All gray areas resolved → 02-CONTEXT.md written

---

## Gray Areas Discussed

### 1. Receptor prep workflow

| Question | Options Presented | Decision |
|----------|------------------|----------|
| Which pdbfixer fixes to apply? | All fixes / Selective (missing atoms only) / Selective (H only) | All fixes (missing atoms + missing residues + H at pH 7.4) |
| Cache receptor PDBQT between runs? | Cache if PDBQT newer than PDB / Always re-run | Always re-run |
| On prepare_receptor4.py failure? | Hard abort with message / Warn and continue / Retry once | Hard abort with full stderr |

### 2. autogrid4 / GPF generation

| Question | Options Presented | Decision |
|----------|------------------|----------|
| GPF source? | Programmatic from DockConfig / User-supplied template / Hybrid | Programmatic from DockConfig |
| HD map guard behavior? | Hard abort / Warn and skip AD4 / Warn and continue | Hard abort with specific message |
| AD4 atom types? | Full peptide set (C A N O S H HD e d) / Minimal (C A N O H HD) | Full peptide set |
| Map file location? | output_dir/maps/ / output_dir/ / Alongside receptor PDBQT | output_dir/maps/ subdirectory |

---

## Items Left to Claude's Discretion

- Ligand batch parallelism (executor type, degree, per-pose error handling)
- Test fixture strategy
- GPF grid spacing (standard 0.375 Å default)
