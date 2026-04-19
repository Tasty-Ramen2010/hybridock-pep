# State: HybriDock-Pep

## Current Position

Phase: Not started (roadmap being created)
Plan: —
Status: Defining roadmap
Last activity: 2026-04-19 — Milestone v1.0 started

## Accumulated Context

- Two-environment architecture is non-negotiable: `rapidock-env` (Python 3.9, PyTorch 2.7, CUDA 12.8) and `score-env` (Python 3.11, Vina 1.2.7, OpenMM 8.4)
- PyTorch 2.7 + CUDA 12.8 required for native Blackwell sm_120 support (not 2.3/12.4 as spec says)
- PULCHRA must be v3.04 exactly — build from source, do not use Bioconda 3.06
- fair-esm 2.0.0 is highest fragility point (abandoned upstream) — smoke test import first
- Use Vina Python API (not subprocess) for per-pose scoring to avoid fork+exec × 100
- scikit-learn clustering: use `average` linkage with precomputed RMSD metric (Ward requires Euclidean)
- Cluster over contact-zone Cα only (not full peptide) to avoid terminal-residue RMSD dominance

## Decisions

| Decision | Rationale |
|----------|-----------|
| PyTorch 2.7 + CUDA 12.8 | First native sm_120 support; 2.3/12.4 is emulation only |
| Vina Python API for scoring | Avoids 100 fork+exec cycles per run |
| contact-zone Cα RMSD for clustering | Terminal residues dominate full-peptide RMSD and corrupt cluster quality |
| Skip PyRosetta relax by default | ref2015 alignment failure on C-terminal cysteine (§16.1) |
| AD4 scoring in parallel with Vina | Provides charge signal Vina ignores; discrepancy flags electrostatics-dominated binding |

## Blockers

(none)

## Pending Todos

(none)
