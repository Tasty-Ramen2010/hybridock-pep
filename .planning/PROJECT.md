# HybriDock-Pep

## Current Milestone: v1.0 Core Pipeline

**Goal:** Build and ship the complete hybrid peptide docking tool — from CLI scaffold to benchmarked, documented pipeline — ready for iGEM 2026 submission.

**Target features:**
- CLI entry point with `dock`, `calibrate`, `benchmark`, `prep` subcommands
- Two-environment pipeline: RAPiDock GPU sampling + physics-based rescoring (Vina + AD4 + entropy correction)
- RMSD clustering, ensemble statistics, convergence/silhouette plots
- Full reproducibility metadata (`run_metadata.json`)
- Smoke test, MDM2/p53 integration test, benchmark suite
- README, INSTALL.md, architecture docs, tutorial notebook

## What This Is

A hybrid peptide docking tool for the iGEM 2026 Best Software Tool award. It combines a diffusion-based generative model (RAPiDock) for stochastic pose sampling with physics-based rescoring (AutoDock Vina, AutoDock4, backbone entropy correction, and optional MM-GBSA) to produce trustworthy binding pose rankings and free energy estimates for any peptide sequence against any receptor PDB. Developed by the Denmark High School iGEM 2026 dry lab team.

## Core Value

Ranking peptide binding poses with physics-backed scores that are more accurate than ML or Vina alone — so the top-1 result can be trusted for real scientific decisions.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Accept any peptide sequence (FASTA or inline) and any receptor PDB as input
- [ ] Run RAPiDock (diffusion model) for N=100 stochastic pose samples on RTX 5070
- [ ] Rescore each pose with AutoDock Vina (`--score_only`) and AutoDock4 (`--scoring ad4`) in parallel
- [ ] Apply backbone entropy correction with calibrated α coefficient
- [ ] Cluster poses by pairwise Cα RMSD (agglomerative) to identify distinct binding modes
- [ ] Output top-10 ranked poses as CSV with hybrid score and ΔG estimate
- [ ] Output best-pose PDB (top cluster centroid)
- [ ] Optional MM-GBSA rescoring on top-K poses via `--refine-topk N` flag (OpenMM + GBn2 implicit solvent)
- [ ] CLI with subcommands: `dock`, `calibrate`, `benchmark`, `prep`
- [ ] Validate all inputs before spawning subprocesses
- [ ] Log reproducibility metadata (git SHA, seeds, software versions, receptor hash) to `run_metadata.json`
- [ ] `--seed N` flag for deterministic runs
- [ ] Cross-platform: Linux, macOS ARM (via Rosetta 2 for ADFRsuite), WSL2
- [ ] Benchmark suite: Pearson r ≥ 0.55 on held-out test set, ≥ 0.10 better than Vina-alone
- [ ] Tutorial notebook runs top-to-bottom on fresh install (iGEM wiki)

### Out of Scope

- GUI or web interface — dry lab CLI users only; no wet lab UX requirement
- General protein–protein docking — peptide docking only
- Recompiling Vina to add Coulomb term — explicitly rejected (see §5.6–5.7 of spec)
- PyRosetta relax step by default — triggers ref2015 cysteine alignment failure on LISDAELEAIFEADC; OpenMM minimization used instead
- Multi-GPU parallelism for RAPiDock — one GPU, sequential inference (forking if parallel is ever needed, not threading)
- Copyleft dependencies — iGEM OSI requirement enforces MIT/Apache-2.0 only
- Bundling ADFRsuite/AutoDock4 binaries — non-redistributable licenses; documented in INSTALL.md instead

## Context

**Primary scientific target:** Malaria rapid-diagnostic peptide LISDAELEAIFEADC targeting PfLDH (PDB 1CZB) with selectivity over hLDH (PDB 1I0Z). HybriDock-Pep is the tool; this target is the headline benchmark result.

**GPU environment:** RTX 5070 (Blackwell, CC 12.0). RAPiDock's upstream pins (CUDA 11.5 / PyTorch 1.11) are incompatible — must use CUDA 12.4+ / PyTorch 2.3+.

**Two-environment architecture:** RAPiDock's pinned stack cannot coexist with the scoring stack. Two separate conda envs: `rapidock-env` (Python 3.9, PyTorch 2.3+, CUDA 12.4) and `score-env` (Python 3.11, Vina 1.2.5+, OpenMM 8.1+). Driver in `score-env` orchestrates RAPiDock via `subprocess` + `conda run`.

**PULCHRA:** Must be exactly v3.04. v3.07 produces incomplete aromatic side-chain atoms from ADCP output — reproducible bug.

**Technical spec:** `docs/HybriDock-Pep_Technical_Specification.pdf` (32 pages) is the source of truth. Sections §4, §5, §8, §11, §12, §16 are load-bearing.

## Constraints

- **GPU**: RTX 5070 (Blackwell CC 12.0) — CUDA 12.4+ required; old PyTorch pins will not run
- **Timeline**: iGEM 2026 Jamboree (November); repo freeze ≥ 2 weeks before submission
- **License**: MIT/Apache-2.0 only — iGEM OSI requirement; no copyleft in own source
- **PULCHRA**: v3.04 exactly — v3.07 has a reproducible side-chain atom bug
- **Python**: 3.11 for score-env (all in-repo code); 3.9 for rapidock-env subprocess only
- **Performance**: Full 100-pose run ≤ 5 min wall-clock on RTX 5070 + modern CPU (excl. MM-GBSA)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use AD4 scoring in parallel (not Vina charge extraction) | Vina ignores `q` column; AD4 uses Gasteiger charges explicitly | — Pending |
| Skip PyRosetta relax step by default | ref2015 alignment failure on C-terminal cysteine (LISDAELEAIFEADC) — documented in §16.1 | — Pending |
| MM-GBSA as optional `--refine-topk` flag, not default | Adds minutes; fast corrected score sufficient for ranking; publication-quality ΔG on demand | — Pending |
| Two separate conda envs (rapidock-env + score-env) | Stack incompatibility between RAPiDock's pins and OpenMM/Vina — cramming into one env causes pain | — Pending |
| Do not recompile Vina for Coulomb term | Considered and explicitly rejected in §5.6–5.7 of spec | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-18 after initialization*
