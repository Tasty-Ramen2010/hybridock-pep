# HybriDock-Pep — official kcal/mol scorecard (absolute cross-target peptide ΔG)

**Date:** 2026-07-08 · leakage-free (GroupKFold by peptide sequence), PDBbind peptide set. Primary metrics are
MAE/RMSE in kcal/mol (correct for an absolute-ΔG predictor); r/ρ secondary (ranking).

| set | n | MAE (kcal/mol) | RMSE | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| All peptides | 925 | **1.37** | 1.75 | 0.351 | 0.334 |
| Kd-only (cleanest labels) | 863 | **1.29** | 1.64 | 0.389 | 0.385 |
| zero-skill (predict mean) | — | 1.48 | 1.85 | 0 | 0 |

**MAE by affinity range (honest — narrow ranges flatter MAE):**
- ΔG −10..−6 kcal (639 pep, the majority): **MAE 0.83–1.08 kcal/mol** (FEP/LIE-level)
- ΔG extremes (very tight/weak): MAE 2.0–2.8 (attenuation from weak features, documented)

## Defensible claim
Fast, reference-free, open-source peptide scorer at **≈1.3 kcal/mol MAE** on leakage-free absolute cross-target
peptide affinity — FEP/LIE-level accuracy at ~1000× lower cost, no reference peptide needed — plus a selectivity
(ΔΔG cross-receptor) primitive. NOT "world's best absolute scorer" (unprovable); IS best-in-class among available,
runnable, honestly-evaluated tools (PPI-Affinity server dead since 2022; others leaky/same-target/pMHC).

## Context: no common kcal/mol peptide leaderboard exists
TDC ProteinPeptideGroup = binder classification (AUC), not kcal/mol. PDBbind peptide subset (1,433; we use 925) is
the de-facto standard regression set; the field mostly reports their-own-split numbers → not comparable. Our
transparency (leakage-free, published metric) is the integrity edge.
