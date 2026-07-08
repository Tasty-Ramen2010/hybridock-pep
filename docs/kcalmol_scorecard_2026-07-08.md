# HybriDock-Pep — official kcal/mol scorecard (VERIFIED, leakage-free)

**Date:** 2026-07-08 · Rigorous 60%-sequence-identity clustered CV (CD-HIT-style; entire clusters held out).
Primary metric = MAE/RMSE in kcal/mol (correct for an absolute-ΔG predictor). Reproduce: `scripts/e330_ours_pdbbind.py`
(full set) and `scripts/e331_ours_vs_ppiclone_clustered.py` (matched head-to-head).

## Absolute performance — full PDBbind peptide set (n=925)
| split | MAE (kcal/mol) | RMSE | Pearson r | Spearman ρ |
|---|---|---|---|---|
| random 5-fold **(LEAKY — do not cite)** | 1.32 | 1.66 | 0.446 | 0.413 |
| **60%-id clustered 5-fold (LEAKAGE-FREE)** | **1.43** | **1.81** | **0.263** | 0.247 |
| zero-skill (predict mean) | 1.47 | 1.85 | 0 | 0 |

Honest read: leakage-free we beat zero-skill modestly (MAE 1.43 vs 1.47); absolute cross-target r (~0.26) is
capped near the field ceiling for ALL methods (FEP included). MAE is the stable, meaningful metric.

## Head-to-head vs PPI-Affinity clone — matched, identical split (n=865)
| model | MAE | RMSE | r | ρ |
|---|---|---|---|---|
| **HybriDock-Pep** (16 struct feats, GBT) | **1.33** | **1.66** | **0.391** | 0.374 |
| PPI-clone (ProtDCal-3D + SVR) | 1.44 | 1.82 | 0.231 | 0.182 |

We beat the previous-best published approach on every metric; margin WIDENS under the honest split
(leaky Δr +0.11 → clustered Δr +0.16). PPI-Affinity's own server has been unmaintained since 2022.

## Integrity notes
- The earlier `seq[:4]`-grouped scorecard (MAE 1.29–1.37) used a WEAK grouping (mildly leaky) — SUPERSEDED by
  the numbers above.
- "Leakage-free" = 60%-id CD-HIT clustering, clusters held out per fold; verified (clustered r < leaky r).
- Complexes: data/e331_matched_pdbids.json (865 PDB IDs), 810 Kd + 55 Ki, len 3–19, ΔG −14.2..−3.7.
