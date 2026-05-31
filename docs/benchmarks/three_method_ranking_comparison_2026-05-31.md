# Three-Method Pose Ranking Comparison: Vina vs OpenMM→Vina vs ref2015
Date: 2026-05-31

## Overview

Three experiments benchmark pose-ranking quality on bench300 peptide docking poses:
1. **Plain Vina** (score_only on raw diffusion pose)
2. **OpenMM→Vina** (AMBER ff14SB peptide minimization, then Vina score_only)
3. **ref2015** (PyRosetta ref2015 on receptor+pose complex, no FastRelax)

## Key Results

| Method | N | Kendall τ | P(best) | Gap Rec | Notes |
|--------|---|-----------|---------|---------|-------|
| Plain Vina | 60 | +0.021 | 16.7% | +1.2% | ~random |
| OpenMM→Vina | 60 | +0.080 | 31.7% | +19.5% | Clash relief helps |
| ref2015 (no relax) | 180 | +0.176 | 24.4% | +22.1% | Best τ, no relaxation needed |

**Best τ**: ref2015 (0.176) > OpenMM→Vina (0.080) > Plain Vina (0.021)
**Best P(best)**: OpenMM→Vina (31.7%) > ref2015 (24.4%) > Plain Vina (16.7%)

## ref2015 Per-Bucket Breakdown (N=180)

| Bucket | N | τ | ρ | P(best) | Top-1 RMSD |
|--------|---|---|---|---------|-----------|
| all | 180 | +0.176 | +0.217 | 24.4% | 4.20Å |
| short | 45 | +0.067 | +0.084 | 17.8% | 3.01Å |
| medium | 45 | +0.213 | +0.251 | 28.9% | 3.42Å |
| long | 45 | +0.222 | +0.251 | 20.0% | 4.46Å |
| very_long | 45 | +0.200 | +0.280 | 31.1% | 5.91Å |
| HELIX | 60 | +0.120 | +0.165 | 28.3% | 4.31Å |
| SHEET | 60 | +0.217 | +0.257 | 23.3% | 4.28Å |
| UNUSUAL | 60 | +0.190 | +0.228 | 21.7% | 4.01Å |

## Key Findings

1. **Plain Vina on raw diffusion poses is near-random** (τ≈0.02) due to clash noise.
2. **OpenMM minimization restores Vina signal** (τ: 0.021→0.080, P(best): 17%→32%).
3. **ref2015 without FastRelax outperforms both Vina variants** (τ=0.176).
   - This is surprising: even unrelaxed complexes provide better ref2015 signal.
   - ref2015 handles steric penalties (fa_rep) more gracefully than Vina in score_only mode.
4. **Medium peptides** are hardest to rank (OpenMM→Vina τ=-0.093), easiest for ref2015 (τ=0.213).

## Implication for HybriDock-Pep Scoring Pipeline

Current pipeline: Vina + AD4 → entropy correction → optional MM-GBSA.
Finding: ref2015 score (no relax, <0.5s/pose) provides better ranking than Vina alone.
Recommendation: Add ref2015 as an optional fast rescoring tier between AD4 and MM-GBSA.

## Confidence Model Comparison (pending training)

Training in progress (~60 epochs, ~3.5h). Results will be added when complete.