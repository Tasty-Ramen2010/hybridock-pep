# ref2015 Scoring: No-Relax vs FastRelax Comparison
Date: 2026-05-31

## Key Finding: FastRelax Hurts Ranking Quality

| Method | N | tau | rho | P(best) | top1 | gap_rec |
|--------|---|-----|-----|---------|------|---------|
| Plain Vina | 60 | +0.021 | +0.034 | 16.7% | nanA | +1.2% |
| OpenMM->Vina | 60 | +0.080 | +0.078 | 31.7% | 4.25A | +19.5% |
| ref2015 no-relax | 180 | +0.176 | +0.217 | 24.4% | 4.20A | +22.1% |
| ref2015 + FR(10) | 180 | +0.163 | +0.196 | 25.6% | 4.22A | +21.7% |
| ref2015 + FR(20) | 180 | +0.139 | +0.171 | 24.4% | 4.19A | +22.6% |

## Why FastRelax Hurts

No-relax ref2015 uses fa_rep clash energy as a discriminative signal:
bad poses have more clashes -> higher (worse) score -> correctly ranked lower.
FastRelax repairs all poses equally, homogenizing scores and destroying signal.

Recommendation: use ref2015 NO-RELAX for ranking, FastRelax ONLY for structure output.