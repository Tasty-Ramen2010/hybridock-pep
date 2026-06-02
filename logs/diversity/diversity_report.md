# Confidence Diversity Campaign — Final Report
Generated: 2026-06-02

## Executive Summary

Five experiments definitively answered whether pose-generation diversity is the
main remaining bottleneck for the confidence model. It is not. Data volume is.

## Exp 1 — Variant Count Sweep

Matched-pairs control (2040 pairs fixed across 1/2/4 variants, 5 seeds each):

| n_variants | mean τ (matched) | mean τ (unmatched) | unmatched pairs |
|---|---|---|---|
| 1 | 0.171 | 0.165 | 2 040 |
| 2 | 0.170 | 0.231 | 9 179 |
| 4 | 0.172 | 0.267 | 38 757 |

**Finding:** Structural diversity alone contributes <0.002 τ. All gains in the
unmatched condition come from 19× larger pair count.

## Exp 2 — Diversity Metrics

| n_variants | RMSD spread | feat std | feat_dist_ratio | τ |
|---|---|---|---|---|
| 1 | 2.52 Å | 19.1 | — | 0.165 |
| 2 | 3.28 Å | 20.4 | 0.919 | 0.231 |
| 4 | 3.89 Å | 22.4 | 0.962 | 0.267 |

feat_dist_ratio → 1.0 with more variants: different model variants produce
nearly indistinguishable encoder-space neighborhoods.

## Exp 3 — Routing Analysis

Best config by slice:

| Slice | Best config | τ | vs bench_only |
|---|---|---|---|
| SHEET | 75B_25G | 0.266 | +0.082 |
| HELIX | bench_only | 0.261 | baseline |
| UNUSUAL | bench_only | 0.371 | baseline |
| very_long | 25B_75G | 0.346 | +0.027 |
| medium | 50B_50G | 0.400 | +0.022 |
| long | bench_only | 0.220 | baseline |
| short | gen_only | 0.248 | +0.132 |

Optimal routing (length-bucket): weighted τ ≈ 0.317 vs bench_only 0.277 (+0.04).
Practical two-rule router: short (≤8 aa) → gen_only, rest → bench_only.

## Exp 4 — Clean REF2015 5-fold CV

Mean τ across folds by blend weight:

| w_conf | mean τ |
|---|---|
| 0.0 (pure ref2015) | 0.174 |
| 0.5 | 0.196 |
| **0.6** | **0.200** ← optimal |
| 1.0 (pure conf) | 0.138 |

REF2015 (τ=0.174) beats pure confidence (τ=0.138). Optimal blend w_conf=0.6.
REF2015 contributes independent signal — do not drop it.

## Exp 5 — Bootstrap Scaling Law

| N | τ | 95% CI |
|---|---|---|
| 63 | 0.219 | ±0.028 |
| 126 | 0.229 | ±0.021 |
| 189 | 0.245 | ±0.014 |
| 253 | 0.265 | ±0.022 |

Projection: 2× data → τ ≈ 0.285–0.295; ceiling ≈ 0.38.
Currently at ~70% of data-scaling ceiling.

## Recommendations (priority order)

1. **Collect ~250 more PepPC complexes** (highest ROI, +0.025 τ projected)
2. **Deploy two-rule routing**: short ≤8 aa → gen_only, rest → bench_only (+0.04)
3. **Lock in REF2015 blend at w_conf=0.6** (already optimal, just deploy it)
4. **Do not generate more variant poses** (diversity adds nothing at fixed pair count)

## Deployed performance roadmap

| Configuration | τ |
|---|---|
| Current (bench_only, no routing) | 0.201 ± 0.019 |
| + optimal REF2015 blend | 0.200 (CV, uncontaminated) |
| + two-rule routing | ~0.24 |
| + 2× data + routing + blend | ~0.28–0.30 |
| Theoretical ceiling | ~0.38 |
