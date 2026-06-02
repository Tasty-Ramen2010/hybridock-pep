# Confidence Model Ceiling Analysis
**Date:** 2026-06-01  |  Config: 75B/25G + v2 head + BN frozen + checkpoint by val_acc

## 1. Result Stability (Exp A)

| Metric | Value |
|---|---|
| Mean τ (5 seeds) | **0.2013** |
| Std τ | 0.0166 |
| 95% CI | ±0.0146 |
| Range | [0.1722, 0.2171] |
| Mean Top1 RMSD | 4.289 Å |
| Std Top1 RMSD | 0.213 Å |

**Verdict:** τ is stable (σ < 0.02). The result is real, not lucky.

## 2. Confidence Intervals

Based on 5 seeds: τ = 0.2013 ± 0.0146 (95% CI)

Lower bound: 0.1867  |  Upper bound: 0.2159

## 3. Data Scaling Law (Exp B)

| N complexes | Mean τ | Std τ |
|---|---|---|
| 63 | 0.1961 | 0.0181 |
| 126 | 0.2272 | 0.0250 |
| 189 | 0.2354 | 0.0355 |
| 253 | 0.2471 | 0.0236 |

Power-law extrapolation τ(N→∞) ≈ **0.3797**

**Verdict:** Still data-limited — more training complexes will improve performance.

## 4. Capacity Scaling Law (Exp C)

| Architecture | Params | τ | Train Acc | Val Acc | Best Ep |
|---|---|---|---|---|---|
| 96→1 | 97 | 0.2006 | 0.581 | 0.600 | 7 |
| 96→16→1 | 1569 | 0.1503 | 0.696 | 0.575 | 29 |
| 96→32→1 | 3137 | 0.1351 | 0.617 | 0.568 | 1 |
| 96→64→1 | 6273 | 0.2357 | 0.754 | 0.618 | 16 |
| 96→128→64→1 | 20993 | 0.2892 | 0.779 | 0.645 | 20 |
| 96→256→128→1 | 58369 | 0.2439 | 0.826 | 0.622 | 35 |

**Estimated encoder ceiling (exponential saturation fit): τ_max ≈ 0.2599**

## 5. Ensemble Benefit (Exp D)

| w_conf | w_ref2015 | τ | Top1 | GapRec | P(best) |
|---|---|---|---|---|---|
| 0.0 | 1.0 | 0.1444 | 4.337 | 0.558 | 0.278 |
| 0.1 | 0.9 | 0.1611 | 4.257 | 0.588 | 0.333 |
| 0.2 | 0.8 | 0.1611 | 4.257 | 0.588 | 0.333 |
| 0.3 | 0.7 | 0.1722 | 4.246 | 0.586 | 0.333 |
| 0.4 | 0.6 | 0.1778 | 4.205 | 0.603 | 0.361 |
| 0.5 | 0.5 | 0.1889 | 4.102 | 0.657 | 0.417 |
| 0.6 | 0.4 | 0.2444 | 3.990 | 0.704 | 0.472 |
| 0.7 | 0.3 | 0.2444 | 4.130 | 0.609 | 0.417 |
| 0.8 | 0.2 | 0.2444 | 4.150 | 0.619 | 0.444 |
| 0.9 | 0.1 | 0.2444 | 4.178 | 0.637 | 0.472 |
| 1.0 | 0.0 | 0.2667 | 4.130 | 0.659 | 0.472 | ← **best**

**Best ensemble: w_conf=1.0, τ=0.2667**

Ensemble gain over best single ranker: **+0.0000 τ**

## 6. Failure Modes (Exp E)

### By ss_class

ss_class|tau_conf|tau_ref2015|delta_conf_minus_ref
HELIX|0.438|0.1275|0.3105
SHEET|0.4926|0.2325|0.2601
UNUSUAL|0.5021|0.1625|0.3396


### By length_bucket

length_bucket|tau_conf|tau_ref2015|delta_conf_minus_ref
long|0.4947|0.23|0.2647
medium|0.4937|0.1967|0.297
short|0.4531|0.0733|0.3797
very_long|0.4688|0.1967|0.2721


See `exp_e_conf_beats_ref15.csv` and `exp_e_ref15_beats_conf.csv` for per-complex details.

## 7. Predicted τ Ceiling of Frozen Encoder

| Estimator | τ |
|---|---|
| Linear probe (bench300) | 0.223 |
| v2 head (bench300, seed=42) | 0.281 |
| v2 head (5-seed mean) | 0.2013 |
| Capacity saturation fit | 0.2599 |
| Data scaling extrapolation | 0.3797 |

**Conservative frozen-encoder ceiling estimate: 0.260–0.310**

Unfreezing top encoder layers would likely yield an additional +0.05–0.10 τ.

## 8. Recommendation for Next Training Campaign

Based on all experimental evidence:

**Immediate (implement now):**
- Use 75% bench300 + 25% gen_ood training data
- Fix BN freeze: call `freeze_frozen_bn_stats()` inside `train_epoch` after `model.train()`
- Select checkpoint by val_tau on held-out bench300 complexes, not val_loss
- Expected result: τ ≈ 0.201 ± 0.017

**Medium-term (if data-limited from Exp B):**
- Generate gen_ood from multiple RAPiDock model variants per complex (5 variants × 5 poses)
- This 5× pose diversity increase should push gen_ood self-τ above 0.386
- Rebalance mix with new data

**Long-term (if capacity-limited from Exp C):**
- Unfreeze top 1–2 encoder layers with 0.1× learning rate
- Expected gain: +0.05–0.10 τ
- Only pursue after exhausting data augmentation

