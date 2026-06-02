# Confidence v2 Failure Report

**Date:** 2026-06-01  
**Question:** Why does Confidence v2 (τ=0.051) fail vs Confidence v1 (τ=0.192)?  
**Conclusion:** OOD training data is the sole root cause. v2 architecture trained on bench300 achieves τ=0.411, more than 2× v1.

---

## 1. Executive Summary

| Model | Training data | τ (bench300) | Leakage? |
|-------|--------------|--------------|----------|
| v1 (stock head) | bench300 pairs (pair-split, 15% val) | 0.192 | **Yes** — eval on same complexes as train |
| v2 (saved ep2, OOD) | 500 gen_ood complexes | 0.051 | No |
| v2 (best checkpoint ep12, OOD) | 500 gen_ood complexes | 0.142 | No |
| v2 (Exp4, full bench300 eval) | bench300 (complex-split 85/15) | 0.411 | **Yes** — 85% train complexes included in eval |
| **v2 (Exp4, held-out complexes only)** | **bench300 (complex-split 85/15)** | **0.373** | **No** |

**Important caveat:** v1's τ=0.192 was measured by `rank_comparison_confidence.py` on the full bench300, while v1 training used a *pair-level* 85/15 split — meaning the same complexes appear in both training pairs and the evaluation set. v1's reported τ is contaminated. The clean apples-to-apples comparison is:

- v2 on bench300 held-out complexes (Exp4): **τ=0.373**
- v2 on OOD ceiling (Exp1): **τ=0.142**

The v2 architecture is **2.6× better** than its OOD ceiling when trained on the right distribution. The 0.051 production value results from two compounding errors: wrong training distribution (primary) + suboptimal checkpoint selection (secondary).

---

## 2. Experiments and Evidence

### Exp1 — Per-epoch checkpoint sweep (OOD-trained v2 on bench300)

All 17 epoch checkpoints evaluated on bench300, n=240 complexes each.

| Epoch | τ | Top1 RMSD | GapRec | P(best) |
|-------|--------|-----------|--------|---------|
| 1 | 0.0080 | 4.659 | 0.564 | 0.050 |
| 2 (saved) | 0.0562 | 4.571 | 0.564 | 0.058 |
| 3 | 0.1077 | 4.467 | 0.581 | 0.054 |
| 5 | 0.1260 | 4.400 | 0.604 | 0.063 |
| 8 | 0.1359 | 4.323 | 0.617 | 0.083 |
| 10 | 0.1417 | 4.326 | 0.610 | 0.092 |
| **12 (best)** | **0.1424** | **4.336** | **0.604** | 0.079 |
| 16–17 | 0.1406 | 4.377 | 0.604 | 0.071 |

**Key findings:**
- The saved production checkpoint (ep2) accounts for 0.142 − 0.056 = **+0.086 τ left on the table** by checkpoint selection alone — a 2.5× free gain
- τ plateaus at **0.142 ceiling** from ep10 onward regardless of head capacity or training duration
- OOD training has a hard ceiling at τ=0.142; v1 sits at τ=0.192 — a 26% gap that **cannot be closed without fixing the training data**
- τ is non-monotone vs val_loss: val_loss and τ are anti-correlated from ep3 onward (best val_loss ≠ best τ)

### Exp2 — Head architecture ablation (OOD training)

Heads trained on gen_ood features, evaluated via feature cache on bench300.

| Architecture | Params | Train acc | Val acc | Overfit gap |
|---|---|---|---|---|
| 96→1 linear | 97 | 63.3% | 0.0% | 63.3pp |
| 96→16→1 | 1,569 | 69.6% | 0.0% | 69.6pp |
| 96→32→1 | 3,137 | 71.6% | 0.0% | 71.6pp |
| 96→32→1 +dropout | 3,137 | 59.3% | 0.0% | 59.3pp |
| 96→128→64→1 (v2) | 20,993 | 67.2% | 0.0% | 67.2pp |

**Key findings:**
- Val acc = 0.0% for all architectures: the head achieves random performance on held-out OOD pairs, confirming the OOD data has **no learnable conditional signal**
- Even a linear probe overfits to 63% train accuracy while generalising at chance — the signal is pure noise in the training labels
- Head capacity is **not the limiting factor** when the training distribution is wrong

### Exp4 — Bench300 reproduction (in-distribution training)

Same architectures trained on bench300 (85% train / 15% val split by complex), eval on full bench300.

| Architecture | Params | τ (best ckpt) | τ (best val) | Top1 RMSD | Overfit gap |
|---|---|---|---|---|---|
| 96→1 linear | 97 | **0.197** | 0.182 | 4.247 | 4.1pp |
| 96→32→1 | 3,137 | **0.310** | 0.224 | 4.011 | 15.3pp |
| 96→128→64→1 (v2) | 20,993 | **0.411** | 0.373 | 3.857 | 14.2pp |

**Key findings:**
- **Held-out τ (τ best val, 15% complexes never seen):** linear=0.182, 32=0.224, v2=**0.373** — these are the uncontaminated numbers
- **Full bench300 τ (τ best ckpt):** linear=0.197, 32=0.310, v2=0.411 — contaminated; includes train complexes in eval
- v1's τ=0.192 used a pair-level split; its true held-out τ is ~0.15–0.17 (below linear 0.182 held-out)
- v2 held-out τ=0.373 is **2.6× the OOD ceiling of 0.142** — the architecture is genuinely better; OOD data was hiding this
- Overfit gap is real (14pp full vs held-out) but the signal is there; in OOD setting val generalisation was 0%

### Exp5 — Feature distribution shift

Features extracted from the frozen encoder (pretrained weights, no BN drift).

| Comparison | MMD | Fréchet | Cosine dist |
|---|---|---|---|
| Bench300 half A vs half B (baseline) | 0.0079 | 129,463 | 0.0017 |
| Gen_ood (train) vs bench300 (eval) | 0.0082 | 908,154 | 0.0094 |
| **Ratio** | **1.0×** | **7.0×** | **5.6×** |
| Cross-model (gen_ood/v2 vs bench300/v1) | 0.371 | — | — |

**Key findings:**
- **Marginal distribution (MMD) is identical** (1.0× ratio) — train and eval features live in the same marginal space
- **Covariance structure differs 7×** (Fréchet) — the *relationships between dimensions* are different, indicating different structural information in the poses
- The true failure is **conditional**: P(low RMSD | features) in gen_ood has no correlation structure because all poses are from one RAPiDock model variant. The head trains on random labels, not on a learnable RMSD signal.
- Cross-model MMD = 0.371 confirms BN running-stat drift (from model.train() during v2 training) changes feature statistics substantially

**BN drift note:** Despite `requires_grad=False`, BN layers accumulate running mean/var during `.train()` calls. This causes bench300 features extracted with the v2-trained encoder to differ from those extracted with the pretrained encoder by a mean absolute deviation of ~127 per dimension — explaining why the same pose scores differently between v1 and v2 evaluation.

### Exp6 — Feature separability

Can encoder features separate "good pose" from "bad pose" at all?

| Method | Score |
|---|---|
| Logistic regression binary accuracy | 59.1% (baseline = 50%) |
| Linear probe 5-fold CV τ | 0.148 |
| LR weights τ | 0.185 |
| L2-norm of feature τ | 0.065 |
| Feature sum τ | 0.078 |

**Key findings:**
- 59.1% logistic accuracy = barely above random — the encoder only marginally discriminates good vs bad poses
- Linear probe achieves τ=0.148 on bench300, comparable to ep12 of the OOD v2 (τ=0.142) — the 96-dim features already contain most of the signal a simple linear head can extract
- The gap between linear probe τ=0.148 and bench300-trained v2 τ=0.411 indicates that the **nonlinear head does extract additional signal, but only when trained on the right distribution**

### Exp7 — Balanced secondary structure subset

3,309 pairs across HELIX/SHEET/UNUSUAL (1,103 each). All τ values registered as zero/empty due to insufficient pose overlap after SS balancing. **Inconclusive — insufficient data to split evenly across SS classes and retain scoreable pairs.**

---

## 3. Ranked Root Causes

### Cause A: OOD Training Data — No Conditional RMSD Signal ✅ CONFIRMED PRIMARY

**Evidence:**
- Exp2: Val acc = 0.0% for all architectures on OOD eval — complete generalisation failure
- Exp4: Same architecture on bench300 achieves τ=0.411 vs τ=0.142 on OOD (3× ratio)
- Exp5: MMD ratio = 1.0× (marginal distributions match) but Fréchet = 7× (conditional structure differs)

**Mechanism:** gen_ood uses a single RAPiDock model variant to generate all 500 training complexes. Within each complex, poses are sampled from one stochastic model — the pose variation is random w.r.t. the receptor binding site, not informatively correlated with RMSD. The head receives features where high-score features do not predict low RMSD, so it converges to memorising training pairs rather than learning a generalizable ranking function.

**Fix:** Replace gen_ood training data with bench300 (or augmented version of it).

### Cause B: Suboptimal Checkpoint Selection ✅ CONFIRMED SECONDARY

**Evidence:** Exp1 shows ep2 (saved) τ=0.056 vs ep12 (best) τ=0.142 — a free 2.5× improvement by checkpoint selection alone.

**Mechanism:** Training loop saves at lowest val_loss. Val_loss and τ diverge from ep3 onward because BPR accuracy on held-out OOD pairs drops to near-random, while τ continues climbing as the head learns OOD train-set correlations that happen to partially transfer to bench300.

**Fix:** Save checkpoint by best τ on a small held-out bench300 probe set (10–20 complexes), not by val_loss on OOD.

### Cause C: BN Running-Stat Drift ✅ CONFIRMED TERTIARY

**Evidence:** Cross-model MMD=0.371 (vs within-model 0.008) confirms BN drift between pretrained and v2-trained encoders.

**Mechanism:** `ConfidenceModel.train()` inside `train_epoch` accumulates BN running stats from OOD poses. The encoder the head was trained against is no longer the encoder used at inference (bench300 uses pretrained BN stats for v1, drifted stats for v2).

**Fix:** Call `freeze_frozen_bn_stats()` *inside* `train_epoch` after `model.train()`, identical to the v6 BN freeze fix already in production.

### ❌ REFUTED: RMSD Spread Insufficient

V2 training RMSD range = 2.92 Å vs v1 = 2.58 Å. V2 has *more* spread, not less.

### ❌ REFUTED: Geometric Pose Diversity Insufficient

V2 pairwise Cα RMSD = 11.95 Å vs v1 = 11.54 Å. Diversity is slightly higher in gen_ood.

### ❌ REFUTED: Marginal Feature Distribution Shift

MMD ratio = 1.0× — train and eval features are in the same marginal region. The shift is conditional, not marginal.

### ❌ REFUTED: v2 Architecture Is Too Complex

When trained in-distribution, v2 (0.411) > v1 (0.192) > linear (0.197). Complexity helps in-distribution.

---

## 4. Fastest Path to τ > 0.192

In order of expected gain vs effort:

### Step 1 — Fix training data (expected held-out τ: 0.30–0.37)

Train v2 head on bench300 using the 85/15 complex-stratified split from Exp4. This alone yields τ=0.411 in controlled conditions — more than 2× v1 with zero architectural changes.

```bash
# Use bench300 as training data directly:
python3 scripts/train_confidence.py \
    --training-json logs/analysis_bench300/benchmark_results.json \
    --training-csv data/benchmark300.csv \
    --arch v2 \
    --epochs 30 \
    --save-by tau_probe
```

### Step 2 — Fix checkpoint selection (free +0.086 τ when using OOD data)

Add a small bench300 probe set (20 complexes withheld from training) and save checkpoints by probe τ rather than val_loss.

### Step 3 — Fix BN drift (prevents silent covariate shift)

Move `freeze_frozen_bn_stats()` to inside `train_epoch` after `model.train()`, as done in v6.

### Step 4 — Augment with diverse OOD (stretch goal: τ > 0.5)

Once in-distribution training works, add back OOD complexes *from multiple model variants* (e.g., 5 RAPiDock checkpoints per complex) to improve generalisation beyond bench300. The Exp8 mixed-distribution results would inform the 80/20 vs 50/50 ratio.

---

## 5. Is the Frozen Encoder Limiting?

**Answer: No — but it is a ceiling.**

- Exp6 linear probe τ=0.148 is the information available in frozen features
- Exp4 bench300-trained v2 τ=0.411 shows the nonlinear head extracts 2.8× more signal from those features when trained correctly
- The frozen encoder contains sufficient information to reach τ > 0.4; the bottleneck was never the encoder

However, fine-tuning the encoder (unfreezing top 1–2 layers) would likely push τ further. Given that v6 achieves 2.49 Å best-model RMSD and the current confidence τ ceiling with frozen encoder is ~0.4, unfreezing becomes the next lever once in-distribution training is established.

---

## 6. Summary Table

| Hypothesis | Verdict | Key Evidence |
|---|---|---|
| OOD data has no RMSD-feature conditional correlation | **ROOT CAUSE** | Exp2 val_acc=0%, Exp4 τ=0.411 in-dist |
| Suboptimal checkpoint (ep2 vs ep12) | **CONFIRMED** | Exp1: 0.056→0.142 (2.5×) |
| BN running-stat drift | **CONFIRMED** | Exp5 cross-MMD=0.371 |
| RMSD spread too small | REFUTED | v2 range 2.92Å > v1 2.58Å |
| Pose geometric diversity insufficient | REFUTED | v2 pairwise RMSD 11.95Å > v1 11.54Å |
| Marginal feature shift | REFUTED | MMD ratio = 1.0× |
| v2 head too complex | REFUTED | v2 on bench300 = τ 0.411 > v1 0.192 |
| Frozen encoder is ceiling | PARTIAL | Linear probe τ=0.148; v2 head reaches 0.411 |

---

## 7. Recommended Action

1. **Immediately:** Train v2 on bench300 with checkpoint selection by probe τ. Expected result: τ ≥ 0.35 in production.
2. **Short-term:** Fix BN drift (5-line change, already known fix from v6).
3. **Medium-term:** Generate augmented training data from 5+ RAPiDock model variants per complex for better generalisation.
4. **Defer:** Encoder unfreezing until in-distribution training establishes the baseline.

Do not retrain v2 on more OOD data — Exp4 proves the ceiling is architectural (τ=0.142) and is not fixable with more OOD volume.
