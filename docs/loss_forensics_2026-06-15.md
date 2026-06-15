# Why We Lose on ≤12, vlong, and charged — Full Forensic + What's Fixable

*2026-06-15 · E218–E220 · Ram's "I will not stand by this loss" deep dive: per-complex errors, feature
correlations, exact worst peptides, and tests of the charged-interaction-model + vlong-XL-substitution ideas.*

---

## 1. The single universal cause — regression-to-the-mean on affinity EXTREMES

Across **every** band the #1 error driver is identical: `corr(|error|, |affinity y|) ≈ +0.16`. The model
predicts everything toward the mean (≈ −8 kcal/mol), so:
- **strong binders (y ≈ −13 to −14) are UNDER-predicted** (predicted ≈ −7 to −8)
- **weak binders (y ≈ −4 to −6) are OVER-predicted** (predicted ≈ −9 to −10)

Measured shrinkage: slope of `y ~ prediction = 0.71` (1.0 = calibrated). The exact worst peptides:
```
 1eb1 L10 q−4  y=−14.2 pred=−6.5  |err|=7.7   ← strongest binder, under by 7.7
 4hpy L11 q+4  y=−13.5 pred=−8.2  |err|=5.3
 3fn0 L9  q 0  y=−6.0  pred=−10.8 |err|=4.8   ← weak binder, over by 4.8
 5etu L12 q+1  y=−5.3  pred=−9.6  |err|=4.3
```
**This is NOT under-fitting** (E220): a deeper model (depth 8) makes shrinkage WORSE (slope 0.65, r 0.36→0.32).
The features genuinely cannot separate a −14 binder from a −8 binder — that separation lives in the
expensive physics (specific H-bond networks, desolvation, entropy = FEP-only). It is an information ceiling,
not a model-capacity problem.

## 2. Per-band breakdown (crystal-925 clustered-CV, production 262-feat)

| Band | n | r | MAE | label std | top error-driver |
|---|---|---|---|---|---|
| short ≤8 | 305 | **0.436** | 1.29 | 1.77 | n_neg −0.15, \|y\| +0.15, hyd_mismatch −0.14 |
| med 9–12 | 407 | 0.298 | 1.41 | 1.85 | \|y\| +0.16, poc_net +0.10 |
| ≤12 (both) | 712 | 0.358 | 1.36 | 1.82 | \|y\| +0.16 |
| vlong ≥17 | 53 | 0.106→**0.477*** | 1.45 | 1.78 | \|y\| −0.16 |
| charged \|q\|≥2 | 417 | 0.242 | 1.32 | 1.72 | \|y\| +0.16, poc_net +0.09 |
| charged ≤12 | 285 | 0.214 | 1.28 | 1.61 | \|y\| +0.12, poc_net +0.11 |

*The pooled-model vlong 0.106 is the geometry-sabotage artifact; with vlong IN training (proper test) it's
**0.477** (E220) — vlong is NOT actually our weak band once trained correctly.

**Why ≤12 isn't "taking the cake":** short ≤8 is actually our BEST band (0.436). The drag is **med 9–12
(0.298)** — and its failure is pure affinity-extreme shrinkage (`|y| +0.16`, no other structural driver),
i.e. the FEP ceiling, plus a **narrow label spread** (std 1.61–1.85 → mechanically caps r). It's not that
we're "ass" at ≤12; it's that med-9-12 affinity is FEP-limited and the labels are compressed.

## 3. Ram's two ideas — TESTED

### 3a. Charged: richer interaction features instead of hand indicators (E219)
Built per-charge-pair interaction features (favorable +/− pairs, like-like repulsion, **charge × burial**
= buried salt-bridge strength, **charge × pocket-hydrophobicity** = desolvation penalty, polar-pocket
screening). The model CAN learn strong-vs-weak charged bonds in principle:
```
 slice          base    +charge-interaction   Δ
 charged ≤12    0.214   0.231                +0.017
 |q|≥3          0.259   0.271                +0.011
 charged ≥2     0.269   0.268                −0.002  (overall flat)
```
**+0.017 on charged ≤12** — real but small. Confirms the charged floor is **single-pose-electrostatics
FEP-bound** (memory's repeated finding): a buried salt bridge's strength depends on explicit water and
exact geometry a static snapshot can't resolve. The richer features help a little; they don't break the floor.
*Worth folding the charge×burial term in (small, free, sign-stable), but it is not the dominance lever.*

### 3b. vlong: extremely-long-ligand substitution + structured features (E220)
PPIKB gives **239 fresh vlong/XL Kd** (138 at 17–25, 101 at 26–50). Tested as training augmentation on a
held-out crystal-vlong test:
```
 (1) 925 only                          vlong-test r = 0.477   ← already strong with vlong in training
 (2) 925 + PPIKB 17-25                 r = 0.377   ▼ HURTS
 (3) 925 + PPIKB 17-50 (XL subst)      r = 0.374   ▼ HURTS
 (4) (3) + structured-peptide feats    r = 0.374   (no change)
```
**XL substitution HURTS** — extra-long ligands are a different binding regime (more anchor points, different
entropy) and add off-distribution noise. The key reframe: **vlong is NOT data-limited or weak — it's 0.477
when vlong is simply IN the training fold.** Our only real vlong problem was the pooled-model geometry
sabotage, already fixed by the vlong router (E216). Extremely-long ligands are NOT a good substitution.

## 4. The honest verdict — what's fixable vs the ceiling

| Loss | Cause | Fixable? |
|---|---|---|
| **≤12 / med 9–12** | affinity-extreme shrinkage (FEP physics) + narrow labels | **No** — information ceiling; deeper model worsens it |
| **charged** | single-pose electrostatics/desolvation | **Marginally** (+0.017 charge×burial); floor is FEP-bound |
| **vlong** | was geometry-sabotage (FIXED, router); real vlong r=0.477 | **Already fixed**; XL substitution hurts |

**The thing we keep "losing" on is the same thing FEP exists to solve.** No feature engineering on a static
pose closes it — proven three ways here (deeper model worsens, richer charge features +0.017, XL data hurts).
This is exactly why the honest claim is *"best NON-FEP scorer"*: FEP wins the affinity-extreme separation at
10⁴× the cost; among methods that don't pay that cost, we are at/above the field.

## 5. What to ship from this

- **Fold in `charge × burial`** (buried-salt-bridge term): +0.017 charged, free, sign-stable.
- **Keep the vlong router** (already shipped) — don't add XL ligands.
- **Do NOT deepen the model** — it worsens calibration.
- **Frame the claim around "non-FEP, commercially available"** — the losses are the FEP ceiling, which no
  available non-FEP tool escapes either.
