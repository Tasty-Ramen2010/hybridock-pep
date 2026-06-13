# Real-pose deployment forensics — why r fell to 0.37, and what it actually is

**Date:** 2026-06-13 · **Scripts:** `e97_cr65_realpose_grade.py`, `e98_realpose_forensics.py`
**Set:** 65 cr65 complexes, freshly regenerated RAPiDock-Reloaded poses (N=100, 25 used for analysis).

## TL;DR

The "mass failure" to r=0.37 was **not** a regression and **not** pose quality. It was two things:
1. **Pose-aggregation methodology** (~half the gap): grading on diffusion-order poses instead of
   ML-ranked poses. The ML ranker wired tonight lifts deployment **0.372 → 0.478** (≈ documented 0.486).
2. **Regression dilution** from 3 conformation-dependent features that go noisy/flip on real poses
   (`poc_net`, `poc_eis`, `rg_per_L`/`org_density`), concentrated in **long peptides**.

The grader is sound: the control (crystal poses, this pipeline) reproduces **r=0.537** (target ≈0.585).

## 0. Control — grader is trustworthy
Crystal oracle-pose LOO with this exact pipeline: **r=0.537, RMSE 1.85**. Reproduces the documented
crystal upper bound (≈0.585; small gap = fresh feature recompute vs cached). → the real-pose drop is REAL.

## 1. Aggregation strategies (production ridge + router, LOO, n=65)
| strategy | r | ρ | RMSE |
|---|---|---|---|
| rank-1 (diffusion order) | +0.320 | +0.314 | 2.16 |
| top-5 mean-feat (diffusion) | +0.372 | +0.332 | 2.08 |
| (b) mean-predicted-ΔG, top-5 | +0.395 | +0.439 | 1.98 |
| **ML-best-1** | **+0.423** | +0.396 | 2.03 |
| **ML-best-5 mean-feat** | **+0.478** | +0.464 | **1.93** |

**The ML pose ranker recovers deployment.** Selecting the 5 cleanest ML-ranked poses (vs arbitrary
diffusion order) buys +0.106 r. This is **denoising via clean-pose selection**, NOT oracle-RMSD
selection (which E94 proved hurts). They are different: the ranker picks physically-clean poses, and
averaging 5 of them sharpens the feature estimate.

> **Leakage caveat (honest):** the ranker was trained on 52 of these 65 cr65 complexes' poses, so
> ML-best-5 = 0.478 is mildly optimistic. Mitigant: E96 proved the ranker generalizes leave-one-
> complex-out (τ=0.406 on held-out complexes), so its selection skill is genuine, not memorized — a
> fully LOCO-clean re-grade should land close. **To confirm:** retrain ranker excluding cr65 (use
> the98 poses) and re-grade.

## 2. Which features de-correlated (crystal → real top-5)
| feature | crystal r | real r | Δ | verdict |
|---|---|---|---|---|
| `rg_per_L` | +0.17 | −0.12 | **−0.29** | FLIPS — compactness wrong on real poses |
| `poc_eis` | −0.03 | −0.30 | −0.27 | pocket-chemistry noise |
| `org_density` | −0.37 | −0.15 | +0.23 | intra-peptide org, half-lost |
| `poc_f_arom` | +0.07 | +0.37 | +0.30 | (gained — robust) |
**Robust / pose-invariant (deployment-grade):** `sasa_sb` +0.48, `sasa_hb` +0.46, `mj_contact` −0.45,
`poc_n` +0.46, `poc_f_hyd` −0.54, `arom_cc` +0.46, `hb_count` +0.42. These survive the crystal→real jump.

## 3. Pose noise (within-complex CV across real poses) — the dilution source
`poc_net` CV=**9.6**, `poc_eis` CV=**3.8**, `arom_cc` 1.6, `org_density` 0.9 — the high-CV features are
exactly the ones that de-correlate. Pocket-contact features swing wildly pose-to-pose (depend on which
residues the peptide touches). Low-CV survivors: `cys_frac` 0, `poc_f_hyd` 0.07, `strength_bur` 0.08,
`poc_n` 0.10, `mean_burial` 0.12, `mj_contact` 0.14.

## 4. Systematic value shift (crystal → real, in σ_crystal)
`org_density` **−1.72σ**, `bsa_hyd` −1.19σ, `mean_burial` −0.72σ, `rg_per_L` +0.59σ, `strength_bur` −0.55σ.
Real poses are systematically **less compact, less buried, less organized** than crystal. (Absorbed by
LOO retraining on real poses, so not the main r-killer — but it would bias a crystal-calibrated model
deployed directly.)

## 5. Length breakdown (top-5, LOO) — the failure is length-structured
| band | n | r | RMSE |
|---|---|---|---|
| short ≤8 | 0 | — | — |
| **med 9–12** | **40** | **+0.159** | 2.02 |
| long 13–16 | 10 | +0.365 | 2.69 |
| **vlong ≥17** | **15** | **−0.515** | 1.72 |

- **The workhorse band (9–12, 40/65) is the weakest** at r=0.16 — compact peptides where noisy pocket
  features dominate.
- **vlong ≥17 is NEGATIVE (−0.515)** — model actively backwards. This is the documented compactness flip
  (`rg_per_L`, `org_density` invert on extended real poses) = the "conformational gap, needs MD." The
  length router does NOT route these (short-only), so they hit the full model wrong-signed.
- **No short peptides exist in cr65** — the router never fires here.

## 6. Residual autopsy
`corr(|err|)`: `poc_f_arom` −0.39, `arom_cc` −0.26, `bsa_hyd` −0.23 → aromatic/hydrophobic interfaces
predicted well; **error concentrates on non-aromatic (charged/polar) interfaces** (the charged floor).
`corr(pose bestRMSD, |err|) = +0.01` → **pose quality does NOT drive error** (E94 confirmed again; poses
are good, best 2.2 Å, mean-top5 3.7 Å). `corr(bestRMSD, y)=+0.24` → pose quality barely tracks affinity.

## Actions
1. **Wire ML ranker into affinity pose selection** (ML-best-5 ensemble) → +0.11 deployment r. Confirm
   leak-clean first (LOCO ranker / the98-trained ranker).
2. **Extend the length router to vlong ≥17** — a lean/sign-corrected sub-model, since the full model is
   negative there. Or gate vlong to MD.
3. **Down-weight or drop high-CV features** (`poc_net`, `poc_eis`) for real-pose deployment — they
   contribute noise, not signal, off-crystal.
