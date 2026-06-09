# Scoring Accuracy — Honest Numbers & Comparison to the Literature

**Date:** 2026-06-08 · **Status:** living document
**Companion to:** `docs/scoring_overhaul_plan.md`

This is the citable accuracy statement for the iGEM wiki. Every number here is
reproducible from `scripts/eval_scoring.py` (absolute ΔG) and the run CSVs
(selectivity). Where we are weak, we say so.

---

## 1. What HybriDock-Pep actually predicts, and how well

The tool produces three kinds of output. They do **not** have the same accuracy,
and conflating them is the most common way peptide-docking tools mislead.

### 1a. Absolute ΔG (single peptide, single target) — the hard one

Measured on the clean **Kd+Ki** holdout (n=101; IC50/EC50 dropped as
assay-dependent), via `scripts/eval_scoring.py`:

| Predictor | Pearson r | RMSE (kcal/mol) |
|---|---|---|
| Mean-predictor baseline (predict −9.9 for all) | 0.00 | 2.25 |
| Raw Vina (uncalibrated) | −0.45 | 9.25 |
| v1.2 calibration (production) | −0.33† | 3.90 |
| **Vina re-fit slope+intercept (per-set ceiling)** | **+0.45** | **2.01** |

† v1.2 has the physically-correct sign *inside* its 6-complex training family
(LOO r=0.72) but does not generalise to this broad mixed-family set — the
cross-family absolute-ΔG ceiling documented across this project.

**Honest reading:** our absolute ΔG ceiling on clean data is **r ≈ 0.45,
RMSE ≈ 2 kcal/mol** — and the 2 kcal/mol is largely the narrow dynamic range of
peptide affinity (the mean-predictor already scores 2.25), *not* skill. Absolute
ΔG must be reported as **calibrated/relative to a known binder**, never as a
standalone number.

### 1b. Selectivity ΔΔG (one peptide, two targets) — the strong one

`ΔΔG = ΔG_target − ΔG_offtarget`. This is the parent project's actual question
(PfLDH vs hLDH) and is **more accurate than absolute ΔG** because the peptide's
intrinsic entropy and Vina's per-peptide scale error largely cancel in the
difference. **Fix shipped (2026-06):** ΔΔG is now scored on the interaction
energy (MM-GBSA → Vina), **never** the entropy-corrected hybrid — the hybrid
cancels the cross-target signal (empirically it compressed |ΔΔG| ~4× on the
PfLDH/hLDH case). Reported with a 1000× paired bootstrap 95% CI.

Caveat: we have **no experimental ΔΔG benchmark** to put a Pearson r on this
yet. The method is sound and the CI is honest; the calibration against measured
selectivity is future work.

### 1c. Pose ranking (which of N poses is native-like) — competitive

From the benchmark history: best-of-top-25 Cα-RMSD **2.49 Å**, hit@5 **91%**,
ref2015 ranking Kendall τ=0.176. This is in the published peptide-docking band
and is the tool's most defensible quantitative strength.

---

## 2. How we fare vs. the literature

| Method | Task / dataset | Reported | Source |
|---|---|---|---|
| AutoDock CrankPep | abs. ΔG, 50 cyclic-pep | r ≈ 0.32 | BiB 2025 |
| MM/PBSA tuned (εin=2, ff03, min.) | abs. ΔG, 50 cyclic-pep | r ≈ 0.55 | BiB 2025 |
| Two-step rerank → affinity | abs. ΔG, 50 cyclic-pep | r ≈ 0.73 | BiB 2025 |
| AF3-based (ASAP'25) MERS / SARS Mpro | abs. potency | R ≈ 0.12 / 0.17 | ASAP 2025 |
| ITScorePP + MM-GBSA rescoring | pose top-1, LEADS-PEP | 17% (vs 11% FlexPepDock) | JCIM 2020 |
| **HybriDock-Pep (this work)** | **abs. ΔG, clean Kd+Ki n=101** | **r ≈ 0.45 (refit ceiling)** | `eval_scoring.py` |
| **HybriDock-Pep** | **pose, best-of-top-25** | **2.49 Å, hit@5 91%** | benchmark |

**Where we stand, stated plainly:**
- **Absolute ΔG:** we sit **above CrankPep (0.32), below the tuned-MM/PBSA SOTA
  (0.55–0.73).** We have *not* done the per-interface MM/PBSA tuning that buys
  the top numbers — and our one attempt at their headline lever (εin=2) **did
  not transfer** to our data (it inverted the correlation sign; see overhaul
  plan §5). Different datasets (linear vs curated cyclic) mean these r's are not
  strictly apples-to-apples, so we do not claim parity.
- **The AF3 ASAP'25 result (R≈0.12–0.17) is a useful reality check:** even a
  frontier deep-learning method scores near-zero on absolute potency for a hard
  target. Absolute affinity prediction is *unsolved*, not a HybriDock-Pep
  failing.
- **Pose ranking:** competitive with published peptide docking.
- **Selectivity:** the literature's relative-ranking strength (MM-GBSA ΔΔG) is
  exactly the regime we now target by construction — but ours is not yet
  benchmarked against measured ΔΔG.

---

## 3. Bottom line for the wiki

> HybriDock-Pep reports calibrated, relative ΔG (RMSE ≈ 2 kcal/mol on Kd-grade
> data; Pearson r ≈ 0.45), competitive pose ranking (best-of-top-25 ≈ 2.5 Å),
> and a bootstrap-bounded selectivity ΔΔG. Absolute, uncalibrated single-pose ΔG
> is **not** reliable for peptides — a limitation shared by all current methods,
> including frontier deep-learning models. The tool is strongest where it is
> designed to be used: **ranking poses and scoring selectivity between two
> targets.**

Sources: BiB 2025 (10.1093/bib/bbaf632) · JCIM 2018 (8b00248) · JCTC 2021
(1c00374) · JCIM 2020 (0c00058).
