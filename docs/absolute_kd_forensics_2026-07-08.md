# Absolute-Kd forensics: where we predict badly, our own pipeline bugs, and the universal cause

**Date:** 2026-07-08 · Method: forensic decomposition of the scorer error + feature inter-correlation + calibration
audit + MM-GBSA/RISM audit + 12 literature searches. Trigger: Ram — "we must be missing something universal…
look deep into our own code for errors and mis-calibrations, physics never lies."

## The universal cause (found): weak features → attenuation bias → compression

- **Every feature is individually weak.** Best univariate corr with affinity: `cys_frac −0.154`, `rg_per_L +0.125`,
  `poc_f_arom −0.121` — **nothing exceeds |0.155|.** 16 features collapse to **~12 independent** (redundant pairs:
  sasa_hb↔hb_count 0.85, poc_n↔mj_contact −0.79, bsa_hyd↔mj_contact −0.76).
- **This mathematically forces compression.** Calibration audit: `y ≈ 0.788·pred − 1.66`; **prediction std 0.84 vs
  real 1.85** — the model predicts *half* the true dynamic range. This is textbook **regression dilution /
  attenuation bias**: *"noise in X biases the slope toward zero… predictions shrink toward the mean"*
  ([Wikipedia](https://en.wikipedia.org/wiki/Regression_dilution); [ML underestimates extremes](https://arxiv.org/html/2412.05806v1)).
- **Consequence = systematic bias at the extremes** (not noise): we **under-predict tight binders by −2.08 kcal**
  and **over-predict weak binders by +1.80 kcal**; the middle (near the mean) is fine (|resid| 0.8). The literature
  calls this out for scoring functions directly: *"discernible dispersion especially at the extremes"* and
  *"predictions compressed vs the experimental range"* ([Frontiers scoring review](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2018.01089/full)).

**The "universal something" is not a missing physics term — it is the statistical signature of weak predictors.**
With max feature |corr| ≈ 0.15, no model can avoid compressing and mispredicting the extremes.

## Our own pipeline bugs (Ram was right to suspect)

1. **MM-GBSA (--ultra) is a SIZE ARTIFACT, not a desolvation term.** On our 64-complex benchmark:
   `corr(mmgbsa_dg, peptide_len) = −0.724`, and **size-normalized it is dead (−0.029)**; worse, it is
   **anti-correlated with experiment: corr(mmgbsa_dg, dg_exp) = −0.434** (scale −152..−13 vs exp −14..−6). This is
   the documented MM-GBSA size bias — *"binding free energy becomes more favorable with increasing size… a
   mathematical artifact"* ([ligand-efficiency size bias](https://pubs.acs.org/doi/10.1021/acsmedchemlett.5c00652)).
   **Answer to "doesn't --ultra solve desolvation?": no — its MM-GBSA is measuring peptide size and points the wrong
   way.** Do not trust --ultra MM-GBSA for absolute affinity.
2. **RISM desolvation-on-binding is the same trap** (E356): Δsolv comes back +192…+499 kcal — a difference of two
   huge exchem values (complex vs rec+pep), size-dominated and unconverged, exactly the catastrophic-cancellation
   problem. The 3-way RISM desolvation is not a usable absolute term at this fidelity.

## The recurring physical thread: entropy / preorganization

Everywhere we look, the signal that *does* exist is entropy/preorganization:
- **Best feature is `cys_frac`** — cysteine/disulfide peptides are cyclic/preorganized: *"the preorganized ring
  reduces entropic binding costs → greater affinity"* ([cyclic disulfide peptides](https://pmc.ncbi.nlm.nih.gov/articles/PMC10096437/)).
- **Worst-predicted tight binders are entropy-driven**: AVGIGA / AIIGLMVGGVV (hydrophobic, membrane-like) —
  *"desolvation entropy from burial of hydrophobic groups ≈ 4 kcal"* ([hydrophobic entropy JACS](https://pubs.acs.org/doi/10.1021/ja101362u)).
- **Worst-predicted weak binders are floppy** (GSSGSGSNGD) — un-modeled conformational entropy penalty.
- **A crude preorganization proxy (cys·2 + pro) is the single biggest improvement we found: r 0.339 → 0.361
  (+0.022)** — larger than de-shrink calibration or regime removal.

## Where we fail, catalogued
| cluster | n | signature | cause |
|---|---|---|---|
| tight hydrophobic/membrane | ~17 | under-pred by **−3.9 kcal** | entropy-driven hydrophobic effect; general scorer blind |
| MHC-groove peptides (ELAGIGILTV…) | few | under-pred | *"a fundamentally different regime"* needing domain-specific scoring ([Sci Rep](https://www.nature.com/articles/s41598-018-22173-4)) |
| floppy/weak binders | ~several | over-pred by **+1.8 kcal** | un-modeled conformational-entropy penalty |
| middle affinity | most | well-predicted (0.8) | near the mean, compression harmless |

## Brainstorm — fixes, ranked by measured payoff (all honest, none breaks the info ceiling)
1. **Add a real preorganization/entropy feature** — the biggest lever (+0.022 r, cheap). Cyclic/disulfide detection
   + proline + a confinement-entropy or Rg-flexibility term. This is the entropy thread made into a feature.
2. **Fix the --ultra MM-GBSA bug** — it is size-dominated and anti-correlated. Either drop it from absolute-affinity
   output, size-normalize (dead → drop), or **replace with a Δ-learning residual model** trained on the systematic
   extreme-bias ([residual/Δ-learning](https://www.biorxiv.org/content/10.1101/2020.07.29.227959.full.pdf)).
3. **Regime router / flag** — detect MHC-groove and membrane-hydrophobic peptides; route or low-confidence-flag
   them (recovers ~0.01 r + honest UX). MHC is a *known* separate regime.
4. **De-shrink calibration (LCC)** — post-hoc linear map on a held-out split to fix the extreme MAE bias
   ([post-hoc calibration](https://arxiv.org/html/2509.23665v1)). r-invariant but improves the *usable* numbers.
5. **Consensus/ensemble** — blend scorer + physics + a sequence iLM (SWING-class); ensembles reliably add a few
   points ([ensemble scoring](https://pmc.ncbi.nlm.nih.gov/articles/PMC7697539/)).

## The honest bottom line
We were **not** wrong that a wall exists, but we found **three things we were doing wrong / could improve**:
(a) our expensive --ultra MM-GBSA is a **size artifact pointing the wrong way** (a real bug), (b) the model
**compresses** because features are weak (fixable at the extremes by calibration, and the cause is now named), and
(c) the one signal that keeps surfacing — **entropy/preorganization** — is under-exploited and gives the biggest
cheap gain (+0.022) when added as a feature. The r-ceiling (~0.35) is still set by weak features + the general-
model information limit, but **the predictions themselves can be made meaningfully better** (fix the MM-GBSA bug,
add preorganization, de-shrink, flag regimes) — and that is worth shipping even though it doesn't smash the ceiling.

---
### Sources
Attenuation/regression dilution: [Wikipedia](https://en.wikipedia.org/wiki/Regression_dilution), [arXiv 2412.05806](https://arxiv.org/html/2412.05806v1), [LCC/Tweedie de-shrink](https://arxiv.org/html/2509.23665v1).
MM-GBSA size bias: [ACS Med Chem Lett](https://pubs.acs.org/doi/10.1021/acsmedchemlett.5c00652), [Chem Rev MM/PBSA](https://pubs.acs.org/doi/10.1021/acs.chemrev.9b00055).
Scoring plateau/compression: [Frontiers](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2018.01089/full), [static→dynamic](https://pmc.ncbi.nlm.nih.gov/articles/PMC11516055/).
Cyclic preorganization entropy: [PMC10096437](https://pmc.ncbi.nlm.nih.gov/articles/PMC10096437/). Hydrophobic entropy: [JACS ja101362u](https://pubs.acs.org/doi/10.1021/ja101362u).
MHC regime: [Sci Rep 22173-4](https://www.nature.com/articles/s41598-018-22173-4), [PNAS 2216697120](https://www.pnas.org/doi/10.1073/pnas.2216697120).
Ensemble/consensus: [AK-Score PMC7697539](https://pmc.ncbi.nlm.nih.gov/articles/PMC7697539/). Δ-learning: [hybrid ML/MM](https://www.biorxiv.org/content/10.1101/2020.07.29.227959.full.pdf).
