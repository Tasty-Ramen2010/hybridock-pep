# Peptide ΔG (kcal/mol): World-Survey, Experiments, and Self-Rebuttal

**Date:** 2026-06-10 · branch `phase-scoring/selectivity-and-entropy-fixes`
**Question we are actually answering:** can HybriDock-Pep's v1.2 hybrid score be
calibrated to a *trustworthy* cross-complex ΔG in kcal/mol — and if so, what is
the correct enthalpy/entropy decomposition to get there?

This document is research + a battery of proposed experiments + explicit
counter-arguments to each. It is meant to be read alongside
`docs/scoring_overhaul_verdict.md` (our own 65-complex finding) — which is the
single most important prior, because it already diagnosed the disease.

---

## 1. How the rest of the world gets peptide kcal/mol — and how well it works

| Class | Representative tool | Mechanism | Best reported accuracy | What it actually buys |
|---|---|---|---|---|
| End-point MM/PBSA·GBSA | Amber `MMPBSA.py`, gmx_MMPBSA | ΔE_MM + ΔG_solv − TΔS over MD/min ensemble | **rp ≈ 0.75** short peptides (εin=2, ff99); 0.735 medium peptides (GBOBC1, ff03) — Chen et al. *MM/GBSA part 9*, the canonical protein-peptide benchmark | Within-set ranking; **per-size-class tuning required**; MUE still 4–21 kcal/mol absolute |
| End-point + interaction entropy | IE (Duan-Gao-Zhang 2016) | −TΔS = kT·ln⟨e^{βΔE_int}⟩ from interaction-energy *fluctuations*; free (no NMA) | Comparable-or-better than truncated-NMA in *MM/GBSA part 7* (1500 systems); NMA usually **worsens** cross-system accuracy | Cheap entropy that doesn't degrade the rank the way NMA does |
| Alchemical (gold std) | ABFE/FEP, FEP-SPell-ABFE (2025), Boltz-ABFE | Thermodynamic cycle, explicit solvent | Boltz-ABFE rp=0.95, MAE 0.42 kcal/mol on a *protein-ligand* set; classic ABFE ~1 kcal/mol **within a congeneric series** | Relative ΔΔG in a series; ~1000× too slow for 100-pose screening; peptides converge poorly (slow backbone DOF) |
| Contact / empirical | **PRODIGY** (protein-protein), PRODIGY-LIG, FoldX | Count interfacial contacts by chemical type (polar/apolar/charged) + non-interacting-surface (NIS) area | PRODIGY **rp = 0.73, RMSE 1.89 kcal/mol** on the PP benchmark | Cheap, structure-only; the one model where *contact composition* (not raw size) carries signal |
| Knowledge-based potentials | DFIRE, ITScorePP, ADCP/AD4 | Statistical pair potentials | Modest, dataset-dependent | Pose ranking; weak ΔG calibration |
| Rosetta | FlexPepDock + InterfaceAnalyzer, flex-ddG | Reweighted REU, ensemble ddG | **r = 0.59** on a 20-peptide influenza series (relative); REU ≠ kcal/mol | Relative within a series; needs per-series refit |
| Deep learning, structure | Boltz-2 (2025), BA-Pred, CORDIAL | Learned interface representation | Boltz-2 rp **0.66** on 4-target FEP subset; **>0.55 on only 3/8 assays**; BA-Pred RMSE ~2.0 kcal/mol (protein-ligand) | SOTA, but variance "typical of FEP"; data-hungry |
| Deep learning, MHC-peptide | NetMHCpan, structure-aware MHC-II | Learned, huge labelled sets | High **within MHC** | Only works because MHC has 10⁴–10⁵ labels; not general peptide ΔG |

### The pattern that matters
Nobody gets a reliable *blind, cross-family, single-shot absolute* peptide ΔG.
Every headline number is one of three things:
1. **within a size/structure-coherent set** (MM/GBSA part 9 tunes εin per peptide-size class),
2. **relative within a congeneric series** (FEP, FlexPepDock r=0.59, flex-ddG), or
3. **contact-composition, not size** (PRODIGY 0.73 — and it explicitly adds the
   non-interacting surface term precisely to *break* the size scaling).

SOTA ML (Boltz-2) lands at 0.66 and is honest that it fails on >half of assays.
So a target of "r≈0.55 cross-family from cheap features" is **at or above the ML
state of the art** — which should make us suspicious of any quick win.

---

## 2. The physical anchor the user is right about

Peptide binding *is* an enthalpy–entropy balance, and the entropy is the hard,
size-scaling half:

- **Conformational entropy loss on binding ≈ 0.7 kcal/mol per residue** for a
  flexible peptide (configurational-entropy study, Tsg101–PTAP). Receptor side
  is ~0.1 kcal/mol/res — negligible by comparison.
- **Secondary-structure dependence:** helix vs sheet residues differ by
  ~0.5 kcal/mol/res (helical residues move less, lose less). This is *exactly*
  why we carry `s_ss_weighted`.
- Translational/rotational loss on the order of a fixed ~few kcal/mol per binding
  event (cratic + rigid-body), roughly constant across peptides.

**The smoking gun in our own calibration:** production `alpha = 0.1` kcal/mol per
contact-residue is **~7× smaller than the physical 0.7**. That is not a model of
entropy. It is a fudge factor calibrated to whatever value made the (backwards)
Vina size-trend cancel on PepSet-6. We have been calling a size-correction
"entropy." The verdict doc proves it: forcing the physically-correct Vina sign
gives *negative* held-out r.

This reframes the whole task. The job is **not** "find a better entropy term."
It is "stop letting enthalpy and entropy both collapse onto interface size, and
calibrate each against its correct physical reference."

---

## 3. Why our numbers have been a mirage (restating the verdict in physics terms)

On the 65-complex crystal set, every cheap feature is the *same vector*:
Vina ↔ n_contact r = −0.877; Vina, AD4, n_contact, s_ss all ≈ one size–burial
axis. In this sample bigger peptides happen to bind weaker, so:

- enthalpy proxy (Vina) ∝ +size,
- entropy proxy (α·N, s_ss) ∝ +size,
- experimental ΔG ∝ −size (sampling accident).

A linear fit "wins" only by riding the −size accident. **Enthalpy and entropy are
collinear, so the balance the user wants to strike is unidentifiable** — you
cannot fit two coefficients to two vectors that are the same vector. This is the
root cause, and it is a *data geometry* problem, not a missing-feature problem.

---

## 4. Experiments — each with its own rebuttal

> Ordering principle: **E0 is a go/no-go gate.** Do not run E1–E5 until E0 says
> there is any signal left after you remove size. Everything we've shipped that
> looked like 0.55 died at this gate retroactively.

### E0 — Length-residualization gate (the decision experiment)
- **Hypothesis:** after regressing both ΔG_exp and every feature on N_res (and on
  buried-SASA), a non-trivial partial correlation survives.
- **Method:** on the 65-set, compute residuals `ΔG_exp ⟂ N_res` and
  `feature ⟂ N_res`; report partial Pearson + Spearman. Repeat controlling for
  buried interface area instead of raw length.
- **Pass bar:** |partial r| ≥ 0.3 for at least one feature, *stable in sign*
  under family-LOO.
- **Rebuttal to myself:** this may simply confirm the ceiling (partial r ≈ 0),
  in which case the honest deliverable is relative-to-reference ΔG, not absolute.
  That is a *result*, not a failure — and it's cheap to get (minutes, no GPU).

### E1 — Physics-anchored two-term model with *fixed* entropy
- **Hypothesis:** if entropy is pinned to physics (0.7·N_res, helix/sheet-scaled
  via s_ss) instead of fit, the residual enthalpy term calibrates to a sane,
  positive Vina slope and generalizes.
- **Model:** `ΔG = a·(Vina or MM-GBSA enthalpy) + b − 0.7·Σ_i w_ss(i)` with the
  entropy term *fixed, not fitted*; fit only `a > 0` (sign-constrained) and `b`.
- **Pass bar:** held-out r > 0 *with a > 0*. We never need 0.55 — we need a model
  whose enthalpy slope points the physically correct way and still ranks.
- **Rebuttal:** if entropy and enthalpy are collinear (§3), fixing entropy just
  re-injects +size, and the sign-constrained enthalpy fit will go to a≈0. Likely
  outcome: this proves the two terms are not separately identifiable on this data
  — which tells us we need length-balanced data (E4), not a cleverer formula.

### E2 — PRODIGY-style contact-type model (the one new feature worth adding)
- **Hypothesis:** contact *composition* (charged/charged, polar/apolar counts) +
  NIS carries signal orthogonal to raw size — it's the only cheap feature class
  that beat size for protein-protein (0.73).
- **Method:** compute IC_charged/charged, IC_polar/polar, IC_apolar/apolar, IC
  mixed, and %NIS_polar/%NIS_charged at each pose's interface (Biopython, no new
  heavy dep). Fit the PRODIGY linear form; evaluate after E0-style residualization.
- **Pass bar:** the contact-type composition (not total IC) retains |partial r| ≥
  0.3 vs N_res in family-LOO.
- **Rebuttal:** PRODIGY was trained/validated on *protein-protein* with rigid,
  large interfaces. Peptide interfaces are small and dominated by a few hotspot
  contacts; the contact-type statistics may be too sparse (n=5–15 residues) to be
  stable. Also our poses are *docked*, not crystal — contact-typing amplifies
  pose error. Mitigate by typing on the top cluster centroid only.

### E3 — Interaction Entropy, but only where it's free and as a *signed* term
- **Hypothesis:** −TΔS_IE varies per-complex with interface *floppiness*, not size,
  so it can break collinearity where α·N cannot.
- **Status from verdict:** already tested and rejected on two grounds — (a)
  impractical in this WSL2/OpenMM env (a single complex didn't finish in 6 min;
  GBn2 system falls back to CPU), and (b) −TΔS_IE ∝ interaction-energy variance ∝
  interface size, i.e. *same dead axis*.
- **Only-if revival:** the IE module exists and is unit-tested. Revisit **only**
  if E0 shows a floppiness signal that survives size-residualization AND we move
  Stage-2 trajectory sampling to the real CUDA box (not WSL2). Otherwise leave it
  as the research-only `--mmgbsa-ie` flag.
- **Rebuttal to reviving it:** even if cheap, the verdict's variance∝size argument
  stands; don't relitigate without E0 evidence of an independent floppiness axis.

### E4 — Build a length-balanced, structurally-resolved benchmark
- **Hypothesis:** the ceiling is a property of *our 65-set's size confound*, not
  of peptides. A benchmark stratified to be flat in N_res (e.g. 5 length bins ×
  equal counts, ΔG spread within each bin) would let enthalpy and entropy
  separate.
- **Method:** mine PDBBind peptide subset + PepBDB + the ITC-curated sets we
  already touched (PEPBI 329, Wang) for complexes where ΔG varies *within* a
  fixed length bin. Target ≥ 15 complexes per bin, decorrelated.
- **Pass bar:** within-bin Pearson of any feature ≥ 0.4 (this is the real test of
  whether physics is recoverable once size is held fixed).
- **Rebuttal:** we have walked this road (284-set, PEPBI, Wang) and the ITC labels
  carry ~7 kcal/mol noise and cross-assay systematics; curation may not yield 15
  clean per-bin complexes. This is the highest-effort, highest-payoff item and the
  *only* one that can genuinely raise the ceiling — but it is a data project, not
  a coding project, and should be scoped honestly.

### E5 — Relative-to-reference ΔG (the defensible product, ship regardless)
- **Hypothesis:** calibrating each prediction against one known binder per target
  cancels the additive size/cratic terms and yields trustworthy ΔΔG, which is what
  the parent iGEM use-case (PfLDH selectivity) actually needs.
- **Method:** `ΔG_pred(pep) = ΔG_known + [score(pep) − score(known)]` with the
  slope from a *local* (per-target) 2–3 point anchor when available. Report a CI
  and the size-confound caveat.
- **Pass bar:** ΔΔG sign accuracy and within-target ranking — both of which the
  verdict already shows survive (selectivity is where the confound cancels).
- **Rebuttal:** requires a known binder per target; for a truly novel target with
  no reference, we fall back to "ranked poses + uncalibrated score," and we must
  *say so* rather than print a false absolute number. That honesty is the feature,
  not a limitation — it's exactly how FEP and PRODIGY-grade tools report.

---

## 5. Recommended path

1. **Run E0 today** (cheap, no GPU). It is the gate. Output: do any features carry
   ΔG signal independent of peptide size, with stable sign under family-LOO?
2. **If E0 passes** → run **E2** (contact-type/NIS) and **E1** (fixed-physics
   entropy). These are the two principled ways to exploit a surviving signal. Add
   the PRODIGY contact-type features to the feature set if and only if they pass
   the residualized, leakage-controlled held-out test from the verdict.
3. **If E0 fails** (most likely, per all prior evidence) → **stop chasing absolute
   cross-family ΔG.** Ship **E5** (relative-to-reference ΔG with CIs) as the
   honest product, keep pose-ranking + ΔΔG-selectivity as the headline claims, and
   open **E4** as a separate, clearly-scoped data-curation effort — the only lever
   that can move the ceiling.
4. **Do not** re-enable IE/MM-GBSA on the production path (E3) without E0 evidence
   of an independent floppiness axis *and* a move off WSL2/OpenMM-CPU.

### The one-line honest framing for the iGEM writeup
> "Absolute peptide ΔG from cheap structure features is capped near the size
> confound — SOTA ML (Boltz-2) only clears r=0.55 on 3/8 assays — so HybriDock-Pep
> reports *calibrated relative* ΔG against a reference binder, plus rank-validated
> poses and ΔΔG selectivity, rather than an over-confident absolute number."

That is a stronger, more defensible claim than a fragile 0.55 that flips sign on
the judges' prospective test case.

---

---

## Appendix — E0/E1/E2 results (run 2026-06-10, this session)

All on the 65-complex crystal benchmark, family-grouped GroupKFold.

**E0 (length-residualization gate, `experiments/e0_residualization_gate.py`):** every
legacy feature (vina, mmgbsa, n_contact, bsa, contact-type counts) dies after
removing length OR survives only as Vina's backwards-sign size artifact. The ONE
survivor of the raw gate: **NIS composition** (`nis_p_frac`/`nis_c_frac`, PRODIGY's
non-interacting-surface). Kd-only partial|L: nis_p −0.63, nis_c +0.60.

**E0b/E0c (`e0b_nis_grouped_cv.py`, `e0c_sign_and_ci.py`):** `nis+vina` hits
0.56/0.61 grouped-oof — but fits Vina at weight −1.26 (physically backwards). Force
the correct Vina sign → collapses to 0.11/0.36. The 0.55 is the size artifact.

**E1 (`e1_nis_vs_bsa.py`):** NIS is orthogonal to BSA (corr −0.05). NIS+BSA reached
0.43 Kd-only — looked like a genuine lift past the documented ~0.28 ceiling.

**E2 (`e2_features.py`, `e2_model.py`, dedup diagnostic) — the rigorous kill:**
With length-residualization INSIDE each fold:
- **BSA flips to −0.50/−0.56** — the handoff's "+0.28 size-controlled" used leaky
  full-set partialling; BSA is the size axis.
- Stacking 12 orthogonal physics axes (area-%NIS, hb_density, salt bridges,
  hydrophobic burial, buried-unsat-polar) DILUTES: z-blend 0.18 < NIS-alone 0.44.
- **NIS dedup test:** Kd-only +0.44 → ALL-minus-largest-family −0.07 (n=48) →
  **one-per-family +0.065 (n=20)**. NIS's 0.44 was intra-family variation in one
  dominant family. **Cross-family, NIS ≈ 0.**

### Final verdict (8th independent confirmation)
No cross-family-generalizable cheap feature for **absolute** peptide ΔG exists,
including NIS. Every number above ~0.1 is leakage: size confound (BSA, MM-GBSA),
backwards-Vina sign trick (the 0.55), or intra-family variation (NIS 0.44,
per-family 0.65). Honest cross-family r ≈ 0.07. **The bottleneck is data geometry
(size confound + near-duplicate families + n), not features.**

### What IS real and shippable
NIS carries genuine **within-target** signal (~0.4) — i.e. ranking peptide variants
against ONE fixed receptor (relative ΔΔG, the iGEM PfLDH selectivity use-case).
Ship: absolute cross-family ΔG = uncalibrated/relative with the ceiling stated;
within-target variant ranking = NIS-based, ~0.4, with CIs. Do NOT ship an absolute
0.55 — it is the backwards-Vina size artifact and inverts on novel targets.

## Sources
- MM/GBSA part 9 (protein-peptide): https://pubmed.ncbi.nlm.nih.gov/31062799/
- MM/GBSA part 7 (entropy effects, IE vs NMA): https://pubs.rsc.org/en/content/articlelanding/2018/cp/c7cp07623a
- Interaction Entropy (Duan-Gao-Zhang 2016): https://pubs.acs.org/doi/abs/10.1021/jacs.6b02682
- IE assessment (JCTC 2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8389774/
- PRODIGY (contacts + NIS, r=0.73): https://academic.oup.com/bioinformatics/article/32/23/3676/2525629
- Peptide configurational entropy ~0.7 kcal/mol/res: https://pmc.ncbi.nlm.nih.gov/articles/PMC2758778/
- Conformational entropy / SS dependence: https://www.pnas.org/doi/abs/10.1073/pnas.1407768111
- FlexPepDock relative affinity (r=0.59): https://pubmed.ncbi.nlm.nih.gov/36512534/
- Boltz-2 (rp 0.66; 3/8 assays > 0.55): https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full.pdf
- ABFE peptide (entropic minNLS): https://pmc.ncbi.nlm.nih.gov/articles/PMC12020361/
