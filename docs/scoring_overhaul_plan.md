# Scoring Accuracy Overhaul — Plan & Rationale

**Status:** PLAN (no code changes yet — approved for write-up, awaiting go for execution)
**Author:** Dry Lab, drafted June 2026
**Scope:** Rebuild peptide ΔG scoring around what is physically achievable and
literature-validated. Covers all three remediation tiers + a full physics
upgrade to MM-GBSA. Touches `scoring/`, `driver.py`, calibration, and docs.

> Gate reminder (CLAUDE.md §7 "Before changing the entropy correction formula"):
> re-read spec §5 and §8, re-calibrate, re-benchmark Pearson r + RMSE
> before/after, commit the calibration JSON alongside the code change.

---

## 1. Problem statement (measured, not asserted)

Two complaints triggered this work: (a) "entropy has too much weight," and
(b) "entropy always favors one direction." Investigation found a deeper issue.

### 1.1 The entropy term is single-signed in every code path
- Legacy (`apply_hybrid_score`): `+α·n_contact`, α>0 → pure penalty, monotone
  in contact count, can never reward.
- Ridge v1.4 (`apply_hybrid_score_ridge`): `−0.34·s_ss_weighted`, s_ss≥0 →
  always favorable, monotone.

Neither can be `+` for one residue and `−` for another. Real binding entropy
is a **difference of competing terms** (configurational loss on rigidification,
`+`; hydrophobic desolvation / water release, `−`) whose net sign varies per
residue and per burial state. The complaint is correct: the model cannot
represent this.

### 1.2 The live calibration (v1.4) is a regression
`cli.py` defaults `--calibration` to `data/calibration_v1_4_balanced.json`.
Leave-one-out CV on its own PepSet-6 training set:

| Calibration | in-sample r | **LOO r** | LOO RMSE |
|---|---|---|---|
| v1.2 (entropy ridge) | 0.934 | **0.715** | 1.51 |
| **v1.4 (LIVE)** | 0.854 | **0.303** | 2.41 |

v1.4's LOO r=0.303 trips the §9 stop-and-flag line ("Pearson r < 0.35 after
full calibration"). It should not be the production default.

### 1.3 Absolute ΔG is at a hard data ceiling
On the 240-complex holdout (`data/eval_holdout_calibrations.json`):

| Predictor | Pearson r | MAE | RMSE (kcal/mol) |
|---|---|---|---|
| Mean-predictor baseline (predict −9.2 for all) | 0.00 | — | **2.29** |
| Raw Vina (uncalibrated) | −0.26 | 5.9 | 8.8 |
| v1.4 (live) | −0.24 | 8.8 | 11.9 |
| Vina re-fit slope+intercept (in-sample ceiling) | +0.26 | 1.8 | **2.21** |

Kd-only subset (n=56): re-fit ceiling r=+0.475, RMSE=2.17; mean-predictor 2.47.

**Key truth about "~2 kcal/mol off":** the number is real but is *almost
entirely the narrow dynamic range of peptide affinities* (ΔG clusters in a
~2.3 kcal/mol band around −9.2), not predictive skill. A constant
mean-predictor already scores RMSE 2.29. The honest limiting metric is
Pearson r (0.26 mixed / 0.47 clean-Kd). The raw uncalibrated scale is 6–12
kcal/mol off because Vina was fit for drug-like small molecules, not
50-rotatable-bond peptides with huge buried interfaces.

### 1.4 Why physics scoring fails here (root causes)
1. **Enthalpy–entropy compensation.** ΔG (−6 to −10) is a small difference of
   two large opposing terms (enthalpy −tens; entropy +tens). 5% error in
   either swamps the answer. Fundamental, not a coding bug.
2. **Vina mis-scaled for peptides.** Flat `Nrot` flexibility term saturates;
   score scales with buried interface → −10 to −15 for µM binders.
3. **OpenMM minimization captures zero entropy** — it finds one enthalpic
   minimum, samples no ensemble. The dominant peptide-binding term is invisible
   to it. (`Stage 1.5` minimization is a geometry/clash fixer, not a ΔG method.)
4. **Single-trajectory MM-GBSA has no entropy + a reference-state bias** — it
   reads "peptide_alone" energy from the bound, ordered geometry; a free
   peptide is disordered.
5. **Noisy labels + narrow range.** Holdout = 107 IC50 + 45 Ki + 56 Kd + 32
   EC50; IC50/EC50 are assay-dependent. Several structures are Rosetta models
   (~10 kcal/mol systematic bias).

---

## 2. What the literature does (and the realistic ceiling)

| Method | Dataset | Pearson r | Source |
|---|---|---|---|
| AutoDock CrankPep | 50 cyclic-pep | 0.32 | BiB 2025 |
| **MM/PBSA tuned (εin=2, ff03, minimized)** | 50 cyclic-pep | **0.55** | BiB 2025 |
| **Two-step rerank → affinity** | 50 cyclic-pep | **0.73** | BiB 2025 |
| ITScorePP + MM-GBSA rescoring | LEADS-PEP | 17% top-1 (vs 11% FlexPepDock) | JCIM 2020 |

**Nobody beats r ≈ 0.73 on curated data.** The levers that moved CrankPep's
0.32 → 0.73 are mostly replicable with our OpenMM stack:

- **Internal dielectric εin = 2** (not 1) — single biggest knob; εin 1.4–2.0
  consistently best (JCIM 2018, 8b00248).
- **Energy minimization beats short MD** — short MD *degraded* their results.
  Validates our minimization-only design.
- **No explicit entropy needed for r=0.73** — but the **Interaction Entropy
  (IE)** method (Duan/Gao/Zhang; JCTC 2021 1c00374) gives cheap, *signed*,
  per-complex entropy from energy fluctuations on the same trajectory, beating
  normal-mode. This is the principled fix for the §1.1 single-sign problem.
- **Two-step workflow** (rerank poses → compute affinity on best) — already our
  architecture (RAPiDock → rescore → `--refine-topk`).
- **Combined knowledge-based + MM-GBSA** — we have **ref2015** (τ=0.176) as the
  knowledge-based half.

**Current-state gap:** `scoring/mmgbsa.py` uses GBn2 with **εin = 1.0 (OpenMM
default), minimization-only, no entropy.** We are one dielectric constant and
one entropy term away from published best practice.

Sources:
- https://academic.oup.com/bib/article/26/6/bbaf632/8361798
- https://pubs.acs.org/doi/10.1021/acs.jcim.8b00248
- https://pubs.acs.org/doi/10.1021/acs.jctc.1c00374
- https://pubs.acs.org/doi/10.1021/acs.jcim.0c00058

---

## 3. Plan

### Phase 0 — Honest test harness (no scoring changes)
- Build a **Kd+Ki-only** evaluation subset from the holdout (drop 139
  IC50/EC50 entries → Tier 3 data hygiene). Persist as
  `data/eval_kd_ki_clean.json`.
- Lock metrics: **Pearson r** (primary) + **per-target-anchored RMSE**
  (secondary). Every later change is judged against this frozen set.
- **Success:** reproducible script `scripts/eval_scoring.py` that prints r/MAE/
  RMSE for any predictor; baseline numbers committed.

### Phase 1 — Tier 2 physics upgrade (replicable SOTA)
1. **Expose & screen internal dielectric `εin`** in `mmgbsa.py`
   (currently hard-defaulted to 1.0). Add `soluteDielectric` param; screen
   εin ∈ {1, 2, 4} on the Kd+Ki set; pick by Pearson. *Expected: εin=2 best.*
2. **Interaction Entropy (IE) term** — short (200–500 ps) OpenMM Langevin run
   around each minimized top-K complex; collect interface interaction-energy
   fluctuations ΔE_int; compute −TΔS = kT·ln⟨e^{βΔE_int}⟩. Signed, per-complex,
   reuses existing OpenMM stack. Gated behind `--refine-topk` so the fast path
   is unaffected. **This is the §1.1 fix.**
3. **Three-trajectory MM-GBSA option** — minimize unbound peptide (and receptor)
   separately to remove the §1.4(4) reference-state bias. `--mmgbsa-3traj` flag;
   more compute, only when the user wants max accuracy.
- **Success:** Pearson r on Kd+Ki set improves measurably over current
  MM-GBSA; target r ≈ 0.5–0.6 (honest, not magic). Document the lift per lever.

### Phase 2 — Tier 1 reframe (cheap, scientifically correct)
- **ΔΔG selectivity as the headline metric** (PfLDH vs hLDH). The cancellation
  makes it the most accurate output we can produce *and* the parent project's
  actual need. Validate ΔΔG accuracy on any available paired data.
- **Per-target intercept calibration**: report ΔG anchored to ≥1 known binder
  for the target. Turns "±2 kcal/mol" into a defensible claim.
- **Explicit two-step pipeline**: ref2015 (rank) → tuned-MM-GBSA (affinity on
  top cluster centroids), mirroring ITScorePP+MM-GBSA.
- **Success:** selectivity ΔΔG reported with CI; per-target ΔG within ~2
  kcal/mol of anchor on validation targets.

### Phase 3 — Tier 3 hygiene + honest reporting
- **Revert production default** v1.4 → v1.2 (or the new Phase-1 calibration
  once validated). One-line CLI change; stops the §1.2 regression.
- **Config hygiene**: calibration loader logs which JSON is active at INFO;
  collapse the 13 stray `data/calibration_*.json` files; document the canonical
  one.
- **`docs/scoring_accuracy_analysis.md`** for the wiki: cite *calibrated/
  relative ΔG, RMSE ≈ 2 kcal/mol on Kd-grade data, r ≈ 0.5; absolute
  uncalibrated ΔG unreliable for peptides*, with the r≈0.73 literature ceiling
  as honest context.
- **Success:** fresh-install run prints active calibration; wiki doc passes a
  factual self-review against the committed numbers.

---

## 4. Risks & stop-flags (tie to CLAUDE.md §9)
- If Phase-1 physics does **not** lift Pearson r above ~0.4 on clean Kd data,
  stop and re-scope to ranking + ΔΔG only — do not ship a worse absolute number
  dressed up.
- Do **not** double-count desolvation: Vina/AD4 already carry a hydrophobic
  term; favorable desolvation must come from the GB solvation term, not a
  hand-added reward (CLAUDE.md §7).
- Do **not** recompile/fork Vina (spec §5.6/§5.7, CLAUDE.md §2.1). The data
  shows the problem is the data ceiling + entropy, not Vina's source.
- Re-calibrate α / ridge weights after any entropy-formula change; commit the
  JSON in the same change.

## 5. Decision log
- **AD4**: demoted to opt-in (`--scoring vina,ad4`), not removed. Production =
  Vina + entropy. Keep as-is.
- **εin**: now a tunable param in `mmgbsa.py` (was hard-locked to 1.0).
  **Screen result (2026-06, `scripts/screen_dielectric.py`, n=4 usable):**
  εin=1.0 → r=+0.579 (correct sign); εin=2.0 → r=−0.685; εin=4.0 → r=−0.667.
  **The literature's εin~2 did NOT transfer** — it inverted the correlation
  sign on our data. **Verdict: KEEP εin=1.0.** Caveats making this inconclusive
  rather than a finding: (a) n=4 after 1nrl/1ywi failed on crop-boundary residue
  breakage; (b) the speed-crop removes charged residues, confounding the very
  electrostatics εin scales. A definitive screen needs full receptors on a
  larger structurally-resolved Kd set (deferred — no docked structures for the
  101-complex clean set).
- **Entropy**: replace single-sign scalar with IE-method signed entropy
  (`scoring/interaction_entropy.py`, Phase 1.2 — DONE, 6/6 unit tests).
  Resolves the original complaint with published physics. Still to wire into
  `refine_topk_poses`.
- **Headline metric**: shift from absolute single-pose ΔG → ΔΔG selectivity +
  per-target-anchored ΔG + pose ranking.

## 6. Session log (2026-06-08)
- Phase 0 DONE: `scripts/eval_scoring.py` + `data/eval_kd_ki_clean.json` (n=101).
  Clean-set baseline: Vina r=−0.45, refit-ceiling r=+0.45 / RMSE 2.01,
  mean-predictor RMSE 2.25.
- Phase 1.1 DONE (code) / INCONCLUSIVE (experiment): εin exposed + screened;
  keep 1.0 (see decision log).
- Phase 1.2 DONE: Interaction Entropy estimator + OpenMM sampler + 6 unit tests.
- Existing suite: 20/20 mmgbsa tests still green after εin edits.
- NEXT: fix crop-boundary failures OR full-receptor re-screen; wire IE into
  `refine_topk_poses` (+`--mmgbsa-3traj`); re-eval on clean set; Phase 2/3.
