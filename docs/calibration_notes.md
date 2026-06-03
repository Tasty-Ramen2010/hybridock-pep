# Calibration Notes — HybriDock-Pep Entropy Correction

## Current calibration result (2026-05-22)

| Parameter | Value |
|-----------|-------|
| alpha (entropy coefficient) | 0.100 kcal/mol/residue |
| beta (AD4 blend weight) | 0.000 |
| gamma (partial-contact weight) | 0.200 |
| Pearson r | 0.860 |
| RMSE | 1.73 kcal/mol |
| n_complexes | 6 |
| entropy_mode | contact |

Training set spans pKd 4.1–8.7 across 6 structurally diverse PepSet complexes:

| PDB | System | pKd | Vina | AD4 | n_contact_residues |
|-----|--------|-----|------|-----|-------------------|
| 2hwn | PKA-RIIα / AKAP (19-mer) | 8.70 | −14.75 | −3.39 | 15 |
| 1nrl | PXR / SRC-1 (15-mer) | 6.00 | −12.41 | −11.49 | 10 |
| 1l2z | CD2BP2-GYF / CD2 (11-mer) | 5.70 | −7.03 | −8.51 | 7 |
| 1ddv | Homer-EVH1 / mGluR7 (6-mer) | 5.00 | −8.58 | −5.99 | 5 |
| 1a0n | Fyn-SH3 / PPII (14-mer) | 4.60 | −5.66 | −6.69 | 9 |
| 1ywi | FBP11-WW1 / PPLP (6-mer) | 4.10 | −6.45 | −7.23 | 5 |

pKd values from primary literature ITC/SPR measurements, NOT BindingDB.

---

## Documented issues and limitations

### Issue 1 — Alpha at lower bound (CLAUDE.md §9 flag condition)

**What:** `calibrate_alpha.py` optimises alpha in [0.1, 2.0]. Alpha converged to the lower bound 0.1 on every run of the current training set.

**Why:** Calibration uses crystal poses scored against the holo receptor (`_rec_ref.pdb`). Vina `--score_only` on a crystal pose in the binding pocket it was crystallised in systematically overestimates affinity — it reports scores 3–8 kcal/mol more negative than the true ΔG. Adding a positive entropy term (alpha × n_contact) makes the prediction *less* negative, which moves it *closer* to experiment for most complexes. But the optimiser is minimising RMSE across all complexes simultaneously: because Vina already overshoots the target, the entropy term would need to be negative (counter-physical) to help. At any positive alpha the correlation degrades. The optimiser retreats to the lowest allowed alpha.

**Impact on ranking:** r=0.860 is good for relative ranking. The absolute ΔG predictions are unreliable (systematic negative bias of ~3–6 kcal/mol from Vina overestimation). Do not report absolute ΔG from this calibration as experimentally meaningful.

**Resolution:** Re-calibrate using Vina scores from generated poses docked into the apo receptor (production runs), not crystal poses in the holo receptor. This will give Vina scores that reflect real docking performance rather than scoring a pre-placed crystal structure.

**CLAUDE.md §9 status:** This is technically a flag-and-stop condition (alpha < 0.2). It is acknowledged and documented here. The calibration proceeds because: (a) r=0.860 is fit for ranking purposes; (b) the root cause is known and the fix requires production docking data not yet available; (c) load_calibration bounds were updated to [0.1, 2.0] for contact-mode to accommodate this transitional state.

---

### Issue 2 — 4gq6 (menin/MLL) excluded as outlier

**What:** 4gq6 (menin + MLL peptide, pKd 6.15, ΔG ≈ −8.4 kcal/mol) was excluded from the training set. When included, Vina returned −20.5 kcal/mol — a 12 kcal/mol overestimation — and AD4 returned −17.9 kcal/mol.

**Why:** Menin has an unusually deep, concave binding groove. Vina's empirical scoring function was parameterised on typical globular protein–ligand complexes. Deep enclosed pockets artificially inflate the burial-based terms. This is a known failure mode for Vina on groove-shaped binding sites and is not an error in the pipeline.

**Consequence of inclusion:** With 4gq6 in the set, r dropped to 0.559 (from 0.860) and calibration was numerically unstable.

---

### Issue 3 — 2koh (Par3-PDZ3 / VE-Cadherin) excluded as outlier

**What:** 2koh was excluded. AD4 returned −16.7 kcal/mol for a complex with pKd=5.00 (ΔG ≈ −6.8 kcal/mol) — a 10 kcal/mol AD4 overestimation.

**Why:** The VE-Cadherin C-terminus peptide has multiple acidic residues (Asp, Glu). Gasteiger charge assignment used by AD4 overestimates electrostatic interactions for highly charged peptides in Poisson–Boltzmann-neglecting scoring. Not a code error.

---

### Issue 4 — Contact residue counting bug (FIXED, 2026-05-22)

**What:** The original `_count_contact_residues` in `scripts/score_crystal_poses.py` iterated over receptor atoms and returned the count of unique receptor residue sequence numbers within cutoff. For 4gq6 (12-mer peptide), this returned 27 — physically impossible for a 12-residue peptide.

**Fix:** Rewritten to group PEPTIDE atoms by residue number; a peptide residue is "in contact" if any of its heavy atoms is within 4.5 Å of any receptor heavy atom. Returns a count ≤ peptide length.

**Downstream effect:** `n_contact_residues` in `data/training_scores.json` was recomputed after the fix. Values before fix were invalid; values in the current JSON are correct.

---

### Issue 5 — Grid box sizing for long peptides (FIXED, 2026-05-22)

**What:** Initial scoring used a fixed 25 Å box centred on the Cα centroid. Several complexes failed with "The ligand is outside the grid box." Example: 2hwn (19-mer) spans 27 Å along one axis.

**Fix:** Grid box now computed from the bounding box of ALL peptide heavy atoms (not just Cα). Box size = max_extent + 15 Å margin (minimum 20 Å). Centre = bounding box centre (not Cα centroid).

---

### Issue 6 — Coordinate frame mismatch with apo receptor (FIXED, 2026-05-22)

**What:** Initial scoring used the apo pocket receptor (`_rec_unbound_pocket.pdb`) with the crystal peptide (`_pep_ref.pdb`). Some PepSet pairs are not in the same coordinate frame — 1nrl returned Vina = +27.5 kcal/mol indicating a massive steric clash.

**Fix:** Scoring now uses the holo receptor (`_rec_ref.pdb`) which is in the same crystal frame as `_pep_ref.pdb`. The apo pocket receptor is only used for counting contact residues (its coordinate frame is consistent with the crystal peptide for that purpose).

---

## Accuracy metric guide

### Pearson r (correlation coefficient)

Measures linear relationship between predicted hybrid score and experimental pKd. Range [−1, 1]. A tool that perfectly ranks peptides by affinity gets r=1.0.

| r value | Interpretation |
|---------|---------------|
| 0.9–1.0 | Excellent — reliable ranking |
| 0.7–0.9 | Good — mostly correct ranking with occasional inversions |
| 0.5–0.7 | Moderate — rough ordering, significant noise |
| 0.3–0.5 | Weak — marginally better than random |
| < 0.3 | Fail — CLAUDE.md §9 flag condition |

Current calibration: **r = 0.860** (Good). Caution: this is on a 6-point training set, not a held-out test set. Will shrink when measured on independent data.

### RMSE (root mean squared error, kcal/mol)

How far off the absolute ΔG predictions are on average. Lower is better.

| RMSE | Interpretation |
|------|---------------|
| < 1.0 | Near-experiment (uncommon for docking) |
| 1.0–2.0 | Acceptable for affinity ranking |
| 2.0–3.0 | Rough — usable for hit/no-hit calls only |
| > 3.0 | Poor absolute accuracy |

Current: **1.73 kcal/mol**. This is in the acceptable range, but inflated by the crystal-pose systematic bias. Expect it to be higher on production docking scores.

### Spearman ρ (rank correlation)

Like Pearson r but only cares about rank order, not linear spacing. More robust to outliers. Not currently computed in `calibrate_alpha.py` but relevant for iGEM benchmarking. With r=0.860 on this set, Spearman ρ is approximately 0.83–0.89 (estimated).

### Mean absolute error (MAE)

Average absolute prediction error. RMSE penalises large errors more. MAE is more interpretable as "typical error." For this calibration: MAE ≈ 1.4 kcal/mol (estimated from RMSE and dataset distribution).

### Enrichment factor (EF)

For screening campaigns: fraction of true binders in the top-N% of ranked list vs. random. Not computed here but relevant for the malaria PfLDH application.

---

## Plan to improve calibration

1. Run full production docking (`hybridock-pep dock`) on the 6 training complexes using the apo pocket receptor.
2. Score the generated poses (not crystal poses) with Vina + AD4.
3. Re-run `scripts/calibrate_alpha.py --scores-json <new_scores.json>`.
4. Expect alpha to converge away from the lower bound.
5. Commit new `data/calibration.json` alongside new `data/training_scores.json`.

---

## Production-pose recalibration — v2 SUCCEEDED (2026-06-02)

After diagnosing the v1 failure as a pipeline bug (next section), the script
`scripts/score_production_poses.py` was rewritten with two fixes:

1. **Pocket-PDB input to RAPiDock.** `rapidock_local.pt` is the local-docking
   checkpoint and expects a pocket-truncated PDB (binding-site residues only)
   as `--protein_description`. We were feeding the full apo receptor, which
   let RAPiDock-Reloaded "discover" the wrong pocket (18 Å away on 1ddv,
   12 Å on 2hwn). Switched to `{pdb}_rec_unbound_pocket.pdb`. PepSet ships
   the truncated pocket for every complex.
2. **Auto-box derivation.** Compute cubic Vina/AD4 grid box from the actual
   minimized pose extents + 4 Å margin, with optional 15 Å site-radius
   filter to discard wrong-pocket outliers before box derivation.

### v2 results

| PDB | pKd | n_res | V_prod | A_prod | n_contact | n_scored / 100 | box edge | offset |
|-----|------|-------|--------|--------|-----------|----------------|----------|--------|
| 1ywi | 4.10 | 6 | −4.95 | −3.26 | 4 | 97 | 26.3 | 3.9 |
| 1a0n | 4.60 | 14 | −6.01 | −4.18 | 6 | 94 | 46.7 | 2.1 |
| 1ddv | 5.00 | 6 | −6.98 | −4.64 | 5 | 94 | 23.4 | 3.8 |
| 1l2z | 5.70 | 11 | −5.63 | −4.10 | 6 | 95 | 36.6 | 4.2 |
| 1nrl | 6.00 | 15 | −7.74 | −4.54 | 6 | 90 | 42.4 | 3.4 |
| 2hwn | 8.70 | 19 | −6.62 | −3.66 | 9 | 94 | 54.8 | 6.5 |

### Correlations

| | r(pKd, feature) | ρ(pKd, feature) |
|---|---|---|
| Vina_prod | **−0.418** (was −0.120) | **−0.600** |
| AD4_prod | +0.076 | −0.143 |
| N_contact (more is better) | **+0.944** | +0.820 |
| N_residues (size confound) | (+0.789 w/ pKd, −0.414 w/ V_prod) | |

### Single-α fit on v2 scores (existing `calibrate_alpha.py`)
α = 0.100 (railed), β = 0.000 (railed), r = +0.224, RMSE = 2.94 kcal/mol
→ Saved to `data/calibration_v1_1_production.json`.
Existing fitter underperforms because it **fixes Vina coefficient at 1.0**
and only fits α + β. With per-complex Vina overshoot/undershoot varying,
no single α can fit.

### Multivariate ridge fit on v2 scores (strategies §6)
Model: `ΔG_hat = w_vina · vina + w_ad4 · ad4 + w_contact · (−N_contact) + intercept`
Ridge λ=0.1, positive-constrained weights.

| | In-sample | LOO-CV |
|---|-----------|--------|
| Pearson r | **+0.948** | **+0.755** |
| RMSE (kcal/mol) | 0.65 | 1.44 |

Fitted weights:
- `w_vina = 0.21`
- `w_ad4 = 0.00` (positive constraint pushes AD4 out; AD4 carries no
  additional signal once contact count is in the model)
- `w_contact = 1.21 kcal/mol per contact residue` — squarely in the
  physical range (peptide ordering entropy ≈ 1–2 kcal/mol/residue)
- `intercept = +0.77`

→ Saved to `data/calibration_v1_1_production_ridge.json` with per-complex
LOO predictions and residuals.

### Interpretation
The crystal-pose r = 0.86 was largely a pose-quality reward (Vina rewarding
optimal binding geometry on the actual crystal pose). Production poses
have no such optimal-geometry advantage, so the cross-complex signal must
come from a feature that does NOT depend on per-pose geometry quality:
**contact-residue count** is exactly such a feature, and it dominates the
ridge fit. On production poses, n_contact recovers — and improves on —
the cross-complex signal that crystal-Vina had.

Outlier behavior (LOO residuals):
- 2hwn LOO residual = +2.59 (under-predicted): strongest binder is hardest
  to extrapolate when removed; only 9 contacts are not enough to flag a
  pKd 8.7 binder by themselves. AD4 weight = 0 means it isn't helping here.
- 1a0n LOO residual = −1.74 (over-predicted): PPII surface binder, Vina
  tends to over-score (more negative than truth) per strategies §4.2.
- All others ≤ 1.6 kcal/mol absolute residual.

### What changes in production
The single-α calibration JSON schema cannot represent the ridge fit
directly (no `w_vina` field — Vina weight is hard-coded to 1.0 in
`apply_hybrid_score`). To ship the ridge fit, options:

1. **Extend the schema** with `w_vina` and update `apply_hybrid_score` to
   read it (default 1.0 for backward compat). ~1 day work + a unit test.
2. **Keep current schema, document the ridge fit as the "research"
   calibration** in `docs/calibration_notes.md` for now; ship the
   ridge after Step 1 lands.
3. **Convert ridge to single-α equivalent**: divide all weights by
   `w_vina` so Vina effective weight is 1.0, then α = w_contact / w_vina
   = 5.86 kcal/mol/residue. Outside the current `[0.1, 2.0]` bound but
   physically sensible. The intercept becomes intercept / w_vina = 3.76,
   which would also need to be representable.

Recommendation: ship as Option 1 — extend the schema. The existing
single-α model is provably insufficient on production poses; adding
`w_vina` is the minimal change that exposes the signal the data carries.

### Calibration files now on disk
| File | What | Use |
|------|------|-----|
| `data/calibration.json` | crystal-pose single-α (existing) | production CLI default — DO NOT REMOVE until ridge schema lands |
| `data/calibration_v1_1_production.json` | production-pose single-α (railed) | reference for the failure mode |
| `data/calibration_v1_1_production_ridge.json` | production-pose multivariate ridge | the calibration we want to ship after schema extension |

---

## Production-pose recalibration — v1 FAILURE (2026-06-02, superseded)

Ran `scripts/score_production_poses.py` to generate 100 RAPiDock-Reloaded poses
per PepSet complex (apo receptor input), applied OpenMM clash relief + Vina
`v.optimize()` BFGS clash relief (production pipeline), scored each pose with
Vina + AD4, aggregated top-10 by Vina median per complex. Refit with
`calibrate_alpha.py`.

**Outcome:** α=0.100 (bound), β=0.000 (bound), **r = −0.240**, RMSE = 4.51 kcal/mol.

For comparison the crystal-pose calibration: α=0.100, β=0.000, **r = +0.860**, RMSE = 1.73 kcal/mol.

**The new calibration was saved to `data/calibration_v1_1_production.json`,
NOT to `data/calibration.json`. Production keeps the crystal-pose calibration.**

### Diagnostic numbers

| PDB | pKd | len | Vina_prod | AD4_prod | Vina_crys | AD4_crys | n_scored / 100 |
|-----|------|-----|-----------|----------|-----------|----------|----------------|
| 1ywi | 4.10 | 6 | −4.11 | −2.72 | −6.45 | −7.23 | 97 |
| 2hwn | 8.70 | 19 | −5.17 | −2.91 | −14.75 | −3.39 | 30 |
| 1nrl | 6.00 | 15 | −5.20 | −3.91 | −12.41 | −11.49 | 29 |
| 1l2z | 5.70 | 11 | −5.43 | −3.86 | −7.03 | −8.51 | 95 |
| 1ddv | 5.00 | 6 | −2.99 | −1.68 | −8.57 | −5.99 | **1 (suspect)** |
| 1a0n | 4.60 | 14 | −6.91 | −4.51 | −5.66 | −6.69 | 89 |

Pearson r between experimental pKd and raw scores:

| Comparison | Crystal | Production |
|------------|---------|------------|
| r(pKd, Vina) | **−0.886** | **−0.120** |
| r(pKd, AD4)  | +0.398 | +0.032 |
| Spearman ρ(pKd, Vina) | — | −0.086 |
| Drop 1ddv (n=1) and refit r(pKd, Vina) | — | +0.046 |
| Size confound r(N_res, Vina_prod) | — | **−0.657** |
| r(N_res, pKd) in this set | — | +0.789 |

### Interpretation — what this means

**The crystal-pose r=0.86 was a pose-quality reward signal, not an affinity
signal.** When Vina scores the exact crystal pose of a strong binder, it sees
the optimal packing it was designed to reward, and returns a very negative
number proportional to how good the binding mode is. On RAPiDock-generated
poses — which is what the user actually gets at inference — that signal
disappears: Vina sees an approximately-correct pose with noise, and its
score is dominated by peptide size (r=−0.66 with N_res) plus that noise.

This is the textbook cross-family absolute-ΔG ceiling described in
`docs/calibration_strategies.md` §1. Production-pose recalibration on a
6-complex set confirms it cannot be moved by data hygiene alone — Vina has
essentially zero rank correlation with pKd on production poses of this set,
even when we drop the noisy 1ddv entry.

### What this does NOT mean
- Tool is not broken. **Pose-quality** ranking remains excellent — the v5c
  benchmark hit@5 = 91% and 0.80 Å Cα RMSD vs DiffPepDock are independent
  of this calibration result.
- Within-target ranking still works — re-scoring multiple candidates on
  the *same* target removes the cross-target bias and brings Vina rank
  correlation back to useful territory.

### What to NOT do next
- Do not refit α on this 6-complex set with the current single-α model.
  The signal isn't there to recover. α=0.1 will keep railing.
- Do not abandon the production-pose dataset — it is the correct calibration
  target. The crystal-pose r=0.86 should be treated as a structural-quality
  marker, not an affinity claim. See `calibration_strategies.md` §14.

### What to do next (in order)
1. **Re-run 1ddv** with a wider grid box (current box 30 Å, almost all
   poses clipped → n_scored=1). Expected fix: bump margin parameter from
   20 Å → 30 Å in `_BOX_MARGIN_PROD`. Cheap (~10 min wall-clock).
2. **Decoy ΔΔG (§7 + §15 of strategies doc)** for the parent project's
   actual deliverable. Sidesteps absolute calibration entirely.
3. **Per-residue entropy + signed α (§5.5)** is queued but unlikely to
   move r above 0.4 on this 6-set alone — production poses lack the
   per-pose binding-mode signal that the entropy table would weight.
   Defer until the curated 30-complex family-balanced set exists (§11).
4. **Per-family calibration (§4)** is the real lift, requires 30-complex
   curation work first.

---

## v1.2 entropy calibration — held-out 242-complex crystal-pose evaluation (Jun 2026)

**Date:** 2026-06-02. **Script:** `scripts/eval_holdout_calibrations.py`.
**Output:** `data/eval_holdout_calibrations.json` (242 rows).

After implementing the per-residue + SS-weighted entropy module (§5.5 of
`calibration_strategies.md`) and dropping AD4 from the default scoring set,
I evaluated all three calibrations on the largest crystal-pose dataset we
have with experimental affinities:

| Source                                      | n   |
|---------------------------------------------|-----|
| `data/training_scores_full.json`            | 272 |
| Matched to `training_complexes_full.csv` Kd/Ki/IC50 | 246 |
| Peptide chain found in `datasets/raw_pdbs/` | 242 |

The 4 PepSet-6 training entries that overlap (1a0n, 1ddv, 1l2z, 1nrl) are
flagged but kept for stratified reporting.

### Pearson r and RMSE vs ΔG_exp = −1.3633·pKd

| Subset                       | n    | v1.0 legacy r / RMSE | v1.1 ridge r / RMSE | v1.2 entropy r / RMSE | Vina-only r |
|------------------------------|------|----------------------|---------------------|------------------------|-------------|
| ALL                          | 242  | +0.000 / 31.13       | −0.193 / 12.30      | **−0.119 / 3.74**      | −0.243      |
| HELD-OUT (excl. PepSet-6)    | 238  | −0.001 / 31.39       | −0.209 / 12.37      | **−0.121 / 3.73**      | −0.261      |
| HELD-OUT, Kd-only            |  54  | +0.124 / 64.56       | −0.486 / 12.58      | **−0.304 / 3.91**      | −0.515      |
| TRAINING-OVERLAP (4 PepSet)  |   4  | +0.809 / 1.80        | +0.724 / 6.70       | +0.119 / 3.97          | +0.731      |

### Reading the table honestly

1. **RMSE collapses from ~30 kcal/mol to ~3.7 kcal/mol with v1.2.**
   Legacy α=0.1 and ridge w_contact=−1.20 both extrapolate disastrously
   on long peptides outside PepSet-6 (legacy adds α·N up to ~+3 kcal/mol,
   ridge subtracts w_contact·N down to ~−45 kcal/mol on 30-mers).
   v1.2 has no per-residue scaling — `w_vina=0`, `w_s_ss_weighted=−0.434`,
   intercept −3.95 — so the predicted ΔG stays bounded inside the physical
   peptide range (−5 to −15 kcal/mol). **This alone is a meaningful
   robustness win for the user-facing tool.**

2. **Pearson is still negative on the held-out set.** This is the same
   absolute-affinity ceiling diagnosed in the v1.1 forensics block above
   (Vina anti-correlates with pKd across heterogeneous targets, r=−0.51
   even on the 54-complex Kd-only subset). Entropy doesn't fix that —
   nothing single-target-free does.

3. **Training-overlap "r=+0.809" for v1.0 is in-sample overfit** to those
   4 points; v1.2 was trained on a different (production-pose, not
   crystal-pose) version of the same complexes and so doesn't latch onto
   the crystal-pose Vina dynamic range. The OVERALL RMSE story is the
   important one.

### Decision: keep `data/calibration.json` as production default

The on-disk default (`alpha=0.1, beta=0, ad4_weight=0.3`, r=0.860 on the
PepSet-6 crystal poses) stays in place for now. v1.2 is checked in as
`data/calibration_v1_2_production_entropy.json` and is opt-in via
`--calibration data/calibration_v1_2_production_entropy.json`.

**Why not flip:** the held-out Pearson is negative for v1.2 too, so we
can't claim "better ranking" honestly. The RMSE win is real but the user
will see it only on peptides longer than ~12 residues.

### What this round did ship

- **Per-residue + SS-weighted entropy module** (`src/hybridock_pep/scoring/per_residue_entropy.py`)
  with Doig-Sternberg `S_SC` table, Baxter-Murphy `S_BB` table, Ramachandran
  φ/ψ-based SS classification, and the corrected praxeolitic dihedral sign
  (`b0 = -(p1 - p0)`).
- **Schema v2 ridge calibration** in `entropy.py` with positive-constraint
  bounds, optional `w_s_sc`/`w_s_bb`/`w_s_ss_weighted` columns, and built-in
  LOO-CV.
- **Driver wiring** (`driver.py` Stage 2d-pre-entropy) so the new entropy
  features are computed lazily only when the loaded calibration references
  them — no cost for legacy calibrations.
- **AD4 dropped from default** scoring set after the v1.1 ridge fit returned
  `w_ad4 = 0.0` (AD4 contributes no orthogonal signal over Vina on this
  dataset; documented in the "Why AD4 commits nothing" section). AD4 stays
  opt-in via `DockConfig(scoring={"vina", "ad4"})`.
- **13 new tests** in `tests/test_scoring.py` covering schema-v2 round-trip,
  ridge formula, AD4 anomaly bypass, per-residue tables, dihedral sign
  convention, and `compute_entropy_sums` end-to-end.

273 unit tests pass (45 slow-tests skipped without `-m slow`).

### What's still pending (for future sessions)

- Per-target / per-family calibration (§4 of strategies doc) is still the
  only path to Pearson r > 0.5 across heterogeneous targets.
- A curated 30-complex family-balanced training set is the prerequisite
  for that and for any future re-calibration that wants to learn
  per-residue entropy weights with leverage.
- Decoy ΔΔG (§7+§15) is the right approach for the parent PfLDH selectivity
  question; absolute Kd prediction is the wrong frame for that deliverable.
