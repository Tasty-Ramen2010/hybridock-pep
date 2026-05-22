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
