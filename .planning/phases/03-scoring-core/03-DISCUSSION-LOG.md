# Phase 3: Scoring Core - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 03-scoring-core
**Areas discussed:** Hybrid score formula, α calibration workflow

---

## Hybrid score formula

| Option | Description | Selected |
|--------|-------------|----------|
| Vina + entropy only | hybrid = vina + entropy_correction. AD4 logged but not in score. | |
| Average(Vina, AD4) + entropy | hybrid = 0.5×vina + 0.5×ad4 + entropy | |
| Vina + β×(AD4−Vina) + entropy | hybrid = vina + β×(ad4−vina) + α×n_residues | ✓ |

**User's choice:** Freeform — "vina + entropy + ad4 electrostatics: Vina is primary scorer. AD4 delta (ad4 - vina) captures charge penalty. Formula: hybrid = vina + β×(ad4-vina) + α×n_residues. Only 2 params to fit (β, α), accounts for electrostatics without overfitting."

**Notes:** User originated this formula design. The (ad4−vina) delta term elegantly captures the electrostatic signal without requiring AD4 and Vina scores to be on the same absolute scale. When β=0, falls back to pure Vina; when β=1, pure AD4. Physically interpretable range.

---

## Entropy term details

| Option | Description | Selected |
|--------|-------------|----------|
| Full peptide length, positive α | n_residues = len(peptide_sequence), positive α | ✓ |
| Full peptide length, negative α | Sign embedded in α value | |
| Contact-zone residues only | n_residues = contact residues count | |

**User's choice:** Full peptide length, positive α (recommended)

---

## β validation range

| Option | Description | Selected |
|--------|-------------|----------|
| 0.0–0.5, abort outside | β > 0.5 means AD4 dominates, abort | ✓ |
| 0.0–1.0, warn outside | Full range, warn if outside | |
| No validation on β | Validate α only per spec | |

**User's choice:** 0.0–0.5, abort outside (recommended)

---

## Scoring failure handling

| Option | Description | Selected |
|--------|-------------|----------|
| Collect as PoseFailure, continue | PoseFailure(stage='scoring'), batch continues | ✓ |
| Hard abort on any failure | Raise immediately on any pose failure | |
| Score None, flag it | None fields in ScoredPose | |

**User's choice:** Collect as PoseFailure, continue (recommended)

---

## α calibration workflow

### Training data format

| Option | Description | Selected |
|--------|-------------|----------|
| PDB ID + experimental pKd | CSV: pdb_id, peptide_sequence, experimental_pkd | ✓ |
| PDB ID + ΔG directly | CSV: pdb_id, peptide_sequence, delta_g_kcal_mol | |
| PDB ID + IC50 + conditions | More raw, more normalization needed | |

**User's choice:** PDB ID + experimental pKd (recommended)

### Fitting method

| Option | Description | Selected |
|--------|-------------|----------|
| scipy.optimize.minimize, least-squares | L-BFGS-B with bounds | ✓ |
| OLS regression (numpy lstsq) | No bounds enforcement | |
| sklearn LinearRegression | Same limitation as OLS | |

**User's choice:** scipy.optimize.minimize, least-squares (recommended)

### calibration.json output

| Option | Description | Selected |
|--------|-------------|----------|
| alpha, beta + fit metadata | Full provenance included | ✓ |
| alpha and beta only | Minimal | |
| Full scipy result dump | Verbose | |

**User's choice:** alpha, beta + fit metadata (recommended)

### Script location

| Option | Description | Selected |
|--------|-------------|----------|
| scripts/ + hybridock-pep calibrate | Both entry points, core in entropy.py | ✓ |
| scripts/ only | No CLI wiring yet | |
| scoring/entropy.py only | No standalone script | |

**User's choice:** scripts/ + hybridock-pep calibrate (recommended)

### Default calibration.json

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, ship a default | α≈0.65, β≈0.22 committed to repo | ✓ |
| No, require calibrate first | abort if file missing | |

**User's choice:** Yes, ship a default (recommended)

---

## Claude's Discretion

- Per-pose Vina+AD4 parallelism strategy — optimize for 5-min wall-clock target
- Vina Python API lifecycle (instance reuse vs per-pose)
- Grid boundary check implementation details for is_clipped

## Deferred Ideas

- MM-GBSA pre-minimization (CLAUDE.md §2.5) — deferred to Phase 7/OPT-01
- Temperature as CLI flag for pKd→ΔG — deferred to v2 (T=298K hardcoded)
