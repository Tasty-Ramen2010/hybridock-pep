# ML Residual-Correction Model — Scoping

Status: scoping (Jun 11). Decision pending e50 (complete-LIE) charged result.

## 1. The thesis (what the model is *for*)

Physics (static terms, ensemble MM-GBSA, LIE) plateaus at the **charged-desolvation floor** —
r≈0.07 on charged complexes for *everyone*, including real GB and (pending) ensemble LIE. The floor
is **bias** (missing physics: explicit-water reorganization, polarization, the desolvation net), not
**variance** — so more sampling can't fix it (e47/e48/e49 confirmed). The only doors are *compute*
(FEP, out of scope) or *data* (learn the bias correction from labelled examples). **This model is
the data door.**

Core design choice: **predict the RESIDUAL of the physics baseline, not raw ΔG.**

    ΔG_pred = ΔG_physics(pose)  +  f_ML(features)
    train f_ML on:  residual = ΔG_exp − ΔG_physics

Physics carries the bulk (vdW, hydrophobic burial, shape); ML only learns the systematic part
physics gets wrong (charged desolvation, hotspot heterogeneity). Smaller learning target = less
data needed = less overfit. This is the opposite of training a model to predict ΔG from scratch.

## 2. Hard constraints from prior results (do NOT relitigate)

- **Within-dataset r is a MIRAGE.** Every validation MUST be leave-DATASET-out, not just
  leave-complex-out. crystal-65 LOO 0.576 → leave-dataset-out −0.28 (E19). Non-negotiable.
- **Data VOLUME is the bottleneck, not diversity** (confidence campaign Jun 2). ~163 labelled
  structured complexes is too few for high-capacity models. Match model capacity to n.
- **High-capacity models overfit here** — ESM per-contact, deep confidence heads all overfit
  65–300 points. Start simple.
- **Realistic ceiling ≈ 0.55** (PPI-Affinity, thousands of complexes). We will not reach FEP. A
  charged-subset lift from 0.07 → 0.30 cross-dataset is the real, honest win to aim for.
- **Net charge is the #1 residual driver** (e49: corr(resid, net_charge) = −0.345 — physics
  over-stabilizes positively-charged peptides). First feature in the door.

## 3. Target & features

**Target:** `residual = ΔG_exp − ΔG_physics_baseline`, baseline = best of {ensemble MM-GBSA (e49),
complete-LIE (e50)} once e50 settles which.

**Features (3 tiers, all already computable):**
| tier | features | source |
|---|---|---|
| physics ensemble | ⟨E_int⟩, reorg/desolv, E_int_std (fluctuation≈entropy), −TΔS_IE, dg_3traj | e49/e50 |
| charge/desolvation | net_charge, charged_frac, n_buried_unsat, e_desolv, buried-charge frac | e42 |
| composition/shape | strength_bur (SKEMPI), mj_contact, bsa_hyd, arom_frac, length, pocket descriptors | geometry_features/e46 |

Net charge / desolvation tier is the priority — it targets the measured failure axis.

## 4. Architecture (matched to data size, staged)

- **Phase A (n≈163, now):** gradient-boosted trees (LightGBM/XGBoost, depth ≤3, heavy
  regularization) **or** ridge regression on the residual. GBT handles the charge×burial
  nonlinearity; ridge is the interpretable floor. Pick by leave-dataset-out CV, not train fit.
- **Phase B (n≈1000s, after a data-generation campaign):** small MLP or — given the 3D structure —
  an equivariant GNN (e3nn) on the interface graph. Only justified once data supports it; at n=163
  it WILL overfit. Do not start here.
- **NOT** a from-scratch equivariant force field / Boltzmann generator (10k+ examples, GPU-cluster,
  PhD-scale — see project_pose_ensemble_jun11). If we want learned ensembles, use a *released*
  model (Boltz-2, BioEmu) as a feature, not train one.

## 5. Data strategy (the actual bottleneck)

- **Have:** crystal-65 + the-98 (~163 structures+Kd). Ensemble features being generated now (e49/e50).
- **Expand (priority order):** PDBbind peptide subset, PPI-Affinity's curated set, SKEMPI-derived
  pairs, Wang/PEPBI ITC. Each needs the e49/e50 ensemble pipeline run on it (GPU campaign, ~40s/complex).
- **Honest scaling:** approaching 0.55 needs ~1000s of labelled complexes WITH ensemble features.
  That is a data-generation campaign, not a modelling trick. Budget it explicitly.

## 6. Validation protocol (the part that keeps us honest)

1. **Leave-dataset-out** primary metric (train crystal-65 → test the-98 and vice versa).
2. Leave-complex-out secondary.
3. **Charge-stratified report**: r on cf≥0.3 subset separately — that's the floor we're attacking.
4. Permutation test (shuffle labels) to confirm signal > chance, as in the free-entropy work.
5. Ship only if it beats the physics baseline **cross-dataset on the charged subset**.

## 7. Milestones

- M0: e50 settles the physics baseline (does complete-LIE help charged?). ← in progress
- M1: assemble feature matrix (physics + charge + composition) on the 163. Phase-A GBT/ridge on
  residual, leave-dataset-out + charge-stratified. Go/no-go: charged cross-dataset r > 0.20.
- M2: if M1 promising, data-generation campaign (PDBbind/PPI peptides through e49/e50) to ~1000.
- M3: Phase-B model if data supports; wire best model as a residual head behind `--refine-topk`.

## 8. Risks / kill criteria

- If M1 charged cross-dataset r stays ≈ physics baseline (~0.07–0.15) → the residual is not learnable
  from these features at this n; the bias needs explicit-solvent FEP (compute door), and we stop
  chasing absolute charged ΔG. Pivot to **relative ΔΔG / selectivity** (flex ddG), where the floor
  cancels — the use case where partial-physics actually pays and our tool differentiates.
