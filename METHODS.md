# Methodology

A short, self-contained description of what HybriDock-Pep does and how it was
validated. Written to be read start-to-finish in a few minutes; every claim
links to the script that produces it.

For *how to run it*, see [README §Quick start](README.md#quick-start--one-command).
For the full research ledger, including refuted ideas, see
[`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).

---

## 1. Problem

Given a **peptide sequence** and a **receptor structure**, predict

1. how the peptide binds (an all-atom pose),
2. how tightly (ΔG in kcal/mol), and
3. whether it prefers one receptor over another (ΔΔG, with a confidence
   interval).

The design target is the iGEM/wet-lab workflow scale: tens of candidate
peptides against one or two targets, minutes per peptide on commodity
hardware — not a proteome-wide screen, and not a single-complex FEP campaign.

## 2. Pipeline

Two stages. Stage 1 proposes structures; Stage 2 turns structures into numbers.
Both run from one CLI (`hybridock-pep dock`); the code path is
`driver.py::run_dock`.

```
peptide sequence + receptor PDB
        │
        │  receptor cleaned with PDBFixer
        ▼
STAGE 1  Diffusion sampling — RAPiDock-Reloaded
        │  N stochastic SE(3)-equivariant denoising passes → N all-atom poses
        │  sampling is receptor-wide; --site only bounds the Stage 2 grid
        ▼
  1.5   Restrained clash relief (OpenMM, heavy-atom harmonic restraints)
  1.7   Drop off-pocket poses · auto-expand the search box if poses need it
        ▼
STAGE 2  Pose prep and ranking
        │  receptor + ligand → PDBQT (meeko)
        │  Vina = clash relief only, NOT the reported score
        │  BSA-fit and ML pose rankers predict native RMSD
        ▼
  3     Cα-RMSD agglomerative clustering → cluster representatives
  3.5   Optional MM-GBSA relaxation of the top-K representatives
        │  ff14SB + GBn2 implicit solvent;  ΔG = E(complex) − E(rec) − E(pep)
  3.6   PRIMARY ΔG: learned-geometry affinity model on the AI pose
        ▼
ranked_poses.csv · best_pose.pdb · cluster_summary.csv · run_metadata.json
```

**What the reported ΔG is.** The headline number is Stage 3.6:
`data/affinity_ai_nofix.joblib`, a `HistGradientBoostingRegressor` over 262
features — a 16-term interface-geometry block (buried hydrophobic area, H-bond
and salt-bridge SASA, aromatic contacts, mean burial, Miyazawa–Jernigan contact
energy, pocket descriptors) plus ~246 ProtDCal sequence/structure descriptors —
trained on **633 real RAPiDock poses**, internal CV r = 0.492.

It is *not* the Vina score. Vina appears only as a clash-relief minimiser in
Stage 2 and its score is logged as telemetry. AutoDock4 scoring is off by
default. MM-GBSA (Stage 3.5) is an independent, physically-grounded absolute
estimate available with `--refine-topk`; it does not override the learned
scorer.

Peptides of ≤ 8 residues are routed to a lean 3-feature hydrophobic ridge
sub-model (`scoring/length_router.py`). Short peptides make too few interface
contacts for the full feature block — 13 of the 16 geometry features have
near-zero dynamic range there — and the standard path collapses to r ≈ 0 on
them; the sub-model recovers r ≈ 0.51.

> **Two different models appear in this document, and they are not the same
> one.** The deployment scorer above is what `hybridock-pep dock` runs. The
> PDBbind benchmark numbers in §4 come from a 16-feature
> `GradientBoostingRegressor` fitted per-fold inside the benchmark scripts on
> *crystal* poses — that is the correct object for a like-for-like comparison
> against other published scorers on crystal structures, but it is not the
> artifact the CLI loads. The number that characterises the deployed path is
> the AI-pose row: **MAE ≈ 1.51 kcal/mol, r ≈ 0.50**. See
> [`MODEL_CARD.md`](MODEL_CARD.md) for every shipped artifact and its role.

**Why a learned scorer on top of a diffusion sampler.** The two failure modes
are separable. Diffusion models place peptides well but do not produce
energies; classical scoring functions produce energies but were fitted to
crystal poses and degrade on generated ones. Training the scorer *on AI poses*
(`data/affinity_ai_nofix.joblib`) rather than on crystals absorbs that
distribution shift: on the same complexes, a crystal-tuned model reaches
r ≈ 0.585 and the AI-pose model reaches 0.49–0.53, versus 0.325 for a
pose-blind sequence-only baseline that cannot read the pose at all.

## 3. Training data and features

| | Deployment scorer (`dock`) | Benchmark model (§4 PDBbind numbers) |
|---|---|---|
| Artifact | `data/affinity_ai_nofix.joblib` | fitted per-fold inside the experiment script |
| Estimator | `HistGradientBoostingRegressor` | `GradientBoostingRegressor` (300 trees, depth 3) |
| Features | 262 (16 geometry + ~246 ProtDCal) | 16 geometry |
| Trained on | 633 **RAPiDock-generated** poses | 925 PDBbind **crystal** poses |
| Labels | Kd/Ki → ΔG (kcal/mol) | Kd/Ki → ΔG (kcal/mol) |

Training on generated rather than crystal poses is deliberate. A crystal-tuned
model collapses on real RAPiDock output ("the AI haircut"), because the feature
distribution it was fitted to does not occur there. The crystal-tuned sibling
is still shipped, exposed only through `crystal-score`, where the input really
is a crystal pose.

Evaluation sets never used in training:

| | |
|---|---|
| Independent database | PPIKB, n = 808 (Kd/Ki-only subset) |
| External holdout | 43 complexes from Wang et al. 2024 supplementary tables — absent from training *and* from every 60%-identity cluster in it |
| Real-pose set | 151 complexes (cr65 + the98) scored on generated poses, LOCO CV |

## 4. Validation and leakage control

This is the part that decides whether any of the numbers mean anything, so it
is specified exactly.

- **Leave-cluster-out cross-validation.** Peptides are clustered by sequence
  identity and *entire clusters* are held out per fold (CD-HIT-style). No
  near-duplicate of a test peptide is ever in the corresponding training fold.
- **Placement-aware identity.** Cluster identity uses a gap-penalised
  alignment, not longest-common-subsequence over the shorter sequence. The
  earlier free-gap metric scored `GGA`≈`ACC` at 0.33 and collapsed 925 peptides
  into 21 clusters at the 30% cutoff; the corrected metric gives 410. Both are
  reproducible in
  [`experiments/e367_gap_penalized_trend.py`](experiments/e367_gap_penalized_trend.py).
- **The whole threshold sweep is reported, not one split.** MAE is flat
  (1.32 → 1.42 kcal/mol) from random splits down to a 30% identity cutoff,
  while Pearson r falls from 0.45 (leaky) and levels off near 0.32. The 60%
  headline and the standard 30% cutoff are both reported.
- **Leakage is demonstrated, not asserted.** Clustered r (0.35) is lower than
  leaky random-CV r (0.44) on the same data — the expected direction, and the
  check that the clustering is doing something.
- **The primary metric is MAE/RMSE in kcal/mol.** Pearson r is secondary: it is
  sensitive to test-set spread and is capped near the field ceiling for every
  method, FEP included.

**Headline results** (leakage-free, 60%-identity leave-cluster-out CV):

```
  crystal poses — comparable to other published scorers
  925 PDBbind peptide-Kd complexes        MAE 1.40   RMSE 1.77   r 0.321
  matched n=865 vs a PPI-Affinity clone   MAE 1.35   RMSE 1.69   r 0.352
                       (clone)            MAE 1.46   RMSE 1.84   r 0.210
  PPIKB independent, n=808                MAE 1.94   RMSE 2.47   r 0.333
  Wang 2024 external holdout, n=43        MAE 1.60   RMSE 1.90   r 0.44

  generated poses — what you actually get from `dock`
  151 real-pose complexes, LOCO CV        MAE 1.51   RMSE 1.87   r 0.501
```

The gap between the two blocks is the pose-quality tax: scoring a pose the
pipeline generated instead of one solved by crystallography costs ~0.1 kcal/mol
of MAE. Quote the second block when describing what the tool does.

**Where it loses.** On PPI-Affinity's own curated T100 set (n = 48) the real
published tool beats us on ranking (r 0.549 vs 0.225), though our MAE (1.54) is
second only to theirs. In-distribution numbers are not comparable across tools;
that is the reason the headline uses a matched leakage-free split, and the
reason this table is in the README rather than omitted.

## 5. Uncertainty, anchoring, and selectivity

- **Uncertainty.** ΔG is reported with a conformal prediction interval,
  calibrated at 65% coverage (±1.64 kcal/mol). The interval is a fixed
  calibrated width — an attempt at adaptive per-peptide error bars was
  evaluated and rejected: nothing available at prediction time predicted the
  error.
- **Anchoring (the same-receptor mode).** With 2–3 measured affinities on the
  actual target, subtracting the per-receptor offset lifts within-receptor
  r from ≈ 0.25–0.47 (cold) to ≈ 0.55–0.71, depending on the dataset. This is
  the same trick FEP relies on — work relative to a reference so the systematic
  per-receptor bias cancels.
- **Selectivity.** `hybridock-pep selectivity` computes ΔΔG between a target
  and an off-target with a bootstrap CI. Measured r ≈ 0.30–0.45. This is a
  structure-based primitive; a sequence-only scorer cannot provide it.

## 6. Cost and limits

| | |
|---|---|
| Peptide length | 3–19 residues |
| Runtime | ~3 min to generate N=100 poses (RTX 5070); ~2.8 s/pose to score |
| Hardware | CUDA · ROCm · Intel XPU · Apple MPS · CPU |
| Determinism | fixed seed reproduces exactly on a pinned stack (CUDA/CPU) |

Known limits, stated plainly:

- Absolute cross-target r is capped around 0.32 in the leakage-free regime.
  That is a property of the task, not a bug we expect to fix; the number we
  stand behind is the kcal/mol error, which is stable across the whole
  identity sweep.
- Absolute ΔG for **charged** peptides sits at the non-FEP ceiling.
- Results depend on Stage 1 pose quality. Scoring AI poses costs ~0.05–0.09 in
  r relative to scoring crystal poses.
- `scikit-learn`/`numpy` versions shift GBT feature values slightly; a single
  score can move by up to ~1 kcal/mol across stacks, though it is deterministic
  within one.

## 7. Reproducing

```bash
make verify                              # math-only checks, 30 s, no data needed
python experiments/e330_ours_pdbbind.py  # the 925-complex headline
python experiments/e366_identity_threshold_trend.py     # the identity sweep
python experiments/e367_gap_penalized_trend.py          # the identity-metric fix
```

**One exception, stated up front.** `experiments/e331_ours_vs_ppiclone_clustered.py`
— the matched ours-vs-PPI-clone comparison — does *not* run from a clean clone.
It needs `data/e180_protdcal3d.jsonl`, which is `.gitignore`d, and that file's
generator needs a module (`e158_overfit_failure_analysis`) that was never
committed. Treat the head-to-head numbers in §4 as reported-but-not-yet-
independently-reproducible until that is restored; everything else on this page
reproduces from what ships. `tests/test_repro_claims.py` pins this so it cannot
recur silently.

Each script prints its own metrics; the head-to-head (`e331`) also writes a JSON
receipt. Datasets, including everything too large for git, are archived on Zenodo
([10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573)) and
indexed in [`docs/DATA_ARCHIVE.md`](docs/DATA_ARCHIVE.md).

## 8. Key citations

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025)
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021)
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017)
- **PPI-Affinity** — Romero-Molina et al., *J. Proteome Res.* 21:1829 (2022)
- **Wang et al.** — *Curr. Med. Chem.* 31(31):4127 (2024)
