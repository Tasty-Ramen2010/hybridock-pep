# Model Card — HybriDock-Pep scoring models

This repository ships **ten** trained artifacts in `data/*.joblib`. Only three are wired
into the CLI; the rest are research/ablation variants kept for transparency. If you read
one number from this repo, read this table first.

## What actually ships (wired into the CLI)

| Artifact | Used by | Role | Set in |
|---|---|---|---|
| **`data/affinity_ai_nofix.joblib`** | `dock` (default) | **Headline ΔG** (`delta_g`, "Best pose ΔG"). Geometry-feature model, **no** size-fix. Tuned on real RAPiDock/AI poses. | `scoring/affinity_model.py` `_DEFAULT_ARTIFACT` |
| **`data/affinity_crystal_ifp.joblib`** | `crystal-score` | Score an existing **crystal-quality** pose (geometry + interaction map). | `scoring/interaction_map.py` `_DEFAULT_ARTIFACT` |
| **`data/pose_ranker_ml.joblib`** | Stage 2 ranking | Predicts per-pose native-RMSD to rank poses (**not** an affinity). | `scoring/pose_ranker_ml.py` `DEFAULT_MODEL_PATH` |

**Vina is clash-relief only. AD4 is off by default. Neither is the reported ΔG.**

## Research / ablation artifacts — do NOT cite as "the model"

`affinity_ai_sizefix`, `affinity_crystal_sizefix`, `affinity_crystal_augmented`,
`affinity_pooled_prodn`, `affinity_realpose`, `affinity_rank_ifp`, `entropy_surrogate`.
These reproduce ablations documented in the development timeline archived on Zenodo:
[10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573). They are not the shipping scorer.

## Intended use

Rank/compare short peptides (3–19 aa) against one or two receptors at iGEM workflow scale
(dozens of candidates, minutes each, commodity hardware). Strongest outputs are **relative**:

- **`selectivity`** — ΔΔG(target − off-target) with bootstrap CI. *Recommended primary output.*
- **double-difference ΔΔG** (same-receptor cycle) — FEP-grade relative accuracy.
- **reference-anchored ΔG** — 2–3 measured anchors lift within-receptor r from ≈0.25 to ≈0.55.

## Performance (leakage-free, kcal/mol primary)

Full numbers, methodology, and reproduce commands: **[RESULTS.md](RESULTS.md)**. Headline:
absolute cross-target **MAE ≈ 1.40** (60%-id clustered CV, n=925); matched head-to-head
**1.35 vs PPI-clone 1.46** (n=865, every metric). Methodology follows standard
leakage-control practice — see the README's *Evaluation methodology* section.

## Known limitation — read before quoting an absolute Kd

Blind, cross-target **absolute** ΔG is confound-limited for *every* cheap non-FEP method
(interface-size / per-system baseline; enthalpy–entropy compensation — FEP hits the same
wall in this regime). MAE is the stable, meaningful metric; absolute Pearson r caps near
the field ceiling (~0.32) for everyone. An **earlier, now-superseded** production scorer
generalized to *negative* correlation on a small holdout — that model is retired; the story
is documented on the record in the Zenodo research archive:
[10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573). **Report relative ΔΔG / selectivity / anchored ΔG
as the accurate paths; treat a standalone absolute Kd as a coarse readout, not a validated
prediction.**
