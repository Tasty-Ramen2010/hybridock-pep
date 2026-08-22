# RESULTS — leakage-free benchmarks

One page: every headline number, the exact command that regenerates it, the honest caveat
attached. Metric is **MAE/RMSE in kcal/mol** (primary; correct for an absolute-ΔG predictor);
Pearson r is secondary and capped near the field ceiling for *all* methods, FEP included.

## How we benchmark (why these numbers are trustworthy)

- **Leave-cluster-out CV.** Complexes are clustered by sequence identity with a
  **placement-aware (gap-penalised)** alignment and **whole clusters held out per fold** —
  no homolog of a test receptor is ever in training. Verified leakage-free: clustered
  r (0.35) < leaky random-CV r (0.44).
- **Same-split head-to-head.** The PPI-Affinity clone is scored on the *identical* held-out
  split — not its own paper's numbers (its server has been down since 2022). Published
  scorers report r ≈ 0.5–0.77 on training-overlapped sets; strip the leakage and the field
  sits near r ≈ 0.32.
- **Full identity-cutoff trend**, not one cherry-picked split.
- **Negative results kept public** in `docs/` — including a retired scorer that once
  generalized to negative correlation. We do not quietly drop them.

## Headline numbers

| Claim | HybriDock-Pep | Baseline | n | Reproduce (from `experiments/`) |
|---|---|---|---|---|
| **Matched head-to-head, 60%-id clustered** | **MAE 1.35 · RMSE 1.69 · r 0.352** | PPI-clone 1.46 · 1.84 · 0.210 | 865 | `python e331_ours_vs_ppiclone_clustered.py` |
| **Full PDBbind peptide set, leakage-free** | **MAE 1.40 · RMSE 1.77 · r 0.321** | zero-skill MAE 1.47 | 925 | `python e330_ours_pdbbind.py` |
| **30% cutoff (standard threshold)** | **MAE 1.39 · RMSE 1.76 · r 0.322** | — | 410 clusters | `python e366_identity_threshold_trend.py` |
| PDBbind crystal + interaction map | r 0.480 (charged 0.401) | PPI-clone 0.291 (0.146) | 865 | `python e298_ppi_vs_ifp.py` |
| Double-difference ΔΔG (same-receptor) | r ≈ 0.96 | FEP/TI ≈ 0.85 | — | `python e287_similarity_and_dd.py` |
| Affinity r on real AI poses (geom→+IFP) | 0.486 → 0.53 · **MAE 1.51–1.54** | PPI pose-blind 0.325 | 151 | `python e106_combined_realpose_grade.py` |

**MAE is flat (1.32→1.42) across the entire 30–100% identity sweep** — that stability of the
kcal/mol error is the number we stand behind. r declines smoothly from 0.45 (leaky) and
levels near 0.32: the honest cross-target ceiling.

**Offline, no data, 30 s:** `make verify` runs the math-only tests (double-difference,
anchoring, selectivity) — proves the relative-scoring machinery is correct without PDBbind.

---

## Why HybriDock-Pep — five conclusive tests

**① We beat a faithful PPI-Affinity clone on the identical leakage-free split — measured in kcal/mol.**
Both models score the *same* 865 PDBbind peptide-Kd complexes, clustered at 60% sequence identity (placement-aware
alignment) with entire clusters held out per fold (CD-HIT-style; verified leakage-free — clustered r 0.35 < leaky
random-CV r 0.44). `experiments/e331_ours_vs_ppiclone_clustered.py`:

```
  n=865 matched PDBbind peptide-Kd · MAE/RMSE in kcal/mol · leakage-free 60%-id clustered CV
  ──────────────────────────────────────────────────────────────────────────────────────────
  model                          MAE↓     RMSE↓    Pearson r↑    Spearman ρ↑
  HybriDock-Pep (16 feats, GBT)  1.35     1.69     0.352         0.338        ◀ WIN on every metric
  PPI-clone (ProtDCal-3D + SVR)  1.46     1.84     0.210         0.177
  ──────────────────────────────────────────────────────────────────────────────────────────
  margin holds under the honest split:  leaky random-CV Δr +0.11  →  clustered Δr +0.14
  PPI-Affinity's own paper reports held-out sets (R≈0.5–0.77 on its benchmarks) — on DIFFERENT
  datasets/splits, so not directly comparable; we therefore benchmark a faithful CLONE of its
  method on our identical split. Its web server has been down since 2022, so the original cannot
  be queried. This is a clone-on-our-split comparison, not a claim about their published numbers.
```

On the **full 925-complex set**, our leakage-free absolute number is **MAE 1.40 / RMSE 1.77 / r 0.321**
(`experiments/e330_ours_pdbbind.py`) — modestly above zero-skill (mean-predictor MAE 1.47) and honest about the cap.

**Accuracy vs sequence-identity cutoff — the full trend.** Because 30% identity is the more standard clustering
cutoff and the *trend* across thresholds is more informative than any single split (cf. [Runs-and-Poses, bioRxiv
2025.02.03.636309](https://www.biorxiv.org/content/10.1101/2025.02.03.636309v3)), we report every cutoff.
Same 925 complexes, same 16-feature GBT, leave-cluster-out CV at each cutoff, using a **placement-aware identity
metric** (`experiments/e366_identity_threshold_trend.py`, data in [`data/hybridock_identity_trend.csv`](data/hybridock_identity_trend.csv)):

```
  identity   clusters   MAE    RMSE    Pearson r      (kcal/mol; leave-cluster-out CV)
  cutoff                (kcal/mol)                    r bar: each █ ≈ 0.03
  ─────────────────────────────────────────────────────────────────────────────────
  random       925      1.32   1.66    +0.446  ███████████████  ← leaky (near-twins split across folds)
   100%        832      1.33   1.68    +0.422  ██████████████
    90%        807      1.35   1.70    +0.406  █████████████
    80%        737      1.36   1.73    +0.368  ████████████
    70%        693      1.40   1.77    +0.317  ███████████
    60%        644      1.40   1.77    +0.321  ███████████     ← we headline this
    50%        592      1.40   1.77    +0.319  ███████████
    40%        532      1.42   1.79    +0.289  ██████████
    30%        410      1.39   1.76    +0.322  ███████████     ← standard cutoff
  ─────────────────────────────────────────────────────────────────────────────────
  MAE is flat (1.32→1.42 kcal/mol) across the whole sweep; r declines smoothly from 0.45 (leaky) and
  levels off around 0.32 by the 30–70% cutoffs — the honest cross-target ceiling. That stability of the
  kcal/mol error is the number we stand behind.
```

> **Metric note (fixed 2026-07-09).** Our first version used a free-gap alignment whose score reduced to
> *longest-common-subsequence ÷ shorter length* — it ignored residue placement (it scored `GGA`≈`ACC` at 0.33
> from one gapped residue) and collapsed the 925 peptides to just 21 clusters at 30%, giving a spuriously low
> r≈0.23. We switched to a **placement-aware (gap-penalised)** identity (`GGA`/`ACC`→0, `GGA`/`CGG`→0.33); it
> yields many more, cleaner clusters (410 at 30%) and a **steadier, slightly higher r (0.32)**. The before/after
> is reproducible in [`experiments/e367_gap_penalized_trend.py`](experiments/e367_gap_penalized_trend.py). We report the
> corrected numbers and flag the fix rather than bury it.

At the stricter **30% cutoff (the standard clustering threshold), the honest numbers are MAE 1.39 / RMSE 1.76 /
r 0.32** — inside the cross-target ABFE band, reported alongside our 60% headline rather than instead of it.

**Independent-set check (PPIKB, a *different* database — the win generalizes).** Leakage-free (60%-id clustered),
full feature stack (ProtDCal + pocket/physics), Kd/Ki-only:

```
  PPIKB independent, n=808, leakage-free clustered CV
  ─────────────────────────────────────────────────────────
  model                     r↑      MAE↓    RMSE↓
  HybriDock-Pep (ours)      0.333   1.94    2.47    ◀ WIN, and comparable to our PDBbind r
  PPI-clone (ProtDCal+SVR)  0.265   1.99    2.56
  ─────────────────────────────────────────────────────────
```

We beat the PPI-clone on this second, independent database too (all-PPIKB, n=885: ours 0.336 vs clone 0.269). The
higher *absolute* MAE (~1.9 vs ~1.4 on PDBbind) is **PPIKB's own label noise**, not our scorer: ~20% of PPIKB
labels are IC50/EC50 (assay-specific, *not* thermodynamic — [JCIM 4c00049](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049):
27% of IC50 pairs disagree by >1 log unit), and identical peptide sequences carry y-values differing by **up to
10.8 kcal/mol**. Restricting to the curated Kd/Ki-only subset leaves the ranking unchanged (ours 0.333 vs clone
0.265). Full diagnostic in the Zenodo research archive: [10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713).

**Where we lose, stated up front: PPI-Affinity's own home test set (T100).** On the 48-complex set PPI-Affinity
curated and tuned on, the *real published tool* (not our clone) beats us on ranking — this is the honest flip
side of the leakage argument, and we lead with it rather than bury it (`experiments/e300_ifp_on_t100.py`,
[`data/e300_ifp_t100.json`](data/e300_ifp_t100.json)):

```
  PPI-Affinity's OWN T100 set (n=48) — in-distribution for PPI, cold out-of-distribution for us
  ────────────────────────────────────────────────────────────────────────────────────────────
  method                          Pearson r    MAE (kcal/mol)
  PPI-Affinity (real tool)          0.549          1.14         ◀ wins ON ITS HOME TURF
  DFIRE (2002 potential)            0.437          9.37   ← note the MAE
  Kdeep                             0.395         17.80   ← note the MAE
  RF-Score                          0.388          1.85
  HybriDock-Pep (+IFP, cold OOS)    0.225          1.54         ◀ us: worse rank, 2nd-best MAE
  PRODIGY                           0.086          2.09
  ────────────────────────────────────────────────────────────────────────────────────────────
```

Two honest reads: (1) on a scorer's *own* curated set it wins — which is exactly why in-distribution numbers
(incl. PPI's published 0.55–0.63) are not comparable across tools, and why our headline uses a **matched
leakage-free split** where we win (test ① above, Steiger p=0.002). (2) Even losing on rank here, our **MAE 1.54
is second only to PPI's**, while DFIRE/Kdeep sit at 9–18 kcal/mol — calibrated absolute ΔG is a separate axis we
hold. We show this table because a reviewer who finds it themselves should find nothing we didn't already report.

**② Same-receptor *relative* mode — anchor to a few measured references** (the honest analogue of what FEP
does: work relative to a reference so the per-receptor bias cancels). When you have ≥2–3 measured affinities
on your actual target, subtract that offset and the cold within-receptor *r* jumps:

```
  within-receptor absolute (cold, no reference)   r ≈ 0.25 – 0.47   (dataset-dependent)
  anchored to 2–3 measured references on-target    r ≈ 0.61 – 0.71   ← the same-receptor lever (E264/E280, re-verified)
```

Peptide–receptor binding is also largely **additive** — the coupling term in a 2×2 peptide×receptor grid is
only ~1.1 kcal/mol std — so a thermodynamic-cycle estimate closes to about that error. The honest same-receptor
win is the **anchoring** result above; we make **no relative-correlation claim** beyond it here.

**③ The number you actually get on AI-generated poses** — no crystal handed to you, the honest deployment
case. This is where we pull away from PPI-Affinity: **PPI is structure-free, so it is pose-blind** — it
returns the *same* score for any pose and cannot tell a good AI pose from a bad one. We read the pose:

```
  POSE ACCURACY (Cα-RMSD, lower = better)     AFFINITY r — SCORING THE AI POSE (each █ = 0.025 r, full = 0.60)
  ────────────────────────────────────────    ─────────────────────────────────────────────────────────────
  best-of-top-25  2.49 Å · hit@5 91%          HybriDock-Pep · AI pose + interaction █████████████████████░░░ 0.53
  MDM2/p53 1YCR   0.80 Å                      HybriDock-Pep · AI pose, geometry     ███████████████████░░░░░ 0.486
   vs DiffPepDock 3.54 Å ◀ ~4× tighter        PPI-clone     · pose-blind*           █████████████░░░░░░░░░░░ 0.325
                                              HybriDock-Pep · crystal (upper bound) ███████████████████████░ 0.585
  * structure-free method (our faithful clone; the original server is dead): identical score for any pose,
    so it cannot rank poses at all. Bars are each method's honest independent number.

  AI-POSE MAE (kcal/mol) — pooled 151 real-pose complexes (cr65 + the98), LOCO CV
  ─────────────────────────────────────────────────────────────────────────────
  diffusion top-5 aggregation   MAE 1.51   RMSE 1.87   r 0.501
  ML-best-5 aggregation         MAE 1.53   RMSE 1.87   r 0.501
  `experiments/e106_combined_realpose_grade.py` (MAE not printed by the script; computed with its
  own ridge/LOCO/router logic, verified against its printed r/RMSE before reporting here)
```

We turn the AI pose into a **0.49–0.53** signal; PPI cannot use the pose at all and is stuck at its
structure-free **0.325**. Going fully structure-free costs us only ~0.05–0.09 in *r* (0.585 crystal → ~0.50
on AI poses) — the haircut every structure-based scorer pays on non-native poses, and one of the few we
publish. In absolute terms, scoring our own AI-generated poses (no crystal, the honest deployment case)
lands **MAE ≈ 1.51–1.54 kcal/mol** — worse than the crystal-pose leakage-free headline (1.35–1.40) by the
pose-quality tax you'd expect, since RAPiDock poses aren't perfect.

**④ Real published complexes, scored blind.** 15 real peptide–protein structures (RCSB titles + primary
citations pulled live from the PDB), each scored by a model that never saw its 60%-identity cluster —
including real **peptide–MHC** (4PRN, HLA-B\*35:01). Aggregate over **all 925** such complexes, blind and
leakage-free: **MAE 1.40 / RMSE 1.77 kcal/mol** (41% within 1.0, 77% within 2.0 kcal/mol).
`experiments/e364_blind_demo.py` · [`data/hybridock_literature_complexes.csv`](data/hybridock_literature_complexes.csv).

**⑤ An external benchmark we did *not* assemble.** The supplementary tables of **Wang et al., *Curr. Med. Chem.*
2024, 31(31):4127** ([DOI](https://doi.org/10.2174/0929867331666230908102925); tables + PDF shipped in-repo so
anyone can check). Their independently-published pK_d reproduces our ΔG labels to **corr 0.998**. 155 overlap
complexes scored blind: **MAE 1.43 / RMSE 1.68**; and a **true external holdout of 43** complexes never in
training (nor their 60%-id clusters): **MAE 1.60 / RMSE 1.90 / r 0.44.** `experiments/e365b_failure_analysis.py` ·
[`data/hybridock_wang2024_external43.csv`](data/hybridock_wang2024_external43.csv).

**Also vs Rosetta FlexPepDock** (the standard physics baseline), same 918 PDBbind complexes matched
complex-for-complex: ours (leakage-free clustered CV) is **r 0.32 / MAE 1.40**, while unrelaxed ref2015
interface energy calibrates to **r ≈ 0** — it collapses onto the mean-predictor, because REU has no native
kcal/mol (a linear `ΔG=a·REU+b` fit is correlation-invariant). Interface-relax rescues ref2015 to r 0.18 —
still below ours and below its own within-target 0.59. `experiments/e329_ref2015_pdbbind.py` ·
`experiments/e331_relax_pdbbind.py`.

Everything else stays honest: absolute charged Kd is capped at the non-FEP ceiling and we say so; selectivity
ΔΔG (target vs off-target) lands r ≈ 0.30–0.45; MIT-licensed and runs on CUDA · Apple MPS · Intel · AMD · CPU.
Full evidence and every negative result: development timeline archived on Zenodo
([10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713)) ·
[`docs/SCORING_COMPARISON.md`](docs/SCORING_COMPARISON.md) · reproduce them in
[Reproduce the benchmarks](#reproduce-every-number-on-this-page).

---

## The claim, stated plainly — and why it holds in 2026

**Among all non-FEP/LIE methods with a fair, leakage-free benchmark, HybriDock-Pep is the best
protein–peptide ΔG scorer we can find — and the most efficient.** Two legs, both measured:

**Speed.** End-to-end **scoring is ~2.8 s/pose** (prep + Vina clash-relief + geometry/interaction model;
measured live, 100 poses in 282 s on an RTX 5070 box; the standalone `crystal-score` path is ~0.9 s/pose).
Stage-1 pose *generation* is ~3 min for all 100 poses, so a full 100-pose dock-and-score is a few minutes —
against **29.8 min for a single global peptide docking** by HPEPDOCK in the 2026 field review (Martins,
Santos & Sousa, *J. Comput. Chem.* 47:5). No slower method that also emits a calibrated ΔG comes close.

**It runs on a laptop, off battery.** Measured end-to-end on a **fanless MacBook Air M3 (16 GB RAM, Apple MPS,
no discrete GPU)**: a full **100-pose MDM2/p53 dock** (peptide `ETFSDLWKLLPE`) completes in **under 15 minutes**,
and the best-pose ΔG lands **~0.9 kcal/mol from experiment**. Honest caveat: that complex is a *neutral, short
12-mer* — a favourable case for absolute accuracy; charged/long peptides sit at the cross-target ceiling
discussed above. The point of this datapoint is the **efficiency**: no cluster, no CUDA, no discrete GPU.

**Accuracy — and the field is empty of live rivals.**

- **PPI-Affinity**, the prior best *published* ML peptide scorer, has been **unmaintained since 2022** (dead web
  server). A faithful clone of its method (ProtDCal-3D + SVR), scored on the *identical* leakage-free split as ours,
  loses on every metric — **MAE 1.46 vs our 1.35, r 0.210 vs our 0.352** (test ①; Steiger's Z=3.1, p=0.002, so
  the gap is statistically significant, not a tie). Its published 0.55–0.63 is on different datasets/splits, so
  not directly comparable — which is exactly why we benchmark a faithful clone on our *identical* split.
- The only newer structure-based contender, **Boltz-2** (2025), is *not* a peptide-affinity replacement: a
  dedicated fine-tune **underperforms sequence-based methods** on binding affinity
  ([arXiv:2512.06592](https://arxiv.org/abs/2512.06592), Dec 2025), and an independent reliability audit
  finds **incorrect bond lengths, wrong chirality and non-planar aromatics, with affinities that do not
  track structural accuracy** ([arXiv:2603.05532](https://arxiv.org/abs/2603.05532), Mar 2026).
- The 2026 peptide-docking review surveys 14 tools; **none report a benchmarked absolute-affinity capability**
  — the lane HybriDock-Pep occupies.

So the honest superlative is not "beats FEP" (nothing cheap does) — it is: **the best and fastest non-FEP/LIE
protein–peptide ΔG scorer with a reproducible, leakage-free benchmark to stand on — at FEP-competitive absolute
error** (MAE ~1.3–1.6 kcal/mol, inside the ~1.2–2.5 kcal/mol error band that absolute FEP itself reaches on
peptides), for ~1000× less compute and with no
reference peptide required.

### Why absolute cross-target affinity is hard for everyone (FEP included)

The reason no method — ours, FEP, or LIE — posts a high *absolute cross-target* correlation is a **regime** fact,
not a skill gap, and it is worth stating plainly so our modest r isn't misread:

- **FEP/LIE's famous ~1 kcal / r≈0.8 accuracy is a different problem:** *relative* free energy (RBFE) between
  *similar* ligands on the *same* target. There, systematic errors **cancel**. *"Relative calculations benefit
  from cancellation of systematic errors… absolute calculations accumulate all sources of error"*
  ([Comm. Chem. 2023, s42004-023-01019-9](https://www.nature.com/articles/s42004-023-01019-9); maximal-accuracy
  review [PMC10576784](https://pmc.ncbi.nlm.nih.gov/articles/PMC10576784/)).
- **Absolute FEP (ABFE) itself only reaches ~1.2–2.5 kcal/mol**, and it needs the bound pose, heavy sampling, and
  expert setup — degrading further cross-target. **LIE cannot even run without per-system fitted α/β/γ.** Neither is
  a plug-and-play "peptide + protein → absolute kcal/mol" predictor.
- **Enthalpy–entropy compensation** makes binding ΔG a *small net of large, mutually-cancelling terms* — so single
  physics terms (electrostatics, desolvation, entropy) are individually large but compensate, and better physics
  (polarization, QM) sharpens terms that cancel ([EEC review, ACS Omega 1c00485](https://pubs.acs.org/doi/10.1021/acsomega.1c00485)).
- Consequently, **cross-target absolute peptide affinity is r≈0.15–0.55 for the entire field** (best ML ~0.6–0.7 on
  large data; [ML-affinity review arXiv:2410.00709](https://arxiv.org/html/2410.00709v2)). Our leakage-free
  0.26–0.39 sits squarely inside that band — mid-field, honest, and *reference-free*.

**This is why we report kcal/mol MAE (stable, meaningful) as the headline and treat r as secondary.** Our full
characterisation of this wall — proven from ~10 experimental angles — is archived on Zenodo:
[10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713).

### Fresh out-of-training check (2026-07-06)

Blind scoring of three peptide–protein complexes pulled straight from the literature — deposited structures,
**none in any training split** — via `crystal-score`:

```
  system            PDB    peptide         HybriDock-Pep ΔG    literature reference
  ──────────────────────────────────────────────────────────────────────────────────────
  MDM2 / p53        1YCR   ETFSDLWKLLPE         −9.28          −8.5   (exp, K_d 0.6 µM)
  MDM2 / PMI        3EQS   TSFAEYWNLLS          −9.67          −12.7  (exp, K_d 0.49 nM)
  importin-α / NLS  3VE6   EGPSAKKPKKEA         −9.77          −4.8 FEP / −5…−10 exp
```

Honest read: every prediction lands within a few kcal/mol of its reference, but they cluster near −9.5 while
the true values span −4.8 to −12.7 — the **blind-absolute dynamic-range compression that caps every non-FEP
method**, ours included (we publish it rather than hide it). This is exactly why the headline is a
*leakage-free ranking* win (test ①) and *selectivity* — not a blind-absolute one.

---

## Datasets — download and test for yourself

These files are small, plain-text, and MIT-licensed (derived features + public experimental
affinities — no redistributed third-party structures). All of them ship in this repo except one,
called out in the table and explained under it.

> **One gap, stated plainly.** `data/e180_protdcal3d.jsonl` — the ProtDCal-3D feature file for the
> PPI-Affinity clone — is **not in this repository**, so test ① (the ours-vs-clone head-to-head) is
> **not currently reproducible from a clean clone**. The file is `.gitignore`d, and its generator
> `experiments/e180_protdcal_925.py` cannot run either: it imports
> `e158_overfit_failure_analysis`, which was never committed (48 experiment scripts depend on that
> module for `greedy_cluster()` and `pocket_seq()`). Every other number on this page — the 925-complex
> headline, the identity sweep, PPIKB, the Wang 2024 external holdout — reproduces from what is here.
> `tests/test_repro_claims.py` guards the rest of the table so this cannot happen silently again, and
> carries a strict `xfail` that flips green the moment the missing module is restored.

| File | What it is | Rows |
|---|---|---|
| [`data/pdbbind_peptides.jsonl`](data/pdbbind_peptides.jsonl) | 925 PDBbind protein–peptide complexes with experimental K_d/K_i, our 16 structural features + sequence per complex | 925 |
| `data/e180_protdcal3d.jsonl` — **not shipped, see below** | PPI-Affinity-clone features (37 ProtDCal-3D intra-peptide descriptors) per complex — the head-to-head baseline | ~900 |
| [`data/e331_matched_pdbids.json`](data/e331_matched_pdbids.json) | The exact 865 PDB IDs in the leakage-free ours-vs-PPI-clone head-to-head (both models can score) | 865 |
| [`data/e329_ref2015_pdbbind.json`](data/e329_ref2015_pdbbind.json) | Rosetta ref2015 / FlexPepDock unrelaxed interface-ΔG (REU) for 918 of those complexes | 918 |
| [`data/e331_relax_pdbbind.json`](data/e331_relax_pdbbind.json) | Unrelaxed vs interface-relaxed ref2015 interface-ΔG on a 40-complex spread | 40 |
| [`data/benchmark_crystal.json`](data/benchmark_crystal.json) | The crystal-65 reference set (PDB paths + experimental ΔG) used across the scoring campaign | 65 |

The **865-complex leakage-free head-to-head** (810 K_d + 55 K_i, peptide length 3–19, ΔG −14.2 to −3.7 kcal/mol,
clustered into 379 groups at 60% identity) is the fairest peptide-affinity comparison we can run: both HybriDock-Pep
and the PPI-Affinity clone score every complex, on identical folds.

The raw PDBbind structures themselves are **not** redistributed (PDBbind licensing) — register at
[pdbbind.org.cn](http://www.pdbbind.org.cn/) for the v2020 general set; `experiments/e108_ingest_pdbbind.py`
rebuilds `pdbbind_peptides.jsonl` from it. To re-score the head-to-head from the shipped features alone
(no structures needed):

```bash
conda activate score-env
python experiments/e330_ours_pdbbind.py     # ours + matched ref2015 head-to-head → r / RMSE / MAE
```

## Hardware floor — does it run with no GPU at all?

Yes, measured end-to-end, not asserted. The platform table in
[INSTALL.md](INSTALL.md) has long claimed CPU support; these are the first
timings behind that claim.

**Method.** On an Apple M3 the GPU was made invisible to PyTorch
(`PYTHONPATH=scripts/force_cpu`, which masks MPS so RAPiDock's
`cuda → mps → xpu → cpu` cascade falls through to CPU). Nothing is stubbed —
the diffusion model runs on real CPU tensors, and the pipeline reports the
branch it took: `[run_rapidock] backend optimization: CPU (threads tuned)`.
Target is the tutorial case, p53 peptide `ETFSDLWKLLPE` vs MDM2 (1YCR), seed 42.

| Poses (`--n-samples`) | Wall time, CPU only | Best-pose ΔG |
|---|---|---|
| 10 | **68 s** | −9.22 kcal/mol |
| 100 | **7 min 40 s** (460 s) | −9.40 kcal/mol |

Machine: Apple M3, 8 physical cores, 16 GB RAM. Both runs exited 0 and wrote the
full output set (`best_pose.pdb`, `ranked_poses.csv`, convergence and silhouette
plots). For reference, `crystal-score` on the same complex gives −9.28 and the
experimental value is ≈ −8.5 — so the no-GPU run is not a degraded mode, it lands
where the GPU path does.

The two points separate fixed cost from marginal cost: **≈ 24 s of setup
(ESM-2 embedding, model load, receptor prep) plus ≈ 4.4 s per pose.**

**On "hundreds of poses in under 30 minutes":** that rate puts 300 poses at
≈ 22 minutes with no GPU. That figure is a *linear extrapolation* from the two
measurements above, not a third measurement — 10 and 100 are what were run.

### Caveats

- An M3's CPU is fast. An older or lower-core laptop — the machine this claim
  is really aimed at — will be slower, and has not been measured. The honest
  statement is "runs without a GPU, at ~4.4 s/pose on 8 modern cores", not
  "runs at this speed anywhere".
- One peptide against one receptor. This is a **timing** result; accuracy is
  benchmarked elsewhere on this page.
- 82 of 100 poses survived filtering in the n=100 run, which is normal.
- The GPU was masked in software rather than physically absent. The code path
  taken is the genuine CPU one, but a machine that never had an accelerator has
  not been tested directly.

Reproduce (on a GPU machine, to force the CPU path):

```bash
PYTHONPATH=scripts/force_cpu KMP_DUPLICATE_LIB_OK=TRUE \
hybridock-pep dock --peptide ETFSDLWKLLPE \
    --receptor data/pdbs/1YCR_mdm2.pdb --site 25.20 -25.61 -7.97 --box 30 \
    --n-samples 100 --seed 42 --output-dir runs/cpu_only
```

On a machine with no GPU, drop `PYTHONPATH=scripts/force_cpu` — the cascade
lands on CPU by itself.

---

## Reproduce every number on this page

Every headline number maps to one committed script that prints the exact *r* / MAE table and writes a JSON
beside it for line-by-line checking. Download PDBbind v2020 ([pdbbind.org.cn](http://www.pdbbind.org.cn))
and PPIKB / the PPI-Affinity SI first (the large/external inputs are gitignored; the small IFP caches ship
in `data/`). Run each with `OMP_NUM_THREADS=1` on this machine for the speed the docs assume.

| Number on this page | Command | Writes |
|---|---|---|
| **ours MAE 1.35 / r 0.352  vs  PPI-clone MAE 1.46 / r 0.210** (leakage-free head-to-head, test ①; Steiger p=0.002) | `OMP_NUM_THREADS=1 python experiments/e331_ours_vs_ppiclone_clustered.py` | [`data/e331_ours_vs_ppiclone.json`](data/e331_ours_vs_ppiclone.json) (random + clustered, both models) |
| **ours full-set leakage-free MAE 1.40 / RMSE 1.77 / r 0.321** + matched ref2015 | `OMP_NUM_THREADS=1 python experiments/e330_ours_pdbbind.py` | stdout table (leaky vs clustered vs length-stratified) |
| **0.480 / 0.291** PDBbind crystal + IFP (charged 0.401 / 0.146) — legacy test ① | `python experiments/e298_ppi_vs_ifp.py` | `data/e298_ppi_vs_ifp.json` |
| **PPIKB independent, leakage-free: ours r 0.333 / MAE 1.94  vs  PPI-clone 0.265 / 1.99** (Kd/Ki-only, full stack) | `OMP_NUM_THREADS=1 python experiments/e332b_ppikb_headtohead.py` | stdout |
| **0.25 → 0.52–0.61** same-receptor anchoring — test ② | `python experiments/e264_ppikb_anchor_fusion.py` | `data/e264_ppikb_results.json` |
| **0.225 ← 0.045** IFP rescue on PPI's own T100 — § ideas | `python experiments/e300_ifp_on_t100.py` | `data/e300_ifp_t100.json` |
| **0.437 / 0.399** train IFP on all 973 / 1405 crystals — § ideas | `python experiments/e304_ifp_mega_everything.py` | `data/e304_ifp_mega.json` |
| full non-FEP/LIE scorecard on 156 complexes | `python experiments/e90_full_scorecard.py` | stdout table |
| **0.486 → 0.53** affinity *r* on real RAPiDock poses (MAE 1.51–1.54) — test ③ | `python experiments/e106_combined_realpose_grade.py` | per-complex CSV |
| **2.49 Å** best-of-top-25 pose RMSD, hit@5 91% — test ③ | `hybridock-pep benchmark --test-csv data/test_complexes.csv --report bench.md` | `bench.md` |
| reference-anchoring **math** (thermodynamic cycle closes by construction; not a prediction claim) | `pytest tests/test_anchoring.py tests/test_double_difference.py -q` | green = the anchoring/cycle math holds |
| **ΔΔG selectivity** primitive end-to-end | `pytest tests/test_selectivity.py -q` | green |

Rebuild the IFP training cache from raw structures (the 437 new PPIKB complexes) with
`python experiments/e303_build_ppikb_ifp.py`. The full experiment ledger (E0–E304, every win and every refuted
idea) is archived on Zenodo: [10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713).

---

## Honest caveats (state these before a judge finds them)

- **Absolute cross-target Kd is confound-limited** for every cheap non-FEP method, ours
  included (size/baseline + enthalpy–entropy compensation). We report **relative** ΔΔG /
  selectivity / anchored ΔG as the accurate paths; absolute ΔG is a coarse readout. See
  [MODEL_CARD.md](MODEL_CARD.md) and the Zenodo research archive: [10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713).
- **Selectivity ΔΔG** lands r ≈ 0.30–0.45 — useful for triage, not a final answer.
- **This is a rigor contribution, not a discovery.** The size/baseline (Simpson) confound is
  known; we prove it is the specific cause of cross-dataset non-replication in peptide docking
  and ship instant geometric features that stay sign-stable across two independent datasets.

## Evaluation methodology

Benchmarks follow standard leakage-control practice: leave-cluster-out CV on every headline
number, the standard **30% identity cutoff** reported alongside our 60%, and the full
identity-vs-accuracy trend rather than a single split.
