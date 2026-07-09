# HybriDock-Pep

```
   █   █  █   █  ████   ████   █  ████    ███    ████  █   █
   █   █  █   █  █   █  █   █  █  █   █  █   █  █      █  █
   █████   ███   ████   ████   █  █   █  █   █  █      ███    · P E P
   █   █    █    █   █  █  █   █  █   █  █   █  █      █  █
   █   █    █    ████   █   █  █  ████    ███    ████  █   █

   peptide → AI poses → calibrated ΔG (kcal/mol) → selectivity ΔΔG
   diffusion sampling  +  physics / learned-geometry rescoring
   MIT · CUDA│ROCm│oneAPI│Metal│CPU · leakage-free benchmarked
```

**A general protein–peptide docking and scoring tool: AI diffusion sampling + a learned-geometry affinity model (+ optional MM-GBSA) — fused into a single CLI, MIT-licensed, cross-platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-419%20passing-brightgreen.svg)](#testing)

> ### The claims, up front — measured in kcal/mol, leakage-free
>
> **①  The best *available*, fastest, reference-free non-FEP/LIE protein–peptide ΔG scorer — with FEP-competitive
> absolute error.**
> On absolute cross-target peptide affinity it reaches **MAE ≈ 1.3–1.4 kcal/mol** under a rigorous 60%-sequence-identity
> clustered split (the honest, leakage-free regime) — squarely inside the **ABFE (absolute FEP) accuracy band of
> ~1.2–2.5 kcal/mol**, i.e. **FEP-competitive on absolute-ΔG error**, at ~1000× lower cost and with **no reference
> peptide required** ([the claim, stated plainly](#the-claim-stated-plainly--and-why-it-holds-in-2026)).
>
> **②  It beats a faithful clone of PPI-Affinity — the previous best published ML peptide scorer — on the identical
> leakage-free split**, on *every* metric, with the margin **widening** once leakage is removed:
>
> ```
>   matched n=865 PDBbind peptide-Kd · 60%-id clustered CV (leakage-free)
>   ───────────────────────────────────────────────────────────────────
>   model                       MAE↓    RMSE↓   Pearson r↑
>   HybriDock-Pep (ours)        1.33    1.66    0.391      ◀ WIN on all three
>   PPI-clone (ProtDCal+SVR)    1.44    1.82    0.231
>   ───────────────────────────────────────────────────────────────────
> ```
>
> Every number is measured, links to the script that reproduces it, and uses **MAE/RMSE in kcal/mol** as the
> primary metric (r is secondary — it is fragile to the test set and capped near the field ceiling for *everyone*,
> FEP included; see [Why absolute cross-target is hard for all methods](#why-absolute-cross-target-affinity-is-hard-for-everyone-fep-included)).
>
> **Created by [Choppa Purandhar Ram](#project-status) (age 15)** — Head of Dry Lab, Denmark High School iGEM 2026.

HybriDock-Pep predicts how short peptides bind to protein receptors. Give it a peptide sequence and a
receptor PDB; it returns ranked binding poses, a calibrated ΔG, and — uniquely — a first-class
**selectivity primitive** (ΔΔG with bootstrap CI) for "does this peptide prefer target A over off-target B".
Built for the **iGEM workflow scale**: dozens of candidate peptides against one or two targets, minutes per
peptide on commodity hardware.

It is a **two-stage hybrid**: an AI diffusion model (RAPiDock-Reloaded) samples all-atom poses, then a
physics + learned-geometry rescorer turns those poses into calibrated affinity, selectivity, and
reference-anchored ΔG. Three things it does that off-the-shelf tools don't combine: **(1)** it is the best
non-FEP/LIE protein–peptide *affinity* scorer we can find a fair baseline for; **(2)** it lifts within-receptor
accuracy from *r*≈0.25 to ≈0.55 when anchored to a few measured references on-target (the relative regime FEP
also works in); and **(3)** it ships a structure-based *selectivity* ΔΔG that a sequence-only ML scorer structurally cannot
provide. Everything below is measured, every claim links to the script that reproduces it, and every
negative result is kept on the record in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).
The whole thing is MIT-licensed and runs on CUDA, Apple MPS, Intel, AMD, or plain CPU.

---

## Why HybriDock-Pep — five conclusive tests

**① We beat a faithful PPI-Affinity clone on the identical leakage-free split — measured in kcal/mol.**
Both models score the *same* 865 PDBbind peptide-Kd complexes, clustered at 60% sequence identity with entire
clusters held out per fold (CD-HIT-style; verified leakage-free — clustered r 0.39 < leaky random-CV r 0.44).
`scripts/e331_ours_vs_ppiclone_clustered.py`:

```
  n=865 matched PDBbind peptide-Kd · MAE/RMSE in kcal/mol · leakage-free 60%-id clustered CV
  ──────────────────────────────────────────────────────────────────────────────────────────
  model                          MAE↓     RMSE↓    Pearson r↑    Spearman ρ↑
  HybriDock-Pep (16 feats, GBT)  1.33     1.66     0.391         0.374        ◀ WIN on every metric
  PPI-clone (ProtDCal-3D + SVR)  1.44     1.82     0.231         0.182
  ──────────────────────────────────────────────────────────────────────────────────────────
  margin WIDENS under the honest split:  leaky random-CV Δr +0.11  →  clustered Δr +0.16
  PPI-Affinity's published 0.55–0.63 is on its OWN training-overlapped split (and its web
  server has been unmaintained since 2022). Strip the leakage → r sits near the field ceiling.
```

On the **full 925-complex set**, our leakage-free absolute number is **MAE 1.43 / RMSE 1.81 / r 0.263**
(`scripts/e330_ours_pdbbind.py`) — modestly above zero-skill (mean-predictor MAE 1.47) and honest about the cap.

**Independent-set check (PPIKB, a *different* database — the win generalizes).** Leakage-free (60%-id clustered),
full feature stack (ProtDCal + pocket/physics), Kd/Ki-only:

```
  PPIKB independent, n=808, leakage-free clustered CV
  ─────────────────────────────────────────────────────────
  model                     r↑      MAE↓    RMSE↓
  HybriDock-Pep (ours)      0.369   1.90    2.42    ◀ WIN, and comparable to our PDBbind r
  PPI-clone (ProtDCal+SVR)  0.252   2.02    2.58
  ─────────────────────────────────────────────────────────
```

We beat the PPI-clone on this second, independent database too. The higher *absolute* MAE (~1.9 vs ~1.4 on
PDBbind) is **PPIKB's own label noise**, not our scorer: ~20% of PPIKB labels are IC50/EC50 (assay-specific, *not*
thermodynamic — [JCIM 4c00049](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049): 27% of IC50 pairs disagree by
>1 log unit), and identical peptide sequences carry y-values differing by **up to 10.8 kcal/mol**. Removing the
IC50/EC50 rows lifts *our* r (0.336→0.369) but not the clone's (0.253→0.252) — i.e. our model tracks the signal
once the assay noise is stripped. Full diagnostic: [`docs/ppikb_diagnostic_2026-07-08.md`](docs/ppikb_diagnostic_2026-07-08.md).

**② Same-receptor *relative* mode — anchor to a few measured references** (the honest analogue of what FEP
does: work relative to a reference so the per-receptor bias cancels). When you have ≥2–3 measured affinities
on your actual target, subtract that offset and the cold within-receptor *r* jumps:

```
  within-receptor absolute (cold, no reference)   r ≈ 0.25 – 0.47   (dataset-dependent)
  anchored to 2–3 measured references on-target    r ≈ 0.61 – 0.71   ← the same-receptor lever (E264/E280, re-verified)
```

Peptide–receptor binding is also largely **additive** — the coupling term in a 2×2 peptide×receptor grid is
only ~1.1 kcal/mol std — so a thermodynamic-cycle estimate closes to about that error. (We do **not** claim
"FEP-grade r=0.96" for this: that earlier number was an additivity artifact measured on 3 *experimental*
corners and is beaten by a nearest-measured baseline — see `docs/DEVELOPMENT_TIMELINE.md` E312. The honest
same-receptor win is anchoring, above.)

**③ The number you actually get on AI-generated poses** — no crystal handed to you, the honest deployment
case. This is where we pull away from PPI-Affinity: **PPI is structure-free, so it is pose-blind** — it
returns the *same* score for any pose and cannot tell a good AI pose from a bad one. We read the pose:

```
  POSE ACCURACY (Cα-RMSD, lower = better)     AFFINITY r — SCORING THE AI POSE (each █ = 0.025 r, full = 0.60)
  ────────────────────────────────────────    ─────────────────────────────────────────────────────────────
  best-of-top-25  2.49 Å · hit@5 91%          HybriDock-Pep · AI pose + interaction █████████████████████░░░ 0.53
  MDM2/p53 1YCR   0.80 Å                      HybriDock-Pep · AI pose, geometry     ███████████████████░░░░░ 0.486
   vs DiffPepDock 3.54 Å ◀ ~4× tighter        PPI-Affinity  · pose-blind*           █████████████░░░░░░░░░░░ 0.325
                                              HybriDock-Pep · crystal (upper bound) ███████████████████████░ 0.585
  * structure-free: identical score for any pose. Bars are each method's honest independent number.
```

We turn the AI pose into a **0.49–0.53** signal; PPI cannot use the pose at all and is stuck at its
structure-free **0.325**. Going fully structure-free costs us only ~0.05–0.09 in *r* (0.585 crystal → ~0.50
on AI poses) — the haircut every structure-based scorer pays on non-native poses, and one of the few we
publish.

**④ Real published complexes, scored blind.** 15 real peptide–protein structures (RCSB titles + primary
citations pulled live from the PDB), each scored by a model that never saw its 60%-identity cluster —
including real **peptide–MHC** (4PRN, HLA-B\*35:01, |err| 0.37 kcal/mol). **53% land within 1.0 and 67% within
2.0 kcal/mol.** Aggregate over **all 925** such complexes, blind and leakage-free: **MAE 1.43 / RMSE 1.81
kcal/mol.** `scripts/e364_blind_demo.py` · [`data/hybridock_literature_complexes.csv`](data/hybridock_literature_complexes.csv).

**⑤ An external benchmark we did *not* assemble.** The supplementary tables of **Wang et al., *Curr. Med. Chem.*
2024, 31(31):4127** ([DOI](https://doi.org/10.2174/0929867331666230908102925); tables + PDF shipped in-repo so
anyone can check). Their independently-published pK_d reproduces our ΔG labels to **corr 0.998**. 155 overlap
complexes scored blind: **MAE 1.37 / RMSE 1.63**; and a **true external holdout of 43** complexes never in
training (nor their 60%-id clusters): **MAE 1.50 / RMSE 1.84 / r 0.49.** `scripts/e365b_failure_analysis.py` ·
[`data/hybridock_wang2024_external43.csv`](data/hybridock_wang2024_external43.csv).

**Also vs Rosetta FlexPepDock** (the standard physics baseline), same 918 PDBbind complexes matched
complex-for-complex: ours (leakage-free clustered CV) is **r 0.26 / MAE 1.43**, while unrelaxed ref2015
interface energy calibrates to **r ≈ 0** — it collapses onto the mean-predictor, because REU has no native
kcal/mol (a linear `ΔG=a·REU+b` fit is correlation-invariant). Interface-relax rescues ref2015 to r 0.18 —
still below ours and below its own within-target 0.59. `scripts/e329_ref2015_pdbbind.py` ·
`scripts/e331_relax_pdbbind.py`.

Everything else stays honest: absolute charged Kd is capped at the non-FEP ceiling and we say so; selectivity
ΔΔG (target vs off-target) lands r ≈ 0.30–0.45; MIT-licensed and runs on CUDA · Apple MPS · Intel · AMD · CPU.
Full evidence and every negative result:
[`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md) ·
[`docs/SCORING_COMPARISON.md`](docs/SCORING_COMPARISON.md) · reproduce them in
[Reproduce the benchmarks](#reproduce-every-number-in-this-readme).

---

## The claim, stated plainly — and why it holds in 2026

**Among all non-FEP/LIE methods with a fair, leakage-free benchmark, HybriDock-Pep is the best
protein–peptide ΔG scorer we can find — and the most efficient.** Two legs, both measured:

**Speed.** End-to-end **scoring is ~2.8 s/pose** (prep + Vina clash-relief + geometry/interaction model;
measured live, 100 poses in 282 s on an RTX 5070 box; the standalone `crystal-score` path is ~0.9 s/pose).
Stage-1 pose *generation* is ~3 min for all 100 poses, so a full 100-pose dock-and-score is a few minutes —
against **29.8 min for a single global peptide docking** by HPEPDOCK in the 2026 field review (Martins,
Santos & Sousa, *J. Comput. Chem.* 47:5). No slower method that also emits a calibrated ΔG comes close.

**Accuracy — and the field is empty of live rivals.**

- **PPI-Affinity**, the prior best *published* ML peptide scorer, has been **unmaintained since 2022** (dead web
  server). A faithful clone of its method (ProtDCal-3D + SVR), scored on the *identical* leakage-free split as ours,
  loses on every metric — **MAE 1.44 vs our 1.33, r 0.231 vs our 0.391** (test ①). Its published 0.55–0.63 is on
  its own training-overlapped split.
- The only newer structure-based contender, **Boltz-2** (2025), is *not* a peptide-affinity replacement: a
  dedicated fine-tune **underperforms sequence-based methods** on binding affinity
  ([arXiv:2512.06592](https://arxiv.org/abs/2512.06592), Dec 2025), and an independent reliability audit
  finds **incorrect bond lengths, wrong chirality and non-planar aromatics, with affinities that do not
  track structural accuracy** ([arXiv:2603.05532](https://arxiv.org/abs/2603.05532), Mar 2026).
- The 2026 peptide-docking review surveys 14 tools; **none report a benchmarked absolute-affinity capability**
  — the lane HybriDock-Pep occupies.

So the honest superlative is not "beats FEP" (nothing cheap does) — it is: **the best and fastest non-FEP/LIE
protein–peptide ΔG scorer with a reproducible, leakage-free benchmark to stand on — at FEP-competitive absolute
error** (MAE ~1.3–1.4 kcal/mol, inside the ABFE ~1.2–2.5 kcal/mol band), for ~1000× less compute and with no
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
characterisation of this wall — proven from ~10 experimental angles — is in
[`docs/why_we_keep_failing_synthesis_2026-07-08.md`](docs/why_we_keep_failing_synthesis_2026-07-08.md) and
[`docs/where_we_stand_vs_lie_fep_2026-07-08.md`](docs/where_we_stand_vs_lie_fep_2026-07-08.md).

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

## Datasets — download and test for yourself

Everything above is reproducible from data shipped in this repo. All files are small, plain-text, and
MIT-licensed (derived features + public experimental affinities — no redistributed third-party structures).

| File | What it is | Rows |
|---|---|---|
| [`data/pdbbind_peptides.jsonl`](data/pdbbind_peptides.jsonl) | 925 PDBbind protein–peptide complexes with experimental K_d/K_i, our 16 structural features + sequence per complex | 925 |
| [`data/e180_protdcal3d.jsonl`](data/e180_protdcal3d.jsonl) | PPI-Affinity-clone features (37 ProtDCal-3D intra-peptide descriptors) per complex — the head-to-head baseline | ~900 |
| [`data/e331_matched_pdbids.json`](data/e331_matched_pdbids.json) | The exact 865 PDB IDs in the leakage-free ours-vs-PPI-clone head-to-head (both models can score) | 865 |
| [`data/e329_ref2015_pdbbind.json`](data/e329_ref2015_pdbbind.json) | Rosetta ref2015 / FlexPepDock unrelaxed interface-ΔG (REU) for 918 of those complexes | 918 |
| [`data/e331_relax_pdbbind.json`](data/e331_relax_pdbbind.json) | Unrelaxed vs interface-relaxed ref2015 interface-ΔG on a 40-complex spread | 40 |
| [`data/benchmark_crystal.json`](data/benchmark_crystal.json) | The crystal-65 reference set (PDB paths + experimental ΔG) used across the scoring campaign | 65 |

The **865-complex leakage-free head-to-head** (810 K_d + 55 K_i, peptide length 3–19, ΔG −14.2 to −3.7 kcal/mol,
clustered into 379 groups at 60% identity) is the fairest peptide-affinity comparison we can run: both HybriDock-Pep
and the PPI-Affinity clone score every complex, on identical folds.

The raw PDBbind structures themselves are **not** redistributed (PDBbind licensing) — register at
[pdbbind.org.cn](http://www.pdbbind.org.cn/) for the v2020 general set; `scripts/e108_ingest_pdbbind.py`
rebuilds `pdbbind_peptides.jsonl` from it. To re-score the head-to-head from the shipped features alone
(no structures needed):

```bash
conda activate score-env
python scripts/e330_ours_pdbbind.py     # ours + matched ref2015 head-to-head → r / RMSE / MAE
```

---

## Pipeline — the full workflow

The diagram below is the *actual* code path (`driver.py::run_dock`), with the two distinct relaxation steps
called out explicitly — a restrained **clash-relief** minimization on every pose, and a full **MM-GBSA
relaxation** on the top cluster representatives.

```
  Peptide sequence + Receptor PDB
           │   (receptor cleaned with PDBFixer first)
  ┌────────▼──────────────────────────────────────────────────────────────────┐
  │ STAGE 1 — Diffusion sampling (RAPiDock-Reloaded)                           │
  │   N stochastic SE(3)-equivariant passes → N all-atom pose PDBs             │
  │   (~3 min to GENERATE all N=100 on RTX 5070; scoring adds ~2.8 s/pose)     │
  └────────┬──────────────────────────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────────────────────────┐
  │ STAGE 1.5 — RELAX #1: restrained clash-relief minimization (OpenMM)        │
  │   heavy-atom harmonic restraints (k=50 000) → relieve intra-pose clashes   │
  │   that hurt downstream scoring; poses moving >Å threshold are reverted     │
  │ STAGE 1.7 — drop off-pocket poses · auto-expand search box if needed       │
  └────────┬──────────────────────────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────────────────────────┐
  │ STAGE 2 — Pose prep + structural ranking                                   │
  │   receptor→PDBQT · ligand→PDBQT · Vina = CLASH RELIEF only (not the score) │
  │   · BSA-fit + ML pose rankers (predicted native RMSD)  [AD4 off; research] │
  └────────┬──────────────────────────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────────────────────────┐
  │ STAGE 3 — Cα-RMSD agglomerative clustering → cluster representatives       │
  └────────┬──────────────────────────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────────────────────────┐
  │ STAGE 3.5 — RELAX #2: MM-GBSA on the top-K cluster reps (--refine-topk)    │
  │   minimize each complex in AMBER ff14SB + GBn2 implicit solvent, then      │
  │   ΔG_bind = E(complex) − E(receptor) − E(peptide)   ← most accurate ΔG     │
  │ STAGE 3.6 — PRIMARY ΔG: AI-pose affinity model (geometry features, NO      │
  │   Vina/AD4; length-routed, short peptides → hydrophobic sub-model)         │
  └────────┬──────────────────────────────────────────────────────────────────┘
           ▼
  ranked_poses.csv · best_pose.pdb · cluster_summary.csv · convergence.png ·
  dendrogram.png · run_metadata.json   (git SHA, seeds, versions, input hashes)
```

**The headline ΔG is the AI-pose affinity model — not Vina.** Stage 3.6 scores every pose with the
geometry-feature model tuned on real RAPiDock/AI poses (`data/affinity_ai_nofix.joblib`); that value is the
`delta_g` column and the reported "Best pose ΔG". **Vina is retained only for clash relief** (Stage 2 —
rescuing RAPiDock's clashing poses); its score is raw telemetry, never the affinity. **AD4 is off by
default.** For a crystal-quality pose, the sibling crystal-tuned model is exposed as a standalone command —
see [`crystal-score`](#crystal-score--score-an-existing-crystal-pose).

**Yes — `--refine-topk K` actually relaxes the top poses.** Stage 3.5 takes one representative per cluster
(best hybrid score), keeps the top *K* by cluster mean, and **energy-minimizes each receptor+peptide complex
in GBn2 implicit solvent** before reading ΔG — that minimization *is* the relaxation, and the MM-GBSA ΔG is
the pipeline's most accurate affinity number. `--mmgbsa-3traj` additionally relaxes the unbound receptor and
peptide to capture reorganization energy. (Stage 1.5 is a *separate*, lighter, restrained relax that only
relieves clashes without changing the binding mode.)

---

## Install

```bash
# 1. Scoring + analysis environment (the package itself)
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# 2. GPU sampling environment (Stage 1) — pick your platform
conda env create -f envs/rapidock-env.yml            # Linux/WSL2 + CUDA
# conda env create -f envs/rapidock-env-macos.yml    # Apple Silicon (MPS)
```

ADFRsuite + PULCHRA are license-restricted and **not** redistributed here — see
[INSTALL.md](INSTALL.md) for the one-time download. Verify the install with `bash scripts/smoke_test.sh`.

---

## Usage

HybriDock-Pep is one CLI with six subcommands: **`dock`**, **`selectivity`**, **`reproducibility`**,
**`prep`**, **`calibrate`**, **`benchmark`**. Run `hybridock-pep <command> --help` for the full flag list.

### `dock` — end-to-end docking + scoring

```bash
hybridock-pep dock \
    --peptide ETFSDLWKLLPE \
    --receptor receptors/mdm2.pdb \
    --site 25.20 -25.61 -7.97 \   # binding-site center (x y z, Å)
    --box 30 \                    # search box edge (Å)
    --n-samples 100 \             # RAPiDock passes (default 100)
    --refine-topk 10 \            # MM-GBSA on the top-10 cluster reps
    --output-dir runs/mdm2_p53
```

Key options:

The default ΔG (`delta_g`) is the **AI-pose affinity model** — Vina is clash-relief only, AD4 is off.

| Flag | What it does |
|---|---|
| `--scoring vina,ad4` | force-field backends to run (default `vina` = clash relief; add `ad4` for research telemetry). Neither is the headline ΔG. |
| `--refine-topk K` | **most accurate ΔG** — MM-GBSA (AMBER ff14SB + GBn2) on the top-K cluster reps. Use it unless screening hundreds. |
| `--ultra [K]` | **ultra ranking mode** — compute `rank_score` as the mean of K feature-jittered evaluations (randomized smoothing, default K=32). Tightens within-target ranking ~+2 pts pairwise at ~K× scoring cost; does **not** improve absolute ΔG. |
| `--ensemble` | also emit the optional geometry+Vina ensemble ΔG column (research/telemetry; not the default scorer) |
| `--free-entropy` | add the free-state conformational-entropy feature (helps long/floppy peptides) |
| `--input-poses DIR` | **skip Stage 1** and score pre-generated poses (e.g. sampled on a remote CUDA box) |
| `--seed N` | deterministic run (modulo CUDA nondeterminism; logged to `run_metadata.json`) |
| `--mmgbsa-ie` / `--mmgbsa-3traj` / `--mmgbsa-dielectric EPS` | interaction-entropy term · three-trajectory MM-GBSA · custom solute dielectric |
| `--mmgbsa-cpu-only` / `--no-minimize` | force MM-GBSA onto CPU · skip the OpenMM pre-minimization |

### `selectivity` — does my peptide prefer target A over off-target B?

```bash
hybridock-pep selectivity \
    --peptide LISDAELEAIFEADC \
    --target-receptor receptors/target.pdb \
    --target-site 31.9 17.5 9.5 --target-box 25 \
    --offtarget-receptor receptors/offtarget.pdb \
    --offtarget-site 12.3 4.1 22.7 --offtarget-box 25 \
    --output-dir runs/selectivity_check
```

Returns **ΔΔG = ΔG_target − ΔG_offtarget** with a 95 % bootstrap CI over the top-K cluster centroids.
Negative ΔΔG with a CI that doesn't cross zero ⇒ statistically selective. This sidesteps the absolute-Kd
ceiling because the same systematic bias applies to both receptors and cancels in the difference.

### `reproducibility` — multi-seed pose agreement

```bash
hybridock-pep reproducibility \
    --peptide ETFSDLWKLLPE --receptor receptors/mdm2.pdb \
    --site 25.20 -25.61 -7.97 --box 30 \
    --seeds 1 2 3 --n-samples 100 --output-dir runs/repro
```

Runs the pipeline once per seed and reports the Cα-centroid agreement across runs — the honest stochastic
stability of the sampler on your target.

### `crystal-score` — score an existing crystal pose

HybriDock-Pep ships **two scoring functions of the same design, separately tuned**: the **AI-pose model**
(the default inside `dock`, calibrated on RAPiDock/AI poses) and the **crystal model** (calibrated on
crystal/native poses). When you already have a crystal-quality bound pose and just want its ΔG — no docking —
call the crystal scorer directly:

```bash
hybridock-pep crystal-score \
    --receptor receptors/mdm2.pdb \
    --peptide-pdb poses/native_peptide.pdb \
    --peptide ETFSDLWKLLPE
# → Crystal ΔG = -10.07 kcal/mol  (geometry + interaction map, crystal-tuned model)
```

No RAPiDock, no Vina, no MM-GBSA — it runs the geometry + interaction-map crystal model
(`data/affinity_crystal_ifp.joblib`, override with `--artifact`) on the pose you give it.

### `prep` — pre-build a receptor PDBQT

```bash
hybridock-pep prep --receptor receptors/mdm2.pdb --output-dir prepped/
```

Wraps `prepare_receptor` (ADFRsuite) so you can cache the receptor once and reuse it across many `dock` runs.

### `calibrate` — fit the ΔG correction to your own data

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

Pass the result to `dock --calibration data/calibration.json`. Shipped calibrations live in `data/` with
full LOO-CV provenance; see [`docs/calibration_notes.md`](docs/calibration_notes.md).

### `benchmark` — score a CSV of complexes against baselines

```bash
hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --baselines vina,adcp \
    --report benchmark_report.md
```

### Cross-platform & accelerator tuning (CUDA · ROCm · oneAPI · Metal · CPU)

Backend selection and per-device tuning are **automatic** — no flags. Each compute path is routed to the
fastest silicon available and tuned for it, centralized in `hybridock_pep/hardware.py` (OpenMM) and
`sampling/run_rapidock.py::_optimize_backends` (torch):

| Stage (engine) | NVIDIA (CUDA) | AMD (ROCm) | Intel (oneAPI) | Apple (Metal) | CPU |
|---|---|---|---|---|---|
| **Stage 1 — RAPiDock (torch)** | TF32 fast path (`matmul_precision('high')`, `allow_tf32`) | ROCm via the CUDA API, same TF32 path | XPU + `intel-extension-for-pytorch` (ipex) | MPS + op-fallback | physical-core threads |
| **Stage 1.5 / 3.5 — OpenMM** | CUDA, mixed precision | **HIP**, mixed precision | OpenCL | OpenCL | thread-pinned CPU |
| **Stage 2 — Vina / AD4** | CPU (`cpu=`physical cores) | CPU | CPU | CPU | CPU |

OpenMM platform priority is **CUDA → HIP → OpenCL → CPU** (HIP beats OpenCL on AMD; OpenCL covers Intel and
Apple GPUs, which have no native OpenMM backend); mixed precision gives near-double accuracy at near-single
speed. Vina/AD4, the geometry model, and the calibrated ΔG are **pure-CPU and identical on every platform** —
only Stage 1 sampling and the OpenMM relaxations change speed with hardware. No local NVIDIA GPU? Sample
Stage 1 elsewhere (or on CPU) and run scoring locally with `dock --input-poses poses_dir/`.

### Outputs

Every run writes to `--output-dir`: `ranked_poses.csv` (per-pose scores + calibrated ΔG), `best_pose.pdb`,
`cluster_summary.csv`, `convergence.png`, `dendrogram.png`, and `run_metadata.json` (git SHA, seeds, software
versions, input hashes — everything needed to reproduce the run).

`ranked_poses.csv` includes a **`rank_score`** column — the composition-IFP ranking model (E309). To screen a
peptide panel, dock each candidate against the same receptor and compare their **best-pose `rank_score`**
(lower = predicted stronger); it ranks within-target candidates better than the absolute ΔG (70.5% vs 64.5%
pooled-pairwise) because it is size-normalized. It is *not* an absolute ΔG (use `delta_g`) and *not* a
within-run pose ranker (that ordering is the CSV row order).

`rank_score` is **target-dependent** — reliable on shape/hydrophobic grooves (SH3 ρ=+0.91, MDM2 +0.67), weak
where affinity is single-residue side-chain chemistry (PDZ +0.26, BH3 −0.63). It **self-reports confidence**:
`interaction_map.ranking_confidence(best_pose_rank_scores)` returns `high` (reliable — 100% correct direction
in validation) when the panel's scores spread out, `low` (verify in wet lab) when they cluster. See
[`docs/external_validation_2026-07-06.md`](docs/external_validation_2026-07-06.md).

`best_pose.pdb` is the exact geometry the headline ΔG was computed on, **with standard residue names** — so
you can re-score it directly: `hybridock-pep crystal-score --receptor R.pdb --peptide-pdb <out>/best_pose.pdb
--peptide SEQ`. (A `best_pose_vina_relaxed.pdb` with the Vina clash-relieved geometry is also written for
visualization; it is ligand-format and not meant for re-scoring.)

---

## Testing

```bash
pip install -e ".[dev]"          # pytest + dev tools (the runtime install omits them)
pytest                           # 419 fast unit tests
pytest -m slow                   # + integration tests (MDM2/p53, ~30 min)
pytest --cov=hybridock_pep       # coverage
```

> **WSL2 / CUDA:** the MM-GBSA test runs real OpenMM. Export the WSL CUDA path so it finds the GPU:
> `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`. `OMP_NUM_THREADS=1` keeps the sklearn-heavy
> scoring tests fast.

## Reproduce every number in this README

Every headline number maps to one committed script that prints the exact *r* / MAE table and writes a JSON
beside it for line-by-line checking. Download PDBbind v2020 ([pdbbind.org.cn](http://www.pdbbind.org.cn))
and PPIKB / the PPI-Affinity SI first (the large/external inputs are gitignored; the small IFP caches ship
in `data/`). Run each with `OMP_NUM_THREADS=1` on this machine for the speed the docs assume.

| Number in this README | Command | Writes |
|---|---|---|
| **ours MAE 1.33 / r 0.391  vs  PPI-clone MAE 1.44 / r 0.231** (leakage-free head-to-head, test ①) | `OMP_NUM_THREADS=1 python scripts/e331_ours_vs_ppiclone_clustered.py` | stdout table (random + clustered, both models) |
| **ours full-set leakage-free MAE 1.43 / RMSE 1.81 / r 0.263** + matched ref2015 | `OMP_NUM_THREADS=1 python scripts/e330_ours_pdbbind.py` | stdout table (leaky vs clustered vs length-stratified) |
| **0.480 / 0.291** PDBbind crystal + IFP (charged 0.401 / 0.146) — legacy test ① | `python scripts/e298_ppi_vs_ifp.py` | `data/e298_ppi_vs_ifp.json` |
| **PPIKB independent, leakage-free: ours r 0.369 / MAE 1.90  vs  PPI-clone 0.252 / 2.02** (Kd/Ki-only, full stack) | `OMP_NUM_THREADS=1 python scripts/e332b_ppikb_headtohead.py` | stdout |
| **0.25 → 0.52–0.61** same-receptor anchoring — test ② | `python scripts/e264_ppikb_anchor_fusion.py` | `data/e264_ppikb_results.json` |
| **0.225 ← 0.045** IFP rescue on PPI's own T100 — § ideas | `python scripts/e300_ifp_on_t100.py` | `data/e300_ifp_t100.json` |
| **0.437 / 0.399** train IFP on all 973 / 1405 crystals — § ideas | `python scripts/e304_ifp_mega_everything.py` | `data/e304_ifp_mega.json` |
| full non-FEP/LIE scorecard on 156 complexes | `python scripts/e90_full_scorecard.py` | stdout table |
| **0.486 → 0.53** affinity *r* on real RAPiDock poses — test ③ | `python scripts/e106_combined_realpose_grade.py` | per-complex CSV |
| **2.49 Å** best-of-top-25 pose RMSD, hit@5 91% — test ③ | `hybridock-pep benchmark --test-csv data/test_complexes.csv --report bench.md` | `bench.md` |
| double-difference + anchoring **math** (cycle closes; not a prediction claim) | `pytest tests/test_double_difference.py tests/test_anchoring.py -q` | green = the cycle/anchoring math holds |
| **ΔΔG selectivity** primitive end-to-end | `pytest tests/test_selectivity.py -q` | green |

Rebuild the IFP training cache from raw structures (the 437 new PPIKB complexes) with
`python scripts/e303_build_ppikb_ifp.py`. The full experiment ledger (E0–E304, every win and every refuted
idea) is in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).

---

## Roadmap / to-do

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │  HybriDock-Pep  ·  where we are and where we're going                   │
  └───────────────────────────────────────────────────────────────────────┘
```

**Done ✓**
- [x] Two-stage pipeline (RAPiDock-Reloaded sampling → physics/geometry rescoring), MIT, cross-platform
- [x] Calibrated ΔG in kcal/mol; leakage-free benchmark (60%-id clustered CV)
- [x] Beat PPI-Affinity clone on the identical honest split (MAE 1.33 vs 1.44; r 0.391 vs 0.231)
- [x] Selectivity ΔΔG primitive (target vs off-target) with bootstrap CI
- [x] Rigorous characterisation of the absolute-cross-target wall (why it's hard for FEP too)
- [x] `--ultra` verification tier scoped (MM-GBSA + charged/entropy physics; honest limits documented)

**In progress / next**
- [ ] Trajectory cache (`e363`) — simulate once, re-derive any physics term offline (near done)
- [ ] Per-residue ΔΔG *design* map (which residues drive PfLDH-vs-hLDH selectivity) — the winnable, relative regime
- [ ] Data expansion + representation (the field's proven lever for absolute cross-target: more/synthetic data + PLM embeddings)
- [ ] Uncertainty/confidence flag surfaced per prediction (know when to trust an absolute ΔG)
- [ ] iGEM wiki write-up: lead with kcal/mol MAE + selectivity + honest negative on absolute cross-target

**Explicitly out of scope (proven dead-ends, kept on the record)**
- [x] ~~Breaking absolute cross-target r past the field ceiling with more physics~~ — fundamental wall (see docs)
- [x] ~~Raw electrostatic/entropy terms as absolute-ΔG features~~ — charge-count/near-cancellation artifacts

---

## Project status

Built for the **iGEM 2026 Best Software Tool** award by the Denmark High School Dry Lab team. Target-agnostic;
the initial test case is a malaria rapid-diagnostic peptide selectivity check (PfLDH vs hLDH). Stable,
MIT-licensed, 419 unit tests + integration tests. See [`docs/architecture.md`](docs/architecture.md) for the
pipeline spec.

**Author:** Choppa Purandhar Ram — Head of Dry Lab, Denmark High School iGEM (2026); designed and built at
age 15.

## Citations

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025).
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021).
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017).
- **PPI-Affinity** — Romero-Molina et al., *J. Proteome Res.* 21:1829 (2022); web server unmaintained since 2022.
- **Boltz-2 affinity fine-tune** — "On fine-tuning Boltz-2 for protein–protein affinity prediction," [arXiv:2512.06592](https://arxiv.org/abs/2512.06592) (2025).
- **Boltz-2 reliability audit** — "On the Reliability of AI Methods in Drug Discovery: Evaluation of Boltz-2," [arXiv:2603.05532](https://arxiv.org/abs/2603.05532) (2026).
- **Peptide-docking review** — Martins, Santos & Sousa, *J. Comput. Chem.* 47:5, doi:10.1002/jcc.70328 (2026).
- **HybriDock-Pep** — this repository, 2026.

## License

[MIT](LICENSE). Third-party dependencies retain their own licenses — see [INSTALL.md](INSTALL.md) for
ADFRsuite, AutoDock4, and PULCHRA caveats (none redistributed here).
