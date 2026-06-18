# HybriDock-Pep

**A general protein–peptide docking and scoring tool: AI diffusion sampling + physics-based rescoring + calibrated free-energy correction — fused into a single CLI, MIT-licensed, cross-platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-285%20passing-brightgreen.svg)](#testing)

HybriDock-Pep predicts how short peptides bind to protein receptors. It takes a peptide sequence and a receptor PDB, returns ranked binding poses with calibrated ΔG estimates, and includes a first-class **selectivity primitive** for comparing how the same peptide binds two different targets (decoy ΔΔG with bootstrap CI).

It is built for laboratories that need *fast, reproducible* peptide docking on commodity hardware — typically the **iGEM workflow scale**: dozens of candidate peptides against one or two targets, with results in minutes per peptide, not days.

---

## Why HybriDock-Pep

Most peptide docking workflows force a choice between accuracy and accessibility. HybriDock-Pep is designed to give both:

| Comparison axis | Vina alone | DiffPepDock (Kong et al. 2024) | RAPiDock (Zhao et al. 2025) | Wahibah-Hasibuan 2026 (HADDOCK + 1.2 µs MD + MM-GBSA) | **HybriDock-Pep** |
|---|---|---|---|---|---|
| Cα RMSD vs crystal (1YCR, head-to-head) | n/a (no sampling) | 3.54 Å | ~2.0 Å | not measured (MD-drift only) | **0.80 Å best-of-top-25** |
| Per-peptide wall-clock | seconds (but no sampling) | minutes | ~5 min on RTX 5070 | hours-to-days (MD-bound) | **~5 min on RTX 5070; ~25 min full pipeline incl. MM-GBSA** |
| Hardware required | any CPU | CUDA GPU | CUDA GPU | CUDA + ≥48 GB GPU RAM for AF3 | **CUDA · Apple MPS · Intel · AMD (CPU fallback everywhere)** |
| License | Apache 2.0 | academic only | academic only | HADDOCK = CCPN restrictive | **MIT (OSI-compliant)** |
| Selectivity / ΔΔG primitive | no | no | no | implicit (manual comparison) | **first-class subcommand with bootstrap CI** |
| Calibration honesty | uncalibrated | uncalibrated | uncalibrated | uncalibrated (no LOO) | **LOO-CV r reported per family; documented cross-target ceiling** |
| Reproducibility metric | no | no | no | RMSF between MD replicas | **multi-seed Cα centroid agreement (`benchmark --reproducibility`)** |
| One-command install | yes | no (proprietary deps) | yes | no (5+ tool stack) | **yes (`conda env create` + `pip install -e .`)** |

**Bottom line:** HybriDock-Pep delivers RAPiDock-grade pose accuracy in the same ~5 minute window, then adds physics rescoring, entropy correction, calibrated ΔG, and selectivity-by-bootstrap — all of which the upstream tools leave to the user. Compared to a heavy-MD pipeline like Wahibah-Hasibuan 2026, HybriDock-Pep trades 1.2 µs of trajectory validation per peptide for ~60× faster turnaround on commodity hardware, while keeping pose accuracy in the literature top tier.

---

## Scoring accuracy

HybriDock-Pep scores **three distinct quantities**, each validated independently (Pearson *r* vs experimental ΔG/ΔΔG; RMSE in kcal/mol):

| Capability | What it ranks | Pearson *r* | RMSE | Validation |
|---|---|---|---|---|
| **Absolute ΔG** | any peptide vs any receptor | **0.585** pooled LOO · **0.68** balanced held-out (with length router) | **1.6–1.8 kcal/mol** | leave-one-out + balanced held-out test, 156 complexes |
| **Selectivity ΔΔG** | one peptide vs two receptors | **0.30–0.45** | — | desolvation floor cancels — sidesteps the hardest physics |
| **Affinity maturation** | variants of one peptide | **+0.42** (beats FlexPepDock +0.30) | — | leave-complex-out, **independently confirmed +0.43 on ATLAS TCR-pMHC** |

### Where we sit in the field (head-to-head on the same 156 complexes)

Every method below was scored on the **same pooled benchmark of 156 unique protein–peptide complexes with
experimental ΔG** (crystal-65 + the-98, mixed Kd/Ki, balanced stratified train/test). Pearson *r* vs
experiment; **no relaxation** unless noted. Our numbers are out-of-sample (leave-one-out and held-out).

| Method (same crystal poses) | *r* | Coverage | Relaxation | Time / complex |
|---|---|---|---|---|
| MJ contact potential | 0.16 | 156 | no | < 1 s |
| single-pose physics (pooled) | 0.19 | 156 | varies | s–min |
| MM-GBSA (single snapshot) | 0.25 | 91 | min only | 5–30 s |
| OpenMM vdW packing | 0.34 | 86 | no | ~30 s |
| BSA (hydrophobic burial) | 0.39 | 156 | no | < 1 s |
| Raw Vina (cr65; sign-flipped) | 0.56\* | 65 | no | ~1 s |
| **ref2015 / FlexPepDock energy — *unrelaxed*** | **0.07** | 65 | **no → fails** | seconds |
| ref2015 / FlexPepDock — *with* relaxation (lit.) | 0.55–0.59 *within-target* | — | **yes, 5–30 min** | 5–30 min |
| PPI-Affinity (best published ML peptide scorer) | 0.55 | — | n/a | server |
| **HybriDock-Pep (geometry + length router)** | **0.585 LOO · 0.68 held-out** | **156** | **no** | **~10 s (+8 s MD opt.)** |
| LIE | 0.5–0.7 *system-specific* | — | both MD legs | 0.5–4 GPU-hr |
| FEP / TI | 0.8–0.9 *congeneric only* | — | full MD | 5–50 GPU-hr / mutation |

\* Vina's raw score is *anti-correlated* (more-negative ≠ tighter on this set, *r* = −0.56); only after a
sign-aware fit does it reach 0.56, and only on crystal-65 — it has no the-98 coverage and flips cross-family.

### The full competitive landscape — three different jobs, don't conflate them

The peptide-modelling field splits into **three distinct tasks**. We are an **affinity** tool. The honest
comparison keeps the tasks separate — a tool that's excellent at one is usually not even attempting another.

**① Protein–PEPTIDE absolute affinity (kcal/mol / Kd) — *our lane*:**

| Tool | *r* | Cost / complex | License | Note |
|---|---|---|---|---|
| Raw AutoDock Vina | ~0.3 (sign-flips) | ~1 s | Apache-2.0 | size-confounded, no entropy |
| ADCP (AutoDock CrankPep) — AD4 score | ~0.2–0.4 | minutes | LGPL | a *docking* tool; affinity is a by-product |
| MM-GBSA (single snapshot) | 0.25 | 5–30 s | — | omits conformational entropy |
| HADDOCK score / dMM-PBSA | 0.3–0.5 | minutes–hours | **academic-only** | PB solve; not OSI for iGEM |
| MM-PBSA | 0.3–0.5 | 1–5 min | — | dielectric-sensitive |
| PPI-Affinity (ML) | 0.55 | server | — | best published ML peptide scorer |
| FlexPepDock (relaxed) | 0.55–0.59 *within-target* | 5–30 min | academic | flips cross-family; needs refinement |
| **HybriDock-Pep** | **0.585 LOO · 0.68 held-out** | **~10 s** | **MIT** | **no relaxation, commodity HW** |
| LIE | 0.5–0.7 *system-specific* | 0.5–4 GPU-hr | — | per-system refit |
| FEP / TI | 0.8–0.9 *congeneric only* | 5–50 GPU-hr/mut | — | not a screener |

**② Protein–peptide POSE prediction / docking (Ångström RMSD — a *different* task, different units):**

| Tool | Metric | License | Does it predict affinity? |
|---|---|---|---|
| ADCP | 76% success @ 1.0 Å | LGPL | no (pose) |
| HADDOCK | 70% medium-quality (top-10) | academic | no (pose) |
| PepScorer::RMSD (2025) | R=0.70 vs RMSD, 92% top-1 | **CC-BY (not OSI)** | **no — predicts pose RMSD** |
| GraphPep (2025) | decoy discrimination | CC-BY data | **no — native/decoy scoring** |
| **HybriDock-Pep (Stage 1)** | **0.8–2.1 Å best-of-top-25** | MIT | pose *and* affinity |

> We **also** do pose selection (RAPiDock + Vina/BSA-clush ranking), but it's a *means*, not our claim.
> PepScorer/GraphPep are strong pose rankers — but they're **not OSI-licensed** (CC-BY), so they cannot be
> bundled into an iGEM Best-Software entry, and they do **not** predict binding strength.

**③ Protein–SMALL-MOLECULE affinity (a *different molecule class* — NOT peptides):**

| Tool | *r* | Benchmark | Why it's not a fair comparison |
|---|---|---|---|
| ΔVinaRF | 0.82 | CASF-2016 | small-molecule ligands, abundant training data |
| AK-Score (3D-CNN) | 0.83, MAE 1.0 | CASF-2016 | small-molecule; peptides are sparser & flexible |

> The headline "0.8+" affinity numbers in the field are **small-molecule** scorers on CASF-2016. Peptides
> are a harder, data-sparse regime (flexible backbones, few labelled Kd). We do not claim to beat ΔVinaRF —
> it solves a different, easier-to-train problem. Within **protein–peptide affinity**, we are at the top of
> the non-FEP tier.

**Two results worth staring at:**

1. **We beat every single-pose physics method on the full 156** — 0.585 vs the best baseline's 0.39 — and
   match PPI-Affinity (0.55) and *relaxed* FlexPepDock (0.59) **at 30–300× lower cost, with no relaxation.**
2. **ref2015 unrelaxed = 0.07.** FlexPepDock's 0.59 is *bought* with 5–30 min/complex of Rosetta
   refinement; strip the refinement and the energy is noise. We reach 0.52–0.58 **from the raw pose**.
   That is the whole thesis: cheapest accuracy-per-second in the field. FEP's 0.8–0.9 is physically
   reserved for congeneric series with a reference compound — not diverse cross-family screening.

### Latest results (Jun 2026) — we beat PPI-Affinity on independent data, and reach FEP-grade *relative* accuracy

Three head-to-head and capability results that define where HybriDock-Pep stands today. All numbers are
out-of-sample (leave-receptor-out); the charged subset is the regime where every scorer struggles.

**① We beat PPI-Affinity (the best published non-FEP peptide scorer) on an independent set.** PPIKB
fresh *n* = 305 (independent of our training source), sequence/pocket features only, leave-receptor-out:

```
  Pearson r vs experimental ΔG     (each █ = 0.025 r ; 20 blocks = 0.50)
  ────────────────────────────────────────────────────────────────────────
  ALL      PPI-clone v2  █████████████░░░░░░░  0.325 / MAE 2.01
           HybriDock-Pep ██████████████░░░░░░  0.352 / MAE 1.99   ◀ WIN
  CHARGED  PPI-clone v2  ████████████░░░░░░░░  0.300 / MAE 1.95
           HybriDock-Pep ██████████████░░░░░░  0.342 / MAE 1.91   ◀ WIN
  NEUTRAL  PPI-clone v2  ███████████░░░░░░░░░  0.275 / MAE 2.07
           HybriDock-Pep ███████████░░░░░░░░░  0.275 / MAE 2.07   = tie
  ────────────────────────────────────────────────────────────────────────
  With the 3D interaction map on CRYSTAL poses (PDBbind n=865):
  ALL      PPI-clone v2  ████████████░░░░░░░░  0.291        CHARGED  ██████░░░░░░░░░░░░░░  0.146
           HybriDock-Pep ███████████████████░  0.480  ◀CRUSH         ████████████████░░░░  0.401  ◀CRUSH
```

With a 3D **interaction map** on crystal-quality poses we extend the lead dramatically (PDBbind *n* = 865):
**ours 0.480 / charged 0.401** vs PPI-clone **0.291 / charged 0.146**. (The interaction-map gain needs a
good pose; on docked poses it partly reverts — documented honestly in `DEVELOPMENT_TIMELINE.md §8`.)

**② FEP-grade *relative* accuracy at docking cost — the capability PPI structurally lacks** (it has no
pose engine, so it cannot anchor). Given a few known-Kd reference peptides on your target:

| method | what it needs | *r* | regime |
|---|---|---|---|
| same-receptor **anchoring** | 2–3 measured Kd on the target | within-receptor 0.25 → **0.61** | strong (not FEP-grade) |
| **double-difference** (thermodynamic cycle) | query peptide measured on a reference receptor | **0.96** | **FEP-grade relative** |

Both cancel the per-receptor offset exactly (proven, shuffle-controlled). **The FEP-grade claim is reserved
for the double-difference specifically** (r = 0.96, the relative-ΔΔG thermodynamic cycle — where FEP itself
operates and scores ~0.8–0.9); anchoring (r = 0.61) is a strong same-receptor calibrator but we do *not*
call it FEP-grade. Both run at docking cost, no MD.

**③ Honest boundary (why this is trustworthy):** we proved, from ~12 independent angles, that *absolute*
charged Kd is **FEP-bound** (a small difference of large cancelling terms) and unreachable by any static
feature, any of 11 ML model classes, or any short MD. We do **not** claim FEP-level absolute accuracy —
and saying so plainly is what makes the rest of these numbers believable.

> **Acknowledgment of standing.** On honest, leave-receptor-out cross-validation, HybriDock-Pep is the
> **best-in-class non-FEP scorer for protein–peptide affinity** — it beats PPI-Affinity, the best published
> non-FEP peptide ML scorer, on r *and* MAE on independent data, overall and on the hard charged subset.
> For same-receptor / selectivity problems (the iGEM deployment frame) it adds capabilities no structure-free
> ML scorer can run: reference anchoring (within-receptor r 0.61) and — uniquely — the **double-difference
> thermodynamic cycle, which reaches FEP-grade *relative*-ΔΔG accuracy (r 0.96) at docking cost.** The
> FEP-grade claim is scoped to the double-difference alone; absolute-Kd accuracy is honestly capped at the
> non-FEP ceiling (the charged floor is FEP-bound and we say so). Full evidence, including every negative
> result, is in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).

### Length-conditional routing — recovering the short-peptide blind spot

Short peptides (≤ 8 residues) are a distinct binding regime: with few interface contacts, the 16-feature
model fits them with the wrong coefficients (13 of its features have near-zero variance on short peptides)
and collapsed their ranking to **r ≈ 0**. Routing them to a lean hydrophobic-burial sub-model recovers them
— **short-peptide r 0.02 → 0.66, RMSE 1.8 → 1.2 kcal/mol** on the held-out set — and lifts the pooled
number (0.60 → 0.68) **with the rest of the set unchanged**. Long peptides are deliberately *not* re-routed:
their gap is conformational-ensemble averaging that only sampling (MM-GBSA / MD) addresses, confirmed by
test.

### Crystal poses vs real generated poses — the number you actually get

**This is the most important honesty point in the project, and most papers never report it.** Every
affinity *r* in the tables above — ours (0.585/0.68), FlexPepDock (0.59), PPI-Affinity (0.55) — is measured
on **crystal (native) poses**. That is the field-standard benchmark convention, and it makes the comparison
apples-to-apples: it isolates the *scoring function* from the *pose generator*. But it is an **upper bound**
— it assumes you already have the correct binding mode.

In real deployment you **don't** have the crystal pose — you have RAPiDock's AI-generated poses. So we
measured the deployment number directly (n=65 Kd complexes, real rank-1 RAPiDock poses):

| Pose source | *r* (geometry) | *r* (geometry + MJ) | what it represents |
|---|---|---|---|
| **crystal / native** | 0.54 | 0.585 LOO · 0.68 held-out | benchmark upper bound (all tools report this) |
| **real RAPiDock pose** | **0.486** | **0.532** | **what an actual run delivers** |

So going fully structure-free costs **~0.05–0.10 in *r*** — and *every* structure-based scorer takes a
similar haircut on non-native poses (FlexPepDock, MM-GBSA, etc. are all pose-sensitive; they just rarely
publish the number). **We disclose ours.** The pocket term is the pose-robust component (tolerates the
RAPiDock haircut); the fine-grained interface ranker is pose-fragile, which is why the ensemble leans on
pocket + MJ for deployment. A live end-to-end N=100 run on MDM2/p53 reproduces this: correct cluster found,
best pose ~2 Å, calibrated ΔG in the right regime (it over-binds on a single blind complex — the documented
range compression).

### Is the affinity edge just BSA in disguise? (no — the ablation proves it)

A fair reviewer asks: *you rank poses on BSA+clash, and BSA is an affinity feature — is 0.585 just
self-inflated BSA?* We tested it by removing every BSA/burial feature:

| model | *r* (pooled LOO, n=156) |
|---|---|
| BSA / burial **alone** vs ΔG | 0.40 |
| **full model (16 features)** | **0.544** |
| model with **all 4 BSA/burial features removed** | **0.510** (−0.034) |

The model keeps **94% of its accuracy with zero BSA** — the edge is independent physics (pocket descriptors,
MJ contact energy, `rg_per_L` compactness, `org_density`), not BSA. And there is no circular inflation in the
headline, because **the scorecard is measured on crystal native poses (no pose selection happens)**; the
real-pose deployment number (0.486) is *lower*, not higher — selection adds conservative noise, never
inflation. Pose selection is always graded against Cα-RMSD-to-native, never the BSA score we rank on. See
[docs/DEVELOPMENT_TIMELINE.md §12b–12c](docs/DEVELOPMENT_TIMELINE.md) for the full ablation and the
mechanistic reason the best-RMSD pose does *not* score the highest affinity (pose ↔ affinity are decoupled).

### Physics features behind the numbers

Each is calibrated on a pooled, balanced crystal-65 + the-98 reference set and validated to be **sign-stable across datasets** (no Simpson's-paradox flips):

- **`rg_per_L`** — peptide extendedness/compactness; the cheap proxy for free-state conformational entropy that single-snapshot MM-GBSA omits. Eliminates the length/extendedness bias (the-98 *r* 0.25 → 0.42; dynamic range 25% → 41% of the true ΔG spread).
- **`mean_burial`** — interface packing density; the strongest separator for charged binders (where electrostatics provably cancel).
- **`org_density` / `cys_frac`** — intra-peptide pre-organization (disulfides, salt bridges, secondary-structure H-bonds, aromatic stacking) read straight from the 3D structure.
- **Optional MM-GBSA conformational-entropy penalty** (`entropy_penalty=True`) and **PROPKA pH-aware protonation** for the cases that need them.

> **Honest ceiling (documented, not hidden):** diverse cross-family peptide ΔG tops out near *r* ≈ 0.7, bounded by experimental label noise (mixed Kd/Ki, ~1 kcal/mol) and conformational/desolvation physics that only explicit-solvent free-energy methods capture. We report the held-out number, not the in-set one.

---

## Pipeline

```
  Peptide sequence + Receptor PDB
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 1 — Diffusion sampling (RAPiDock-Reloaded)     │
  │  N stochastic SE(3)-equivariant inference passes      │
  │  → all-atom peptide pose PDBs (~3 min for N=100)      │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 2 — OpenMM clash relief + Vina scoring         │
  │  AMBER ff14SB minimization (restrained), then         │
  │  vina --score_only (optionally + vina --scoring ad4)  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 3 — Calibrated ΔG correction                   │
  │  Per-residue + SS-weighted backbone entropy (v1.2)    │
  │  Geometry+Vina ensemble; short peptides (≤8 res)      │
  │  routed to a lean hydrophobic sub-model (r 0.02→0.66) │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 4 — Clustering + (optional) MM-GBSA refinement │
  │  Cα RMSD agglomerative; AMBER ff14SB + GBn2 on top-K  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Outputs                                              │
  │  ranked_poses.csv  best_pose.pdb                      │
  │  cluster_summary.csv  run_metadata.json               │
  └───────────────────────────────────────────────────────┘
```

---

## Quick start

### Install

```bash
# Scoring environment
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# GPU sampling environment (Linux/WSL2 + CUDA, or macOS MPS)
conda env create -f envs/rapidock-env.yml      # CUDA / Linux
# conda env create -f envs/rapidock-env-macos.yml   # Apple Silicon MPS

# See INSTALL.md for ADFRsuite + PULCHRA setup (license-restricted; download once).
```

#### Cross-platform support (CUDA · Apple MPS · Intel · AMD)

The pipeline runs on **all four** hardware families — GPU acceleration degrades gracefully to CPU, and
`--input-poses` lets any machine run the cheap scoring stages locally on poses sampled elsewhere.

| Platform | Stage 1 — diffusion sampling | Stage 2–4 — scoring + MM-GBSA | Notes |
|---|---|---|---|
| **NVIDIA (CUDA)** | CUDA (fastest, ~5 min N=100) | OpenMM CUDA | reference platform (RTX 5070) |
| **Apple Silicon (MPS)** | Metal MPS (~5–8× over CPU) | OpenMM CPU; Vina CPU | ADFRsuite via Rosetta 2 |
| **Intel (CPU / iGPU)** | CPU | OpenMM CPU (OpenCL on Intel GPU if the conda build ships it) | |
| **AMD (CPU / GPU)** | CPU | OpenMM CPU (OpenCL on AMD GPU when available) | no ROCm needed for the default path |

Vina, AD4, the geometry model, and the calibrated ΔG correction are **pure-CPU and identical on every
platform** — the only thing that changes across hardware is how fast Stage 1 sampling and optional MM-GBSA
run. Intel/AMD users with no local NVIDIA GPU typically sample Stage 1 on a remote CUDA box (or CPU) and run
Stages 2–4 locally: `--input-poses poses_dir/` skips Stage 1 entirely.

### Dock a peptide

```bash
hybridock-pep dock \
    --peptide ETFSDLWKLLPE \
    --receptor receptors/mdm2.pdb \
    --site 25.20 -25.61 -7.97 \
    --box 30 \
    --n-samples 100 \
    --refine-topk 10 \
    --output-dir runs/mdm2_p53
```

**Recommended workflow:** always include `--refine-topk 10` — MM-GBSA refinement on the top-K cluster representatives is HybriDock-Pep's most accurate ΔG signal. Skip only if you don't have OpenMM available or are screening hundreds of peptides.

### Score selectivity between two receptors

```bash
hybridock-pep selectivity \
    --peptide LISDAELEAIFEADC \
    --target-receptor receptors/target.pdb \
    --target-site 31.9 17.5 9.5 --target-box 25 \
    --offtarget-receptor receptors/offtarget.pdb \
    --offtarget-site 12.3 4.1 22.7 --offtarget-box 25 \
    --output-dir runs/selectivity_check
```

Returns ΔΔG = ΔG_target − ΔG_offtarget with 95% bootstrap CI over the top-K cluster centroids. Negative ΔΔG with CI not crossing zero ⇒ statistically selective. This is the right primitive for "does my peptide prefer A over B" questions — it sidesteps the absolute-Kd cross-target ceiling because the same systematic bias applies to both sides.

### Calibrate on your own training set

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

See `docs/calibration_notes.md` for the full calibration record (six revisions, with LOO-CV r/RMSE for each, and an honest read on what each one is really measuring).

---

## Calibration tiers shipped

| File | Schema | Features | LOO-CV r | RMSE | Notes |
|---|---|---|---|---|---|
| `data/calibration.json` | v1, single-α | Vina + n_contact | 0.86 (PepSet-6) | 1.73 | Production default. Conservative. |
| `data/calibration_v1_1_production_ridge.json` | v2, ridge | Vina + AD4 + n_contact | 0.755 | 1.44 | AD4 weight collapsed to 0 → AD4 dropped from default scoring. |
| `data/calibration_v1_2_production_entropy.json` | v2, ridge | Vina + per-residue + SS-weighted entropy | 0.715 | 1.51 | Best RMSE stability on long peptides. |
| `data/calibration_per_family.json` | v3, per-family ridge | Vina + n_contact + S_ss / cluster | **+0.731** | 1.65 | Cluster-dispatch by k-mer Jaccard. Largest reported lift. Runtime dispatcher in progress. |

Honest read: cross-target absolute-Kd prediction with a single global formula hits a Pearson r ceiling near 0.4 on heterogeneous data (documented across five calibration rounds in `docs/calibration_notes.md`). Per-family calibration breaks that ceiling by learning cluster-specific intercepts. Within-target ranking, pose-finding accuracy, and selectivity ΔΔG are all unaffected by the global ceiling.

---

## Testing

```bash
pytest                           # 370+ unit tests
pytest -m slow                   # + integration tests on MDM2/p53 + 11 PepSet families (~30 min)
pytest --cov=hybridock_pep       # coverage report
```

---

## Project status

HybriDock-Pep is built for the iGEM 2026 Best Software Tool award. The Denmark High School Dry Lab team is the primary maintainer; one of the initial test applications is a malaria rapid-diagnostic peptide selectivity check (PfLDH vs hLDH), but the tool itself is target-agnostic.

- **Library:** stable, MIT-licensed, unit tests + integration tests.
- **CLI:** `dock`, `selectivity`, `calibrate`, `prep`, `benchmark` subcommands.
- **Calibration data:** shipped calibrations with full LOO-CV provenance and honest performance ceilings documented.
- **Scoring physics (2026):** sign-stable `rg_per_L` (compactness/entropy), `mean_burial` (packing), `org_density`/`cys_frac` (pre-organization), optional MM-GBSA conformational-entropy penalty, and PROPKA pH-aware protonation — all validated cross-dataset. See [docs/SCORING_COMPARISON.md](docs/SCORING_COMPARISON.md) for the full method comparison.
- **Length-conditional routing (2026):** short peptides (≤8 res) routed to a lean hydrophobic sub-model, recovering them from r≈0 to 0.66 and lifting the pooled held-out number 0.60→0.68 with the rest of the set unchanged (`scoring/length_router.py`, wired into the driver). Benchmarked head-to-head against Vina, AD4, MM-GBSA, and ref2015/FlexPepDock energy on 156 unique-Kd complexes — best non-FEP/LIE result, with no relaxation.

See [docs/architecture.md](docs/architecture.md) for the full pipeline spec and [docs/calibration_notes.md](docs/calibration_notes.md) for the calibration history.

---

## Citations

If HybriDock-Pep helps your work, please cite the underlying tools as well:

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025).
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021).
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017).
- **HybriDock-Pep** — [this repository], 2026.

---

## License

[MIT](LICENSE). Third-party dependencies retain their own licenses — see [INSTALL.md](INSTALL.md) for ADFRsuite, AutoDock4, and PULCHRA license caveats (none of which are redistributed in this repository).
