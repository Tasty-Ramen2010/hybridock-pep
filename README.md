# HybriDock-Pep

**A general protein–peptide docking and scoring tool: AI diffusion sampling + physics-based rescoring + calibrated free-energy correction — fused into a single CLI, MIT-licensed, cross-platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-409%20passing-brightgreen.svg)](#testing)

HybriDock-Pep predicts how short peptides bind to protein receptors. Give it a peptide sequence and a
receptor PDB; it returns ranked binding poses, a calibrated ΔG, and — uniquely — a first-class
**selectivity primitive** (ΔΔG with bootstrap CI) for "does this peptide prefer target A over off-target B".
Built for the **iGEM workflow scale**: dozens of candidate peptides against one or two targets, minutes per
peptide on commodity hardware.

It is a **two-stage hybrid**: an AI diffusion model (RAPiDock-Reloaded) samples all-atom poses, then a
physics + learned-geometry rescorer turns those poses into calibrated affinity, selectivity, and
reference-anchored ΔG. Three things it does that off-the-shelf tools don't combine: **(1)** it is the best
non-FEP/LIE protein–peptide *affinity* scorer we can find a fair baseline for; **(2)** it reaches
*FEP-grade relative* accuracy on the same-receptor double-difference cycle at ordinary docking cost; and
**(3)** it ships a structure-based *selectivity* ΔΔG that a sequence-only ML scorer structurally cannot
provide. Everything below is measured, every claim links to the script that reproduces it, and every
negative result is kept on the record in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).
The whole thing is MIT-licensed and runs on CUDA, Apple MPS, Intel, AMD, or plain CPU.

---

## Why HybriDock-Pep — two conclusive tests

**① We beat PPI-Affinity (the best published ML peptide scorer) on independent, leakage-free data.**
Both methods, same complexes, leave-receptor-out CV (no homology leak in either direction):

```
  Pearson r vs experimental ΔG          each █ = 0.025 r
  ───────────────────────────────────────────────────────────────────
  PPIKB  n=305     HybriDock-Pep  ██████████████░░░░░░  0.352   ◀ WIN
  (independent)    PPI-Affinity   █████████████░░░░░░░  0.325
  ───────────────────────────────────────────────────────────────────
  PDBbind crystal  HybriDock-Pep  ███████████████████░  0.480   ◀ CRUSH
  + interaction    PPI-clone      ████████████░░░░░░░░  0.291
  map (n=865)                              charged: 0.401 vs 0.146  ◀ cracks the hard case
  ───────────────────────────────────────────────────────────────────
  PPI's headline 0.55–0.63 is on its OWN training-overlapped test set. Strip the
  leakage and everyone sits near r≈0.35 — where we are #1.
```

**② FEP-grade *relative* accuracy at docking cost** — the double-difference thermodynamic cycle, the one
place we operate where FEP itself does (and the one place we say "FEP-grade"):

```
  ΔG(P,R) ≈ ΔG(P,R_ref) + ΔG(P_ref,R) − ΔG(P_ref,R_ref)    cancels the per-receptor bias exactly
  ──────────────────────────────────────────────────────────  each █ = 0.04 r
  double-difference  ████████████████████████░  r = 0.96   ← FEP-grade, no MD, ~docking cost
  FEP / TI (the bar) █████████████████████░░░░  r ≈ 0.85   (5–50 GPU-hr / mutation)
```

**③ The number you actually get on AI-generated poses** — no crystal handed to you, the honest deployment
case. This is where we pull away from PPI-Affinity: **PPI is structure-free, so it is pose-blind** — it
returns the *same* score for any pose and cannot tell a good AI pose from a bad one. We read the pose:

```
  POSE ACCURACY (Cα-RMSD, lower = better)      AFFINITY r — SCORING THE AI POSE (each █ = 0.025 r)
  ──────────────────────────────────────      ──────────────────────────────────────────────────
  best-of-top-25   2.49 Å  ·  hit@5 91%        HybriDock-Pep · AI pose + interaction █████████████████████░ 0.53
  MDM2/p53 1YCR    0.80 Å                       HybriDock-Pep · AI pose, geometry     ███████████████████░░░ 0.486
   vs DiffPepDock  3.54 Å  ◀ ~4× tighter        PPI-Affinity  · pose-blind*           █████████████░░░░░░░░░ 0.325
                                                HybriDock-Pep · crystal (upper bound) ███████████████████████ 0.585
  * structure-free: identical score for any pose. Bars are each method's honest independent number.
```

We turn the AI pose into a **0.49–0.53** signal; PPI cannot use the pose at all and is stuck at its
structure-free **0.325**. Going fully structure-free costs us only ~0.05–0.09 in *r* (0.585 crystal → ~0.50
on AI poses) — the haircut every structure-based scorer pays on non-native poses, and one of the few we
publish.

Everything else stays honest: absolute charged Kd is capped at the non-FEP ceiling and we say so; selectivity
ΔΔG (target vs off-target) lands r ≈ 0.30–0.45; MIT-licensed and runs on CUDA · Apple MPS · Intel · AMD · CPU.
Full evidence and every negative result:
[`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md) ·
[`docs/SCORING_COMPARISON.md`](docs/SCORING_COMPARISON.md) · reproduce them in
[Reproduce the benchmarks](#reproduce-the-benchmarks).

---

## Pipeline

```
  Peptide sequence + Receptor PDB
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 1 — Diffusion sampling (RAPiDock-Reloaded)     │
  │  N stochastic SE(3)-equivariant passes → pose PDBs    │
  │  (~3 min for N=100 on an RTX 5070; CPU fallback)      │
  └────────┬──────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 2 — OpenMM clash relief + Vina/AD4 scoring     │
  └────────┬──────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 3 — Calibrated ΔG (entropy + geometry,         │
  │  length-routed; short peptides → hydrophobic model)   │
  └────────┬──────────────────────────────────────────────┘
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 4 — Cα-RMSD clustering + optional MM-GBSA      │
  └────────┬──────────────────────────────────────────────┘
           ▼
  ranked_poses.csv · best_pose.pdb · cluster_summary.csv · run_metadata.json
```

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

| Flag | What it does |
|---|---|
| `--scoring vina,ad4` | which rescoring backends to run (default `vina`; `ad4` adds the charge-aware term) |
| `--refine-topk K` | **most accurate ΔG** — MM-GBSA (AMBER ff14SB + GBn2) on the top-K cluster reps. Use it unless screening hundreds. |
| `--ensemble` | add the geometry+Vina ensemble ΔG column (the calibrated best-accuracy number) |
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

### Cross-platform (CUDA · Apple MPS · Intel · AMD · CPU)

Backend selection and per-device tuning are **automatic** (priority CUDA/ROCm → Intel XPU → Apple MPS → CPU):
TF32 fast path on NVIDIA/AMD, ipex on Intel XPU, MPS op-fallback on Apple, thread-pinned CPU otherwise.
Stages 2–4 (Vina, AD4, geometry, calibrated ΔG) are **pure-CPU and identical on every platform** — only
Stage 1 sampling and optional MM-GBSA change speed with hardware. No NVIDIA GPU? Sample Stage 1 elsewhere
(or on CPU) and run scoring locally with `dock --input-poses poses_dir/`.

### Outputs

Every run writes to `--output-dir`: `ranked_poses.csv` (per-pose scores + calibrated ΔG), `best_pose.pdb`,
`cluster_summary.csv`, `convergence.png`, `dendrogram.png`, and `run_metadata.json` (git SHA, seeds, software
versions, input hashes — everything needed to reproduce the run).

---

## Testing

```bash
pip install -e ".[dev]"          # pytest + dev tools (the runtime install omits them)
pytest                           # 409 fast unit tests
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
| **0.480 / 0.291** PDBbind crystal + IFP (charged 0.401 / 0.146) — test ① | `python scripts/e298_ppi_vs_ifp.py` | `data/e298_ppi_vs_ifp.json` |
| **0.352 / 0.325** PPIKB independent, charge-routed — test ① | `python scripts/e294_production_stack.py` | stdout table |
| **0.96** double-difference FEP-grade ΔΔG — test ② | `python scripts/e287_similarity_and_dd.py` | stdout table |
| **0.225 ← 0.045** IFP rescue on PPI's own T100 — § ideas | `python scripts/e300_ifp_on_t100.py` | `data/e300_ifp_t100.json` |
| **0.437 / 0.399** train IFP on all 973 / 1405 crystals — § ideas | `python scripts/e304_ifp_mega_everything.py` | `data/e304_ifp_mega.json` |
| full non-FEP/LIE scorecard on 156 complexes | `python scripts/e90_full_scorecard.py` | stdout table |
| **0.486 → 0.53** affinity *r* on real RAPiDock poses — test ③ | `python scripts/e106_combined_realpose_grade.py` | per-complex CSV |
| **2.49 Å** best-of-top-25 pose RMSD, hit@5 91% — test ③ | `hybridock-pep benchmark --test-csv data/test_complexes.csv --report bench.md` | `bench.md` |
| **r 0.96** module-level double-difference + selectivity | `pytest tests/test_double_difference.py tests/test_anchoring.py -q` | green = the cycle/anchoring math holds |
| **ΔΔG selectivity** primitive end-to-end | `pytest tests/test_selectivity.py -q` | green |

Rebuild the IFP training cache from raw structures (the 437 new PPIKB complexes) with
`python scripts/e303_build_ppikb_ifp.py`. The full experiment ledger (E0–E304, every win and every refuted
idea) is in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).

---

## Project status

Built for the **iGEM 2026 Best Software Tool** award by the Denmark High School Dry Lab team. Target-agnostic;
the initial test case is a malaria rapid-diagnostic peptide selectivity check (PfLDH vs hLDH). Stable,
MIT-licensed, 409 unit tests + integration tests. See [`docs/architecture.md`](docs/architecture.md) for the
pipeline spec.

## Citations

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025).
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021).
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017).
- **HybriDock-Pep** — this repository, 2026.

## License

[MIT](LICENSE). Third-party dependencies retain their own licenses — see [INSTALL.md](INSTALL.md) for
ADFRsuite, AutoDock4, and PULCHRA caveats (none redistributed here).
