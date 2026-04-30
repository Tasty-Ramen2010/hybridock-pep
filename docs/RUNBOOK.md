# HybriDock-Pep — Runbook: Running a Real Docking Test

This document explains how to run a complete end-to-end docking pipeline test
against a real receptor, from environment setup through interpreting results.
It assumes INSTALL.md has been completed and all tools are on PATH.

---

## Quick sanity check first

Before any real run, verify the environment:

```bash
conda activate score-env
bash scripts/smoke_test.sh
```

All three checks must PASS. If any FAIL, see INSTALL.md.

---

## Option A — Single-complex dock (for development/debugging)

This runs the full Stage 1 + Stage 2 pipeline on one receptor/peptide pair.
Use this to validate a new receptor or debug a specific issue.

```bash
conda activate score-env

hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor receptors/1czb.pdb \
    --site 22.5 14.1 38.7 \
    --box 25 \
    --n-samples 100 \
    --scoring vina,ad4 \
    --seed 42 \
    --output-dir runs/pfldh_run1
```

**Key flags:**

| Flag | What it controls |
|------|-----------------|
| `--peptide` | Single-letter amino acid sequence |
| `--receptor` | Raw PDB file from RCSB or prepared structure |
| `--site X Y Z` | Grid box center in Ångstroms (Cα centroid of the known binding peptide) |
| `--box` | Grid edge length in Å. Use 25 for a known site, 40 for blind docking |
| `--n-samples` | RAPiDock inference passes. 100 for production; 5–10 for dev |
| `--scoring` | `vina` only, `ad4` only, or `vina,ad4` (recommended for full pipeline) |
| `--seed` | Integer seed for reproducibility (modulo CUDA nondeterminism) |
| `--output-dir` | All outputs land here |

**Getting the `--site` coordinates:**

The grid center should be the binding pocket centroid. Two ways to get it:

1. **From crystal structure:** If you have a PDB with the peptide co-crystallised,
   compute the Cα centroid of the peptide chain:
   ```python
   from scripts.benchmark import get_peptide_center
   from pathlib import Path
   center = get_peptide_center(Path("structure.pdb"), peptide_chain="B")
   print(center)  # (cx, cy, cz)
   ```

2. **Blind:** If no structure is available, use `--site 0 0 0 --box 60` and
   RAPiDock will sample the entire accessible surface. Score convergence will be
   slower — use `--n-samples 200+`.

**Expected outputs after a successful run:**

```
runs/pfldh_run1/
├── ranked_poses.csv         # All scored poses, ranked by hybrid_score
├── best_pose.pdb            # Best cluster centroid (all-atom PDB)
├── cluster_summary.csv      # Per-cluster statistics
├── convergence_plot.png     # Score variance vs N samples (checks convergence)
├── silhouette_plot.png      # Cluster quality
├── run_metadata.json        # Git SHA, seeds, timing, software versions
├── receptor.pdbqt           # pdbfixer-cleaned + ADFRsuite-prepared receptor
├── receptor_for_rapidock.pdb  # pdbfixer-cleaned receptor (heavy atoms only, for RAPiDock)
├── maps/                    # AutoDock4 affinity maps
├── poses/                   # Renamed pose PDBs from RAPiDock (pose_0.pdb … pose_N.pdb)
└── pdbqt/                   # Meeko-prepared pose PDBQTs
```

**Reading ranked_poses.csv:**

```
rank,hybrid_score,vina_score,ad4_score,entropy_correction,delta_g,cluster_id,...
1,-8.42,-7.91,-9.12,-0.31,-8.42,0,...
2,-7.85,-7.43,-8.61,-0.29,-7.85,0,...
```

- `hybrid_score`: combined Vina + AD4 + entropy correction (lower = stronger binding)
- `vina_score`: AutoDock Vina score only (kcal/mol)
- `ad4_score`: AutoDock4 score (uses Gasteiger charges)
- `entropy_correction`: backbone entropy penalty (α × n_residues)
- `is_clipped`: True if any pose atom fell outside the scoring grid (treat with caution)
- `is_ad4_anomaly`: True if AD4 score is positive (non-physical; indicates clash)

**Interpreting the result:**

The benchmark target for PfLDH is `hybrid_score ≤ −6 kcal/mol` for the top
cluster centroid. If `ranked_poses.csv` is empty or all poses have
`is_clipped=True`, the binding site coordinates are wrong — recompute `--site`.

---

## Option B — Full benchmark suite (10 held-out complexes)

This runs the complete accuracy benchmark defined in §14 of the technical spec.
It downloads PDB structures from RCSB, runs the full pipeline for each complex,
and computes Pearson r against experimental pKd values.

```bash
conda activate score-env

hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --output-dir runs/benchmark/ \
    --seed 42 \
    --n-samples 100
```

**Runtime:** ~5–8 minutes per complex on RTX 5070 (100 samples). 10 complexes
total ≈ 50–80 minutes. Stage 2 (scoring) is CPU-bound and runs in parallel with
the next complex's Stage 1.

**Expected output — `runs/benchmark/benchmark_report.md`:**

```markdown
| pdb_id | Peptide         | Exp. pKd | Hybrid ΔG | Vina-only ΔG | Status |
|--------|-----------------|----------|-----------|--------------|--------|
| 3EQS   | TSFAEYWNLLSP    | 8.48     | -7.81     | -6.22        | ok     |
| 3DAB   | SQETFSDLWKLL    | 5.60     | -5.49     | -4.91        | ok     |
...

| Metric                        | Value  | Target | Result |
|-------------------------------|--------|--------|--------|
| Pearson r (hybrid vs pKd)     | 0.62   | ≥ 0.55 | PASS   |
| Pearson r (Vina-only vs pKd)  | 0.44   | —      | —      |
| Δ improvement (hybrid−Vina)   | +0.18  | ≥ 0.10 | PASS   |
| Overall                       |        |        | PASS   |
```

**What PASS/FAIL means:**

| Target | Description |
|--------|-------------|
| Pearson r ≥ 0.55 | Hybrid scores correlate with experimental binding affinities on the held-out test set |
| Δ ≥ 0.10 | Hybrid scoring improves over Vina-alone by at least 0.10 in Pearson r |

These are the iGEM Best Software Tool accuracy targets from CLAUDE.md §8.

**Diagnosing `skipped_scoring` entries:**

A complex shows `skipped_scoring` when all N=100 poses land outside the Vina
scoring grid. This means RAPiDock placed poses far from the crystal peptide
position. Causes and fixes:

1. **Wrong peptide chain in meta CSV** — check `data/test_complexes_meta.csv`
   and verify `peptide_chain` matches the short peptide, not the receptor chain.
2. **Binding site is in a disordered region** — some RCSB structures have
   missing loop density; the centroid may be wrong.
3. **Grid too small** — increase `--box-size` (default 40 for benchmark;
   try 50 if many complexes still fail).

---

## Option C — Bypass Stage 1 (macOS / pre-generated poses)

If poses are already available (generated on a CUDA machine and transferred),
skip RAPiDock entirely:

```bash
hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor receptors/1czb.pdb \
    --site 22.5 14.1 38.7 \
    --box 25 \
    --input-poses /path/to/poses_dir/ \
    --scoring vina,ad4 \
    --output-dir runs/pfldh_rescore
```

`--input-poses` must point to a directory of `pose_N.pdb` files (the same
format that Stage 1 produces in `{output-dir}/poses/`).

---

## Option D — Calibrate the entropy coefficient

The entropy correction coefficient α is fitted once on the training set. Recalibrate
after any change to the scoring pipeline:

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

The output `calibration.json` stores `alpha` and `beta`. CLAUDE.md §9 flags
as suspicious: α > 1.2 or α < 0.2 kcal/mol/residue.

---

## Checking convergence

After a 100-sample run, open `convergence_plot.png`. The plot shows the running
mean of `hybrid_score` vs the number of poses processed. A well-converged run
will flatten by pose 60–80. If the curve is still trending at pose 100, the
binding site is under-sampled — increase `--n-samples`.

CLAUDE.md §9 flags as suspicious: "Convergence curves still drifting at N=100
for a 15-mer." If you see this, check that the receptor PDB was cleaned
correctly and that RAPiDock is finding the binding pocket.

---

## Expected score ranges for the validation complex

The integration test baseline (CLAUDE.md §4):

| Complex | Peptide | System | Expected |
|---------|---------|--------|----------|
| MDM2/p53 (PDB 2OY2) | ETFSDLWKLLPE | K_d ≈ 0.6 µM | corrected ΔG ≤ −3 kcal/mol |

Run:

```bash
hybridock-pep dock \
    --peptide ETFSDLWKLLPE \
    --receptor receptors/2oy2.pdb \
    --site <cx> <cy> <cz> \
    --box 25 \
    --n-samples 100 \
    --output-dir runs/mdm2_p53_validation
```

If `ranked_poses.csv` shows `hybrid_score > −3 kcal/mol` for the best cluster
centroid, something is broken in the scoring pipeline.

---

## Common runtime warnings and what they mean

| Warning | Meaning | Action |
|---------|---------|--------|
| `addMissingHydrogens failed (HIS residue …)` | Non-standard histidine in PDB; hydrogen addition falls back to ADFRsuite | Harmless; verify `receptor.pdbqt` was written |
| `Pose N: atoms outside grid bounds (is_clipped=True)` | That pose landed outside the scoring box | Check `--site` coordinates; increase `--box` |
| `AD4 anomaly (positive score=… kcal/mol)` | AD4 scored a clashing conformation | With 100 samples this resolves; a few anomalies are normal |
| `RAPiDock pose shortfall: requested 100, generated M` | RAPiDock returned fewer poses than requested | Check `rapidock stderr` in debug mode (`-vv`) |
| `WARNING Search space volume > 27000 Å³` | Box size > 30 Å (Vina advisory only) | Informational only; does not affect scoring |

---

## Debug mode

Run any command with `-vv` to see full DEBUG output, including every RAPiDock
stderr line (diffusion progress, ESM embedding, model loading):

```bash
hybridock-pep -vv dock --peptide TSFAEYWNLLSP \
    --receptor 3eqs.pdb --site -20.8 44.1 9.5 --box 40 \
    --n-samples 5 --output-dir runs/debug
```

This is the fastest way to diagnose RAPiDock failures without editing code.
