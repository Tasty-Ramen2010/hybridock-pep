# HybriDock-Pep — Architecture

---

## 1. Top-Level Pipeline

```
    CLI entry point (score-env)
    hybridock-pep dock [flags]
          |
          v
    cli.py:_run_dock()
    ├─ Input validation (DockConfig via Pydantic)
    ├─ Resolve: input_poses_dir, calibration_path
    └─ driver.run_dock(config, input_poses_dir, calibration_path)
          |
     ┌────┴────────────────────────────────────────────┐
     │ Stage 0: write_metadata_skeleton()              │
     │   → run_metadata.json (skeleton)                │
     ├────────────────────────────────────────────────┤
     │ Stage 1a [if no --input-poses]:                 │
     │   rapidock_runner.run_sampling(config)          │
     │   └── subprocess: conda run -n rapidock-env    │
     │         python run_rapidock.py [args]           │
     │         → poses/pose_*.pdb                     │
     │ Stage 1b [if --input-poses]:                    │
     │   read poses directly from input_poses_dir      │
     ├────────────────────────────────────────────────┤
     │ Stage 2a: pose_io.parse_poses(poses_dir)        │
     │   → list[PoseRecord] + list[PoseFailure]        │
     ├────────────────────────────────────────────────┤
     │ Stage 2b: prep.receptor.prepare_receptor()      │
     │   pdbfixer + prepare_receptor4.py (ADFRsuite)   │
     │   → receptor.pdbqt                             │
     ├────────────────────────────────────────────────┤
     │ Stage 2c: prep.grids.generate_ad4_maps()        │
     │   autogrid4 + HD.map existence guard            │
     │   → maps_dir/receptor.{HD,C,...}.map           │
     ├────────────────────────────────────────────────┤
     │ Stage 2d: prep.ligand.prepare_ligand_batch()    │
     │   Meeko (ProcessPoolExecutor) per pose          │
     │   → pdbqt/pose_*.pdbqt                         │
     ├────────────────────────────────────────────────┤
     │ Stage 2e: scoring.vina.score_vina_batch()       │
     │   Vina Python API --score_only per pose         │
     │   → ScoredPose.vina_score                      │
     ├────────────────────────────────────────────────┤
     │ Stage 2f: scoring.ad4.score_ad4_batch()         │
     │   vina --scoring ad4, load_maps() per pose      │
     │   → ScoredPose.ad4_score, is_ad4_anomaly        │
     ├────────────────────────────────────────────────┤
     │ Stage 2g: scoring.entropy.apply_hybrid_score()  │
     │   hybrid = vina + beta*(ad4-vina) + alpha*n     │
     │   → ScoredPose.hybrid_score                    │
     ├────────────────────────────────────────────────┤
     │ Stage 3: analysis.clustering.cluster_poses()    │
     │   contact-zone Cα RMSD + AgglomerativeCluster   │
     │   silhouette k-search → ClusterResult           │
     │   + statistics.compute_cluster_stats()          │
     │   + plotting.plot_convergence/silhouette()       │
     │   → cluster_summary.csv, *.png                 │
     ├────────────────────────────────────────────────┤
     │ Stage 4: output.csv_writer.write_ranked_csv()   │
     │   + write_best_pose_pdb() + finalize_metadata() │
     │   → ranked_poses.csv, best_pose.pdb,           │
     │      run_metadata.json (finalized)              │
     └────────────────────────────────────────────────┘
```

The driver orchestrates both environments via subprocess — no Python objects cross the conda run boundary. All paths are resolved to absolute before the `subprocess.run()` call (CLAUDE.md §7).

---

## 2. Module Breakdown

Each module has a single responsibility. The table below maps every source file to its key functions and data contracts.

| Module | File | Key Functions | Data In | Data Out |
|--------|------|--------------|---------|----------|
| Models | models.py | DockConfig, PoseRecord, ScoredPose, PoseFailure | — | frozen Pydantic + dataclasses |
| CLI | cli.py | main(), _build_parser(), _run_dock(), _run_calibrate(), _run_prep(), _run_benchmark() | argparse.Namespace | calls driver.run_dock() |
| Driver | driver.py | run_dock() | DockConfig, input_poses_dir, calibration_path | tuple[list[ScoredPose], ClusterResult\|None] |
| Receptor prep | prep/receptor.py | prepare_receptor() | DockConfig | Path (receptor.pdbqt) |
| Ligand prep | prep/ligand.py | prepare_ligand_batch() | list[Path], output_dir | list[Path], list[PoseFailure] |
| Grid gen | prep/grids.py | generate_ad4_maps() | DockConfig, receptor_pdbqt | Path (maps_dir); aborts if HD.map missing |
| Prep errors | prep/errors.py | PrepError(RuntimeError) | — | exception class |
| Vina scoring | scoring/vina.py | score_vina_batch() | list[ScoredPose], DockConfig, receptor_pdbqt | list[ScoredPose], list[PoseFailure] |
| AD4 scoring | scoring/ad4.py | score_ad4_batch() | list[ScoredPose], maps_dir | list[ScoredPose], list[PoseFailure] |
| Entropy | scoring/entropy.py | load_calibration(), apply_hybrid_score(), fit_calibration() | ScoredPose, calibration.json | modifies ScoredPose.hybrid_score in place |
| Clustering | analysis/clustering.py | cluster_poses(), ClusterResult | list[ScoredPose], DockConfig | ClusterResult |
| Statistics | analysis/statistics.py | compute_cluster_stats(), write_cluster_summary_csv() | ClusterResult, ScoredPose | cluster_summary.csv |
| Plotting | analysis/plotting.py | plot_convergence(), plot_silhouette() | list[ScoredPose], ClusterResult | convergence_plot.png, silhouette_plot.png |
| CSV writer | output/csv_writer.py | write_ranked_csv(), write_best_pose_pdb() | list[ScoredPose], ClusterResult, DockConfig | ranked_poses.csv, best_pose.pdb |
| Metadata | output/metadata.py | write_metadata_skeleton(), finalize_metadata() | DockConfig, metadata_path | run_metadata.json |
| RAPiDock runner | sampling/rapidock_runner.py | run_sampling() | DockConfig | poses/pose_*.pdb (via subprocess) |
| Pose I/O | sampling/pose_io.py | parse_poses() | poses_dir | list[PoseRecord], list[PoseFailure] |
| RAPiDock subprocess | sampling/run_rapidock.py | main() | argv (from conda run) | poses written to disk |

---

## 3. Subprocess Orchestration

HybriDock-Pep uses two conda environments because RAPiDock's PyTorch/CUDA stack is incompatible with score-env. The subprocess boundary is strict: only file paths and integer/string flags cross it.

```
driver.py:run_dock()
    └── sampling/rapidock_runner.py:run_sampling(config)
            └── subprocess: conda run --no-capture-output -n rapidock-env
                    python sampling/run_rapidock.py
                    --output-dir <abs_path>
                    --peptide <seq>
                    --receptor <abs_path>
                    --n-samples 100
                    --seed <N>
```

All paths passed to the subprocess — `--output-dir`, `--receptor`, `--peptide` (a string, not a path) — are converted to absolute form via `str(Path(...).resolve())` before `subprocess.run()`. Relative paths fail silently because conda's subprocess CWD is unpredictable.

---

## 4. Data Models

### DockConfig

Pydantic model with `frozen=True`. Fields: `peptide_sequence` (str, validated AA-only), `receptor_path` (Path, must exist), `site_coords` (tuple[float, float, float]), `box_size` (float, >0), `n_samples` (int, default 100), `seed` (int|None), `scoring` (set[Literal["vina","ad4"]], default {"vina","ad4"}), `output_dir` (Path), `run_id` (str, auto-generated from timestamp+seed hash), `verbosity` (int). `run_id` is auto-generated via `@model_validator(mode='before')` so it resolves before field validators fire. `frozen=True` prevents mutation across the subprocess boundary.

### PoseRecord

Fields: `pose_idx` (int), `pdb_path` (Path), `sequence` (str), `ca_coords` (np.ndarray shape (n,3)). Parsed by `pose_io.parse_poses()` from Stage 1 output PDB files.

### ScoredPose

Extends PoseRecord (dataclass inheritance). Additional fields: `pdbqt_path` (Path), `vina_score` (float|None), `ad4_score` (float|None), `is_ad4_anomaly` (bool), `entropy_correction` (float|None), `hybrid_score` (float|None). `hybrid_score = vina_score + beta*(ad4_score - vina_score) + alpha * n_residues`

### ClusterResult

Fields: `k_optimal` (int), `silhouette_score` (float), `per_cluster_stats` (list[dict]). Each dict in `per_cluster_stats` contains: `cluster_id`, `n_poses`, `mean_hybrid_score`, `std_hybrid_score`, `ci95_lower`, `ci95_upper`, `best_pose_idx`. Written by `analysis/clustering.py`; consumed by `csv_writer.py` and `metadata.py`.

---

## 5. Config and Calibration Flow

The entropy correction parameters (α, β) live in `data/calibration.json` and are loaded once at the start of every dock run.

```
scripts/calibrate_alpha.py [--training-csv] [--scores-json] [--output]
          |
          v
scoring.entropy.fit_calibration()
    scipy L-BFGS-B minimization over (alpha, beta)
    loss = 1 - pearsonr(predicted_hybrid, RT * pKd)
          |
          v
data/calibration.json
    { "alpha": 0.65, "beta": 0.22, "n_complexes": 10,
      "pearson_r": 0.71, "rmse_kcal_mol": 1.2, ... }
          |
          v
driver.run_dock() calls load_calibration(calibration_path)
    validates alpha in [0.2, 1.2], beta in [0.0, 0.5]
    applies to each ScoredPose via apply_hybrid_score()
```

α must be in [0.2, 1.2] kcal/mol/residue and β in [0.0, 0.5]. `load_calibration()` aborts with a diagnostic message if either bound is violated.
