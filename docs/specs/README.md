# Design Decision Records (D-XX Reference)

Source code comments like `(D-01)`, `(D-07)`, `(D-11)` reference decisions made during
architecture and code review. These are **module-local** identifiers — the same number in
two different modules refers to two different decisions. They are not a global spec with a
single authoritative document; this file is the authoritative document.

The broader technical spec lives at `docs/HybriDock-Pep_Technical_Specification.pdf`.
If a D-XX decision conflicts with the PDF, the PDF wins.

---

## Module: `src/hybridock_pep/sampling/rapidock_runner.py`

| ID   | Decision |
|------|----------|
| D-01 | Use `subprocess.Popen` (not `run`/`communicate`) so stdout/stderr stream in real-time instead of buffering until the process exits. Stderr is drained on a background daemon thread to prevent pipe-buffer deadlock when RAPiDock emits lots of warnings. |
| D-02 | Drain stdout via `iter(pipe.readline, b"")` sentinel loop on the main thread so every output line is logged promptly. |
| D-03 | Non-zero subprocess exit is always a fatal `RuntimeError` — no retry, no fallback. Retrying a broken RAPiDock run wastes GPU time without fixing root cause. |
| D-07 | All file/directory paths passed to the RAPiDock subprocess are resolved to absolute paths via `Path.resolve()` before the call. `conda run` sets an unpredictable working directory; relative paths will break. |
| D-08 | Seed is forwarded as `--seed N` only when `DockConfig.seed is not None`. Callers that omit a seed get non-deterministic runs; the seed is recorded in `run_metadata.json` either way. |
| D-09 | `RuntimeError` is raised (not `Warning`) when the subprocess produces fewer poses than requested. Caller decides whether to continue with partial results. |
| D-10 | Output `rank*.pdb` files from RAPiDock are renamed to `pose_0.pdb … pose_N.pdb` (zero-indexed, zero-padded) immediately after the subprocess exits. This gives downstream code a stable, sortable naming convention. |
| D-11 | Zero poses produced after a successful (exit 0) subprocess is always a fatal `RuntimeError`. A zero-pose run means something is silently wrong — fail loud. |

---

## Module: `src/hybridock_pep/sampling/run_rapidock.py`

| ID   | Decision |
|------|----------|
| D-07 | All paths resolved to absolute before passing to RAPiDock internals. `conda run` cwd is unpredictable. |
| D-08 | Seed all RNGs (torch, torch.cuda, numpy, random) **before** importing any RAPiDock module so that ESM embedding computation and all diffusion steps are deterministic. |

---

## Module: `src/hybridock_pep/prep/receptor.py`

| ID   | Decision |
|------|----------|
| D-01 | Run all three pdbfixer fixes unconditionally (missing residues, missing atoms, non-standard residues). Do not skip fixes based on whether the input looks clean — input quality is unpredictable and silent partial fixes cause hard-to-debug errors. |
| D-02 | Always regenerate the PDBQT from scratch — no mtime/cache check. Caching caused subtle failures when users re-ran with a modified receptor PDB but the same filename. The regeneration is fast enough to not matter. |
| D-03 | Capture full stderr from `prepare_receptor`. ADFRsuite emits non-fatal warnings on stderr that are essential for diagnosing partial failures. |

---

## Module: `src/hybridock_pep/prep/grids.py`

| ID   | Decision |
|------|----------|
| D-05 | Hard abort (raise immediately) if the HD (hydrogen-bond donor) `.map` file is absent after `autogrid4`. Vina `--scoring ad4` will silently produce wrong scores if map files are missing — catching this early saves hours of debugging. |
| D-07 | All `autogrid4` output files (`.map`, `.gpf`, `.glg`) go to `output_dir/maps/` — never the working directory. Keeps output deterministic and avoids file-not-found bugs when running multiple complexes in parallel. |

---

## Module: `src/hybridock_pep/scoring/entropy.py`

| ID   | Decision |
|------|----------|
| D-01 | Hybrid score formula: `hybrid = vina + beta*(ad4 - vina) + alpha*n_eff_residues`. This formula is defined once and applied everywhere via `apply_hybrid_score()`. Do not inline it in other modules. |
| D-09 | Use `RT = 0.592` kcal/mol (298 K) as a hardcoded constant. Temperature variability is not modelled in v1. |
| D-10 | Optimisation starting point `x0 = [0.65, 0.22]` (alpha, beta). Chosen empirically from preliminary calibration on 6 complexes. Re-calibrate with the full 284-entry set before publication. |
| D-11 | Calibration JSON schema: `{alpha, beta, pearson_r, rmse, n_complexes, calibration_date, contact_dist_ang}`. The `contact_dist_ang` field is required so calibration files are self-documenting about which cutoff they used. |

---

## Module: `src/hybridock_pep/scoring/ad4.py`

| ID   | Decision |
|------|----------|
| D-06 | AD4 scores > 0 are flagged as anomalies (`is_ad4_anomaly=True`) but kept in the scored list. They are informational — the hybrid score formula forces `beta=0` for these poses so the anomalous signal does not corrupt the final score. |
| D-07 | Per-pose exceptions in the scoring loop are caught and recorded as `PoseFailure` objects. The batch never aborts on a single bad pose. |

---

## Module: `src/hybridock_pep/scoring/vina.py`

| ID   | Decision |
|------|----------|
| D-07 | Same per-pose isolation policy as `ad4.py` — batch never aborts. |

---

## Module: `src/hybridock_pep/analysis/clustering.py`

| ID   | Decision |
|------|----------|
| D-01 | Build the RMSD distance matrix using contact-zone Cα atoms only (see D-02 for which atoms). |
| D-02 | Contact-zone = peptide residues whose Cα is within `CONTACT_DIST_ANG` of any receptor heavy atom. Fallback: if fewer than 3 residues qualify, use the full peptide Cα set instead. This prevents degenerate clustering on very short or peripheral peptides. |
| D-05 | Select the number of clusters `k` by maximising silhouette score over `k ∈ [2, min(n_poses//4, 12)]`. |
| D-06 | Minimum silhouette threshold: if best silhouette < 0.15, fall back to `k=1` (all poses in one cluster). |
| D-08 | Assign `cluster_id` by mutating `ScoredPose` objects in-place. Same pattern as `apply_hybrid_score()`. |
| D-10 | Delegate CSV and dendrogram output to `statistics.py` and `plotting.py` respectively — clustering.py does not write files. |

---

## Module: `src/hybridock_pep/analysis/plotting.py`

| ID   | Decision |
|------|----------|
| D-03 | Convergence plot shows how the **score distribution** stabilises (running cumulative mean of top-10 scores) — not how frequently each pose arrives. Score convergence is the meaningful signal for this pipeline. |
| D-07 | Silhouette score plot annotates the chosen `k_optimal` with a vertical dashed line. |

---

## Module: `src/hybridock_pep/output/csv_writer.py`

| ID   | Decision |
|------|----------|
| D-04 | `delta_g` column in the output CSV is identical to `hybrid_score` (same numerical value). The column exists because reviewers and downstream tools expect a `delta_g` field; the label is the scientifically correct name for what hybrid_score approximates. |

---

## Module: `src/hybridock_pep/output/metadata.py`

| ID   | Decision |
|------|----------|
| D-15 | Two-write pattern: write initial `run_metadata.json` before sampling starts (partial, marks run as `in_progress`), then overwrite with final counts and `status=complete` after scoring. Enables crash recovery diagnostics — a JSON missing the final write marks an incomplete run. |
| D-16 | Required fields in `run_metadata.json`: `git_sha`, `rapidock_commit_sha`, `cli_args`, `seed`, `software_versions` (Vina, OpenMM, CUDA), `receptor_sha256`, `peptide_hash`, `wallclock_seconds`. |

---

## Module: `src/hybridock_pep/sampling/pose_io.py`

| ID   | Decision |
|------|----------|
| D-12 | Per-pose parse exceptions are caught and recorded as `PoseFailure` objects. The batch-load function never raises; callers inspect the returned list for failures. |
| D-13 | Receptor coordinates are extracted once and cached in-memory so contact-count computations across 100 poses happen in O(1) disk I/O. |
| D-14 | Peptide sequence extraction: try SEQRES records first; fall back to ATOM record residue names if no SEQRES found. This is the locked decision — do not change the order without updating all callers. |

---

## Module: `src/hybridock_pep/driver.py`

| ID   | Decision |
|------|----------|
| D-01 | `--input-poses` bypass: if this flag is set, skip Stage 1 (RAPiDock) entirely and read pre-generated poses from the specified directory. Required for macOS users who ran Stage 1 on a remote GPU. |
| D-02 | Full pipeline orchestration order: prep receptor → prep grids → sample poses → score poses → cluster → write outputs. This order is fixed; do not reorder without updating the driver's stage documentation. |

---

## Module: `scripts/calibrate_alpha.py`

| ID   | Decision |
|------|----------|
| D-08 | Training CSV schema: three columns — `pdb_id`, `peptide_sequence`, `experimental_pkd`. This is the minimum required schema; additional columns are ignored. |

---

## Module: `scripts/benchmark.py`

| ID   | Decision |
|------|----------|
| D-03 | Benchmark results CSV schema: one row per complex, columns for all scoring methods, Pearson r, and RMSE. |
| D-04 | Two-invocation pattern for Vina-only comparison: run the full HybriDock-Pep pipeline on a complex, then re-score the **same** poses with Vina-only. This controls for RAPiDock stochasticity — both methods use identical poses so pose quality is not a confound. |

---

*Last updated: 2026-05-26. Add new decisions here as the codebase grows.*
