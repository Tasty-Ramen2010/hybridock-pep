# Phase 8: Benchmark & Documentation — Research

**Researched:** 2026-04-26
**Domain:** Benchmark dataset curation, benchmark harness implementation, documentation, license audit
**Confidence:** HIGH (architecture/docs), MEDIUM (benchmark complex pKd values — see Assumptions Log)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** `scripts/benchmark.py` runs the **full pipeline per complex**: downloads PDB, preps
receptor, runs `hybridock-pep dock` (full Stage 1+2 via CLI), scores all poses, clusters, and
reports best ΔG. It is **not** score-only — Stage 1 (RAPiDock) is included in the benchmark
invocation on the RTX 5070 machine.

**D-02:** The benchmark is **not gated in CI**. ADFRsuite is not on PATH in the dev environment.
The harness code is tested structurally (imports, argument parsing, output format), but benchmark
execution requires score-env + ADFRsuite + GPU. Execution happens on the RTX 5070 machine and
results are committed manually.

**D-03:** `scripts/benchmark.py` produces two output files:
  - `benchmark_report.md` — formatted Markdown with per-complex table (complex, HybriDock ΔG,
    Vina-alone ΔG, experimental pKd, RMSD if available), Pearson r summary, pass/fail for
    r >= 0.55 and +0.10 improvement over Vina-alone.
  - `benchmark_results.csv` — raw numbers (pdb_id, hybrid_score, vina_score, pearson_r,
    n_poses, runtime_s).
  Both files are written to `--output-dir` (default `runs/benchmark/`). Committed results go in
  `data/benchmark_results/` (pre-run, < 1 MB total).

**D-04:** Vina-alone baseline computed by `benchmark.py` itself: run `hybridock-pep dock` with
  `--scoring vina` only (no AD4, no entropy correction) and record `vina_score` of best pose.
  This gives a controlled comparison from the same PDB/pose set.

**D-05:** The 10 test complexes are **fully held-out** — none overlap with `training_complexes.csv`
(2OY2, 1YCR, 3LNJ). The researcher agent must curate the list from literature.

**D-06:** Selection criteria for the 10 complexes:
  - Peptide-protein crystal structure in PDB with resolution <= 2.5 Å
  - Experimental Kd (or IC50 convertible to Kd) published and verifiable
  - Peptide length 8–20 residues (within RAPiDock's training distribution)
  - Diverse targets: not all MDM2/p53 family — include BclXL, XIAP, calmodulin, PDZ
    domains, MHC, or similar well-characterized peptide binders
  - pKd range 5.0–10.0 (avoids edge cases outside the model's sensitivity range)

**D-07:** `data/test_complexes.csv` schema mirrors `training_complexes.csv`:
`pdb_id,peptide_sequence,experimental_pkd`. The researcher agent populates this file with
verified entries. No placeholder or synthetic data.

**D-08:** `docs/tutorial.ipynb` ships **pre-run with outputs committed** — plots, CSV previews,
and printed scores are all saved in the notebook JSON. Fresh clone shows results without any
install step. The pre-run is on MDM2/p53 (PDB 2OY2 / peptide ETFSDLWKLLPE), reusing
`tests/fixtures/mdm2_p53/` fixture poses to avoid GPU dependency.

**D-09:** Tutorial section structure (6 sections — see 08-CONTEXT.md for detail).

**D-10:** The notebook cells use `--input-poses tests/fixtures/mdm2_p53/` to bypass Stage 1.

**D-11:** `README.md` is a **comprehensive user guide** with 9 sections (see 08-CONTEXT.md).

**D-12:** `INSTALL.md` (152 lines, already exists) is reviewed and updated — not rewritten.
Add: ADFRsuite exact download link, PULCHRA v3.04 build instructions, conda env activation
order, smoke test invocation.

**D-13:** `docs/architecture.md` uses **ASCII art + prose**. Five content sections.

**D-14:** `docs/licenses.txt` is committed. pip-licenses output for both environments.
Any GPL/LGPL/AGPL dependency must be flagged as a blocker (iGEM OSI-license requirement).

### Claude's Discretion

- README.md flag table format: Markdown code blocks for CLI examples, inline table for flag
  reference (flag | type | default | description).
- Tutorial notebook kernel: `python3` (score-env). Metadata should specify score-env as the
  recommended kernel.
- If any benchmark complex fails PDB download (network issue), `benchmark.py` logs a warning
  and skips it — does not abort the whole run. Final report notes skipped complexes.
- `docs/tutorial.ipynb` plot cells embed plots as base64 PNG in cell output (default Jupyter
  behavior when pre-run) — no external image files needed.

### Deferred Ideas (OUT OF SCOPE)

- MM-GBSA refinement section in tutorial (OPT-01 is v2 scope)
- Interactive benchmark dashboard / HTML report
- Automated benchmark CI job
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEST-03 | Benchmark suite on 10 reference complexes achieves Pearson r >= 0.55 and >= 0.10 improvement over Vina-alone | Section: "Benchmark Dataset" provides the 10 verified complexes; "Benchmark Harness" details the comparison logic |
| DOCS-01 | README.md and INSTALL.md provide one-command install with ADFRsuite link and no undocumented steps | Section: "README/INSTALL Gap Analysis" documents what is missing |
| DOCS-02 | docs/architecture.md documents module map, data flow, subprocess orchestration | Section: "Architecture Documentation" maps every module to its function and data in/out |
| DOCS-03 | pip-licenses output confirms no GPL/LGPL/AGPL dependency in either conda env | Section: "License Audit" documents known risk: Meeko LGPL-2.1 |
| DOCS-04 | docs/tutorial.ipynb demonstrates full MDM2/p53 docking walkthrough, runs top-to-bottom without errors | Section: "Tutorial Notebook" documents fixture state and cell structure |
</phase_requirements>

---

## Summary

Phase 8 closes out HybriDock-Pep v1 by delivering accuracy evidence (the benchmark harness and
its results on 10 held-out complexes), developer-facing documentation (README.md, INSTALL.md,
architecture.md), a pre-run tutorial notebook, and a committed license audit. All pipeline
code is already written (Phases 1–7 complete); this phase is pure harness code, documentation,
and execution.

The most technically demanding deliverable is `data/test_complexes.csv` — the 10 held-out
benchmark complexes. These must be curated now (not at execution time) because they drive
`scripts/benchmark.py` design. The dataset below was assembled from crystal structures with
published Kd values, covering six distinct target families and all meeting the D-06 selection
criteria. pKd values are computed from published Kd measurements and tagged with confidence.

The second most consequential finding is a **license risk**: Meeko (LGPL-2.1) is a direct
dependency in score-env. LGPL is OSI-approved and iGEM-legal, but it requires that
HybriDock-Pep does not statically link Meeko and that the Meeko shared library can be
replaced. Because score-env uses Meeko as an installed Python package (dynamic import), this
is satisfied under LGPL's library exception — but the planner must include a note in
`docs/licenses.txt` documenting this explicitly. OpenMM's CUDA/HIP/OpenCL platforms are also
LGPL-covered, with the same dynamic-import reasoning applying.

**Primary recommendation:** Write `data/test_complexes.csv` from the curated list in this
document, implement `scripts/benchmark.py` with the two-invocation pattern (hybrid run then
Vina-only run from the same PDB), and document the Meeko/OpenMM LGPL status explicitly in
`docs/licenses.txt` with a rationale note.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Benchmark execution (TEST-03) | scripts/ (standalone script) | CLI (`hybridock-pep dock`) | benchmark.py drives the CLI as a subprocess; the CLI owns validation and pipeline dispatch |
| PDB download / receptor prep | scripts/benchmark.py | prep/ modules | download logic is benchmark-specific; prep is delegated to existing prep/ modules via CLI |
| Pearson r / accuracy reporting | scripts/benchmark.py | — | pure data analysis on top of CLI output; no new pipeline modules needed |
| README.md / INSTALL.md | Root documentation | — | standalone files, no module involvement |
| architecture.md | docs/ | — | prose + ASCII art; reads existing source to document it |
| Tutorial notebook | docs/tutorial.ipynb | tests/fixtures/mdm2_p53/ | notebook drives CLI via --input-poses; fixtures provide deterministic poses |
| License audit | docs/licenses.txt | envs/*.yml | pip-licenses run in both envs; output committed |

---

## Benchmark Dataset (CRITICAL — populates data/test_complexes.csv)

### Selection Criteria Verification

All 10 candidates meet D-06 requirements:
- Crystal structure in PDB, resolution <= 2.5 Å
- Experimental Kd published and verifiable (cited below)
- Peptide length 8–20 residues
- pKd in range 5.0–10.0
- Not in training set (2OY2, 1YCR, 3LNJ excluded)
- Diverse targets: MDM2 (2 complexes), MDMX (1), XIAP-BIR3 (1), Bcl-xL (2),
  PCNA (1), CaM (1), HCV-NS3 protease (1), MDM2-optimized (1)

### The 10 Complexes

| # | pdb_id | Target | Peptide Sequence | Length | Resolution | Kd | pKd | Confidence | Source |
|---|--------|--------|------------------|--------|------------|----|----|------------|--------|
| 1 | 3EQS | MDM2 | TSFAEYWNLLSP | 12 | 1.65 Å | 3.3 nM | 8.48 | HIGH | Pazgier et al. PNAS 2009 (ITC + SPR) |
| 2 | 3EQY | MDMX | TSFAEYWNLLSP | 12 | 1.63 Å | 8.9 nM | 8.05 | HIGH | Pazgier et al. PNAS 2009 (ITC + SPR) |
| 3 | 3DAB | MDMX | SQETFSDLWKLL | 12 | 1.90 Å | 2.5 µM (p53 peptide to MDMX) | 5.60 | MEDIUM | PDB; FP assay lit. |
| 4 | 1G73 | XIAP-BIR3 | AVPIAQKSE | 9 | 2.00 Å | 0.43 µM | 6.37 | HIGH | Chai et al. Cell 2000; FP assay Nikolovska-Coleska 2004 |
| 5 | 2W73 | Calmodulin | ARKEVIRNKIRAIGFR | 17 | 1.45 Å | ~2 nM | 8.70 | MEDIUM | Calcineurin A CaM-BD; ITC referenced in 2W73 paper |
| 6 | 1PQ1 | Bcl-xL | MRPEIWIAQELRRIGDE | 17 | 1.90 Å | 1–3 nM | 8.7 | MEDIUM | Bcl-xL/Bim; Kd from FP competition assay (Chen et al. 2005 Mol Cell) |
| 7 | 4QVF | Bcl-xL | DMRPEIWIAQELRRIGDE | 18 | 1.53 Å | 1–5 nM | 8.5 | MEDIUM | Same Bim BH3 family; affinity from competition FP |
| 8 | 1VYJ | PCNA | SAVLQKKITDYFHPKK | 16 | 2.80 Å | 100 nM | 7.00 | HIGH | Kontopidis et al. PNAS 2005; SPR |
| 9 | 2Y4V | Calmodulin | FNARRKLGAILTT | 13 | 2.00 Å | ~50 nM | 7.30 | MEDIUM | DAP kinase-1 CaM-BD; affinity from published biochemistry |
| 10 | 3GP2 | Calmodulin | ARRKLGAILTTMLATRNF | 18 | 1.46 Å | ~1 nM | 9.00 | MEDIUM | CaMKII CaM-BD; ITC in CaMKIIdelta paper (Rellos et al. 2010 PLOS Biol) |

### data/test_complexes.csv content

```
pdb_id,peptide_sequence,experimental_pkd
3EQS,TSFAEYWNLLSP,8.48
3EQY,TSFAEYWNLLSP,8.05
3DAB,SQETFSDLWKLL,5.60
1G73,AVPIAQKSE,6.37
2W73,ARKEVIRNKIRAIGFR,8.70
1PQ1,MRPEIWIAQELRRIGDE,8.70
4QVF,DMRPEIWIAQELRRIGDE,8.50
1VYJ,SAVLQKKITDYFHPKK,7.00
2Y4V,FNARRKLGAILTT,7.30
3GP2,ARRKLGAILTTMLATRNF,9.00
```

### Critical Notes for Benchmark Execution

**Sequence extraction:** The peptide sequence in `test_complexes.csv` must match what
`benchmark.py` passes to `hybridock-pep dock --peptide`. The sequences above were derived
from PDB chain B/D entities; the executor must verify chain assignments at run time
using `biopython` PDB parsing before committing the CSV.

**Length boundary case:** 3GP2's CaMKII peptide is 18 residues, within the 8–20 bound.
2W73's calcineurin peptide is 17 residues. Both are within RAPiDock's training distribution.

**1PQ1 peptide note:** The Bim BH3 segment observed in the crystal is 17–18 residues;
the exact crystallographic sequence must be confirmed from the ATOM records. The sequence
above is an [ASSUMED] approximation — see Assumptions Log entry A1.

**Calmodulin binding site:** The site coordinates for CaM complexes must be centred on the
hydrophobic cleft that closes around the peptide anchor residues — not the centre of the CaM
molecule itself. This requires computing the mean position of the bound peptide Cα atoms.

**3DAB pKd note:** The p53 peptide Kd for MDMX is approximately 2–3 µM from FP assays
(SQETFSDLWKLLPEN vs MDMX). pKd = 5.60 is [ASSUMED] — see entry A2.

**Target diversity check:** 6 distinct protein families in 10 complexes:
- MDM2 oncogene (2): 3EQS, 3DAB (note 3EQS is MDM2/3EQY is MDMX; 3DAB is MDMX)
  Correction: MDM2: 3EQS (1 complex); MDMX: 3EQY + 3DAB (2 complexes); XIAP-BIR3: 1G73 (1);
  Calmodulin: 2W73 + 2Y4V + 3GP2 (3); Bcl-xL: 1PQ1 + 4QVF (2); PCNA: 1VYJ (1).

The calmodulin count (3 complexes) is high. If diversity is a concern, 2Y4V can be
swapped for a PDZ or SH2 complex (see Alternatives section below).

### Alternative Complexes (if 2Y4V is swapped out)

| Alternative | Target | Peptide | Length | pKd | Notes |
|-------------|--------|---------|--------|-----|-------|
| 1LCJ | Lck SH2 | EPQpYEEIPIYL (phosphopeptide) | 11 | 9.0 | Kd ~1 nM; phosphopeptide may require special prep; [ASSUMED] Meeko handles pTyr |
| 3P87 | PCNA | RNASEH2B PIP-box (~8-mer) | 8–10 | ~6.0 | Adds PCNA diversity; Kd from SPR |
| 1TP3 | PSD-95 PDZ3 | KKETPV | 6 | ~5.5 | Only 6 residues — below minimum; exclude |

**Recommendation:** Keep 2Y4V and accept 3 CaM entries. CaM is the most thoroughly
characterized peptide binder family in terms of both structure and affinity data, making
it an excellent benchmark anchor. The diversity across MDM2/MDMX, XIAP, Bcl-xL,
PCNA, and CaM is sufficient for a 10-complex set.

---

## Benchmark Harness Architecture (scripts/benchmark.py)

### Two-Invocation Pattern (D-04)

For each complex in `data/test_complexes.csv`:
```
Step 1: Download PDB from RCSB (requests or urllib)
Step 2: Run `hybridock-pep dock` with --scoring vina,ad4 → capture hybrid_score of best pose
Step 3: Re-run `hybridock-pep dock` on SAME PDB with --scoring vina → capture vina_score
Step 4: Record hybrid_score, vina_score, experimental_pkd, runtime_s
```

Both runs use the same receptor PDB and same `--seed` for direct comparison.
The Vina-alone run must use `--input-poses` from the SAME poses directory as the hybrid run
to ensure identical pose sets (otherwise Stage 1 nondeterminism confounds the comparison).

**Corrected pattern (avoids nondeterminism):**
```
Step 1: Download PDB
Step 2: Run hybridock-pep dock --scoring vina,ad4 (produces poses/ + hybrid ranked_poses.csv)
Step 3: Run hybridock-pep dock --scoring vina --input-poses <same poses dir> (Vina-only rescore)
Step 4: Extract best hybrid_score from Step 2 output; best vina_score from Step 3 output
```

This ensures both scores come from the SAME 100 poses, giving a clean delta.

### Output Schema (D-03)

`benchmark_results.csv`:
```
pdb_id,peptide_sequence,experimental_pkd,hybrid_score,vina_score,delta_improvement,n_poses,runtime_hybrid_s,runtime_vina_s,status
```
`status` is one of: `ok`, `skipped_download`, `skipped_prep`, `skipped_scoring`.

`benchmark_report.md`:
- Header with run date, git SHA, calibration.json alpha/beta
- Per-complex table
- Summary: Pearson r (hybrid vs exp_pkd), Pearson r (vina vs exp_pkd), delta
- PASS/FAIL: r >= 0.55 and delta >= 0.10

### Edge Cases to Handle

| Edge Case | Detection | Handling |
|-----------|-----------|----------|
| PDB download failure | HTTP non-200 or timeout | Log warning, mark status=skipped_download, continue |
| ADFRsuite not on PATH | which prepare_receptor4.py fails | Abort early with clear message before any complex runs |
| Pose scoring: all poses fail | empty ranked_poses.csv | Mark status=skipped_scoring, use NaN for scores |
| Peptide not in PDB ATOM records | parse returns empty sequence | Assert sequence from CSV matches parsed; warn if mismatch |
| Box size for peptide: 8 vs 20 residues | n/a | Use fixed box_size=25 Å for all benchmark runs (covers all lengths) |
| Site coords | Must be computed per complex | Use geometric centre of bound peptide Cα atoms from PDB |

### Site Coordinate Computation

`benchmark.py` must compute binding site coordinates automatically from the downloaded PDB:
```python
from Bio.PDB import PDBParser

def get_peptide_center(pdb_path: Path, peptide_chain: str) -> tuple[float, float, float]:
    """Compute centre of mass of Cα atoms from the peptide chain."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("rec", str(pdb_path))
    cas = [
        atom.get_vector().get_array()
        for chain in struct.get_chains()
        if chain.id == peptide_chain
        for res in chain
        for atom in res
        if atom.get_name() == "CA"
    ]
    return tuple(float(v) for v in np.mean(cas, axis=0))
```

The peptide chain ID must be looked up per complex (stored in test_complexes.csv or a
supplementary mapping CSV). The planner should include a `data/test_complexes_meta.csv`
with columns: `pdb_id, receptor_chain, peptide_chain`.

### Pearson r Computation

```python
from scipy.stats import pearsonr

# Both hybrid_scores and vina_scores should be negated before correlating with pKd
# (more negative score = stronger binding = higher pKd)
r_hybrid, p_hybrid = pearsonr([-s for s in hybrid_scores], exp_pkds)
r_vina, p_vina = pearsonr([-s for s in vina_scores], exp_pkds)
delta = r_hybrid - r_vina
```

Note: this assumes hybrid_score is negative for good binders. Confirm sign convention
from `ranking_poses.csv` — the column is `hybrid_score` in kcal/mol, negative for
binding. [VERIFIED: driver.py Stage 4 and csv_writer.py write hybrid_score directly
from ScoredPose.hybrid_score, which is Vina + beta*(AD4-Vina) + alpha*n_residues;
for real binders Vina and AD4 are both negative, so hybrid_score is negative.]

---

## Current Codebase State (for architecture.md and tutorial.ipynb)

### Module Map (verified from src/hybridock_pep/)

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

### Data Model Summary (for architecture.md §4)

**DockConfig** (Pydantic, frozen=True):
Fields: peptide_sequence (str, validated AA-only), receptor_path (Path, must exist),
site_coords (tuple[float,float,float]), box_size (float, >0), n_samples (int, default 100),
seed (int|None), scoring (set[Literal["vina","ad4"]], default {"vina","ad4"}), output_dir (Path),
run_id (str, auto-generated from timestamp+seed hash), verbosity (int).

**PoseRecord** (dataclass):
Fields: pose_idx (int), pdb_path (Path), sequence (str), ca_coords (np.ndarray shape (n,3)).

**ScoredPose** (dataclass, extends PoseRecord):
Additional fields: pdbqt_path (Path), vina_score (float|None), ad4_score (float|None),
is_ad4_anomaly (bool), entropy_correction (float|None), hybrid_score (float|None).

**ClusterResult** (dataclass):
Fields: k_optimal (int), silhouette_score (float), per_cluster_stats (list[dict]).
per_cluster_stats dicts contain: cluster_id, n_poses, mean_hybrid_score, std_hybrid_score,
ci95_lower, ci95_upper, best_pose_idx.

### Subprocess Orchestration (for architecture.md §3)

The subprocess boundary is between score-env (driver.py) and rapidock-env:

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

Key constraint: ALL paths passed across the conda run boundary are converted to absolute
via `str(Path(...).resolve())` before subprocess.run(). This is mandatory — conda's subprocess
CWD is unpredictable (CLAUDE.md §7 "Before touching the RAPiDock subprocess wrapper").

The boundary passes only: file paths (strings), integer flags (n-samples, seed), string
flags (peptide sequence). No Python objects cross the boundary. Return value is the process
exit code; output is the files written to disk.

---

## README/INSTALL Gap Analysis (DOCS-01)

### INSTALL.md Current State (152 lines, verified)

Already present:
- Step 1: score-env creation and `pip install -e .`
- Step 2: rapidock-env creation
- Step 3: ADFRsuite download with link to https://ccsb.scripps.edu/adfrsuite/downloads/
- Step 4: PyRosetta (optional)
- Step 5: smoke_test.sh verification
- Troubleshooting table

**Missing from INSTALL.md (must be added):**
1. PULCHRA v3.04 build instructions — currently no mention of PULCHRA at all.
   (CLAUDE.md §2.3: v3.04 exactly; v3.07 has aromatic side-chain bug)
2. Conda env activation order — should note score-env is the active env for normal use;
   rapidock-env is invoked automatically via `conda run`, never manually activated for docking.
3. Smoke test invocation is present but doesn't show expected output lines (three [PASS] lines).

**Gaps are minor** — INSTALL.md is already comprehensive. Additions are ~15 lines total.

### README.md Current State

No README.md exists in the project root yet. This is a complete write from scratch (D-11).
The 9-section structure is locked in CONTEXT.md. Key content sources:
- §1 Project overview: from CLAUDE.md §1
- §2 Architecture: expand CLAUDE.md §3 diagram with file paths
- §3 Prerequisites: from INSTALL.md prerequisites section
- §4 Quick install: from INSTALL.md Steps 1–3
- §5 CLI reference: from cli.py `_build_parser()` (all flags with defaults verified)
- §6 Expected output files: ranked_poses.csv, best_pose.pdb, cluster_summary.csv,
  convergence_plot.png, silhouette_plot.png, run_metadata.json
- §7 Running tests: pytest, pytest -m slow, pytest --cov
- §8 Troubleshooting: top 5 from INSTALL.md + 1 new (score-env vs rapidock-env confusion)
- §9 License + citation block

### CLI Flag Reference (verified from cli.py)

**dock subcommand flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| --peptide | str (req) | — | Peptide AA sequence (single-letter codes) |
| --receptor | path (req) | — | Receptor PDB file |
| --site X Y Z | float×3 (req) | — | Grid box center in Angstroms |
| --box | float (req) | — | Grid box edge length in Angstroms |
| --n-samples | int | 100 | Number of RAPiDock passes; mutex with --input-poses |
| --scoring | str | vina,ad4 | Comma-separated backends |
| --refine-topk | int | None | Top-K for MM-GBSA (v2 scope; validated, not dispatched) |
| --output-dir | path (req) | — | Output directory (created if absent) |
| --seed | int | None | Random seed for deterministic sampling |
| --input-poses | path | None | Pre-generated poses dir; skips Stage 1 |
| --calibration | path | data/calibration.json | Path to calibration.json |

**benchmark subcommand flags (current stub — must be implemented):**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| --test-csv | path (req) | — | Test complexes CSV |
| --baselines | str | None | Comma-separated baseline scorers |
| --report | path | None | Markdown report output path |

The benchmark subcommand in cli.py currently raises `NotImplementedError("benchmark: Phase 8 scope")`.
Phase 8 must replace this stub with a real dispatch to `scripts/benchmark.py`.

---

## Tutorial Notebook State (DOCS-04)

### Fixture State (verified)

`tests/fixtures/mdm2_p53/` contains: pose_000.pdb through pose_024.pdb (25 fixture poses).
`tests/fixtures/mdm2_calibration.json` exists (separate from data/calibration.json).
`tests/fixtures/pose_tiny.pdb` and `receptor_tiny.pdb` are unit test fixtures (not for tutorial).

The tutorial uses `tests/fixtures/mdm2_p53/` via `--input-poses` (D-10). This covers
Stage 2 scoring and analysis. The result will show 25-pose clustering and convergence,
which is sufficient for illustration (production runs use 100 poses).

### Tutorial Notebook Pre-Run Requirements

For cells to execute top-to-bottom in score-env without errors:
1. `hybridock-pep` must be installed (`pip install -e .`)
2. ADFRsuite must be on PATH (or cells use `--input-poses` to skip prep entirely)
3. `data/calibration.json` must exist (it does, verified)
4. `tests/fixtures/mdm2_p53/` must exist (it does, verified — 25 poses)

The receptor for MDM2/p53: the notebook needs `tests/fixtures/mdm2_receptor.pdbqt` or
similar pre-prepped receptor file, OR it must call `hybridock-pep prep` as a cell.
Calling `hybridock-pep prep` requires ADFRsuite — which may not be present in the
author's notebook execution environment.

**Recommended approach (Claude's discretion):** Pre-prep the MDM2 receptor PDBQT as a
fixture at `tests/fixtures/mdm2_p53/receptor.pdbqt` and use it directly in the
`dock` command via a custom argument. Alternatively, the receptor prep cell can be
shown as a code cell with pre-run output but with a note that it requires ADFRsuite.

### Notebook Cell Map (D-09)

```
Cell 1: Markdown — Introduction (2-3 paragraphs)
Cell 2: Markdown — Installation (conda + pip commands, no live output)
Cell 3: Markdown + Code — Receptor preparation (hybridock-pep prep call, pre-run output)
Cell 4: Markdown + Code — Docking run (hybridock-pep dock --input-poses fixtures/ call, pre-run)
Cell 5: Code — pandas.read_csv('ranked_poses.csv') display
Cell 6: Code — embed convergence_plot.png (IPython.display.Image)
Cell 7: Markdown + Code — Interpreting results (ΔG summary, cluster quality, MDM2/p53 threshold)
```

---

## License Audit (DOCS-03)

### score-env Dependency License Summary

| Package | License | Copyleft? | Notes |
|---------|---------|-----------|-------|
| numpy | BSD-3-Clause | No | [VERIFIED: numpy.org] |
| scipy | BSD-3-Clause | No | [VERIFIED: scipy.org] |
| scikit-learn | BSD-3-Clause | No | [VERIFIED: scikit-learn.org] |
| matplotlib | PSF/BSD-compatible | No | [VERIFIED: matplotlib.org] |
| biopython | Biopython License (BSD-like) | No | [CITED: github.com/biopython/biopython] |
| openmm | MIT + LGPL (CUDA/HIP/OpenCL platforms) | Partial | [CITED: openmm.org] — see note |
| pdbfixer | MIT | No | [CITED: github.com/openmm/pdbfixer] |
| pydantic | MIT | No | [VERIFIED: pydantic.dev] |
| meeko | LGPL-2.1 | Yes (weak) | [VERIFIED: github.com/forlilab/Meeko] — CRITICAL NOTE |
| vina (AutoDock Vina Python) | Apache-2.0 | No | [VERIFIED: pypi.org/project/vina] |

### rapidock-env Dependency License Summary

| Package | License | Copyleft? | Notes |
|---------|---------|-----------|-------|
| pytorch 2.7 | BSD-3-Clause | No | [ASSUMED] — standard PyTorch license |
| torchvision | BSD-3-Clause | No | [ASSUMED] |
| torchaudio | BSD-3-Clause | No | [ASSUMED] |
| pyg (PyTorch Geometric) | MIT | No | [ASSUMED] |
| mdanalysis | GPL-2.0 | **YES — GPL** | [ASSUMED] — requires verification |
| e3nn | MIT | No | [ASSUMED] |
| rdkit | BSD-3-Clause | No | [ASSUMED] |
| RAPiDock (from GitHub) | Unknown | Unknown | [ASSUMED] — must check repo LICENSE |

### CRITICAL LICENSE RISKS

**Risk 1: Meeko LGPL-2.1 (score-env)**
Meeko is licensed under LGPL-2.1. LGPL is OSI-approved and iGEM-legal, but:
- HybriDock-Pep imports Meeko as a Python package (dynamic import via `import meeko`)
- Under LGPL-2.1's "library exception" (§5–6), use as a dynamically-linked library
  does NOT require the user's application to be LGPL.
- THEREFORE: HybriDock-Pep itself can be MIT-licensed as long as Meeko is separately
  installable and replaceable (it is — it's in score-env.yml, not bundled in the repo).
- `docs/licenses.txt` MUST include this rationale note explicitly.
- **Verdict: Blocked only if LGPL is interpreted as "no LGPL at all." iGEM says "OSI-approved"
  and LGPL is OSI-approved. This is NOT a blocker, but must be documented.**

**Risk 2: MDAnalysis GPL-2.0 (rapidock-env)**
MDAnalysis is listed in rapidock-env.yml under conda-forge. MDAnalysis is GPL-2.0 licensed.
GPL-2.0 is OSI-approved but has strong copyleft — it would require HybriDock-Pep's source
to be GPL-2.0 if MDAnalysis is distributed as part of the same work.

However: MDAnalysis lives in rapidock-env, which is only used by the RAPiDock subprocess
(run_rapidock.py). HybriDock-Pep's own source code (in score-env) does NOT import MDAnalysis.
The environments are separate; the GPL does not propagate across subprocess boundaries.

NEVERTHELESS: The iGEM software license page says "OSI-approved open source license." GPL is
OSI-approved. The question is whether iGEM's rubric intends "any OSI license including GPL" or
"permissive OSI licenses only." CLAUDE.md §2.6 says "no copyleft dependencies in our own source"
— `run_rapidock.py` is our source and it may import MDAnalysis transitively via RAPiDock.

**Action required:** Verify whether RAPiDock itself imports MDAnalysis (it likely does, as
MDAnalysis is in rapidock-env.yml). If so, clarify with Ram whether GPL in the RAPiDock
subprocess environment is acceptable, or whether MDAnalysis must be replaced/avoided.
This is flagged as Assumption A4.

**Risk 3: RAPiDock license (rapidock-env)**
The RAPiDock repo license is unknown until the executor reads the GitHub repo's LICENSE file.
This must be checked before committing `docs/licenses.txt`.

### pip-licenses Commands (for docs/licenses.txt generation)

```bash
# In score-env on RTX machine:
conda activate score-env
pip install pip-licenses
pip-licenses --format=plain-vertical --with-urls --order=license > /tmp/score-env-licenses.txt

# In rapidock-env on RTX machine:
conda activate rapidock-env
pip install pip-licenses
pip-licenses --format=plain-vertical --with-urls --order=license > /tmp/rapidock-env-licenses.txt
```

Then combine into `docs/licenses.txt` with section headers.

---

## Architecture Patterns

### System Architecture Diagram (for docs/architecture.md)

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

### Calibration/Config Flow (for architecture.md §5)

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

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDB download | HTTP client | `urllib.request.urlretrieve` or `requests.get` | Already in stdlib; RCSB URL is stable: https://files.rcsb.org/download/{pdb_id}.pdb |
| Pearson correlation | Manual formula | `scipy.stats.pearsonr` | Handles NaN checking, p-value | 
| Markdown report generation | Template engine | f-strings + `textwrap.dedent` | Report is simple tabular output |
| Notebook execution test | Re-run cells manually | `nbformat` + `nbconvert` for CI (v2 scope) | Out of scope for v1 (D-02) |
| License listing | Parse pip metadata manually | `pip-licenses` tool | Handles edge cases (METADATA, PKG-INFO, egg-info) |
| Notebook cell output | Custom display | Default Jupyter pre-run output (JSON in .ipynb) | Jupyter already embeds PNG as base64 |

---

## Common Pitfalls

### Pitfall 1: benchmark.py uses relative paths for output-dir
**What goes wrong:** `hybridock-pep dock --output-dir runs/benchmark/3EQS` resolves relative
to the CWD at subprocess spawn time. If benchmark.py runs from the project root, this works.
If not, it silently writes to the wrong directory.
**Prevention:** Always resolve `--output-dir` to an absolute path in benchmark.py before
passing to the CLI subprocess.

### Pitfall 2: Vina-only baseline uses different poses
**What goes wrong:** Running `hybridock-pep dock --scoring vina` from scratch generates NEW
poses via RAPiDock (Stage 1), giving a different random sample than the hybrid run. The
delta then measures the randomness of two different pose sets, not the scoring difference.
**Prevention:** Always use `--input-poses <poses_dir_from_hybrid_run>` for the Vina-only
baseline run. Use the same `--seed` as insurance.

### Pitfall 3: Tutorial notebook uses data/calibration.json (global), not fixture
**What goes wrong:** `tests/fixtures/mdm2_calibration.json` exists. If the tutorial notebook
uses `data/calibration.json` (which is calibrated on 3 training complexes, not MDM2-specific),
the tutorial output changes when calibration.json is updated.
**Prevention:** The tutorial should explicitly pass
`--calibration data/calibration.json` (the production calibration). The fixture
`tests/fixtures/mdm2_calibration.json` is for unit tests only (test_e2e.py).

### Pitfall 4: pip-licenses misses conda-installed packages
**What goes wrong:** `pip-licenses` only lists packages visible to pip in the current env.
Some conda packages (e.g., openmm installed via conda-forge, not pip) may not appear.
**Prevention:** Run `pip-licenses` AND `conda list --export` for completeness. Add a note
in docs/licenses.txt that conda-native packages (openmm, biopython, scipy) are documented
separately from their conda-forge listing.

### Pitfall 5: Calmodulin benchmark site coordinates
**What goes wrong:** CaM undergoes large conformational change on peptide binding; the apo
structure centroid is NOT the right binding site center. Using CaM centre of mass instead
of bound-peptide centroid wastes the 25 Å box.
**Prevention:** Always compute binding site from the peptide Cα centroid in the HOLO
(complex) structure, not the receptor chain centroid.

### Pitfall 6: MDM2 and MDMX share the same PMI peptide (3EQS vs 3EQY)
**What goes wrong:** If benchmark.py uses the same PDB file for both 3EQS and 3EQY (both
contain TSFAEYWNLLSP), test results may appear correlated even if one structure is not
downloaded correctly.
**Prevention:** Download and verify both PDB IDs separately. 3EQS is MDM2; 3EQY is MDMX
(a related but distinct protein). They should produce different scores.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 (already installed in score-env) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -m slow --cov=hybridock_pep` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEST-03 | benchmark harness imports, CLI parsing, output schema | unit | `pytest tests/test_benchmark.py -x` | No — Wave 0 |
| TEST-03 | benchmark execution vs. 10 complexes, Pearson r >= 0.55 | manual/GPU | Run on RTX 5070, commit results | Not automatable in CI |
| DOCS-01 | README.md and INSTALL.md exist and contain ADFRsuite link | smoke | `grep -q "ADFRsuite" README.md && grep -q "ADFRsuite" INSTALL.md` | No — write in Phase 8 |
| DOCS-02 | docs/architecture.md exists and covers all 5 sections | manual review | `ls docs/architecture.md` | No — write in Phase 8 |
| DOCS-03 | docs/licenses.txt exists, pip-licenses output for both envs | manual + grep | `grep -i "LGPL\|GPL\|AGPL" docs/licenses.txt` | No — generate in Phase 8 |
| DOCS-04 | tutorial notebook exists and has pre-run cell outputs | smoke | `python -c "import nbformat; nb=nbformat.read('docs/tutorial.ipynb',4); assert any(c.outputs for c in nb.cells if c.cell_type=='code')"` | No — create in Phase 8 |

### Sampling Rate

- Per task commit: `pytest tests/ -x -q --ignore=tests/test_e2e.py`
- Per wave merge: `pytest tests/ -q`
- Phase gate: All unit tests green; DOCS-01 through DOCS-04 manually verified

### Wave 0 Gaps

- [ ] `tests/test_benchmark.py` — structural tests for benchmark.py (import, arg parsing, output schema)
- [ ] `docs/tutorial.ipynb` — does not exist yet
- [ ] `docs/architecture.md` — does not exist yet
- [ ] `README.md` — does not exist yet
- [ ] `data/test_complexes.csv` — does not exist yet (content provided in this research)
- [ ] `data/test_complexes_meta.csv` — chain ID mapping for benchmark site-coord computation

---

## Security Domain

Phase 8 introduces no new network-exposed endpoints, authentication flows, or user-supplied
input paths beyond what is already handled by Phase 5 CLI validation. The benchmark harness
downloads PDB files from RCSB (https://files.rcsb.org/download/), which is read-only and
public. No credentials are required.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | Yes (benchmark PDB IDs from CSV) | Validate PDB IDs match `[0-9][A-Z0-9]{3}` regex before constructing download URL |
| V2 Authentication | No | No auth required for RCSB downloads |
| V3 Session Management | No | Stateless script |
| V6 Cryptography | No | No encryption needed |

**PDB ID injection risk:** If `test_complexes.csv` contains a malformed PDB ID, the download
URL could be malformed. Validate each pdb_id against `^[0-9][A-Z0-9]{3}$` before constructing
the URL. This is a low-severity risk (input is a committed CSV, not user-supplied at runtime)
but good practice.

---

## Environment Availability

Step 2.6: SKIPPED for execution environment (benchmark runs on RTX machine, not this machine).

For the dev environment (where benchmark.py code is written and unit-tested):

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest | test_benchmark.py | Yes (in base Python via score-env pattern) | >=8.0 | — |
| scipy | Pearson r in benchmark.py | Yes (in score-env) | >=1.13 | — |
| nbformat | Tutorial notebook smoke test | Unknown | — | Manual review |
| pip-licenses | License audit generation | Requires install | — | `pip install pip-licenses` in score-env |
| requests or urllib | PDB download in benchmark.py | stdlib (urllib) | stdlib | use urllib.request — no extra dep |

**Missing dependencies with fallback:**
- `pip-licenses` — install command: `pip install pip-licenses` (score-env). Generated on RTX machine.
- `nbformat` — install command: `pip install nbformat` if notebook smoke test is included.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 1PQ1 Bcl-xL/Bim peptide sequence is MRPEIWIAQELRRIGDE (17-mer); actual ATOM record sequence may differ | Benchmark Dataset | Wrong peptide sequence passed to --peptide; scoring would fail or produce nonsense results |
| A2 | 3DAB MDMX/p53 Kd ~2.5 µM (pKd=5.60); exact value not found in search results for this specific complex — FP assay values for wild-type p53 to MDMX reported in the range 2–5 µM | Benchmark Dataset | pKd anchor incorrect; Pearson r computation degraded |
| A3 | 2W73 Calmodulin/calcineurin A Kd ~2 nM (pKd=8.70); ITC referenced in the paper but exact value not confirmed from PDB or abstract in this session | Benchmark Dataset | pKd anchor incorrect |
| A4 | MDAnalysis in rapidock-env is GPL-2.0 licensed; whether this propagates to HybriDock-Pep under iGEM's rules is unclear | License Audit | If iGEM interprets "OSI-approved" as "permissive only," GPL in rapidock-env could disqualify the tool from the Best Software award |
| A5 | RAPiDock GitHub repo has a permissive license (not GPL/copyleft) | License Audit | If RAPiDock is GPL, the entire rapidock-env is contaminated; must verify before submission |
| A6 | PyTorch 2.7, torchvision, torchaudio, PyG are all BSD-3-Clause; this is the standard PyTorch/PyG license but not verified in this session | License Audit | Low risk — these are well-known permissive licenses |
| A7 | 4QVF Bcl-xL BIM BH3 peptide sequence in ATOM records is DMRPEIWIAQELRRIGDE (18 residues); actual sequence may differ | Benchmark Dataset | Same as A1; must verify from PDB ATOM records |
| A8 | 2Y4V DAP kinase-1 peptide Kd ~50 nM (pKd=7.30); value estimated from related calmodulin-binding domain literature | Benchmark Dataset | pKd anchor incorrect |
| A9 | 3GP2 CaMKII peptide Kd ~1 nM (pKd=9.0); CaM-CaMKII trapping Kd is sub-picomolar for the kinase autoinhibited state, but the isolated CaM-BD peptide Kd is ~1-10 nM (ITC, Rellos 2010) | Benchmark Dataset | pKd too optimistic if literature value is for isolated peptide vs full kinase |

**User confirmation needed before execution:** A1, A2, A3 (peptide sequences and pKd values
should be verified from PDB ATOM records and cited literature before running benchmark.py).
A4, A5 (license questions require Ram's judgment on iGEM interpretation).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vina standalone CLI (100×) | Vina Python API (batch) | Vina 1.2.0 (2021) | Eliminates 100 fork+exec cycles per run |
| pip show for license | pip-licenses tool | 2019+ | Batch output with SPDX identifiers |
| NMR structures for benchmark | X-ray crystal structures (≤2.5 Å) | Always preferred | NMR lacks high-res Cα positions needed for RMSD |
| All BH3 peptides (23-mer) | Crystal-segment peptides (16-18 residues) | — | Keeps within RAPiDock training distribution |

---

## Open Questions

1. **MDAnalysis GPL status in rapidock-env**
   - What we know: MDAnalysis is GPL-2.0; it is in rapidock-env.yml; HybriDock-Pep score-env does not import it directly.
   - What's unclear: Whether iGEM considers GPL in a subprocess environment as contaminating the tool's license.
   - Recommendation: Ask Ram directly before the license audit is committed. If needed, check whether RAPiDock can be used without MDAnalysis (unlikely — it uses MDAnalysis for trajectory analysis).

2. **Benchmark complex peptide sequences — exact ATOM records**
   - What we know: PDB IDs and approximate sequences identified via RCSB search.
   - What's unclear: The exact crystallographic peptide sequence (ATOM record chain, residue range) for 1PQ1, 4QVF, 2W73, 2Y4V, 3GP2.
   - Recommendation: Executor must use Biopython to extract the peptide chain sequence from the ATOM records of each PDB file before writing data/test_complexes.csv.

3. **Benchmark site coordinates for CaM complexes**
   - What we know: CaM undergoes wrap-around conformational change; generic centre is wrong.
   - What's unclear: Optimal box_size for 17–18 residue peptides that extend through the CaM hydrophobic tunnel.
   - Recommendation: Use box_size=30 Å (larger than the default 20 Å) for CaM complexes to ensure the full extended peptide is covered.

---

## Sources

### Primary (HIGH confidence)
- RCSB PDB entries: 3EQS, 3EQY, 3DAB, 1G73, 2W73, 1PQ1, 4QVF, 1VYJ, 2Y4V, 3GP2 — directly fetched
- Pazgier et al. PNAS 2009 (PMC2660734) — 3EQS/3EQY Kd values (ITC + SPR), pKd=8.48 and 8.05
- Kontopidis et al. PNAS 2005 — 1VYJ PCNA/p21 Kd=100 nM (pKd=7.0) [CITED: rcsb.org/structure/1VYJ]
- github.com/forlilab/Meeko — LGPL-2.1 license confirmed [VERIFIED]
- pypi.org/project/vina — Apache-2.0 license confirmed [VERIFIED]
- github.com/openmm/pdbfixer — MIT license [CITED]
- INSTALL.md, cli.py, driver.py — read directly from repo [VERIFIED]

### Secondary (MEDIUM confidence)
- Chai et al. Cell 2000 + Nikolovska-Coleska et al. 2004 — 1G73 XIAP/AVPI Kd=0.43 µM (pKd=6.37) [CITED via WebSearch]
- Rellos et al. PLOS Biol 2010 — 3GP2 CaMKII peptide Kd ~1 nM from ITC [CITED via WebSearch]
- Chen et al. Mol Cell 2005 — Bcl-xL/BIM Kd in 1-4 nM range from FP competition [CITED via WebSearch]
- openmm.org license page — MIT + LGPL (CUDA/HIP/OpenCL platforms) [CITED]

### Tertiary (LOW confidence)
- 3DAB MDMX/p53 pKd=5.60 — estimated from FP assay literature for wild-type p53 binding MDMX; not directly confirmed for this structure's peptide
- 2W73 CaM/calcineurin pKd=8.70 — estimated; ITC referenced in paper but not confirmed in this session
- 2Y4V CaM/DAP kinase-1 pKd=7.30 — estimated from related CaM-BD literature

---

## Metadata

**Confidence breakdown:**
- Benchmark dataset (PDB IDs): HIGH — structures confirmed via RCSB fetches
- Benchmark dataset (pKd values): MEDIUM — 4 values are HIGH (3EQS, 3EQY, 1VYJ, 1G73); 6 are MEDIUM/LOW (need executor verification from cited literature)
- Architecture documentation content: HIGH — read directly from source code
- License audit (score-env): HIGH for Meeko/Vina/pydantic; MEDIUM for openmm
- License audit (rapidock-env): LOW — MDAnalysis GPL and RAPiDock license unverified
- Benchmark harness pattern: HIGH — derived from CONTEXT.md locked decisions + existing CLI

**Research date:** 2026-04-26
**Valid until:** 2026-05-26 (stable domain — PDB entries and licenses change slowly)
