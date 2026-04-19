# Pitfalls Research: HybriDock-Pep

**Domain:** Hybrid ML + physics peptide docking pipeline
**Researched:** 2026-04-18
**Overall confidence:** HIGH (known pitfalls from spec + MEDIUM-HIGH from literature verification)

---

## Summary — Top 3 Most Dangerous

1. **CUDA/PyTorch version mismatch on Blackwell (CC 12.0)** — RAPiDock's pinned stack (CUDA 11.5, PyTorch 1.11) flat-out will not load on an RTX 5070. This is a day-one blocker. The fix is confirmed in the RAPiDock repo itself: they ship a CUDA 12.4 YAML. But the old pins in the paper's requirements.txt will mislead anyone who reads the paper first.

2. **Vina `--score_only` on poses that clip the grid box** — When RAPiDock generates a pose that extends even one atom outside the box boundaries, Vina 1.2.5 raises a fatal "ligand outside grid box" error and scores nothing. For a 15-residue peptide across 100 stochastic poses, this will happen repeatedly. Without explicit boundary validation before scoring, your pipeline silently drops poses and produces biased ensemble statistics.

3. **Entropy coefficient alpha calibrated on wrong ensemble or wrong temperature** — If `calibrate_alpha.py` runs on a dataset with a different length distribution from LISDAELEAIFEADC, or at a reference temperature that does not match the implicit solvent temperature in OpenMM, alpha will be systematically wrong. The benchmark will look plausible until you compare against held-out data, at which point Pearson r collapses. This is a calibration scope failure, not a code bug, and is hard to detect without the full benchmark.

---

## Critical Pitfalls — Project-Killers

### Pitfall C1: CUDA 11.5 / PyTorch 1.11 Incompatible with Blackwell CC 12.0

**What goes wrong:** RAPiDock's original `requirements.txt` pins `torch==1.11.0` and `cudatoolkit==11.5.1`. NVIDIA Blackwell (RTX 5070, compute capability 12.0) requires CUDA 12.4+ drivers. PyTorch 1.11 was built before Blackwell existed and contains no kernel support for CC 12.0. The environment will install cleanly, then fail at the first `torch.cuda.is_available()` or model load with a CUDA driver mismatch.

**Why it happens:** The RAPiDock paper was published against an older GPU generation. The GitHub repo has since added a CUDA 12.4 environment YAML (`conda env create -f environment.yml`), but anyone following the paper's supplementary methods will hit the old pins.

**Consequences:** Total inference failure. No poses generated. The pipeline cannot proceed past Stage 1.

**Prevention:**
- Use `envs/rapidock-env.yml` from this repo, not the upstream paper's requirements.
- Pin `pytorch>=2.3, cudatoolkit>=12.4`.
- Run `python -c "import torch; print(torch.cuda.get_device_capability())"` as a smoke test before any inference. Abort loudly if < (12, 0).
- Fall back to a CUDA 11.8 / Ada-generation machine rather than the old pins — the old pins are the bug, not the fix.

**Detection:** `RuntimeError: CUDA error: no kernel image is available for execution on the device` or `torch.cuda.is_available()` returns False despite installed GPU.

**Confidence:** HIGH — confirmed from RAPiDock GitHub and NVIDIA CC compatibility tables.

---

### Pitfall C2: Vina Ignores Partial Charges in PDBQT `q` Column

**What goes wrong:** AutoDock Vina's scoring function entirely ignores the `q` (partial charge) column in PDBQT files. Writing code to "preserve Gasteiger charges across poses for Vina scoring" or "extract per-atom Coulombic contributions from the Vina score" is a no-op. The Vina score is purely empirical and does not include an explicit electrostatics term.

**Why it happens:** Vina uses an empirical scoring function based on interaction terms (gauss, repulsion, hydrophobic, hydrogen bonding, torsional), not a force-field Coulomb term. This is documented in the Vina manual but widely misunderstood because AutoDock4 (the predecessor) does use charges.

**Consequences:** Misleading code that appears to run but produces the same output regardless of charge manipulation. More dangerously: if AD4 mode (which does use charges) is inadvertently skipped because "Vina already handles charges," the charge-sensitivity signal is lost entirely.

**Prevention:**
- Run AD4 scoring in parallel via `vina --scoring ad4` — AD4 uses Gasteiger charges explicitly.
- For top-K poses, run MM-GBSA via OpenMM (AMBER ff14SB + GBn2), which is the strongest accuracy lever for electrostatics.
- Do not recompile Vina with a Coulomb term. That path was explicitly evaluated and rejected in the spec (§5.6–5.7).

**Detection:** No code-level symptom — scores appear normal. The diagnostic is checking whether `ranked_poses.csv` has meaningful AD4 scores alongside Vina scores.

**Confidence:** HIGH — documented in official Vina manual.

---

### Pitfall C3: Poses Clipping the Grid Box Silently Drop from Ensemble

**What goes wrong:** `vina --score_only` requires the entire ligand to be inside the grid box. RAPiDock is a diffusion generative model with stochastic inference — some poses will extend outside the box, especially for a 15-residue peptide at the edge of a shallow binding groove. Vina 1.2.5 (vs. 1.2.2) made the boundary check stricter and raises a fatal error for out-of-bounds poses.

**Why it happens:** The box is defined once at job submission around the known binding site. Diffusion poses are unconstrained; the model can place atoms outside the user-specified region. Without upstream validation, these poses crash the scoring step.

**Consequences:** Silent loss of some fraction of the 100 poses, biased ensemble statistics, skewed convergence curves. If the best poses happen to extend slightly outside the box, you lose signal.

**Prevention:**
- Before calling `vina --score_only`, validate all atom coordinates against the grid boundaries. If any atom is outside, expand the box by 2 Å and re-score, or log and skip.
- Set grid box to `--box 20` with a 2 Å padding beyond the expected peptide extent.
- Log every skipped pose with the reason; surface in `run_metadata.json`.

**Detection:** `Vina runtime error: The ligand is outside the grid box` in subprocess stderr.

**Confidence:** HIGH — documented in AutoDock Vina GitHub issues #112 and #309.

---

### Pitfall C4: PULCHRA v3.07 Produces Incomplete Aromatic Side-Chain Atoms

**What goes wrong:** PULCHRA v3.07 (the most commonly distributed version) silently produces incomplete aromatic side-chain atoms (Phe, Tyr, Trp, His) when reconstructing all-atom models from ADCP Cα-only output. Incomplete atoms crash downstream atom-type assignment in `prepare_ligand` and produce incorrect AD4 maps.

**Why it happens:** A reconstruction bug in v3.07's aromatic ring closure algorithm. Not caused by input — reproducible on any structure containing these residue types.

**Consequences:** Vina/AD4 scoring will either crash on missing atoms or silently assign wrong atom types (UNK), producing garbage scores with no error signal. LISDAELEAIFEADC contains Phe (F), which will trigger this on every run.

**Prevention:**
- Pin PULCHRA to exactly v3.04. Build from source or use a verified conda recipe.
- Run `pulchra --version` in the smoke test and abort if not 3.04.
- After reconstruction, validate atom count per residue against expected values from a reference lookup table before proceeding to PDBQT preparation.

**Detection:** Unexpected missing ATOM records for ring atoms (CD1, CD2, CE1, CE2, CZ in Phe) in PULCHRA output PDB.

**Confidence:** HIGH — documented in spec §16 as a reproducible bug with a clear version fix.

---

### Pitfall C5: ref2015 Scoring Function Cysteine RMSD Alignment Failure in PyRosetta

**What goes wrong:** The C-terminal cysteine in LISDAELEAIFEADC triggers a Rosetta ref2015 RMSD alignment failure in RAPiDock's optional PyRosetta post-relax step. The failure is silent in some configurations — it can produce a partially-relaxed structure with incorrect terminal geometry rather than an error.

**Why it happens:** ref2015 has edge-case handling of terminal cysteine that interacts badly with RMSD alignment during the relax protocol when the C-terminus is in a particular torsion state.

**Consequences:** If not caught, relaxed poses have corrupted terminal geometry, which propagates into downstream Vina/AD4 scoring and produces outlier scores for poses that happen to trigger the bug.

**Prevention:**
- Skip the PyRosetta relax step by default. Add a brief OpenMM minimization (10–50 steps with ff14SB + GBn2) before Vina scoring instead.
- If PyRosetta relax is ever needed, apply the workaround documented in spec §16.1.

**Detection:** Unusual terminal bond lengths in relaxed PDB (> 2 Å for C–N, C–S). The OpenMM minimization replacement produces cleaner geometry reliably.

**Confidence:** HIGH — documented in CLAUDE.md §2.5 and spec §16.1 as a reproducible, project-specific bug.

---

### Pitfall C6: Two Python Stacks in One Conda Environment

**What goes wrong:** RAPiDock requires Python 3.9, PyTorch 2.3+, CUDA 12.4, PyG 2.x, E3NN 0.5.1, MDAnalysis 2.6, and RDKit. The scoring stack requires Python 3.11, OpenMM 8.1+, Meeko, scikit-learn, and ADFRsuite binaries on PATH. Merging these into one environment produces irreconcilable dependency conflicts — PyG versions that need Python 3.9 conflict with OpenMM's Python 3.11 requirement, and MKL/OpenMP version constraints interact across PyTorch and scientific Python.

**Why it happens:** ML frameworks and biophysics simulation tools have historically evolved separately. PyTorch's Intel MKL dependency (mkl 2024.1+ conflicts with PyTorch < 2.x) is a known documented conda issue that appears even in 2024-2025.

**Consequences:** The environment installs but produces wrong numerical results from mismatched BLAS/OpenMP libraries, or fails at import time with obscure shared-library errors.

**Prevention:**
- Maintain exactly two environments: `rapidock-env` (Python 3.9) and `score-env` (Python 3.11).
- The driver script in `score-env` orchestrates Stage 1 via `conda run -n rapidock-env ...`.
- Never import RAPiDock modules from `score-env` directly.

**Confidence:** HIGH — this is a structural design constraint, not a hypothesis.

---

## Common Mistakes

### Pitfall M1: PDBQT Peptide Fragmentation After Vina Scoring

**What goes wrong:** AutoDock Vina reorders atomic coordinates based on the torsion tree (rotatable bond branching). For a peptide with many residues, this scrambles the coordinate order relative to the input PDB, making the output PDBQT look "fragmented" when visualized. This is not a score error — it is a coordinate-ordering artifact.

**Prevention:** Do not interpret PDBQT atom ordering as residue ordering. Always reconstruct residue identity from atom names + residue numbers, not coordinate position. Use Meeko's output parsing, not raw PDBQT atom-order assumptions.

**Confidence:** HIGH — documented in AutoDock Vina docs and community reports.

---

### Pitfall M2: Meeko Sidechain Truncation for Receptor Preparation

**What goes wrong:** Meeko truncates sidechains at Cα when preparing receptor PDBQT files in certain modes. This is documented behavior (PyPI 0.6.0a3 docs: "Sidechains are truncated at the C-alpha") but widely misread as a preparation error. For a rigid receptor, this is a design choice; for flexible residue docking, it silently removes all sidechain interaction information.

**Prevention:** Verify that receptor PDBQT contains full sidechain atoms when using `prepare_receptor4.py` from ADFRsuite (not Meeko's receptor mode). Meeko is the correct tool for ligand/peptide preparation, not receptor preparation in this pipeline.

**Confidence:** HIGH — documented in official Meeko docs.

---

### Pitfall M3: autogrid4 "Unknown Receptor Type" and Missing HD Map

**What goes wrong:** Two distinct autogrid4 failures. First: if the receptor PDB contains non-standard atom types (selenomethionine, zinc in metalloproteins, modified residues) autogrid4 silently fails to generate map files for those types. Second: autogrid4 frequently fails to generate the HD (hydrogen donor) map, causing `--scoring ad4` to abort with `Affinity map for atom type HD is not present`.

**Why it happens:** PDB files from the PDB sometimes contain alternate occupancy atoms, engineered residues, or bound metals not in AutoDock's parameter library. HD map generation requires explicit polar-hydrogen handling in the `.gpf` grid parameter file.

**Consequences:** AD4 scoring cannot run. The pipeline proceeds with Vina-only scoring, silently degrading result quality without warning.

**Prevention:**
- Validate receptor PDBQT atom types against the AutoDock4 parameter library before running autogrid4.
- Explicitly include `map receptor.HD.map` in the `.gpf` file and verify it exists after autogrid4 runs before launching AD4 scoring.
- Strip alternate occupancy atoms (keep only occupancy A) and remove non-water HETATM records during receptor preparation.

**Confidence:** HIGH — documented in AutoDock-Vina GitHub issues #105, #259, #297, #308.

---

### Pitfall M4: AD4 Positive Scores Under Certain Exhaustiveness Settings

**What goes wrong:** When using `vina --scoring ad4` for pose rescoring (`--score_only`), some poses receive positive (unfavorable) binding energies at certain exhaustiveness settings, then correct negative energies at others. The behavior is non-deterministic and reported as unresolved in GitHub issue #48.

**Prevention:** For `--score_only` mode (no search, pure scoring), exhaustiveness has minimal effect and the issue is less likely to manifest. However, explicitly set `--exhaustiveness 32` as a baseline and flag any positive AD4 score (> 0 kcal/mol) as a scoring anomaly in `ranked_poses.csv`. Do not silently accept positive scores.

**Confidence:** MEDIUM — reported but unresolved in Vina GitHub, behavior appears limited to full docking mode rather than pure rescoring.

---

### Pitfall M5: CUDA Nondeterminism Defeats `--seed` Reproducibility

**What goes wrong:** Even with `--seed N`, RAPiDock runs on CUDA operations that are inherently nondeterministic: cuDNN convolution algorithm selection via benchmark (different algorithm may be chosen across runs), and atomicAdd operations in gradient accumulation that use nondeterministic floating-point order. Two runs with identical seeds on the same machine can produce slightly different poses.

**Why it happens:** PyTorch's CUDA backend benchmarks convolution algorithms at runtime, selecting the fastest for current hardware state. This benchmarking is seeded separately from the user-visible RNG.

**Consequences:** `run_metadata.json` logs a seed, creating a false impression of full reproducibility. Benchmark comparisons across different runs are confounded.

**Prevention:**
- Set `torch.backends.cudnn.benchmark = False` and `torch.use_deterministic_algorithms(True)` in the RAPiDock runner subprocess.
- Flag in `run_metadata.json` that CUDA nondeterminism is suppressed only when these flags are set. Document that cross-machine reproducibility is not guaranteed even with identical seeds.
- Accept ~1–2% pose-level variation as irreducible if deterministic mode incurs unacceptable performance cost.

**Confidence:** HIGH — documented in official PyTorch reproducibility notes.

---

### Pitfall M6: Receptor Missing Atoms / Chain Breaks From PDB Download

**What goes wrong:** PDB files for crystal structures (including 1CZB, 1I0Z) frequently have missing residues, alternate occupancies, engineered mutations, or crystallographic water/ion records. If these are passed directly to `prepare_receptor4.py`, the PDBQT will have unconnected chain fragments. AutoDock scores these silently — no error — but scores are physically meaningless in the gap region.

**Prevention:**
- Run a receptor validation step before preparation: check for chain breaks (gap > 4 Å between consecutive Cα atoms), missing residues relative to SEQRES records, multiple occupancy atoms.
- Repair with PDBFixer (OpenMM ecosystem, OSI-licensed) or MODELLER for homology-based loop completion.
- Hash the processed receptor file and log to `run_metadata.json` for traceability.

**Confidence:** HIGH — standard practice in all AutoDock tutorials; confirmed in PMC4868550.

---

### Pitfall M7: Cα-Only RMSD Clustering Is Sensitive to N-/C-Terminal Flexibility

**What goes wrong:** Pairwise Cα RMSD between 100 peptide poses is used as the distance matrix for agglomerative clustering. Terminal residues of flexible peptides are highly mobile and dominate the RMSD calculation, causing the cluster algorithm to group poses by terminal position rather than binding-mode similarity in the binding groove core.

**Why it happens:** Cα RMSD after rigid-body alignment weights all residue positions equally. Terminal residues of a 15-mer that flap freely outside the pocket contribute the same to RMSD as core binding residues that are tightly constrained.

**Consequences:** Cluster centroids may not represent the dominant binding mode — they represent the centroid of terminal-position clusters. The "best cluster" selected for MM-GBSA may not be the most physically meaningful.

**Prevention:**
- Compute RMSD over only the 5–8 residues that contact the receptor (defined by the site parameter), not the full peptide.
- Alternatively, weight Cα contributions by contact frequency across the ensemble before computing RMSD.
- Validate cluster quality with silhouette scores and by visual inspection of cluster populations.

**Confidence:** MEDIUM-HIGH — well-established in RMSD clustering literature (PMC4925300); specific application to peptide termini is domain-specific extrapolation.

---

### Pitfall M8: MM-GBSA Implicit Solvent Inconsistency With Explicit-Solvent-Generated Poses

**What goes wrong:** RAPiDock generates poses in vacuo (no explicit solvent). Running MM-GBSA on these poses with OpenMM + GBn2 implicit solvent does not account for the change in solvation environment. Mean absolute energy differences of 6–7 kJ/mol have been reported when comparing GB energies of snapshots from explicit- vs. implicit-solvent simulation. For small differences in ΔΔG between peptide variants or poses, this noise may dominate the signal.

**Prevention:**
- Always precede MM-GBSA with a short OpenMM energy minimization (50–200 steps) in the same implicit solvent model (GBn2) to relax poses into a consistent solvation environment before computing single-point energies.
- Do not use MM-GBSA for absolute binding free energies — use it only for relative ranking within the same peptide on the same receptor.
- Flag in output metadata that MM-GBSA scores have ~1.5–2 kcal/mol noise floor from the implicit-solvent approximation.

**Confidence:** MEDIUM — supported by PubMed 23595060 and JCTC cysteine/GB paper; OpenMM-specific details extrapolated.

---

## iGEM-Specific Pitfalls

### Pitfall I1: Non-OSI-Licensed Binaries Bundled in Repo

**What goes wrong:** ADFRsuite and AutoDock4 binaries have non-redistributable licenses (Scripps Research non-commercial). If they are committed to the repository, the repo is immediately ineligible for the iGEM Best Software Tool award, which requires an OSI-approved license for all submitted code.

**Prevention:**
- Never commit ADFRsuite or AutoDock4 binaries. Link to official download in `INSTALL.md`.
- Run a license audit on all Python dependencies in `score-env.yml` and `rapidock-env.yml` before final submission. Flag any GPL/LGPL/AGPL dependency (copyleft disqualifies).
- Confirm PyRosetta's license (academic non-commercial) — it is optional and used only for the relax step that is skipped by default. If included, it must be optional with a clear note that the core pipeline runs without it.

**Detection:** `pip-licenses` or `conda list --export | grep -i gpl` in both environments.

**Confidence:** HIGH — iGEM competition rules explicitly require OSI-approved license; confirmed in competition deliverables page.

---

### Pitfall I2: Wiki Tutorial Notebook Fails to Run on Fresh Install

**What goes wrong:** Jupyter notebooks in iGEM wiki submissions are expected to run top-to-bottom on a fresh install. Common failure modes: hardcoded absolute paths to local PDB files, undeclared dependencies that are in the developer's environment but not in the conda YAML, cells that require the RTX 5070 GPU and produce no graceful fallback.

**Prevention:**
- Test the tutorial notebook on a fresh conda environment (not the development machine) before submission.
- Provide CPU-mode fallback for RAPiDock inference (even if with N=5 samples for demo purposes) so the notebook runs without a GPU.
- Use relative paths everywhere in the notebook.

**Confidence:** HIGH — iGEM Best Software Tool rubric explicitly evaluates documentation and reproducibility.

---

### Pitfall I3: Benchmark Pearson r Inflated by Training Set Leakage

**What goes wrong:** If any of the 10 benchmark complexes in `data/test_complexes.csv` share high sequence or structural similarity with the complexes used to calibrate alpha in `data/training_complexes.csv`, the reported Pearson r is inflated relative to generalization performance. iGEM judges and reviewers may not catch this, but the tool will underperform on novel targets (PfLDH binding groove for LISDAELEAIFEADC).

**Prevention:**
- During alpha calibration, ensure training and test sets have < 30% receptor sequence identity (use BLAST or MMseqs2).
- Document the train/test split method in `docs/benchmarking.md`.

**Confidence:** MEDIUM — general ML best practice; specific to calibration step design.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|---|---|---|
| Environment setup | CUDA version mismatch (C1), MKL/OpenMP conflicts (C6) | Smoke test: `python -c "import torch; print(torch.cuda.get_device_capability())"` on first commit |
| Receptor preparation | Missing atoms, chain breaks (M6), autogrid4 HD map failure (M3) | Validate receptor before every run; check `.HD.map` exists |
| Ligand/peptide PDBQT prep | Meeko sidechain truncation (M2), coordinate fragmentation (M1) | Use ADFRsuite `prepare_receptor4.py` for receptor; Meeko only for ligand |
| Stage 1 RAPiDock inference | PULCHRA v3.07 (C4), seed/nondeterminism (M5), PyRosetta relax (C5) | Skip relax by default; add `--version` check in smoke test |
| Vina scoring | Grid box clip (C3), charge misunderstanding (C2) | Validate pose coordinates before calling Vina; run AD4 in parallel |
| AD4 scoring | Autogrid4 HD map missing (M3), positive score anomaly (M4) | Explicit HD map check; flag positive scores |
| RMSD clustering | Terminal flexibility bias (M7) | Cluster over contact-zone Cα only |
| MM-GBSA (optional) | Implicit solvent inconsistency (M8), GBn2 slower than OBC (2x) | Minimize before scoring; warn on energy > noise floor |
| Alpha calibration | Train/test leakage (I3), wrong temperature or length distribution (C-Summary) | Enforce < 30% receptor identity; calibrate at physiological T=310 K |
| Final submission | License violations (I1), notebook fails on fresh install (I2) | `pip-licenses` audit; test notebook from scratch |

---

## Sources

- RAPiDock GitHub: https://github.com/huifengzhao/RAPiDock
- AutoDock Vina docs: https://autodock-vina.readthedocs.io/en/latest/docking_basic.html
- Vina GitHub issue #48 (positive AD4 scores): https://github.com/ccsb-scripps/AutoDock-Vina/issues/48
- Vina GitHub issue #112/309 (outside grid box): https://github.com/ccsb-scripps/AutoDock-Vina/issues/309
- Vina GitHub issue #105 (missing HD map): https://github.com/ccsb-scripps/AutoDock-Vina/issues/105
- Meeko docs (sidechain truncation): https://meeko.readthedocs.io/en/release-doc/lig_overview.html
- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- MM/GBSA explicit vs implicit solvent: https://pubmed.ncbi.nlm.nih.gov/23595060/
- GB models and peptide secondary structure: https://pubs.acs.org/doi/10.1021/acs.jctc.1c01172
- Cluster analysis of MD trajectories (RMSD pitfalls): https://pmc.ncbi.nlm.nih.gov/articles/PMC4925300/
- Peptide docking challenges overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC10392694/
- Frontiers docking quality paper: https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1536504/full
- iGEM software deliverables: https://competition.igem.org/deliverables/project-software
- Vina peptide docking assessment (Briefings in Bioinformatics): https://academic.oup.com/bib/article/16/6/1045/225862
- AutoDock suite review (PMC4868550): https://pmc.ncbi.nlm.nih.gov/articles/PMC4868550/
- RAPiDock paper (Nat. Mach. Intell.): https://www.nature.com/articles/s42256-025-01077-9
