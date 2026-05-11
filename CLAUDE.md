# CLAUDE.md

Authoritative project guidance for Claude Code working on **HybriDock-Pep**.
If this file and the technical spec PDF disagree, the PDF wins (it's the source
of truth); flag the discrepancy and ask.

> **Technical spec**: `docs/HybriDock-Pep_Technical_Specification.pdf` (32 pages).
> Read §4, §5, §11, §12, §16 before writing any code. Don't skim §4–5;
> there are load-bearing corrections in there that invalidate naive approaches.

---

## 1. What this project is

A hybrid peptide docking tool for the iGEM 2026 Best Software Tool award.
Two-stage pipeline:

1. **RAPiDock** (diffusion generative model, Zhao et al. *Nat. Mach. Intell.* 7:1308, 2025) runs N=100 stochastic inference passes on a local RTX 5070 → 100 all-atom peptide pose PDBs.
2. **Physics-based rescoring**: AutoDock Vina (`--score_only`) + AutoDock4 scoring (`--scoring ad4`) in parallel per pose → backbone entropy correction → RMSD clustering → ensemble statistics.

Optional `--refine-topk N` flag runs MM-GBSA via OpenMM on top cluster
centroids for higher-accuracy ΔG on the top candidates.

Output: ranked CSV, best-pose PDB, convergence plot, cluster dendrogram,
run metadata JSON.

**Parent project** (not this repo's scope, but drives requirements):
malaria rapid-diagnostic peptide LISDAELEAIFEADC targeting PfLDH (PDB 1T2D)
over hLDH (PDB 1I0Z).
Note: 1CZB is ASV Integrase (wrong). 1T2D = PfLDH ternary complex with NAD+/oxalate.

---

## 2. Non-negotiable constraints — read before coding

These are the things that will bite if you skip them. They're in the PDF
in full; here's the short version.

### 2.1 Vina does NOT use partial charges

AutoDock Vina ignores the `q` column in PDBQT. The manual says so. Do not
write code that "preserves charges across poses for Vina" or "extracts
per-atom charge contributions from the Vina score" — those are no-ops.

**What we do instead**:
- Run **AutoDock4 scoring in parallel** via `vina --scoring ad4` (AD4
  *does* use the Gasteiger charges, explicitly). This gives us the charge
  signal for free.
- For top-K poses, run **MM-GBSA post-processing via OpenMM** (AMBER
  ff14SB + GBn2 implicit solvent). This is the strongest accuracy lever.

**Do not** recompile Vina to add a Coulomb term. It was considered
and explicitly rejected in §5.6 of the spec. If you think it's a good
idea, re-read §5.7 and ask before starting.

### 2.2 RTX 5070 is Blackwell (CC 12.0)

RAPiDock's `requirements.txt` pins CUDA 11.5 / PyTorch 1.11. These
**will not run** on a 5070. Use **CUDA 12.4+ / PyTorch 2.3+** instead.
Validate RAPiDock inference loads and produces sane output before
committing to the upgraded stack. If inference breaks, fall back to
a CUDA 11.8 / Ada-gen machine (not to the old pins — those pins are
the problem, not the fix).

### 2.3 PULCHRA must be v3.04 exactly

v3.07 produces incomplete aromatic side-chain atoms when fed ADCP
output. If `pulchra --version` reports anything but 3.04, rebuild from
source or pin the conda recipe. The bug is reproducible and documented.

### 2.4 Two Python environments, not one

RAPiDock's pinned stack (even the updated Blackwell-compatible version)
is different enough from the scoring stack that cramming them into one
env causes pain. Keep them separate:

- `rapidock-env` (Python 3.9, PyTorch 2.3+, CUDA 12.4, PyG, MDAnalysis, E3NN, RDKit, PyRosetta)
- `score-env` (Python 3.11, Vina 1.2.5+, OpenMM 8.1+, scikit-learn, Meeko, ADFRsuite binaries on PATH)

The driver script (Python, in `score-env`) orchestrates both via
`subprocess` + `conda run -n rapidock-env ...`. Do not try to import
RAPiDock from `score-env`.

### 2.5 ref2015 cysteine issue

The C-terminal cysteine in LISDAELEAIFEADC triggers a Rosetta ref2015
RMSD alignment failure in RAPiDock's optional PyRosetta post-relax step.
Default behavior: **skip the PyRosetta relax step** and add a brief
OpenMM minimization before Vina scoring instead. Full workaround list
in §16.1 of the PDF.

### 2.6 Never commit secrets or licensed binaries

ADFRsuite and AutoDock4 binaries have non-redistributable licenses.
Link to the official download in `INSTALL.md`; do not bundle them in
the repo. iGEM's OSI-license requirement means we need MIT/Apache-2.0
code only — no copyleft dependencies in our own source.

---

## 3. Architecture at a glance

```
               ┌──────────────────────────────────────────┐
               │  Driver (Python, score-env)              │
               │  hybridock_pep.cli:main                  │
               └────┬───────────────────────────┬─────────┘
                    │                           │
            ┌───────▼────────┐         ┌────────▼─────────┐
            │ Stage 1: GPU   │         │ Stage 2: CPU     │
            │ (rapidock-env) │         │ (score-env)      │
            └───────┬────────┘         └────────┬─────────┘
                    │                           │
        subprocess: │                           │
        conda run rapidock-env                  │
        RAPiDock × N (default 100)              │
                    │                           │
                    ▼                           ▼
         poses/pose_{i}.pdb  ──────────▶ prepare_ligand
                                         vina --score_only          ─┐
                                         vina --scoring ad4          │
                                         entropy correction          │ per pose
                                         (optional: MM-GBSA top-K)   │
                                                                    ─┘
                                                 │
                                                 ▼
                          pairwise Cα RMSD  →  AgglomerativeClustering
                                                 │
                                                 ▼
                          ranked_poses.csv, cluster_summary.csv,
                          best_pose.pdb, convergence.png,
                          dendrogram.png, run_metadata.json
```

### Target repo layout

```
hybridock-pep/
├── CLAUDE.md                      # this file
├── README.md                      # user-facing install + quickstart
├── LICENSE                        # MIT
├── pyproject.toml                 # score-env package
├── docs/
│   ├── HybriDock-Pep_Technical_Specification.pdf   # the spec
│   ├── architecture.md
│   ├── benchmarking.md
│   └── tutorial.ipynb
├── envs/
│   ├── rapidock-env.yml
│   └── score-env.yml
├── src/hybridock_pep/
│   ├── __init__.py
│   ├── cli.py                     # argparse entry point
│   ├── driver.py                  # orchestrates two envs via subprocess
│   ├── prep/
│   │   ├── receptor.py            # prepare_receptor wrapper
│   │   ├── ligand.py              # prepare_ligand wrapper (per-pose)
│   │   └── grids.py               # autogrid4 for AD4 mode
│   ├── sampling/
│   │   ├── rapidock_runner.py     # subprocess wrapper, runs in rapidock-env
│   │   └── pose_io.py             # PDB parsing, pose validation
│   ├── scoring/
│   │   ├── vina.py                # vina --score_only wrapper
│   │   ├── ad4.py                 # vina --scoring ad4 wrapper
│   │   ├── entropy.py             # backbone entropy correction
│   │   └── mmgbsa.py              # OpenMM + GBn2 (optional)
│   ├── analysis/
│   │   ├── clustering.py          # pairwise Cα RMSD + agglomerative
│   │   ├── statistics.py          # ensemble stats, convergence
│   │   └── plotting.py            # matplotlib figures
│   └── output/
│       ├── csv_writer.py
│       └── metadata.py
├── scripts/
│   ├── calibrate_alpha.py         # fits entropy coefficient on training set
│   ├── benchmark.py               # runs full suite against 10 reference complexes
│   └── smoke_test.sh
├── tests/
│   ├── fixtures/                  # toy pdb + fasta files
│   ├── test_prep.py
│   ├── test_scoring.py
│   ├── test_clustering.py
│   └── test_e2e.py                # integration test on MDM2/p53
└── data/
    ├── training_complexes.csv     # PDB IDs + pK_d for α calibration
    └── test_complexes.csv
```

---

## 4. Development conventions

### Language / style
- **Python 3.11** for all in-repo code (score-env). 3.9 only for the
  RAPiDock subprocess.
- - **CRITICAL:** `src/hybridock_pep/sampling/run_rapidock.py` is executed by `rapidock-env` (Python 3.9). **Do not use Python 3.10+ syntax** (no `match`/`case`, no `X | Y` unions) in this specific file, or Stage 1 will crash with a SyntaxError. Keep it strictly 3.9 compatible.
- **Type hints everywhere**. `from __future__ import annotations` at
  the top of every module. mypy strict mode on CI.
- **Ruff** for linting, **black** for formatting. Line length 100.
- Docstrings in Google style, with at least `Args`, `Returns`, `Raises`.
- No bare `except:`. Catch specific exceptions. If you don't know what
  might be raised, wrap narrowly and reraise with context.

### Testing
- **pytest** with `pytest-cov`. Target ≥ 70% line coverage before
  merging anything to main.
- Fast unit tests in `tests/test_*.py`. Slow integration tests in
  `tests/test_e2e.py`, skipped by default, opt-in via `pytest -m slow`.
- **Integration test baseline**: MDM2/p53 (PDB 1YCR) — peptide
  `ETFSDLWKLLPE`, known K_d ≈ 0.6 µM. If the pipeline ever returns
  a corrected ΔG > −3 kcal/mol on this complex, something is broken.
  Receptor: `data/pdbs/1YCR_mdm2.pdb` (chain A only). Binding site
  center for docking: `--site 25.20 -25.61 -7.97 --box 30`.
  (Box was 20 Å; empirically 47% of GPU-diffusion poses had heavy atoms
  0.1–3.8 Å outside on this 12-mer — increased to 30 Å to contain full
  peptide extent. See RTX_DEBUG.md Fix I.)
- Fixture PDBs live in `tests/fixtures/`. Don't regenerate them on the
  fly; deterministic inputs matter.

### Reproducibility
- Every pipeline run logs: git SHA, RAPiDock commit SHA, all CLI args,
  random seeds, software versions (Vina, OpenMM, CUDA), receptor SHA256,
  peptide sequence hash, wallclock. Written to `run_metadata.json`.
- `--seed N` flag on the CLI. Setting it makes the run deterministic
  modulo CUDA nondeterminism (flag that in the JSON).

### CLI
- `argparse` with subcommands: `dock`, `calibrate`, `benchmark`, `prep`.
- Help strings on every flag, with units.
- Validate inputs *before* spawning subprocesses. Failing 30 minutes
  into a run because the FASTA had a non-amino-acid character is
  unacceptable UX.

### Logging
- `logging` module, not `print`. Level INFO by default, DEBUG on `-v`.
- Every subprocess call logs the full command (with PATH sanitized)
  before execution.

### Git hygiene
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- One logical change per commit. No "WIP" or "stuff" commits on main.
- Branch naming: `phase-N/short-description` matching the roadmap phase
  (see §13 of the spec).

---

## 5. Common commands

Once the package is installed in `score-env`:

```bash
# End-to-end docking run
hybridock-pep dock \
    --peptide LISDAELEAIFEADC \
    --receptor data/pdbs/1T2D_receptor.pdb \
    --site 31.9 17.5 9.5 \
    --box 20 \
    --n-samples 100 \
    --scoring vina,ad4 \
    --refine-topk 10 \
    --output-dir runs/pfldh_run1

# Calibrate the entropy coefficient α
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --pdbs-dir data/pdbs/ \
    --output calibration.json

# Run the benchmark suite
hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --baselines vina,adcp,rapidock \
    --report benchmark_report.md

# Dev-side: smoke test that the environments are wired up
bash scripts/smoke_test.sh
```

### Environment setup

```bash
# GPU inference environment (Blackwell-compatible)
conda env create -f envs/rapidock-env.yml
conda activate rapidock-env
pip install git+https://github.com/huifengzhao/RAPiDock.git@<pinned-sha>

# Scoring + analysis environment
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .
```

### Running tests

```bash
pytest                          # fast unit tests
pytest -m slow                  # include integration tests (~2 min)
pytest --cov=hybridock_pep      # with coverage
```

---

## 6. Tooling — claude-mem and graphify

Ram has **claude-mem** and **graphify** installed and wants them used where
they help. Treat them as first-class tools, not decorative add-ons.

### claude-mem

Use claude-mem to persist architecture decisions, open questions,
and session-to-session context. Specifically:

- **After any non-trivial design decision**, write a memory entry
  summarizing the decision and the reasoning. Example:
  *"Decided to skip PyRosetta relax step by default (re ref2015 cys
  alignment bug). OpenMM minimization used instead. See PDF §16.1."*
- **Before starting a new session**, query claude-mem for any
  relevant prior context — don't re-derive decisions already made.
- **Track open questions** as memory entries with a `status: open`
  tag so they surface next session.
- Don't dump entire files into claude-mem. Summaries and decisions only.
  If you need file content, read the file.

### graphify

Use graphify to visualize the module dependency graph when refactoring
or when orienting to an unfamiliar area of the code. Useful moments:

- Before a cross-cutting refactor, generate a graph to check nothing
  unexpected imports the module you're touching.
- When onboarding a new section of the code (e.g., you haven't touched
  `analysis/` yet this session), graphify it first.
- For documentation: graphify output can be embedded in `docs/architecture.md`
  to keep the architecture diagram honest.

Don't graphify on every tool call. It's a perception aid, not a reflex.

---

## 7. Before you do X — playbook

### Before writing a new scoring term
1. Re-read §4 and §5 of the PDF.
2. Ask: is this already covered by AD4 or MM-GBSA? If yes, don't write
   a new term — wire up the existing route.
3. If you genuinely need a new term, propose the math + calibration
   protocol in a memory entry before implementing.

### Before adding a new dependency
1. Is it available on conda-forge or PyPI with wheels for
   linux/macOS-arm64/macOS-x86_64? If no, bias toward not adding it.
2. Is it OSI-licensed? (iGEM requirement.) Copyleft is a no.
3. Does it duplicate functionality we already pull in via OpenMM /
   MDAnalysis / Biopython? If yes, use the existing one.

### Before touching the RAPiDock subprocess wrapper
1. Verify the CUDA/PyTorch combo against a `test_inference.py` smoke
   test. Blackwell compatibility is fragile.
2. Preserve seed propagation — it's the only handle we have on
   reproducibility.
3. Do not parallelize RAPiDock across GPUs speculatively. One GPU,
   sequential inference. If you want parallelism, fork the process,
   don't thread inside one.
4. **Absolute Paths Only:** When passing file paths or directories across the `conda run` boundary to RAPiDock, ALWAYS convert them to absolute paths using `str(Path(...).resolve())`. Conda's subprocess working directory behavior is unpredictable and will break relative paths.

### Before changing the entropy correction formula
1. Re-read §8 of the PDF.
2. Re-calibrate α on the training set (run `scripts/calibrate_alpha.py`).
3. Run the benchmark suite and compare Pearson r and RMSE before/after.
4. Commit the new calibration JSON alongside the code change.

### Before refactoring the CLI
1. Existing flag names are part of the public interface. Don't rename
   flags without a deprecation shim.
2. Run the benchmark suite end-to-end after CLI changes to catch
   arg-passing regressions.

### Before writing a big output file to disk
1. Is it in `runs/` or `tests/fixtures/`? Those are the only committable
   output locations.
2. Files > 1 MB don't go in git. Period.

---

## 8. What good looks like on this project

- **Benchmark suite** (10 complexes, §14 of PDF) with Pearson r ≥ 0.55
  on held-out test set, ≥ 0.10 better than Vina-alone.
- **Pose accuracy**: best-of-top-25 Cα RMSD ≤ 2.0 Å on ≥ 7 of 10
  benchmark complexes.
- **Runtime**: full 100-pose run ≤ 5 min wall-clock on RTX 5070 +
  modern CPU, not including optional MM-GBSA.
- **Cross-platform**: `hybridock-pep dock --help` works on Linux, macOS ARM (via Rosetta 2 for ADFRsuite), and WSL2. **Note:** macOS ARM is supported for Stage 2 (scoring) only. The CLI must support an `--input-poses` bypass flag so Mac users can run scoring on pre-generated poses. Full end-to-end runs require CUDA and must fail gracefully with a clear error on macOS.
- **One-command install**: `conda env create -f envs/score-env.yml &&
  pip install -e .` works with no manual intervention.
- **iGEM wiki**: Best Software Tool page documents the tool to the
  rubric in §15 of the PDF. Tutorial notebook runs top-to-bottom without
  errors on a fresh install.

---

## 9. What failure modes look like — flag and stop

If any of these happen, **stop and ask** rather than working around them:

- Benchmark Pearson r < 0.35 after full calibration.
- Cluster populations on real targets consistently < 10 of 100.
- RAPiDock's learned top-1 pose is systematically *better* than
  HybriDock-Pep's top cluster centroid — means the rescoring is
  destroying signal.
- Convergence curves still drifting at N=100 for a 15-mer.
- α calibrates to > 1.2 kcal/mol/residue or < 0.2 kcal/mol/residue.
- Vina and AD4 scores disagree in sign (one negative, one positive)
  on > 20% of poses.

None of these are normal. They mean something in the pipeline is wrong,
and patching around them will ship a broken tool.

---

## 10. Contact / context

- **Maintainer**: Ram, Head of Dry Lab, Denmark High School iGEM Team
  (class of 2028).
- **iGEM target**: 2026 competition, November Jamboree. Freeze repo
  ≥ 2 weeks before submission.
- **Style of feedback Ram wants**: direct, no sugarcoating. If code or
  design is wrong, say so plainly. Don't pad pushback with hedges.
- **Scope discipline**: tool is for the iGEM Best Software Tool award.
  Scope creep (turning this into a general-purpose protein–protein
  docker, adding a GUI, etc.) is out of bounds unless explicitly
  requested.

---

*Last updated: April 2026 · v0.1 · spec version matches
`docs/HybriDock-Pep_Technical_Specification.pdf` v0.1*
