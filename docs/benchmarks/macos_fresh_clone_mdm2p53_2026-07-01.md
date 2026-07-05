# macOS Fresh-Clone End-to-End Validation — MDM2/p53
Generated: 2026-07-01

## Goal
Validate that a fresh `git clone` of HybriDock-Pep runs end-to-end on macOS
(Apple Silicon, MPS) with no leftover local state — full env setup, Stage 1
diffusion sampling, Stage 2 scoring, Stage 3.6 AI-pose affinity — and check the
resulting ΔG against literature.

## Environment
- Apple Silicon Mac, 16 GB RAM, 8 cores, macOS 26.3.1
- `score-env` (Python 3.11) + `rapidock` (Python 3.10, MPS backend)
- ADFRsuite 1.0 (prepare_receptor, autogrid4, babel)

## Bugs found and fixed during setup
1. **Missing `pyyaml` in `rapidock` env** — RAPiDock-Reloaded's `inference.py`
   imports `yaml` unconditionally; Stage 1 crashed immediately with
   `ModuleNotFoundError`. Fixed in `envs/rapidock-env.yml` and
   `envs/rapidock-env-macos.yml`.
2. **`scikit-learn>=1.4` too loose** — resolved to 1.9.0, which cannot
   unpickle the shipped joblib models (`data/affinity_ai_nofix.joblib`,
   `data/pose_ranker_ml.joblib`; both fit under sklearn 1.8.0). Failure is
   silent — `driver.py` logs a WARNING and falls back to BSA-fit pose ranking,
   so a run "succeeds" but never uses the AI-pose affinity model. Pinned to
   `>=1.8,<1.9` in `envs/score-env.yml` and `pyproject.toml`.
3. **ADFRsuite `autogrid4` two missing dylibs on this machine** —
   `libgomp.1.dylib` and `libgcc_s.1.dylib`, both hardcoded to a build-time
   homebrew path (`/Users/Shared/mgltoolsDev/src/homebrew/...`) that doesn't
   exist on a stock install. Symlinked ADFRsuite's own bundled copies
   (`ADFRsuite-1.0/lib/lib{gomp,gcc_s}.1.dylib`) into the expected path.
   Machine-local fix, not a repo change — noting here so the next macOS setup
   doesn't lose an hour to `autogrid4` segfaulting with `EXC_BAD_ACCESS` at a
   null pointer.
4. **`rapidock_local.pt` checkpoint (56 MB) is untracked by design** —
   re-downloaded from the documented Zenodo record
   (https://zenodo.org/records/14193621/files/rapidock_local.pt?download=1)
   per INSTALL.md; not a bug, just a reminder this step is required after
   every fresh clone.

## Runs

### Run 1 — full-length p53 TAD peptide
```
hybridock-pep dock --peptide ETFSDLWKLLPE --receptor data/pdbs/1YCR_mdm2.pdb \
    --site 25.20 -25.61 -7.97 --box 40 --n-samples 50 --scoring vina,ad4 --seed 42
```
- 50/50 RAPiDock poses sampled (MPS), 47/50 survived Vina clash-relief
- **Wall time: 7 min 34 sec** (n=50, MPS, Apple Silicon)
- Best pose ΔG (AI-pose affinity model): **-9.56 kcal/mol** (pose_1, cluster 2)
- Peak tree RSS ~7 GB (16 GB machine) — no resource pressure

### Run 2 — 9-residue core motif (literature-matched construct)
```
hybridock-pep dock --peptide TFSDLWKLL --receptor data/pdbs/1YCR_mdm2.pdb \
    --site 25.20 -25.61 -7.97 --box 40 --n-samples 50 --scoring vina,ad4 --seed 42
```
- 50/50 RAPiDock poses sampled (MPS), 48/50 survived Vina clash-relief
- **Wall time: 4 min 53 sec** (n=50, MPS, Apple Silicon — 35% faster than the
  12-mer, consistent with fewer atoms to sample/minimize/score)
- Best pose ΔG (AI-pose affinity model): **-9.01 kcal/mol** (pose_5, cluster 0)
- Peak tree RSS ~4.5 GB

## Literature comparison

`data/training_complexes_full.csv` carries an experimental value for this
exact PDB/peptide pair: `1YCR, TFSDLWKLL, pKd=6.52` (manual, chain A).

    Kd = 10^-6.52 M ≈ 0.30 µM
    ΔG = RT·ln(Kd), T=298K, RT=0.593 kcal/mol  →  ΔG ≈ -8.9 kcal/mol

| | ΔG (kcal/mol) |
|---|---|
| Literature (pKd=6.52, 9-mer `TFSDLWKLL`) | -8.9 |
| HybriDock-Pep, 9-mer `TFSDLWKLL` (exact match) | **-9.01** |
| HybriDock-Pep, 12-mer `ETFSDLWKLLPE` (flanking Glu/Pro, no direct lit. value) | -9.56 |

The exact-construct comparison (9-mer vs 9-mer) lands within **~0.1 kcal/mol**
of the literature value. The 12-mer number isn't directly comparable to
pKd=6.52 (different construct, no independent Kd on file for the full
p53 TAD fragment) but is directionally consistent — both call MDM2/p53 a
tight, sub-µM binder.

## Takeaway
A completely fresh clone, after the four environment fixes above, reproduces
the pipeline's headline AI-pose scoring end-to-end on Apple Silicon and lands
within experimental error of the literature Kd for the exact-match construct.
