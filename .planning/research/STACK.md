# Stack Research: HybriDock-Pep

**Project:** HybriDock-Pep
**Researched:** 2026-04-18
**Overall confidence:** HIGH (core constraints), MEDIUM (version pins), LOW (Blackwell edge cases)

---

## Summary

HybriDock-Pep is a two-environment pipeline. The constraints already established in CLAUDE.md and
the technical spec are load-bearing — this document fills in the supporting library layer and
flags version-compatibility landmines, especially around the RTX 5070 (Blackwell, CC 12.0).

**Critical insight discovered during research:** RAPiDock's documented CUDA 12.4 compatibility
path targets H800/Hopper architecture, not Blackwell. PyTorch 2.3+ with CUDA 12.4 runs in
"emulation mode" on sm_120 — functional but with a performance penalty. Full native sm_120
support arrived in **PyTorch 2.7 / CUDA 12.8**. This is a meaningful upgrade from the spec's
"PyTorch 2.3+ / CUDA 12.4" baseline. Recommendation: target PyTorch 2.7 + CUDA 12.8 for
`rapidock-env` from the start.

**Second critical insight:** `fair-esm` (pinned at 2.0.0 in RAPiDock) is effectively abandoned
on PyPI and has no recent PyTorch 2.x compatibility testing. RAPiDock uses it for sequence
embeddings. This is the most likely import-time breakage point when upgrading. Test early.

---

## Recommended Stack

### Pose Generation Environment (`rapidock-env`)

Python 3.9 (hard requirement from RAPiDock's code; it uses typing syntax that breaks on 3.10+
without a migration).

| Package | Pinned Version | Purpose | Rationale |
|---------|---------------|---------|-----------|
| Python | 3.9.x | Runtime | RAPiDock hard requirement |
| pytorch | 2.7.0 | Deep learning runtime | First stable release with native Blackwell sm_120 via CUDA 12.8 |
| pytorch-cuda | 12.8 | CUDA toolkit | Required for sm_120; 12.4 works in emulation but is suboptimal |
| torch-geometric (PyG) | 2.6.x | Graph neural networks | RAPiDock's bi-scale graph architecture; cu128 wheels available for PyG |
| e3nn | 0.5.1 | Equivariant neural networks | RAPiDock's Clebsch-Gordan tensor products; 0.5.1 tested against PyTorch 2.x |
| MDAnalysis | 2.9.x | Protein structure I/O | Used by RAPiDock for pose handling; Python 3.9 compatible |
| biopython | 1.84 | PDB parsing | Pinned by RAPiDock; do not upgrade in this env |
| rdkit | 2024.03.x | Molecular featurization | Replace deprecated `rdkit-pypi`; use `rdkit` package from conda-forge |
| fair-esm | 2.0.0 | Sequence embeddings (ESM-2) | Pinned by RAPiDock; mostly abandoned upstream but still functional |
| pyrosetta | 2024.10+ | Optional relax step | Skip by default (ref2015 cys bug per §16.1); install but don't invoke |

**PyG installation note:** PyG cu128 wheels are community-maintained. If prebuilt binaries are
absent for torch-2.7.0+cu128, install from source:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```
Fall back to source build if the wheel URL 404s.

**PULCHRA:** Not a Python package. Pin to v3.04 exactly. Available via bioconda (`bioconda::pulchra`)
but bioconda ships 3.06 which has the aromatic side-chain bug. Build 3.04 from source and add to
PATH manually inside `rapidock-env`. See PITFALLS.md for the full rebuild procedure.

---

### Scoring and Analysis Environment (`score-env`)

Python 3.11 (all in-repo code lives here).

| Package | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| Python | 3.11.x | Runtime | Chosen for modern typing, long support window |
| vina | 1.2.7 | Vina Python bindings + CLI | Latest stable; 1.2.5+ required for `--scoring ad4`; 1.2.7 released Feb 2025 |
| meeko | 0.7.1 | PDBQT preparation | Forli lab canonical tool; handles ligand → PDBQT for Vina and AD4 |
| openmm | 8.2.0 | MM-GBSA, OpenMM minimization | Minimum for GBn2; 8.4.0 is current stable, use that |
| openmmforcefields | 0.15.1 | AMBER ff14SB + GBn2 XML | Ships AmberTools 24 parameters; required for `amber14-all.xml` |
| parmed | 4.x | AMBER topology manipulation | Required bridge between Biopython/MDAnalysis structures and OpenMM |
| MDAnalysis | 2.9.x | Cα RMSD computation, clustering input | Primary tool for pairwise RMSD matrix construction |
| biopython | 1.86 | PDB I/O, sequence validation | Current version; use `Bio.PDB` for receptor/peptide parsing |
| scikit-learn | 1.5.x | AgglomerativeClustering | Pass precomputed RMSD matrix as affinity; use `average` linkage (Ward requires Euclidean) |
| numpy | 2.1.x | Numerical core | Underpins everything; avoid 2.0.x which had breaking dtype changes |
| scipy | 1.15.x | Linkage matrix, dendrogram | `scipy.cluster.hierarchy` for dendrogram generation |
| matplotlib | 3.9.x | Convergence plot, dendrogram rendering | 3.9+ has improved layout engine |
| pandas | 2.2.x | CSV output, ranked_poses.csv | 2.x API is stable and well-supported |
| rich | 13.x | Progress bars, logging formatting | Optional but strongly recommended for UX; zero-cost dependency |
| pdbfixer | 1.9+ | Receptor PDB cleaning | Fills gaps, adds missing residues before ADFRsuite prep |

**ADFRsuite:** Not pip-installable. Non-redistributable license. Provides `prepare_receptor`,
`prepare_ligand`, `autogrid4` binaries. Must be on PATH inside `score-env`. Link to official
download in INSTALL.md: https://ccsb.scripps.edu/adfr/downloads/

**AutoDock4 binary:** Bundled inside ADFRsuite as `autogrid4`. Do not install separately; it will
conflict. The `vina --scoring ad4` flag calls the AD4 scoring functions that are compiled into
Vina itself — it does NOT call the standalone AutoDock4 binary. Grids are generated by `autogrid4`
from ADFRsuite.

---

### Development and Testing

All dev tooling runs in `score-env` (Python 3.11).

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| pytest | 8.x | Test runner | Standard; `pytest-cov` for coverage |
| pytest-cov | 6.x | Coverage reporting | Target ≥70% line coverage per CLAUDE.md |
| ruff | 0.8+ | Linting + formatting | Replaces black + flake8 + isort; configured in pyproject.toml |
| mypy | 1.13+ | Static type checking | Strict mode per CLAUDE.md; run on `src/hybridock_pep/` only |
| pre-commit | 3.x | Git hook runner | Runs ruff + mypy before commit |

**Note on black:** CLAUDE.md says "Ruff for linting, black for formatting." In 2025/2026 the
standard practice is to use Ruff's formatter (`ruff format`) instead of black — they are
style-compatible and Ruff is faster. Either works. If following CLAUDE.md literally, keep both;
if rationalizing, replace black with `ruff format`. The spec does not prohibit this.

---

## What NOT to Use

| Avoid | Why | What to Use Instead |
|-------|-----|-------------------|
| `rdkit-pypi` | Deprecated PyPI package, unmaintained; RAPiDock pins it but score-env should use `rdkit` | `rdkit` from conda-forge |
| PyTorch 2.3 + CUDA 12.4 on RTX 5070 | sm_120 support is emulation-only; degraded performance | PyTorch 2.7 + CUDA 12.8 |
| standalone AutoDock4 binary for scoring | `vina --scoring ad4` has AD4 functions built in; installing the binary separately causes PATH confusion | Use the AD4 scoring path built into Vina 1.2.7 |
| AmberTools (standalone) | Provides `antechamber`, `tleap`, etc.; none needed here; pulls in a huge dependency tree | Use `openmmforcefields` which ships converted AMBER params |
| GROMACS | Occasionally suggested for MM-GBSA pipelines; no benefit here, adds complexity | OpenMM + GBn2 as specced |
| ProDy | Feature-overlapping with MDAnalysis; adds weight for no gain in this pipeline | MDAnalysis throughout |
| PyRosetta relax step | ref2015 cysteine bug breaks on LISDAELEAIFEADC; skipped by default per §16.1 | OpenMM minimization (already in stack) |
| `fair-esm` upgrade to ESM3 | RAPiDock's architecture is trained against ESM-2 embeddings; switching models changes inference behaviour | Pin fair-esm 2.0.0 in rapidock-env |
| `conda install -c hcc adfr-suite` | HCC channel version is stale and may not include `autogrid4` required for AD4 mode | Official download from ccsb.scripps.edu |
| Ward linkage in AgglomerativeClustering | Ward requires Euclidean distances; RMSD matrices are not Euclidean in the strict sense | Use `average` linkage with precomputed metric |

---

## Version Compatibility Matrix (rapidock-env)

| Component | Original Pin | Recommended Pin | Risk if Kept at Original |
|-----------|-------------|----------------|--------------------------|
| CUDA | 11.5.1 | 12.8 | RTX 5070 will not run |
| PyTorch | 1.11.0 | 2.7.0 | RTX 5070 will not run |
| PyG | 2.1.0 | 2.6.x | May not install against PT 2.7 |
| e3nn | 0.5.1 | 0.5.1 (keep) | No change needed |
| MDAnalysis | 2.6.1 | 2.9.x | Minor API changes, negligible |
| biopython | 1.84 | 1.84 (keep) | No reason to upgrade in this env |
| fair-esm | 2.0.0 | 2.0.0 (keep) | Upgrading breaks ESM-2 embeddings |

---

## Installation Sketch

### rapidock-env
```bash
conda create -n rapidock-env python=3.9 \
    pytorch torchvision torchaudio pytorch-cuda=12.8 \
    -c pytorch -c nvidia
conda activate rapidock-env
# PyG: attempt prebuilt, fall back to source
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html || \
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv --no-index
pip install e3nn==0.5.1 fair-esm==2.0.0 MDAnalysis==2.9.* biopython==1.84
pip install git+https://github.com/huifengzhao/RAPiDock.git@<pinned-sha>
# PULCHRA 3.04: build from source, add to PATH (not conda)
```

### score-env
```bash
conda env create -f envs/score-env.yml
conda activate score-env
# ADFRsuite: manual download from ccsb.scripps.edu, add to PATH
pip install -e .
```

---

## Confidence Notes

| Area | Confidence | Basis |
|------|-----------|-------|
| RAPiDock GitHub deps | HIGH | Direct repository fetch |
| PyTorch 2.7 Blackwell support | HIGH | Official PyTorch release blog |
| Vina 1.2.7 on PyPI | HIGH | PyPI direct lookup |
| Meeko 0.7.1 | HIGH | PyPI + Forli lab docs |
| MDAnalysis 2.9.x Python 3.11 | HIGH | MDAnalysis release blog |
| OpenMM 8.4 on conda-forge | HIGH | conda-forge package page |
| openmmforcefields 0.15.1 | HIGH | GitHub releases + conda-forge |
| PyG CUDA 12.8 wheels available | MEDIUM | Community wheel page; verify at install time |
| e3nn 0.5.1 with PyTorch 2.7 | MEDIUM | Changelog shows PT 2.x compat but not 2.7 explicitly tested |
| fair-esm 2.0.0 with PyTorch 2.7 | LOW | Package is abandoned upstream; may need monkey-patching |
| PULCHRA 3.04 bioconda | LOW | Bioconda ships 3.06; 3.04 must be built from source |

---

## Sources

- [RAPiDock GitHub](https://github.com/huifengzhao/RAPiDock)
- [RAPiDock paper, Nat. Mach. Intell. 7:1308 (2025)](https://www.nature.com/articles/s42256-025-01077-9)
- [PyTorch 2.7 Release Notes (Blackwell sm_120)](https://pytorch.org/blog/pytorch-2-7/)
- [Vina 1.2.7 on PyPI](https://pypi.org/project/vina/)
- [Meeko on PyPI (0.7.1)](https://pypi.org/project/meeko/)
- [Meeko documentation](https://meeko.readthedocs.io/)
- [MDAnalysis 2.9.0 release](https://www.mdanalysis.org/2025/03/11/release-2.9.0/)
- [OpenMM on conda-forge](https://anaconda.org/conda-forge/openmm)
- [openmmforcefields GitHub](https://github.com/openmm/openmmforcefields)
- [PyG installation docs](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html)
- [e3nn GitHub](https://github.com/e3nn/e3nn)
- [fair-esm on PyPI](https://pypi.org/project/fair-esm/)
- [PULCHRA on Bioconda](https://anaconda.org/channels/bioconda/packages/pulchra/overview)
- [ADFRsuite downloads](https://ccsb.scripps.edu/adfr/downloads/)
- [scikit-learn 1.8.0 clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scientific Python SPEC 0 (version support)](https://scientific-python.org/specs/spec-0000/)
