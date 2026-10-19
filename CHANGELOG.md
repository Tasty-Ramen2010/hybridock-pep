# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are commit dates.
Only two versions have been tagged; everything since `v0.2.0` is listed under *Unreleased* and
`pyproject.toml` still says `0.2.0`. For exact history, use `git log v0.2.0..master`.

## [Unreleased]

### Added
- **Browser UI** for `hybridock-pep serve`: a run survives a page reload or a dropped connection,
  Home shows recent predictions with reopen and download, run errors are explained in plain language
  with the raw message under "Technical details", and a guide with real screenshots. (Oct 6–12)
- **Static demo site** on GitHub Pages, built from the same UI. (Oct 8)
- **`serve --public`**, a locked-down mode for sharing a computer with a hosted page: requests are
  rebuilt from an allow-list, no file paths cross the wire, runs and uploads are capped and
  rate-limited. See [SECURITY.md](SECURITY.md) and [docs/SHARING.md](docs/SHARING.md). (Oct 12)
- **`weights/longer_local.pt`**, an epoch-10 fine-tune for peptides of 13+ residues, selected
  automatically by length; a *Models* line in the status panel. (Oct 11)
- **Pocket search for `dock --blind`**, with a length-routed checkpoint. (Aug 11)
- **Colab**: the full pipeline on a free T4 and a shell path in the notebook; setup prewarms
  RAPiDock tables and prefetches ESM-2. (Aug 21, Oct 17)
- `--lite` and a measured no-GPU timing; Metal (Apple GPU) detection and time estimates. (Aug 21, Oct 10)
- **RAPiDock checkpoints vendored** in `weights/`, so install no longer fetches them from Zenodo. (Sep 8)
- Project paperwork: [NOTICE.md](NOTICE.md), [DATA_LICENSES.md](DATA_LICENSES.md),
  [CITATION.cff](CITATION.cff), [codemeta.json](codemeta.json), [CONTRIBUTING.md](CONTRIBUTING.md),
  [SECURITY.md](SECURITY.md), [ARTIFACTS.sha256](ARTIFACTS.sha256) with
  `scripts/verify_artifacts.py` (checked by a test, so CI enforces it), and `.gitlab-ci.yml`.

### Changed
- Default demo peptide is now `LISAAALAAIFAAALAC`, against a rebuilt 1T2D tetramer receptor with
  the real active-site coordinates. (Aug 10–11)
- Vina scoring runs in a memory-aware worker (34% faster), and Stage 1 caps batch size by free VRAM
  and no longer fragments the CUDA allocator. (Oct 8)

### Fixed
- Peptides of 13+ residues failed because the shipped model code could not load `longer_local.pt`.
  (Oct 12)
- `dock` works from any directory; calibration and rank models now ship inside the package. (Oct 9)
- Install: prebuilt `vina` where it exists, pinned CPU wheels for CPU-only Linux/Windows, PyG import
  verified after install. (Oct 8)
- Windows: web bundle served as JavaScript, source hashing across line endings, several CI failures.
  (Aug 7, Sep 23, Oct 8)
- Memory: an OpenMM `Context` was leaked per pose on Colab; an MPS leak and a stale batch-cap formula
  caused out-of-memory at `n=100`. (Aug 6, Sep 9)
- Blind docking: correct `TER` records and a single `END` in cropped receptors. (Aug 14)
- Docs: the `r = 0.96` double-difference row in RESULTS.md was **retracted** — it used three measured
  values to estimate a fourth and does not measure the scorer. (Sep 23)

## [0.2.0] — 2026-08-02

### Fixed
- **Packaging:** a wheel/PyPI install was broken because data files were unreachable; `rdkit`,
  `matplotlib` and `gemmi` were used but never declared as dependencies.
- The benchmark was unseeded and therefore not reproducible.

### Added
- Cross-platform CI (Linux x86-64 and ARM64, macOS ARM64, Windows).
- `dock --blind`, an install path with a sidecar environment for tools the scoring environment
  cannot hold, and a resumable runner for the ProP-PD benchmark.

### Changed
- Shipped model artifacts moved into `src/hybridock_pep/data/`.
- The research ledger moved off GitHub to the Zenodo source snapshot
  ([10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713)).

## [0.1.0] — 2026-08-02

First tagged release, archived on Zenodo as the source snapshot
[10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713).

[Unreleased]: https://github.com/Tasty-Ramen2010/hybridock-pep/compare/v0.2.0...master
[0.2.0]: https://github.com/Tasty-Ramen2010/hybridock-pep/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Tasty-Ramen2010/hybridock-pep/releases/tag/v0.1.0
