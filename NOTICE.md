# Notice and third-party attribution

HybriDock-Pep's own source code is MIT-licensed ([LICENSE](LICENSE)). This file records the
third-party material the repository **ships** or **requires**, and the terms it comes under.
Data sources are in [DATA_LICENSES.md](DATA_LICENSES.md); the reasoning behind individual
licence decisions is in [docs/licenses.txt](docs/licenses.txt).

"Checked" means the licence was read from the project's own `LICENSE` file, its Zenodo record, or
the metadata of the exact package version that installs on Python 3.11, on **2026-10-19**.
Nothing here is legal advice; re-check before relying on it, especially for commercial use.

## Shipped in this repository

| Component | Where | Licence | Credit | Checked |
|---|---|---|---|---|
| RAPiDock-Reloaded (git submodule) | `third_party/RAPiDock` | MIT | A cross-platform fork of RAPiDock — Zhao *et al.*, *Nat. Mach. Intell.* **7**:1308 (2025); upstream <https://github.com/huifengzhao/RAPiDock>, Copyright (c) 2024 huifengzhao | Yes (see *Known gaps*) |
| Modified RAPiDock sources | `third_party/RAPiDock_finetuned/` (`inference.py`, `models/diffusion.py`, `train_lastlayer.py`, `utils/transform.py`) and `third_party/RAPiDock_v2/models/diffusion_v2.py` | MIT | Derived from RAPiDock, Copyright (c) 2024 huifengzhao. The upstream notice is reproduced [below](#upstream-mit-notice) because these copies ship without their own `LICENSE`. | Yes |
| RAPiDock checkpoints | `weights/rapidock_local.pt`, `weights/rapidock_global.pt` | CC BY 4.0 | Zhao, H. *RAPiDock*, Zenodo, doi:[10.5281/zenodo.14193621](https://doi.org/10.5281/zenodo.14193621). Unmodified; hashes in [`weights/SHA256SUMS`](weights/SHA256SUMS) and [`ARTIFACTS.sha256`](ARTIFACTS.sha256). | Yes (Zenodo record) |
| Long-peptide checkpoint | `weights/longer_local.pt` | Derived from the CC BY 4.0 checkpoints above | Our fine-tune of the RAPiDock local checkpoint. The attribution above applies and the changes are stated in [`weights/README.md`](weights/README.md). | Yes |
| ProtDCal-style descriptors | `third_party/protdcal/` (`protdcal_spec.py`, `generate_features.py`, `protdcal_aa_table.csv`) | Our reimplementation | Reconstructed from the supplementary tables of the ProtDCal paper (*BMC Bioinformatics*, 2015). The paper's PDF is **not** redistributed. | Not checked |
| Trained scoring models | `src/hybridock_pep/data/*.joblib` | MIT (our work) | Trained on the datasets in [DATA_LICENSES.md](DATA_LICENSES.md); see [MODEL_CARD.md](MODEL_CARD.md). | n/a |
| Example structures | `data/pdbs/`, `src/hybridock_pep/data/pdbs/`, `datasets/raw_pdbs/` | Coordinates from the RCSB PDB (see [DATA_LICENSES.md](DATA_LICENSES.md)) | Some files are our own prepared or relaxed models of PDB entries. | See DATA_LICENSES.md |

## Python dependencies (installed, never bundled)

Licences below are those of the versions that resolve on Python 3.11 from `pyproject.toml`.
Nothing in this table is copied into the repository or modified.

| Package | Licence | Package | Licence |
|---|---|---|---|
| numpy | BSD-3-Clause (plus bundled permissive parts) | scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause | joblib | BSD-3-Clause |
| pydantic | MIT | matplotlib | PSF-based |
| rdkit | BSD-3-Clause | biopython | Biopython License Agreement (BSD-style) |
| vina | Apache-2.0 | pdbfixer | MIT |
| openmm | PSF-style / BSD-like, as declared on PyPI | psutil | BSD-3-Clause |
| **meeko** | **LGPL-2.1** | **gemmi** | **MPL-2.0** |
| prompt_toolkit | BSD-3-Clause | pytest, pytest-cov | MIT |

`meeko` (LGPL-2.1) and `gemmi` (MPL-2.0) are weak-copyleft, OSI-approved licences. HybriDock-Pep
imports them as separately installed packages and does not copy, modify or statically link them,
so the MIT licence of this repository is unaffected. The same reasoning is recorded for MDAnalysis
(GPL-2.0) in [docs/licenses.txt](docs/licenses.txt): it lives in the separate RAPiDock conda
environment and is reached only across a subprocess boundary.

## Not distributed — you install these yourself

| Tool | Why it is not here |
|---|---|
| ADFRsuite (`prepare_receptor4`, `autogrid4`) and AutoDock4 | Distributed by its authors under their own terms; see [INSTALL.md](INSTALL.md) |
| PULCHRA | Obtain from its authors; see [INSTALL.md](INSTALL.md) |
| ESM-2 weights | Downloaded at run time by RAPiDock from their publisher; not checked here |
| PDBbind, MISATO, PepPC and other large datasets | See [DATA_LICENSES.md](DATA_LICENSES.md) |

## Known gaps

- **Upstream notice in the RAPiDock-Reloaded fork.** The `LICENSE` of the fork (the
  `third_party/RAPiDock` submodule, a separate repository) names only its own 2026 copyright
  holder. MIT requires the upstream notice to be kept in copies of the work, so that file should
  also carry `Copyright (c) 2024 huifengzhao`. It is reproduced below so that this repository is
  not missing it in the meantime.
- **ProtDCal reimplementation.** The terms of the supplementary tables it was reconstructed from
  have not been checked.

## Upstream MIT notice

The RAPiDock-derived files under `third_party/RAPiDock_finetuned/` and `third_party/RAPiDock_v2/`
are used under the following licence:

```text
MIT License

Copyright (c) 2024 huifengzhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
