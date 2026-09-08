# RAPiDock model checkpoints (vendored)

The two Stage 1 diffusion checkpoints live here, in the repository, rather than
being downloaded at install time.

| File | Size | Used by |
|---|---|---|
| `rapidock_local.pt`  | 54 MB | every site-directed dock (`--site`), and every per-pocket refinement pass |
| `rapidock_global.pt` | 54 MB | only the exploratory whole-receptor pass of `dock --blind` (`sampling/pocket_search.py`) |

## Why they are committed

They used to be fetched from Zenodo record
[14193621](https://doi.org/10.5281/zenodo.14193621) by `install.sh` and
`scripts/colab_setup.sh`. That download is the single most fragile step of a
first install: it is the one thing that has to reach a host outside GitHub and
PyPI, and it is blocked outright on some school and institutional networks —
which turns a working Colab notebook into a hard failure at step 4.

Both files are 54 MB, comfortably under GitHub's 100 MB per-file limit, so a
`git clone` now carries them. There is no checkpoint download step any more:
`install.sh` and `scripts/colab_setup.sh` link these into
`third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/` (hard link
where the filesystem allows it, copy otherwise), and fall back to Zenodo only if
this directory is somehow missing them.

`sampling/rapidock_runner.py` carries the same fallback at runtime, so a plain
`git clone --recursive` works even if neither setup script was run.

## Provenance and license

Trained and published by the RAPiDock authors — Zhao et al., *Nature Machine
Intelligence* **7**:1308 (2025) — and deposited at
[10.5281/zenodo.14193621](https://doi.org/10.5281/zenodo.14193621) under
**CC-BY-4.0**, which permits redistribution with attribution. These files are
byte-identical to that record; the checksums below are the same ones the setup
scripts have always verified against.

```
d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693  rapidock_local.pt
a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5  rapidock_global.pt
```

Verify with `cd weights && sha256sum -c SHA256SUMS`.

They are unmodified upstream weights — none of the fine-tuning campaigns in this
project's history produced a checkpoint that beat them, so none shipped. The
long-peptide checkpoint `longer_local.pt` referenced by `select_checkpoint()`
does not exist as a published artifact; peptides ≥ 13 residues fall back to
`rapidock_local.pt` with a warning, which is expected behaviour.
