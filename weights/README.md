# RAPiDock model checkpoints (vendored)

The Stage 1 diffusion checkpoints live here, in the repository, rather than
being downloaded at install time.

| File | Size | Used by |
|---|---|---|
| `rapidock_local.pt`  | 54 MB | every site-directed dock (`--site`), and every per-pocket refinement pass |
| `rapidock_global.pt` | 54 MB | only the exploratory whole-receptor pass of `dock --blind` (`sampling/pocket_search.py`) |
| `longer_local.pt` | 58 MB | peptides of 13+ residues (`select_checkpoint()`); falls back to `rapidock_local.pt` if removed |

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

`rapidock_local.pt` and `rapidock_global.pt` are unmodified upstream weights.

### `longer_local.pt` (this project's own fine-tune)

A derivative of the CC-BY-4.0 `rapidock_local.pt` (Zhao et al., *Nat. Mach. Intell.* 7:1308, 2025), redistributed under
the same CC-BY-4.0 terms with that attribution. It is the **epoch-10 checkpoint of the `longft_tanh` run** (inference
weights only: model and EMA tensors, optimizer state dropped, tensors bit-identical to the training checkpoint).
Recipe: start from `rapidock_local.pt`, train on crystal-pose targets of 13-25 residue complexes at lr 1e-5 (cosine),
batch-norm statistics and the magnitude-gate layers frozen, Tanh torsion heads.

Measured against the unmodified `rapidock_local.pt` pushed through the same code path, on held-out complexes of 25
residues or fewer (best-of-24 samples):

| Set | n | <=5 A | <=2 A | Paired result |
|---|---|---|---|---|
| 13-25 aa (epoch 10) | 57 | 24 -> 35 | 0 -> 5 | median 5.28 -> 4.60 A, sign test p = 0.0002 |
| <=12 aa (epoch 20, same run) | 69 | 62 -> 64 | 25 -> 33 | no regression, not significant on its own |

Caveats: one training run, one checkpoint; the metric is best-of-24 oracle RMSD, not a ranked top-1; the benchmark was
not homology-filtered against the training set; validation loss did not predict the gain. Peptides longer than 25
residues are out of scope and were not trained on. Verify with `cd weights && sha256sum -c SHA256SUMS.longer`.

```
58f9cbe6822c0be62be28c4a230424f1f87650b309577f3423686d2cc6c645f9  longer_local.pt
```
