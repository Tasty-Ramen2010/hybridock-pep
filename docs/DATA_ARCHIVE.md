# Data archive — everything too large for git

This repo keeps `data/` and `experiments/` small on purpose (nothing over ~1 MB is committed —
see `CLAUDE.md` §7). Anything bigger lives outside git. This page is the map: what's archived on
Zenodo, what you re-download yourself, and why some things are deliberately **not** redistributed.

## On Zenodo — our own derived data

**DOI: [10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573)**

| Archive | Size | Contents |
|---|---|---|
| `traj_cache.tar.gz` | 27 MB | `data/traj_cache/*.npz` — cached MD trajectories for the entropy/free-energy work (E363 and downstream). Unpack to `data/traj_cache/` at the repo root. |

This is a small, deliberately narrow bundle: only data we generated ourselves, with no
third-party redistribution restrictions.

## Not on Zenodo — re-download from the original source

These are excluded because they're either too large to usefully mirror, publicly available
already, or under a license that forbids redistribution:

| Directory (gitignored) | What it is | How to get it |
|---|---|---|
| `data/misato` (~124 GB) | Full copy of the public **MISATO** MD dataset | Download directly from [misato-dataset.github.io](https://misato-dataset.github.io/) — no point mirroring a dataset that's already public and this large |
| `data/drive_pull` (~20 GB) | PDBbind general-set mirror (`P-L/<year-range>/...` layout) | **Do not redistribute.** PDBbind's license prohibits third-party redistribution — register at [pdbbind.org.cn](http://www.pdbbind.org.cn/) and download it yourself. This is the same restriction already noted in the README's Datasets section. |
| `datasets/pdb_2024_2026/`, `pdb_2019_2023/`, `pdb_2010_2018/`, `pdb_pre2010/`, `family_targeted/`, `ppii_enriched/`, `ppii_extended/` | RCSB PDB structures used in the RAPiDock fine-tuning / benchmark campaigns | Regenerate with `python scripts/download_from_manifests.py` — reads the committed `datasets/*/manifest.csv` files and re-pulls each structure from RCSB (`files.rcsb.org`) |
| `datasets/training_formatted_peppc/` (~28 GB) | Formatted PepPC training data | Not redistributed pending confirmation of PepPC's own license terms. Rebuild from `PepPC_raw_data.tar.gz` / `PepPC-F_raw_data.tar.gz` (kept locally, not in git) using the ingestion scripts in `experiments/`. |
| `datasets/frag_raw_data_final/`, `nat_raw_data_final/` (~1.5 GB) | Processed PDB loop/helix fragment structures | Source database license not yet confirmed — do not redistribute until checked. |

## Scripts

Almost all scripts that produce or consume the above are in this git repo. **One known exception:**
`e158_overfit_failure_analysis.py` was never committed, and 48 experiment scripts import it for
`greedy_cluster()` and `pocket_seq()` — including the chain that regenerates
`data/e180_protdcal3d.jsonl` (`e180_protdcal_925.py` → `e179_protdcal_3d.py` → `e158`). That makes
the ours-vs-PPI-clone head-to-head unreproducible from a clean clone; see the note in the README's
[Datasets](../README.md#datasets--download-and-test-for-yourself) section. Searched for and not
found on the Zenodo record, the DGX, or the RTX PRO 6000 box.

Research/experiment scripts: [`experiments/`](../experiments/) (E0–E37x, cited
throughout [RESULTS.md](../RESULTS.md)) — this stays in git; it's the reproducibility trail,
not the research ledger below. Operational tooling: [`scripts/`](../scripts/).

## Research ledger — moved to Zenodo only, not in this git tree

As of the `v0.1.0` release, the live repo shows the completed tool, not the full research
process. Three categories moved out of git and now live **only** in the
**v0.1.0 source snapshot**, [10.5281/zenodo.21764713](https://doi.org/10.5281/zenodo.21764713)
— the Zenodo–GitHub release archive of commit `af86f00`, the last commit before the cleanup.
It contains all 88 `docs/*.md` including `DEVELOPMENT_TIMELINE.md`.

> Note the two records are different things and are not interchangeable:
> **21764713** is the source snapshot holding the research ledger; **21680573** (above) is
> the MD trajectory cache and nothing else. Only *this* version of the release carries the
> ledger — later releases are built from the cleaned tree.

| Moved | Count | What it was |
|---|---|---|
| `docs/*` dated notes + `DEVELOPMENT_TIMELINE.md` | ~82 files | Brainstorms, forensics write-ups, refuted approaches, per-milestone verdicts — the prose research journal. |
| `scripts/*` one-off campaign scripts | 59 files | Uncited helpers from specific investigations (RAPiDock fine-tune campaign, one-off benchmark builders) — not referenced by `README.md`/`RESULTS.md`/`MODEL_CARD.md`/`tests/`, and not imported by anything in `experiments/` or the scripts that remain. |
| `data/*` one-off outputs | 47 files | Intermediate CSVs/JSONs from specific experiments, superseded fine-tune training splits, and 7 non-shipped ablation `.joblib` variants no longer cited by name. |

None of this touches `experiments/` or the specific `scripts/`/`data/*` files that
`README.md`, `RESULTS.md`, `MODEL_CARD.md`, or `tests/test_repro_claims.py` cite as
reproducible evidence — those stayed, and a fresh clone still reproduces every number in
`RESULTS.md`. Only the surrounding prose notes and dead-end scripts moved.

## Code archive (not the data archive)

The code itself (everything in this git repo) gets a separate, automatic DOI via the
Zenodo–GitHub integration on every tagged Release — see the badge at the top of the README. That
covers `src/`, `experiments/`, `scripts/`, `tests/`, and all committed `data/*` files. This page is
only about the large files that never make it into git in the first place.
