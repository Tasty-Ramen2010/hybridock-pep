# docs/ — what to read, and what to ignore

This directory holds the documentation for the shipped tool. The dated research ledger
(~80 working notes written as the scoring campaign happened — brainstorms, forensics,
refuted approaches, `DEVELOPMENT_TIMELINE.md`) has been moved off the live git tree and
lives only in the Zenodo archive: [10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573)
(see [`DATA_ARCHIVE.md`](DATA_ARCHIVE.md)). `experiments/` (the reproducibility trail cited
by `RESULTS.md`/`MODEL_CARD.md`/tests) stays in git — only the prose notes moved.

If you are trying to *understand or use* HybriDock-Pep, you want six files.

## Start here

| File | What it is |
|---|---|
| [`../METHODS.md`](../METHODS.md) | The method in a few minutes: pipeline, what the reported number is, how it was validated, limits. |
| [`../README.md`](../README.md) | Install, commands, the claims and the evidence behind them. |
| [`../MODEL_CARD.md`](../MODEL_CARD.md) | Which trained artifacts ship, what each is for, and what not to quote. |
| [`../RESULTS.md`](../RESULTS.md) | Every leakage-free number and the command that reproduces it. |

## Reference

| File | What it is |
|---|---|
| [`architecture.md`](architecture.md) | Module-by-module design: what each stage does and where it lives in `src/`. |
| [`RUNBOOK.md`](RUNBOOK.md) | Walkthrough of a complete end-to-end docking run on real inputs. |
| [`SCORING_COMPARISON.md`](SCORING_COMPARISON.md) | Where this sits in the hierarchy of affinity methods (docking → ML → MM-GBSA → FEP). |
| [`DATA_ARCHIVE.md`](DATA_ARCHIVE.md) | Datasets and training sets too large for git, archived on Zenodo. |
| [`calibration_notes.md`](calibration_notes.md) | The current calibration and how to refit it on your own measurements. |

## The research ledger — Zenodo only

`DEVELOPMENT_TIMELINE.md` (the complete honest development record, E0 → E304) and every
dated working note (`*_2026-06-15.md`, `overnight_*.md`, `*_brainstorm.md`,
`why_we_keep_failing_*.md`, etc.) are preserved in the `v0.1.0` Zenodo snapshot —
[10.5281/zenodo.21680573](https://doi.org/10.5281/zenodo.21680573) — not in this directory.
They were individually accurate as of their date and collectively out of date; read them
only to check that a specific claim was arrived at honestly.

A dated note is a snapshot of what was believed that day. If it disagrees with
`METHODS.md` or `MODEL_CARD.md`, those win.
