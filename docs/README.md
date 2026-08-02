# docs/ — what to read, and what to ignore

This directory holds ~80 files. **Most of them are a dated research ledger, not
documentation.** They were written as the work happened and are kept because
every negative result stays on the record — but they are not a place to start,
and several describe approaches that were later refuted.

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

## The research ledger

| File | What it is |
|---|---|
| [`DEVELOPMENT_TIMELINE.md`](DEVELOPMENT_TIMELINE.md) | The complete honest development record, E0 → E304. **This is the index to everything else here.** |

Every other file in this directory is a dated working note from a single
investigation — `*_2026-06-15.md`, `overnight_*.md`, `*_brainstorm.md`,
`why_we_keep_failing_*.md`, and so on. They are individually accurate as of
their date and collectively out of date. Read them only when
`DEVELOPMENT_TIMELINE.md` points you at one, or when you want to check that a
specific claim was arrived at honestly.

A dated note is a snapshot of what was believed that day. If it disagrees with
`METHODS.md` or `MODEL_CARD.md`, those win.
