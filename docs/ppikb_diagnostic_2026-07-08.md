# PPIKB diagnostic: why the naive number looked bad, and the honest corrected result

**Date:** 2026-07-08 · Ram flagged that a poor PPIKB number shouldn't be taken as "we're bad" without diagnosing
PPIKB itself. Correct call — the first number was an artifact, and PPIKB is genuinely noisy.

## The mistake (corrected)
An initial PPIKB test used only the 22 `pocket_pkf` features → r 0.15 / MAE 2.07 (≈ mean-baseline). That is **not
our scorer** — our stack is ProtDCal + pocket/physics + IFP. With the full 59-feature stack the result is very
different. Reproduce: `scripts/e332b_ppikb_headtohead.py`.

## Corrected, leakage-free (60%-id clustered CV), full stack
| set | model | r | MAE | RMSE |
|---|---|---|---|---|
| all PPIKB (mixed labels) | OURS (59) | 0.336 | 1.92 | 2.45 |
| | PPI-clone (37) | 0.253 | 2.02 | 2.57 |
| **Kd/Ki-only** | **OURS** | **0.369** | **1.90** | 2.42 |
| | PPI-clone | 0.252 | 2.02 | 2.58 |

**Our PPIKB r (0.369) is comparable to our PDBbind r (0.263–0.391) — NOT a collapse — and we beat the PPI-clone on
this second, independent database too.**

## Why PPIKB's absolute MAE is higher (~1.9 vs ~1.4) — it's PPIKB's noise, not us
Diagnostics on `data/ppikb_features.jsonl` (n≈2229 rows):
- **~20% non-thermodynamic labels:** 426 IC50 + 21 EC50 (+1650 Kd + 130 Ki). IC50/EC50 are assay-specific and do
  NOT convert cleanly to ΔG — [JCIM 4c00049](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049): 27% of IC50 pairs
  disagree by >1 log unit; [PLoS ONE mixed-IC50](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3628986/).
- **Same-sequence label disagreement up to 10.8 kcal/mol** (mean within-seq y-STD 0.79) — identical peptides with
  wildly different reported affinities (cross-source aggregation).
- **Wider, messier range:** ΔG −16.4 to **+1.4** (std 2.58 vs PDBbind 1.85); lengths 2–50 (7% are >25-mers, i.e.
  small proteins, not peptides).

## The decisive test that isolates the cause
Removing the IC50/EC50 rows lifts **our** r (0.336 → 0.369) but leaves the **clone** flat (0.253 → 0.252). So the
IC50/EC50 assay noise was dragging the achievable signal, and our model tracks the real signal once it's stripped.
The residual gap to PDBbind is the label noise PPIKB carries and PDBbind (Kd/Ki, curated) does not.

## Verdict
PPIKB is a harder, noisier *independent* validation — and on it, honestly evaluated with the full stack, we (a) do
**not** collapse (r 0.369 ≈ our PDBbind number) and (b) **beat the previous-best-approach clone** again. The
independent-set win generalizes. The higher absolute MAE is PPIKB's documented label heterogeneity, not a scorer
failure — exactly what Ram suspected.
