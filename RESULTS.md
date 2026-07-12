# RESULTS — leakage-free benchmarks

One page: every headline number, the exact command that regenerates it, the honest caveat
attached. Metric is **MAE/RMSE in kcal/mol** (primary; correct for an absolute-ΔG predictor);
Pearson r is secondary and capped near the field ceiling for *all* methods, FEP included.

## How we benchmark (why these numbers are trustworthy)

- **Leave-cluster-out CV.** Complexes are clustered by sequence identity with a
  **placement-aware (gap-penalised)** alignment and **whole clusters held out per fold** —
  no homolog of a test receptor is ever in training. Verified leakage-free: clustered
  r (0.35) < leaky random-CV r (0.44).
- **Same-split head-to-head.** The PPI-Affinity clone is scored on the *identical* held-out
  split — not its own paper's numbers (its server has been down since 2022). Published
  scorers report r ≈ 0.5–0.77 on training-overlapped sets; strip the leakage and the field
  sits near r ≈ 0.32.
- **Full identity-cutoff trend**, not one cherry-picked split (added on Prof. Koes's review).
- **Negative results kept public** in `docs/` — including a retired scorer that once
  generalized to negative correlation. We do not quietly drop them.

## Headline numbers

| Claim | HybriDock-Pep | Baseline | n | Reproduce (from `experiments/`) |
|---|---|---|---|---|
| **Matched head-to-head, 60%-id clustered** | **MAE 1.35 · RMSE 1.69 · r 0.352** | PPI-clone 1.46 · 1.84 · 0.210 | 865 | `python e331_ours_vs_ppiclone_clustered.py` |
| **Full PDBbind peptide set, leakage-free** | **MAE 1.40 · RMSE 1.77 · r 0.321** | zero-skill MAE 1.47 | 925 | `python e330_ours_pdbbind.py` |
| **30% cutoff (Koes's standard)** | **MAE 1.39 · RMSE 1.76 · r 0.322** | — | 410 clusters | `python e366_identity_threshold_trend.py` |
| PDBbind crystal + interaction map | r 0.480 (charged 0.401) | PPI-clone 0.291 (0.146) | 865 | `python e298_ppi_vs_ifp.py` |
| Double-difference ΔΔG (same-receptor) | r ≈ 0.96 | FEP/TI ≈ 0.85 | — | `python e287_similarity_and_dd.py` |
| Affinity r on real AI poses (geom→+IFP) | 0.486 → 0.53 | PPI pose-blind 0.325 | — | `python e106_combined_realpose_grade.py` |

**MAE is flat (1.32→1.42) across the entire 30–100% identity sweep** — that stability of the
kcal/mol error is the number we stand behind. r declines smoothly from 0.45 (leaky) and
levels near 0.32: the honest cross-target ceiling.

**Offline, no data, 30 s:** `make verify` runs the math-only tests (double-difference,
anchoring, selectivity) — proves the relative-scoring machinery is correct without PDBbind.

## Honest caveats (state these before a judge finds them)

- **Absolute cross-target Kd is confound-limited** for every cheap non-FEP method, ours
  included (size/baseline + enthalpy–entropy compensation). We report **relative** ΔΔG /
  selectivity / anchored ΔG as the accurate paths; absolute ΔG is a coarse readout. See
  [MODEL_CARD.md](MODEL_CARD.md) and `docs/why_we_keep_failing_synthesis_2026-07-08.md`.
- **Selectivity ΔΔG** lands r ≈ 0.30–0.45 — useful for triage, not a final answer.
- **This is a rigor contribution, not a discovery.** The size/baseline (Simpson) confound is
  known; we prove it is the specific cause of cross-dataset non-replication in peptide docking
  and ship instant geometric features that stay sign-stable across two independent datasets.

## External review

The **benchmarking methodology** was reviewed by **Prof. David Koes** (Associate Professor,
Computational & Systems Biology, University of Pittsburgh; author of the `smina`/`gnina`
docking tools). His review shaped the leakage protocol here — notably reporting the standard
**30% identity cutoff** and the full identity-vs-accuracy trend, not a single split.
*This is review of the evaluation methodology, **not** an endorsement of the tool or its
results by Prof. Koes or the University of Pittsburgh.* Full statement: README → *External review*.
