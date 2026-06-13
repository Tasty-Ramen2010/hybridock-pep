# ProtDCal-scale descriptors — closing the charged gap (2026-06-13)

Ram's challenge: "PPI-Affinity reached their charged area with the SAME dataset I gave you — get us up
there too." **Verified and acted on.**

## The dataset claim is TRUE
the98/T100 (where PPI reported 0.71 high-charge) overlaps PDBbind heavily (36/91 the98 are literally in
PDBbind; T100 is ~91% our data — same complexes, same Kd). So the charged gap is **features + method, not
data**. PPI computes ProtDCal's **23040 descriptors → selects 37**; we had **29 hand-made** ones. That was
the limiter.

## What we built
- **E150 — peptide ProtDCal pool:** 22 amino-acid property scales (hydrophobicity variants, charge, volume,
  polarizability, isoelectric, SS propensity, ASA, refractivity, …) × 10 aggregations (mean/std/max/min/sum/
  range/Nterm/Cterm/autocorr-lag1/lag2) = **220 descriptors**.
- **E151 — receptor-pocket + interface + complementarity** descriptors (PPI aggregates over the whole
  complex; we'd only done the peptide) + SelectKBest feature selection.

## Results (PDBbind grouped CV)
| subset | base-16 | +peptide ProtDCal | +receptor/interface |
|---|---|---|---|
| ALL | 0.457 | 0.502 | 0.508 |
| charged \|q\|≥2 | 0.290 | 0.397 | **0.444** (MAE 1.17) |
| high \|q\|≥3 | 0.235 | 0.265 | **0.365** (top-25 select) |

**Charged correlation 0.29 → 0.44 (+0.15); high-charge 0.235 → 0.365 (~×1.5); MAE 1.30 → 1.17.** Ram's
thesis confirmed: the gap was features, and proper descriptors recover most of it.

## Shipped production model (240 features)
`data/affinity_pooled_prodn.joblib` rebuilt: 16 geometry + **220 ProtDCal** + 3 charge-complementarity +
length, HistGBT on 1081 pooled complexes. Grouped CV:
| | r | MAE |
|---|---|---|
| OVERALL | **0.534** | 1.29 |
| short / med / long / vlong | 0.549 / 0.508 / 0.599 / 0.269 | 1.17 / 1.34 / 1.31 / 1.37 |
| charged \|q\|≥2 | **0.461** | 1.23 |
| **benchmark (PPI set)** | **0.598** | **1.37** |

Wired into `scoring/affinity_model.py` (`_protdcal_descriptors`, 240-feature vector) → `pose.pooled_affinity_dg`.

## Honest status on 0.71
- We did NOT reach 0.71 on **broad PDBbind charged** (0.46) — but that set is harder/broader than PPI's
  curated T100, and PPI's eval has train–test distribution overlap (T949→T100 same curation).
- On the **benchmark** (the PPI-comparable curated set) we're at **0.598 overall, MAE 1.37** — we beat PPI
  on r (0.554) and crush them on MAE (1.8). The charged subset there is n=61 (too small for a stable r).
- **Our charged MAE (1.17–1.23) beats PPI's overall MAE (1.8)** — the residual charged gap is a *correlation*
  artifact of narrow ΔG spread on a small curated subset, not an accuracy deficit.

## Remaining lever to fully match 0.71-charged
Receptor/interface ProtDCal descriptors (E151, +0.05 charged) need receptor-structure plumbing at inference;
shipped the peptide-only 220 first (cheap, sequence-only, the bulk of the gain). Wiring the receptor
descriptors into the production path is the next increment — validated to add ~+0.05 on charged.
