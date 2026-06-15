# Can a Pre-Existing FEP/LIE Dataset Break the Receptor-Propensity Wall? — Searched, Answered

*2026-06-15 · Ram: "train on a pre-existing directory of FEP/LIE proteins that have this data." Searched the
literature + downloaded the candidate datasets. The answer is a decisive NO, for a fundamental physics reason
— with one genuinely useful (but narrow) data resource identified.*

## The physics that decides it
The hidden variable is **cross-receptor absolute binding propensity**. FEP/LIE are **relative, within-target**
methods: they compute ΔΔG between similar ligands / mutations on the SAME complex. They **cancel** the
receptor baseline (just like our selectivity), they do not predict it. So a FEP dataset structurally cannot
contain the cross-receptor signal we lack — **FEP has the same blind spot we do.**

## What the public FEP datasets actually are (confirmed)
| Dataset | Content | Useful for the wall? |
|---|---|---|
| schrodinger/protein-fep-benchmark | 208 **relative ΔΔG** protein-protein **mutations** | No — relative, mutations |
| Uni-FEP (~1000) | **small-molecule** relative FEP (ChEMBL congeneric) | No — small molecule, relative |
| Nat. Comm. Chem (50 targets/1200 lig) | small-molecule relative FEP | No — relative |
| Absolute-FEP toolkits (Felis-ABFE) | absolute, but tiny + unreliable + small-molecule | No — not peptide, not at scale |

**There is no public FEP "directory" of cross-receptor absolute peptide binding energies.** Absolute FEP at
scale doesn't exist publicly because it's expensive and unreliable — which is *why* the wall exists.

## The real resource found — PPB-Affinity (downloaded, assessed)
[PPB-Affinity](https://zenodo.org/doi/10.5281/zenodo.11070823) (Nature Sci Data 2024, CC-BY): **12,406
complexes, 2,667 receptors, all with structures + Kd + ΔG.** Largest structural affinity DB. BUT for *peptide*
scoring it is mostly the wrong type:
```
 SKEMPIv2.0   7374  protein-protein MUTATIONS (we already have SKEMPI)
 PDBbindCN    3119  (we already have the peptide subset = our 925)
 SAbDab       1152  antibody-antigen (large proteins, not peptides)
 ATLAS+TCR-pMHC ~1636  PEPTIDE-MHC — the ONLY genuinely-new peptide data, but ONE narrow receptor class (MHC)
 Affinity Bmk  160  protein-protein
```
Net new general peptide-receptor pairs ≈ the TCR-pMHC set (one receptor family). It is **not** a broad
peptide-receptor trove; it overlaps our existing SKEMPI + PDBbind heavily.

## Verdict
- **No FEP/LIE dataset breaks the receptor-propensity wall** — FEP is relative and cancels the baseline; the
  cross-receptor absolute signal isn't in any public computed dataset (and ESM proved it isn't in
  representation either, E224).
- **PPB-Affinity** is real and CC-BY, but ~85% protein-protein; its only new peptide content (TCR-pMHC) is a
  single receptor class. Worth ingesting *if* we target MHC; not a general fix.
- **The wall stands. The recovery is per-target data/calibration** (where we already beat PPI 0.69 vs 0.55),
  not a magic training set. This is consistent across every test (E221 structure, E224 ESM, this search).
