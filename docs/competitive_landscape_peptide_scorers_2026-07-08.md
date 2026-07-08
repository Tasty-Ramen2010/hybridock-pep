# Competitive landscape: peptide scorers and where we stand

**Date:** 2026-07-08 · Compiled from literature. Goal: name every peptide scorer + its reported score, normalize
by *regime* (the numbers are NOT apples-to-apples), and place us honestly.

## The critical caveat: there is NO clean common "cross-target absolute peptide Kd" leaderboard
The field is fragmented across **four different tasks** that get called "peptide scoring." A number only means
something with its regime attached:
1. **Pose/docking accuracy** — % of poses within RMSD cutoff (NOT affinity).
2. **Affinity regression** — Pearson/Spearman vs experimental Kd/ΔG (our task) — but usually *their-split* or *same-target*.
3. **Binder classification** — AUC (binder vs non-binder), e.g. TDC.
4. **pMHC** — Spearman on immunopeptidomics (a specialized niche).

## A) Affinity predictors (our comparison class)
| tool | reported score | regime / caveat |
|---|---|---|
| **PPI-Affinity** (SVM web tool) | R=**0.63** (HTRA1, in-domain); R=**0.02** (HTRA3, out-of-domain); ~0.55 their SKEMPI split | **wildly target-dependent** — 0.02→0.63 by target. Same-target/in-domain only |
| protein-LM ranker (bioRxiv 2024) | ranking (Spearman), various | protein-peptide, sequence-based |
| **ESMCBA** (pMHC) | Spearman **0.62** across HLA | pMHC niche; beats NetMHCpan 0.56, MHCflurry 0.49 |
| general peptide/ligand, low-identity | **Pearson 0.165–0.550**, Spearman 0.152–0.553 | **cross-target / 30% seq-id — the honest hard regime** |
| BindPred / ProBASS / DeepProBind | "strong" (no clean cross-target r published) | sequence-based, absolute logKd |
| FoldX / Rosetta (ref2015) | physics baselines; ~10 kcal systematic bias on peptides (our tests) | need per-system calibration |
| **OURS (fast scorer)** | **r≈0.36–0.53** leakage-free (0.38 grouped, up to 0.63 on some subsets) | absolute, cross-target, leakage-free |

## B) Docking / pose scorers (different task — pose RMSD, not affinity)
| tool | success rate | benchmark |
|---|---|---|
| **ADCP** (AutoDock CrankPep) | 85.7% top-10 | LEADS-PEP |
| **HPEPDOCK** | 66.7% top-10 (drops to 35% for ≥9-mers) | peptiDB |
| **FlexPepDock** (Rosetta) | 17% | LEADS-PEP |
| **HPEPDOCK2.0** (cyclic) | 44% top-1 | cyclic set |
| **OURS (RAPiDock Stage 1)** | HDP 0.80 Å vs DiffPepDock 3.54 Å on 1YCR (our bench) | our benchmarks |

## C) The honest placement — where we actually stand
1. **Our absolute cross-target r (0.36–0.53) sits squarely INSIDE the field's honest cross-target range (0.15–0.55).**
   We are **not behind** — we're mid-field for the hard regime.
2. **The "competitors beat us" impression is a regime artifact.** PPI-Affinity's 0.63 is *in-domain, same-target* —
   it collapses to **0.02** out-of-domain. ESMCBA's 0.62 is *pMHC*. Nobody posts a robust cross-target absolute
   peptide-Kd number much above ~0.55, because it doesn't exist (the wall we characterized).
3. **We are NOT "world's best"** (that claim is false and unprovable) — but we are a legitimate mid-field,
   *reference-free, fast, open-source* entrant with an **honest leakage-free evaluation** (most competitors report
   leaky/their-split numbers, per the CASF-CleanSplit critique).
4. **Our genuine edge is orthogonal to this table:** speed, reference-free operation, the **selectivity primitive**
   (ΔΔG across two receptors — few tools do this), and honest uncertainty flagging.

## D) Benchmarks we can test on (to get a directly-comparable number)
| benchmark | task | access | note |
|---|---|---|---|
| **TDC ProteinPeptideGroup** | binder classification (AUC) | `from tdc.benchmark_group... ProteinPeptideGroup` | standardized, but AUC not regression |
| **PDBbind peptide subset** | affinity regression (1,433, 5–50 aa, −logKd) | pdbbind.org.cn | what we already use (~925) |
| **PepBenchmark** (2026) | 29 datasets, leaderboard | arXiv 2604.10531 | newest standardized peptide ML benchmark; PLM models lead |
| **LEADS-PEP / peptiDB** | pose RMSD | published | for docking (RAPiDock) comparison |

## E) Concrete plan to nail "where we stand"
1. **Head-to-head vs PPI-Affinity (our closest peer, has a web server):** run PPI-Affinity on OUR leakage-free test
   set, and our scorer on THEIR set — the cleanest apples-to-apples affinity comparison.
2. **Run on PepBenchmark / TDC** for a standardized leaderboard number (even if AUC/regression differ).
3. **Report the leakage-free number prominently** — it's our integrity edge; most published numbers are leaky.
4. **Lead with selectivity + speed + reference-free**, not absolute r, since absolute is capped for everyone.

---
### Sources
PPI-Affinity: [JPR 2c00020](https://pubs.acs.org/doi/10.1021/acs.jproteome.2c00020), [PubMed 35654412](https://pubmed.ncbi.nlm.nih.gov/35654412/).
Protein-LM ranker: [bioRxiv 2024.11.14.623613](https://www.biorxiv.org/content/10.1101/2024.11.14.623613v1). ESMCBA/pMHC: [arXiv 2507.13077](https://arxiv.org/html/2507.13077).
Docking: [ADCP Bioinformatics](https://academic.oup.com/bioinformatics/article/35/24/5121/5510553), [peptide-protein modelling review PMC10392694](https://pmc.ncbi.nlm.nih.gov/articles/PMC10392694/).
Benchmarks: [TDC ProteinPeptide](https://tdcommons.ai/benchmark/proteinpeptide_group/overview/), [PepBenchmark arXiv 2604.10531](https://arxiv.org/abs/2604.10531), [PDBbind](http://pdbbind.org.cn/).
Cross-target range: [ML affinity review arXiv 2410.00709](https://arxiv.org/html/2410.00709v2).
