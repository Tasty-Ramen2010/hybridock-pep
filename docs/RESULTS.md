# HybriDock-Pep — current results

All numbers measured on the Tanh-restored tree (the SiLU regression of 18-21 Sep is fixed and
guarded by `scripts/exp_runner.py::assert_torsion_nonlinearity`). Last updated 29 Sep 2026.

## 1. Similarity-stratified benchmark — 247 real peptide complexes

The defensible headline. Built to be leakage-clean: every candidate outside the training pools,
interface-quality filtered, PDB-id filtered against every pool, one complex per 90%-identity
MMseqs cluster, spread across all nine binding-site-identity bins.

| arm | median | best-24 ≤5 Å | best-24 ≤2 Å | top-1 ≤5 Å |
|---|---|---|---|---|
| **HybriDock-Pep** | **3.51 Å** | **71.5%** | **22.4%** | **57.3%** |
| RAPiDock (base) | 3.63 Å | 68.7% | 16.3% | 52.4% |
| Boltz-2 *(training-set leaked)* | 5.36 Å | 48.4% | 23.0% | 48.4% |
| HDP full-corpus (unfiltered) | 4.41 Å | 61.0% | 8.5% | 35.8% |
| ADCP | 6.74 Å | 32.4% | 12.2% | 18.5% |
| ESMFold *(leakage-clean)* | 23.38 Å | 10.5% | 3.6% | 10.5% |

We lead RAPiDock in every identity bin. A further 158 PPI-interface fragments (`peppc`) are
excluded and reported separately: they are windows excised from larger partner chains, not
peptide complexes, and every arm scores far worse on them.

## 2. Length-balanced benchmark — 387 complexes, best-of-24

| model | median | ≤2 Å | ≤5 Å |
|---|---|---|---|
| HybriDock-Pep | 2.01 Å | 190/385 | 348/385 |
| HDP quality-filtered (ep3/ep5) | 2.01 Å | 192/385 | 346/385 |
| RAPiDock (base) | 2.21 Å | 165/385 | 341/385 |
| HDP full-corpus (unfiltered) | 3.03 Å | 67/385 | 317/385 |

Paired vs RAPiDock: better on 236/385, Wilcoxon p < 0.0001. Short is a tie (61%/61%); the gain
is medium (56% vs 44%) and long (55% vs 45% at ≤5 Å).

**Read with the caveat in §6.** 60% of this bench has a >90%-identity training homolog.

## 3. Blind docking — 120 full receptors, no binding site given

| | rank-1 median | rank-1 ≤5 Å | best-24 median | best-24 ≤5 Å |
|---|---|---|---|---|
| **HybriDock-Pep global_v2** | **4.81 Å** | **61/120** | **2.39 Å** | **102/120** |
| RAPiDock baseline | 6.50 Å | 45/120 | 3.89 Å | 80/120 |

Paired best-of-24: better on 101/120, p < 0.0001.

## 4. Pose ranking

Top-1 ≤5 Å, 387 complexes. Every candidate is blind; RMSD only scores the pick afterwards.

| ranker | short | med | long | macro |
|---|---|---|---|---|
| random (no ranker) | 83.1% | 71.8% | 13.3% | 56.1% |
| pose_ranker_ml (shipped) | 79.7% | 72.2% | 16.7% | 56.2% |
| **consensus** | **91.5%** | **85.6%** | **36.7%** | **71.3%** |
| oracle (ceiling) | 97.5% | 95.7% | 55.0% | 82.7% |

`pose_ranker_ml` is indistinguishable from random. Consensus is +15 points and the same gain
appears on RAPiDock's poses (62.8% → 77.3%), so it is not a quirk of our checkpoint. ADCP's own
affinity ranker reaches 26.7% macro.

## 5. Stage-1 rescoring

Top-1 Hit@2 Å within ref2015's top 5, 57 complexes: ref2015 **10.5%**, BSA+clash **10.5%**
(equal, but not the same poses — identical top-1 on 16/57), MM-GBSA 15.8%, oracle 21.1%.

## 6. Leakage, stated plainly

The length-balanced bench is **not** homology-free: 60% of its targets have a training receptor
above 90% aligned-region identity, and 27 share a PDB entry outright with `longft_train`.
Stratified by binding-site identity (interface residues only — whole-chain identity is 91%
scaffold, since the pocket crop is a median 185 residues of which 17 touch the peptide):

| binding-site identity | n | ours | RAPiDock | delta |
|---|---|---|---|---|
| >90% | 194 | 72.7% | 56.7% | **+16.0** |
| 70-90% | 54 | 64.8% | 59.3% | +5.6 |
| 50-70% | 34 | 41.2% | 47.1% | −5.9 |
| 30-50% | 23 | 17.4% | 26.1% | −8.7 |
| <30% | 82 | 34.1% | 34.1% | **+0.0** |

Removing the 27 hard leaks moves the headline only slightly (+7.8 → +7.5, still p < 0.0001).
The §1 benchmark exists because of this, and is the one to cite.

## 7. Negative and null results

- **Capacity is not the limit.** ns=48 (7.56 M params) final train loss 0.5142; ns=96 (47.46 M)
  0.5230. 6.3× the parameters buys nothing.
- **Clash penalty is null.** λ=0 vs λ=0.05 at matched epoch 20: 63/120, p = 0.62, minimum
  detectable effect 0.46 Å.
- **Sampling is not saturated.** N=24 → N=100 on 58 long peptides: +10.3 points for us
  (56.9% → 67.2%), +13.8 for RAPiDock. At N=100 we win 38/58, p = 0.001, and 6/58 vs 0/58 ≤2 Å.
- **More data alone hurts; filtered data does not.** The unfiltered 17,232-complex corpus scores
  3.03 Å; the same corpus minus the bottom 20% of interface burial per length class (13,875)
  reaches 2.01 Å at its best epoch, matching hand curation on 3× the data.
- **Designed binders remain hard.** Coventry 14 cognate pairs at N=100: ours 5.93 Å median,
  5/14 ≤5 Å, 0/14 ≤2 Å (RAPiDock 6.65 Å, 3/14).

## 8. Model size, for context

HybriDock-Pep and RAPiDock are **7.56 M parameters**. ESMFold is 3.53 B (466×). Of the complexes
we miss, Boltz-2 — far larger and trained on these exact structures — misses 44-65% of the same
ones, which bounds how much of the residual is reachable by scale.
