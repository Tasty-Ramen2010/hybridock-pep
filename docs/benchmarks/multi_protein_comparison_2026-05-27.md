# Multi-Protein Scoring Comparison Report
**Date:** 2026-05-27 | **Pipeline:** HybriDock-Pep v1.0 (RAPiDock-Reloaded + Vina/AD4)

---

## 1. Crystal-Pose Vina Scoring (54-Complex Kd Training Set)

Crystal-pose Vina scores measure the scoring function's ability to rank binding
affinity for pre-docked (near-native) conformations. This is an upper bound on
what HDP's diffusion-generated poses can achieve.

### PepSet-6 (Production Calibration Set, r=0.860 on hybrid)

| PDB  | Peptide         | Len | pKd  | Exp ΔG (kcal/mol) | Vina (kcal/mol) | AD4 (kcal/mol) | NC |
|------|-----------------|-----|------|-------------------|-----------------|-----------------|----|
| 2hwn | EELAWKIAKMIVSDVMQQC | 19 | 8.70 | −5.15 | −14.75 | — | — |
| 1nrl | SLTERHKILHRLLQE | 15 | 6.00 | −3.55 | −12.41 | −11.75 | 16 |
| 1l2z | SHRPPPPGHRV | 11 | 5.70 | −3.37 | −7.03 | −8.73 | 7 |
| 1ddv | TPPSPF | 6 | 5.00 | −2.96 | −8.77 | −5.94 | 5 |
| 1a0n | PPRPLPVAPGSSKT | 14 | 4.60 | −2.72 | −5.66 | −6.37 | 7 |
| 1ywi | PPPLPP | 6 | 4.10 | −2.43 | −6.45 | — | — |

**PepSet-6 Vina Pearson r:** −0.68 (anti-correlated; corrected by AD4 + entropy → r=0.860 on hybrid)

### Full 54-Complex Kd Set (Crystal Poses)

**Pearson r (Vina vs exp ΔG):** −0.381 (p=0.005)  
This anti-correlation is expected: Vina systematically overestimates binding for
peptides (longer peptides have more contacts = larger |Vina|, but affinity
doesn't scale linearly with length). The entropy correction (alpha × n_contact)
compensates.

**Representative extremes:**

| PDB  | Peptide (len) | pKd  | Exp ΔG | Vina | AD4 | Note |
|------|---------------|------|--------|------|-----|------|
| 6GTZ | ADVTITVNGKVVA (13) | 5.64 | −3.34 | **−34.74** | −29.70 | Size confound |
| 3VQH | TTYADFIASGRTGRRNAIHD (20) | 7.64 | −4.52 | **−27.53** | −22.82 | Large peptide |
| 1YDT | TTYADFIASGRTGRRNAIHD (20) | 7.64 | −4.52 | −22.66 | −22.10 | Same complex |
| 1T65 | HKILHRL (7) | 9.60 | −5.68 | −4.30 | −3.43 | Short, weak Vina |
| 1LCK | EGQYQPQPA (9) | 3.40 | −2.01 | −12.78 | −8.08 | Low affinity, high Vina |

---

## 2. MDM2/p53 (1YCR) — HDP vs DiffPepDock vs Crystal Pose

| Method | Sequence | Vina (kcal/mol) | AD4 (kcal/mol) | Hybrid (kcal/mol) | Note |
|--------|----------|----------------|----------------|-------------------|------|
| Crystal pose | TFSDLWKLL (9-mer) | **−19.62** | −15.19 | — | PDB crystal |
| HDP (old, GPU) | ETFSDLWKLLPE (12-mer) | −12.42 | — | −3.39 | Pre-migration |
| **HDP (new, CPU)** | ETFSDLWKLLPE (12-mer) | **−10.79** | −6.96 | −9.70 | RAPiDock-Reloaded, 31/100 clean |
| DiffPepDock + PyRosetta | SQETFSDLWKLPEN (14-mer) | −3.85 | — | — | FastRelax + Vina optimize |
| Experimental | — | — | — | — | ΔG ≈ −8.2 kcal/mol (K_d 0.6 µM) |

**Key finding:** HDP top-1 is within 2.6 kcal/mol of experiment. DiffPepDock+PyRosetta is 4.4 off.

---

## 3. PfLDH Target (1T2D) — Prior Run

| Run | Peptide | Vina top-1 | Note |
|-----|---------|-----------|------|
| runs/pfldh_1t2d | LISDAELEAIFEADC | — | Check existing run |

---

## 4. Vina Score Distribution by Peptide Length (54 complexes)

| Length range | N | Mean Vina | Std | Mean exp ΔG |
|--------------|---|-----------|-----|-------------|
| 4–8 | 11 | −9.0 | 2.3 | −3.9 |
| 9–12 | 21 | −12.4 | 3.2 | −4.3 |
| 13–16 | 14 | −13.6 | 4.1 | −4.4 |
| 17–20 | 6 | −21.2 | 6.8 | −4.6 |
| 21–30 | 2 | −13.0 | 0.9 | −3.5 |

Length confound is clear: Vina scales with peptide size but exp ΔG does not.
This is the primary motivation for the contact-residue entropy correction.

---

## 5. Known Limitations

1. **Crystal-pose scoring ≠ docking**: The 54-complex analysis uses crystal poses
   prepared from PDB structures. HDP's diffusion-generated poses will always score
   somewhat worse than crystal poses.

2. **RAPiDock-Reloaded CPU mode**: 51/100 poses outside grid, 18/100 clash-relief
   failures. GPU mode (RTX 5070) expected to yield ~100/100.

3. **AD4 anomalies**: 2/31 poses have AD4 > +100k kcal/mol. Source: unknown atom
   type or HOH/cofactor in AD4 grid after clash optimization moves atoms into
   non-parameterized regions.

4. **DiffPepDock**: Uses crystal-structure-derived backbone. Side-chain packing
   requires receptor-aware tool (PyRosetta). Not a fair de novo docking comparison.
