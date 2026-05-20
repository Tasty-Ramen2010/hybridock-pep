# RAPiDock Dataset Analysis

*Generated from `datasets/` — PDB structures committed to machine only, not to repo.*

---

## 1. Dataset Overview

Three dataset collections reside under `datasets/`:

| Directory | Entries | Role |
|-----------|---------|------|
| `pepset/` | 185 complexes | Primary benchmark (PepSet, Wen et al. 2020) |
| `RefPepDB-RecentSet/` | 523 complexes | Recent holdout set (PDB entries ≥ 2020) |
| `cases/` | 4 protein families | Deep-dive case studies with extra metadata |

Each entry provides: receptor pocket PDB, peptide PDB, and plain-text peptide sequence.
`pepset/` additionally supplies unbound receptor structures and a `benchmark.csv` with
difficulty labels and apo-structure PDB codes.

---

## 2. PepSet Benchmark (185 complexes)

### 2.1 Difficulty distribution

| Difficulty | Count | Avg length | Length range |
|------------|-------|------------|-------------|
| Easy | 132 (71%) | 9.5 residues | 5–20 |
| Medium | 28 (15%) | 12.7 residues | 8–19 |
| Difficult | 25 (14%) | 15.3 residues | 11–20 |

Difficulty correlates with peptide length and conformational flexibility, not
directly with protein family.

### 2.2 Protein family breakdown

| Family | Approx. count | Representative PDBs | Dominant peptide character |
|--------|--------------|--------------------|-----------------------------|
| **PDZ domain** | ~33 | 1jq8, 1jd5, 2fgr, 2fka, 2hpl, 4r5i | Short C-terminal motifs, 5–8 residues, [ST]x[VLI] tail |
| **SH3 domain** | ~13 | 1a0n, 1cqg, 1ddv, 1eg4, 1l2z, 2jkg | Proline-rich PXXP, poly-Pro type II helix |
| **WW domain** | ~6 | 1ywi, 1ywo, 1mv0, 2r7g, 3v2o | PPXY / PPXP proline-rich, 5–12 residues |
| **SH2 domain** | ~8 | 1jw6, 2aq9, 2qos, 2fym, 4gq6 | pTyr-surrogate, acidic/aromatic core |
| **Calmodulin / EF-hand** | ~9 | 1f47, 1j2x, 1nrl, 3bej, 3et3 | Amphipathic IQ/1-8-14 helix, bulky hydrophobics |
| **Bromodomain / histone reader** | ~7 | 2ke1, 3shb, 4lk9, 4ouc, 4qbr | H3/H4 tail with Ac-Lys; ARTKQTA(RK) motif |
| **Kinase / signalling** | ~15 | 2khh, 2e4h, 5wfv, 6b27, 4m5s | Phosphorylation-site peptides, Ser/Thr-rich |
| **Amphipathic helix binders** | ~15 | 1yfn, 4txr, 5v2p, 6dei, 3plu | 14–19 residues, helical, charged/hydrophobic patterning |
| **ARM / HEAT repeat** | ~7 | 2cny, 1vpp, 2hwn, 5fml | Long curved-surface binders, 14–20 residues |
| **BCL-2 family / BH3** | ~5 | 2vzg, 5d94, 2koh, 4iga | BH3 α-helix, LXXXDE/DXXXL motif |
| **MDM2 / MDMX** | ~2 | 2nnu, 1pmx | Trp/Phe/Leu triad in hydrophobic cleft |
| **Protease substrate** | ~4 | 4c5a (TEV), 5fv6 (thrombin) | Canonical recognition sequences |
| **Hsp90 / TPR domain** | ~1 | 3fp2 | MEEVD C-terminal pentapeptide |
| **Collagen receptor** | ~1 | 4bt9 | (PPG)_n collagen triple-helix fragment |
| **Neuropeptide** | ~1 | 5e33 | YGGFM methionine-enkephalin |
| **Amyloid / aggregation** | ~1 | 4mvi | Aβ(16–28) KLVFFAEDVGSNK fragment |
| **Miscellaneous** | ~37 | various | Mixed/unclassified modular domains |

---

## 3. RefPepDB-RecentSet (523 complexes)

Structures deposited in PDB 2020–2023 (PDB IDs 7xxx–8xxx). All entries supply protein
pocket + peptide PDB only (no unbound receptor or difficulty label).

### 3.1 Length statistics

| Bin | Count | % |
|-----|-------|---|
| Short (≤ 8 residues) | 143 | 27% |
| Medium (9–14 residues) | 312 | 60% |
| Long (≥ 15 residues) | 68 | 13% |
| Overall average | — | 10.3 residues |

### 3.2 Dominant protein families

**MHC class I / pHLA (~45% of set)**
The single largest group. Classical 8–10 residue T-cell epitopes anchored in the
peptide-binding groove of HLA-A and HLA-B alleles. Highly represented because of
the post-2020 surge in MHC structural studies. Pocket geometry is largely fixed by
the MHC α1/α2 domains; residue identity matters more than backbone conformation.

**Calmodulin-binding peptides (~7%, 36 entries)**
Two distinct IQ-motif subfamilies:
- CaMKII-like: 25 entries sharing the KILHRLL core (e.g., 7aos, 7b88, 7bk4)
- CaMKI/IV-like: 11 entries with GLEAIIR core (e.g., 8aqm, 8b8w–8b95)

**Collagen-receptor peptides (~3%, ~15 entries)**
GPP/GXY-repeat collagen triple-helix fragments binding fibronectin, collagen-
binding bacterial adhesins, etc. (7bdu PGPPGPPGPRGLPGPPGPPG, 7bee, 7bfi).

**Proline-rich / SH3/WW (~2%, ~12 entries)**
Canonical PXXP ligands and WW domain substrates.

**NLS / nuclear-import basic peptides (~3%, ~17 entries)**
Short Arg/Lys-dense sequences recognising importin ARM repeats (7bcy, 7d6r, 7ea1).

**Histone / chromatin readers (~1%, ~3 entries)**
H3 tail peptides (7d87 ARTKQTARK) for PHD, chromodomain, or bromodomain partners.

**HIV V3 loop / immunology (~0.4%, 2 entries)**
GPGR-containing V3 loop peptides (7dng, 7urf) binding antibody or co-receptor surrogates.

**Amyloid-related (~0.2%, ~2 entries)**
KLVFF-containing Aβ fragments (7y3j, similar).

---

## 4. Special Case Studies (`cases/`)

### 4.1 pHLA — MHC class I

8 HLA alleles represented: HLA-A\*02:01, A\*24:02, A\*29:02, B\*07:02, B\*08:01,
B\*15:01, B\*35:01, B\*53:01.

Each allele has a fixed receptor pocket PDB. The accompanying
`pHLA_binding_affinity.csv` provides binary binding labels (0 = non-binder, 1 = binder)
for hundreds of 9-mer peptides per allele, enabling classification benchmarking.
Pocket geometry is dominated by anchor-residue pockets (P2 and P9); mid-sequence
residues bulge out and contribute minimally to binding energy.

### 4.2 PLK1 Polo-box Domain (PBD)

31 structures, all containing phospho-modified peptides with non-standard residues:
- **[TPO]** = phospho-threonine (most cases)
- **[SEP]** = phospho-serine (3P35, 3Q1I)

Consensus polo-box recognition motif: `(S/T)-pS/pT-P`.
Peptides 4–9 residues. Covers the full range of known PLK1 mitotic substrate sequences
(Cdc25C, Wee1, PBIP1, etc.).

### 4.3 SHP2 N-SH2 Domain

9 structures with phospho-tyrosine [PTR] peptides (4–12 residues).
Canonical SH2 pY+1/pY+3 specificity determinants present. Representative
sequences: `LN[PTR]AQLW` (3TL0), `V[PTR]ENVGLM` (4JMG), `ASPEPI[PTR]ATIDFD` (5X94).

### 4.4 Importin-α

Single structure 7M60 with bipartite NLS peptide **ANPRKRHR**.
Two binding modes resolved: major-pocket and minor-pocket conformations.

---

## 5. Where RAPiDock Performs Best

The following assessment is based on (a) the PepSet difficulty labels,
(b) the length and conformational character of each family, and (c) general
properties of SE(3)-equivariant diffusion models trained on PDB complexes.

### 5.1 Strong performance expected

| Family | Reason |
|--------|--------|
| **PDZ domain** | Deep, well-defined groove; canonical C-terminal 5–8 mer motifs with clear geometrical constraints; most represented family in training data |
| **SH2 domain** | Structured electropositive pocket; pY-recognition geometry is stereotyped; short peptides (6–12 residues) |
| **Bromodomain / histone reader** | Deep Kac-binding hydrophobic cavity; short H3/H4 tail peptides with a single pharmacophoric anchor residue |
| **MDM2 / MDMX** | Narrow hydrophobic cleft; three-anchor (Trp/Phe/Leu) pharmacophore makes diffusion sampling convergent |
| **BCL-2 family / BH3** | Well-defined α-helical binding groove; BH3 helix forms reproducibly in the 9–16 residue range |
| **Calmodulin / EF-hand** | Amphipathic IQ helix formation is a strong signal; bulky hydrophobic anchors (Trp, Phe) dock reliably into EF-hand cleft |
| **Protease substrates** | Extended β-strand substrate geometry in well-shaped active site cleft (TEV, thrombin) |

### 5.2 Moderate performance

| Family | Reason |
|--------|--------|
| **Kinase substrates** | Highly variable peptide-binding geometry; ATP pocket context differs by kinase subfamily; moderate difficulty |
| **Amphipathic helix binders** | Longer peptides (14–19 residues) introduce conformational degeneracy; helical register shifts are hard to correct |
| **ARM / HEAT repeats** | Extended curved surface with distributed contacts; peptide adopts extended conformation, entropy penalty is high |
| **MHC class I / pHLA** | Backbone geometry is fixed by groove; diffusion can place the peptide, but allele-specific anchor discrimination requires sidechain precision beyond typical RMSD metrics |

### 5.3 Poor performance / difficult cases

| Family | Reason |
|--------|--------|
| **SH3 / WW (proline-rich PPXP)** | Poly-Pro type II helix is rare in the PDB training distribution; diffusion models under-sample this backbone geometry; consistently classified as "easy" in PepSet but with systematic errors |
| **Collagen (GPP repeat)** | Triple-helix conformation is structurally unique and almost absent from non-collagen training data; highly likely to generate incorrect backbone topology |
| **Long flexible peptides (> 15 residues)** | Conformational sampling degrades; the "difficult" label in PepSet correlates with length > 15 and ~50% lower success rate |
| **pHLA allele discrimination** | Sequence-level binding differences between alleles are subtle; coarse pose accuracy does not translate to binding-affinity prediction without per-residue sidechain accuracy |
| **Phospho-modified peptides (PLK1-PBD, SHP2)** | [TPO], [SEP], [PTR] are non-standard residues; most diffusion models were trained on standard amino acids and handle modified residues poorly or not at all |
| **Cyclic / disulfide-bridged peptides** | Topology is not enforced during diffusion; sampled conformations ignore ring-closure or disulfide constraints |
| **Poly-Lys / basic NLS peptides** | Minimal structural constraints; over-reliance on electrostatic surface matching; geometry is poorly determined |

---

## 6. Obtaining the Data

The `datasets/` directory is excluded from the repository (`.gitignore`) because it
contains large binary PDB files (~300 MB total) that should not be versioned.

| Dataset | Source | Notes |
|---------|--------|-------|
| PepSet | RAPiDock paper supplementary (Zhao et al., *Nat. Mach. Intell.* 7:1308, 2025) | 185 complexes, benchmark splits included |
| RefPepDB-RecentSet | [RefPepDB](https://github.com/DMCB-GIST/RefPepDB) GitHub | Recent-set slice (PDB 7xxx–8xxx) |
| cases/pHLA | [NetMHCpan](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) or PDB direct | Binding affinity CSV is a curated subset |
| cases/PLK1-PBD | PDB direct download + manual curation | 31 phosphopeptide structures |
| cases/SHP2 | PDB direct download + manual curation | 9 pTyr structures |
| cases/Importin-α | PDB direct (7M60) | Bipartite NLS, 2 pocket modes |

Place the extracted directories under `datasets/` and the pipeline will find them
automatically via the paths configured in `data/training_complexes.csv` and
`data/test_complexes.csv`.
