# Data Augmentation Plan — Items 1, 2, 6

**Status:** Spec only. Hand this to Sonnet to implement.
**Owner:** Ram (Head of Dry Lab) · iGEM 2026 Denmark HS
**Last updated:** 2026-05-21
**Source context:** docs/dataset_analysis.md (RAPiDock training gaps); CLAUDE.md (project conventions)

---

## 0. Scope and motivation

The dataset analysis (`docs/dataset_analysis.md` §7, §8) identified three concrete weaknesses in RAPiDock's training distribution and HybriDock-Pep's calibration pipeline:

| # | Gap | Symptom | Fix in this plan |
|---|-----|---------|------------------|
| 1 | RAPiDock training cutoff is ~late 2023; PDB has grown by ~15 k entries since | We lose fresh ground-truth signal | PDB query for protein–peptide complexes deposited after the cutoff |
| 2 | PPII helix (φ ≈ −75°, ψ ≈ +150°) is ~2 % of PDB residues; only ~19 SH3/WW entries in PepSet | Vina score per-contact on SH3 (1A0N) is 0.81 kcal/mol/contact vs 1.05 on PDZ | Targeted PPII-enriched subset extraction with Ramachandran validation |
| 6 | Phospho residues `[TPO]`, `[SEP]`, `[PTR]` are not handled by Meeko/babel | PLK1-PBD, SHP2, phosphopeptide kinase substrates systematically underscored by 3–5 kcal/mol | Add phospho-residue parametrisation to `prep/ligand.py` |

These three items together do **not** require retraining RAPiDock. Items 1 and 2 produce data; item 6 unlocks scoring of an entire family. Combined with the BindingDB calibration expansion (item 3 below), this is the highest ROI work achievable in 1–2 weeks with no new infrastructure.

Total scope:
- **2 new scripts** (~250 lines each): `scripts/fetch_pdb_complexes.py`, `scripts/bindingdb_calibration_join.py`
- **2 modified source files**: `src/hybridock_pep/prep/ligand.py`, `src/hybridock_pep/prep/receptor.py`
- **1 modified env file**: `envs/score-env.yml` (one new dep)
- **1 new test file**: `tests/test_phospho_residues.py`
- **3 new data outputs**: `datasets/pdb_2024_2026/` manifest, `datasets/ppii_enriched/` manifest, `data/training_complexes_expanded.csv`

All file paths, function signatures, and acceptance criteria below are normative. Sonnet should follow them rather than re-deriving choices.

---

## 1. Item 1 — Fresh PDB protein–peptide complexes

### 1.1 Acceptance criteria

After this item is done:
- A manifest CSV `datasets/pdb_2024_2026/manifest.csv` exists with ≥ 800 rows
- Each row has: `pdb_id`, `peptide_chain`, `peptide_seq`, `peptide_len`, `receptor_chain`, `receptor_len`, `resolution_A`, `deposition_date`, `family_hint`, `excluded_reason`
- PDB files are downloaded under `datasets/pdb_2024_2026/structures/{pdb_id}.pdb.gz`
- Entries already in PepSet (185 IDs) and RefPepDB-RecentSet are excluded with `excluded_reason="duplicate_pepset"` or `excluded_reason="duplicate_refpepdb"`
- `datasets/pdb_2024_2026/` is added to `.gitignore` (large binaries)
- Script is idempotent: re-running skips already-downloaded structures and only refreshes the manifest

### 1.2 Query criteria (passed to RCSB Search API)

| Criterion | Value | RCSB attribute |
|-----------|-------|----------------|
| Deposition date | `>= 2023-09-01` | `rcsb_accession_info.initial_release_date` |
| Experimental method | X-ray diffraction OR cryo-EM | `exptl.method in ("X-RAY DIFFRACTION","ELECTRON MICROSCOPY")` |
| Resolution | `<= 2.5 Å` | `rcsb_entry_info.resolution_combined` |
| Polymer entities | exactly 2 protein entities, no nucleic acids | post-filter |
| Short polymer length | `5 <= length <= 30` | `entity_poly.rcsb_sample_sequence_length` |
| Long polymer length | `>= 50` | (the receptor) |

The "exactly 2 protein entities" rule keeps complexes simple. Multi-chain complexes (homodimers + peptide) are handled at a later iteration; flag them with `family_hint="multimer"` and `excluded_reason="non_binary_complex"` in the manifest so the data isn't lost.

### 1.3 Family hint heuristics

Compute `family_hint` from receptor sequence motifs / structural classification:
- Query CATH classification via `rcsb_polymer_entity_annotation.annotation_lineage` (if available)
- Fallback: sequence-motif search against a small dictionary in the script:
  - PDZ → "GLGF" or "GYGF" in receptor sequence
  - SH3 → "FRR" + ~60 residue domain
  - WW → contains 2 Trp ~25 residues apart
  - SH2 → "FLVRES"
  - Bromodomain → "WPF" motif in receptor
  - MHC-class-I → receptor length 270–300 AA, contains "GSHSMRYF" or similar
  - Otherwise → "unclassified"

This is heuristic; don't over-invest. The point is to make the manifest filterable, not to perfectly classify.

---

## 2. Item 2 — PPII-enriched subset

### 2.1 Why PPII separately

PPII complexes are the highest-leverage data because RAPiDock fails worst on SH3/WW (see `docs/dataset_analysis.md` §5.3). We want to pull these out of the full PDB corpus (not just the post-2023 slice) and store them in their own directory.

### 2.2 Acceptance criteria

After this item is done:
- A manifest `datasets/ppii_enriched/manifest.csv` exists with ≥ 150 rows
- Each row has the columns from §1.1 plus `ppii_residues` (int), `ppii_fraction` (float in [0,1]), `consecutive_pro` (int)
- Each entry passes ALL of:
  - Peptide contains ≥ 2 consecutive Pro residues, OR ≥ 4 Pro residues anywhere
  - ≥ 30 % of peptide residues fall in PPII region of the Ramachandran plot (definition below)
- PepSet/RefPepDB duplicates excluded as in §1.1
- Structures downloaded under `datasets/ppii_enriched/structures/{pdb_id}.pdb.gz`

### 2.3 PPII Ramachandran definition

A residue is considered PPII if:
- `−90° ≤ φ ≤ −20°` AND
- `+110° ≤ ψ ≤ +180°` (handle wrap-around: also accept `−180° ≤ ψ ≤ −170°`)

Use Biopython's `Bio.PDB.PPBuilder` to extract polypeptides, then `internal_coord` or manual dihedral calculation. The first and last residues of the peptide have undefined φ or ψ — skip them in the fraction calculation.

### 2.4 Query strategy

The PPII query is a superset query on the PDB (no date filter), then post-filter by:
1. Peptide chain has length 5–25 and contains ≥ 2 consecutive Pro (sequence-level filter via RCSB)
2. Resolution ≤ 2.5 Å, X-ray only (cryo-EM tends to have noisy sidechain φ/ψ)
3. Compute Ramachandran fraction locally and apply 30 % threshold

The RCSB sequence query is:
```
entity_poly.rcsb_polymer_entity_polymer_type = "Protein"
AND entity_poly.rcsb_sample_sequence_length BETWEEN 5 AND 25
AND <regex match for PP in sequence>
```

The "≥2 consecutive Pro" filter can be approximated by an RCSB substring search for "PP", then verified locally after download.

---

## 3. Item 6 — Phospho-residue parametrisation

### 3.1 The actual gap

Currently `src/hybridock_pep/prep/ligand.py` uses `babel -h -xr` from ADFRsuite. When the input PDB contains `TPO`/`SEP`/`PTR` residues, babel either:
- Drops the phosphate group silently (most common), OR
- Errors with a non-standard residue warning, OR
- Adds incorrect atom types to the PDBQT (phosphate as generic P, no formal charge)

Net effect: the Vina/AD4 score loses the dominant electrostatic interaction (phosphate → Arg/Lys), underscoring affinity by 3–5 kcal/mol.

### 3.2 Approach (chosen — do not deviate without discussion)

We will **not** patch babel. Instead, we add a Python pre-processing step that runs *before* babel:

1. **Detect** phospho residues in the input PDB by residue name (`TPO`, `SEP`, `PTR`).
2. **Pre-parametrise** them using Meeko's `PolymerTopology` (Meeko 0.7+ supports custom residue templates via `MoleculePreparation` and `polymer_residues` extension JSON).
3. **Generate** a proper PDBQT directly with Meeko, bypassing babel for phospho-containing peptides only. Standard (non-phospho) peptides continue through the babel path.
4. **Verify** in tests that scored phospho peptides recover the expected ~3–5 kcal/mol electrostatic contribution.

### 3.3 Acceptance criteria

After this item is done:
- `src/hybridock_pep/prep/ligand.py` has a new function `_has_phospho_residues(pdb_path: Path) -> bool` (boolean detection)
- `_prepare_single_ligand` routes to one of two paths based on the detection:
  - **Path A** (no phospho): existing babel pipeline — unchanged
  - **Path B** (has phospho): new Meeko-based pipeline
- A new module `src/hybridock_pep/prep/phospho.py` contains the Meeko logic and the residue templates
- Residue templates exist for `TPO` (phospho-Thr), `SEP` (phospho-Ser), `PTR` (phospho-Tyr), all with formal charge −2 on the phosphate at pH 7
- New test file `tests/test_phospho_residues.py` covers:
  - Detection: PDBs with/without phospho residues are correctly classified
  - Round-trip: TPO/SEP/PTR pose PDB → PDBQT → re-parse, verify P atom present with correct charge
  - Integration: scoring a real SHP2 phospho-peptide pose (we have 4JMG sequence `V[PTR]ENVGLM` — get the structure from the PDB) returns AD4 < −7 kcal/mol (vs ~−4 if phospho is dropped)
- `tests/fixtures/shp2_4jmg/` holds the SHP2 fixture (extracted with the existing `extract_pepset_fixtures.py` pattern; needs a small extension to handle phospho residue names)

### 3.4 Meeko residue template format

Meeko 0.7 expects polymer residue templates as JSON in a specific schema. Reference: https://github.com/forlilab/Meeko/blob/develop/meeko/data/residue_chem_templates.json

Skeleton for one residue:
```json
{
  "TPO": {
    "atom_name": ["N","CA","C","O","CB","OG1","CG2","P","O1P","O2P","O3P","H","HA","HB","HG21","HG22","HG23"],
    "atom_type": ["NA","C","C","OA","C","OA","C","P","OA","OA","OA","HD","HD","HD","HD","HD","HD"],
    "charge":    [-0.35,0.07,0.55,-0.55,0.15,-0.55,0.05,1.10,-0.75,-0.75,-0.75,0.16,0.05,0.05,0.05,0.05,0.05],
    "bonds": [[0,1],[1,2],[2,3],[1,4],[4,5],[4,6],[5,7],[7,8],[7,9],[7,10],[0,11],[1,12],[4,13],[6,14],[6,15],[6,16]]
  }
}
```

These charges are **starting estimates** — exact values should match Gasteiger output for the same topology. Validate by running babel on a single TPO residue (in isolation, no peptide context) and comparing.

The full templates for TPO/SEP/PTR are not yet known to me; Sonnet should:
1. Build the templates by running `babel -h -xr` on isolated TPO/SEP/PTR amino acid model compounds (PDB fragments with N-acetyl and N-methylamide caps), extracting atom types and Gasteiger charges from the resulting PDBQT, and translating to Meeko JSON format
2. If that fails (babel rejects modified residues outright), derive charges from antechamber/ANTE / AmberTools — `antechamber -i TPO.pdb -o TPO.mol2 -fi pdb -fo mol2 -c gas` gives Gasteiger charges
3. As a last resort: use literature values from PARM99 or ff14SB phospho extensions (Homeyer et al. 2006 for TPO/SEP)

### 3.5 Edge cases to handle

- Multiple phospho residues in one peptide (PLK1 substrate has 2 phospho-Thr) — both must be parametrised
- N- or C-terminal phospho — terminal atom types differ; templates need terminal variants
- Mixed phospho + standard residues in same peptide (the common case) — route through Path B if ANY phospho present
- PDB file lists phospho as HETATM rather than ATOM (older entries) — handle both record types

---

## 4. Script spec — `scripts/fetch_pdb_complexes.py`

Covers Items 1 + 2 in one script (they share infrastructure: RCSB query, PDB download, manifest writing, deduplication).

### 4.1 Header

```python
"""Fetch protein–peptide complexes from the RCSB PDB.

Two modes:
  --mode recent  : Item 1 — protein-peptide complexes deposited after 2023-09-01
  --mode ppii    : Item 2 — PPII-enriched complexes (any deposition date)
  --mode both    : run both (default)

Outputs:
  datasets/pdb_2024_2026/manifest.csv  (recent mode)
  datasets/pdb_2024_2026/structures/{pdb_id}.pdb.gz
  datasets/ppii_enriched/manifest.csv  (ppii mode)
  datasets/ppii_enriched/structures/{pdb_id}.pdb.gz

Idempotent: skips structures already downloaded; only refreshes manifests.

Usage:
    conda run --no-capture-output -n score-env python scripts/fetch_pdb_complexes.py \
        --mode both \
        --max-workers 4 \
        --resolution-cutoff 2.5

Requires (score-env):
  - rcsbsearch>=2.3 (NEW dependency, add to envs/score-env.yml)
  - biopython>=1.83 (already in env)
  - requests>=2.32 (already in env)
  - pandas>=2.0 (already in env)
"""
```

### 4.2 Module structure

```
scripts/fetch_pdb_complexes.py
├── CONFIG (constants)
├── RCSB query builders
│   ├── _query_recent_complexes(since: str) -> list[str]
│   └── _query_ppii_candidates() -> list[str]
├── PDB download
│   ├── _download_pdb_gz(pdb_id: str, dest: Path) -> bool
│   └── _download_batch(pdb_ids: list[str], dest_dir: Path, max_workers: int)
├── PDB parsing & filtering
│   ├── _parse_chain_info(pdb_path: Path) -> dict
│   ├── _is_protein_peptide_complex(chains: dict) -> tuple[bool, str]
│   ├── _classify_family(receptor_seq: str) -> str
│   └── _compute_ramachandran_ppii(pdb_path: Path, chain_id: str) -> tuple[int, float, int]
├── Deduplication
│   ├── _load_pepset_ids() -> set[str]      # parse from datasets/pepset/
│   └── _load_refpepdb_ids() -> set[str]    # parse from datasets/RefPepDB-RecentSet/
├── Manifest writers
│   ├── _write_recent_manifest(rows: list[dict], out: Path)
│   └── _write_ppii_manifest(rows: list[dict], out: Path)
└── CLI entry point (argparse → main)
```

### 4.3 Key functions — exact signatures

```python
def _query_recent_complexes(
    since: str = "2023-09-01",
    resolution_cutoff: float = 2.5,
    short_chain_min: int = 5,
    short_chain_max: int = 30,
    long_chain_min: int = 50,
) -> list[str]:
    """Return PDB IDs of complexes meeting Item 1 criteria. Uses rcsbsearch."""

def _query_ppii_candidates(
    resolution_cutoff: float = 2.5,
    short_chain_min: int = 5,
    short_chain_max: int = 25,
) -> list[str]:
    """Return PDB IDs of complexes whose short chain contains 'PP' substring."""

def _download_pdb_gz(pdb_id: str, dest: Path) -> bool:
    """Download via https://files.rcsb.org/download/{pdb_id}.pdb.gz.
    Skip if dest exists and is non-empty. Return True on success, False on failure.
    Use requests with timeout=30 and 3 retries with exponential backoff."""

def _parse_chain_info(pdb_path: Path) -> dict:
    """Return dict with keys per chain ID:
      {
        chain_id: {
          'length': int,
          'sequence': str (1-letter code; X for nonstd),
          'is_protein': bool,
          'nonstd_residues': list[str] (e.g. ['TPO', 'PTR'])
        }
      }
    Uses Biopython PDBParser. Handles gzip via gzip.open."""

def _compute_ramachandran_ppii(
    pdb_path: Path,
    chain_id: str,
) -> tuple[int, float, int]:
    """Compute PPII fraction for given chain.
    Returns (n_ppii_residues, ppii_fraction, max_consecutive_pro).
    Uses Bio.PDB.PPBuilder for polypeptide extraction, internal_coord for phi/psi.
    PPII region: -90 <= phi <= -20 AND (110 <= psi <= 180 OR -180 <= psi <= -170)."""
```

### 4.4 Manifest schemas

**Recent manifest** (`datasets/pdb_2024_2026/manifest.csv`):

| Column | Type | Notes |
|--------|------|-------|
| pdb_id | str | upper case |
| peptide_chain | str | chain ID of short polymer |
| peptide_seq | str | 1-letter code, `X` for nonstandard |
| peptide_len | int | residue count |
| peptide_nonstd | str | comma-separated nonstd residue names (e.g. "TPO,PTR") |
| receptor_chain | str | chain ID of long polymer |
| receptor_len | int | residue count |
| receptor_seq_md5 | str | MD5 hex of receptor sequence (for dedup) |
| resolution_A | float | reported resolution |
| method | str | "X-RAY DIFFRACTION" or "ELECTRON MICROSCOPY" |
| deposition_date | str | ISO date |
| family_hint | str | from `_classify_family` |
| excluded_reason | str | "" if included, otherwise enum: `duplicate_pepset`, `duplicate_refpepdb`, `non_binary_complex`, `bad_resolution`, `chain_count` |

**PPII manifest** (`datasets/ppii_enriched/manifest.csv`):

Same columns plus:

| Column | Type | Notes |
|--------|------|-------|
| ppii_residues | int | count of PPII residues |
| ppii_fraction | float | n_ppii / (peptide_len − 2) |
| consecutive_pro | int | longest run of consecutive Pro |
| passes_ppii_filter | bool | True if `ppii_fraction >= 0.30` AND `consecutive_pro >= 2` |

### 4.5 Error handling

- Network errors during download → 3 retries with backoff, then log to stderr and mark row as `excluded_reason="download_failed"`
- Malformed PDB → catch parsing exception, mark `excluded_reason="parse_failed"`
- Empty short chain (length 0 after filtering nonstd) → mark `excluded_reason="empty_peptide"`
- All errors logged via `logging.warning`, never raise

### 4.6 Performance

- Use `concurrent.futures.ThreadPoolExecutor` for downloads (network-bound)
- Default max_workers=4; expose `--max-workers N` CLI flag
- Expected runtime on a fresh machine: ~30 min for recent mode (~1000 structures), ~20 min for ppii mode (~200 structures after filter)

---

## 5. Script spec — `scripts/bindingdb_calibration_join.py`

### 5.1 Purpose

Expand `data/training_complexes.csv` from 2 entries to ~200+ by joining BindingDB Kd measurements to PDB protein–peptide structures.

### 5.2 Acceptance criteria

After this script is run:
- `data/training_complexes_expanded.csv` exists with ≥ 100 valid rows (target 200)
- Columns: `pdb_id`, `peptide_sequence`, `receptor_chain`, `experimental_pkd`, `kd_nM`, `source`, `family_hint`
- All rows have `3 <= experimental_pkd <= 12` (physically reasonable Kd range)
- Source column is one of: `bindingdb_kd`, `bindingdb_ki_converted`, `manual` (existing rows from `training_complexes.csv` carried over with `source="manual"`)
- Header preserves backward compat with current 3-column `training_complexes.csv` so calibrate_alpha.py keeps working

### 5.3 Data source

BindingDB All Data download, TSV format:
- URL: `https://www.bindingdb.org/bind/downloads/BindingDB_All_202504_tsv.zip` (use the latest monthly snapshot — discover URL at runtime by scraping https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
- Size: ~2 GB compressed, ~9 GB uncompressed
- Key columns (note: BindingDB column names are unstable — verify against the actual header):
  - `BindingDB MonomerID`
  - `Ligand SMILES`
  - `Ligand InChI`
  - `BindingDB Ligand Name` — sometimes a peptide sequence in 1-letter code
  - `PDB ID(s) of Target Chain` — the join key
  - `Kd (nM)`, `Ki (nM)`, `IC50 (nM)`, `EC50 (nM)`
  - `Target Name`
  - `Target Source Organism According to Curator or DataSource`
  - `BindingDB Reactant_set_id`

### 5.4 Module structure

```
scripts/bindingdb_calibration_join.py
├── CONFIG
├── BindingDB download/cache
│   ├── _get_latest_url() -> str         # scrape Download.jsp
│   ├── _download_and_extract(url, dest_dir: Path) -> Path
│   └── _load_cached(cache_path: Path) -> pd.DataFrame | None
├── Peptide detection
│   ├── _is_peptide_smiles(smiles: str) -> bool
│   └── _smiles_to_sequence(smiles: str) -> str | None  # extracts 1-letter code from peptide SMILES
├── PDB cross-check
│   ├── _load_existing_pepset_pdbs() -> set[str]
│   ├── _fetch_pdb_metadata(pdb_id: str) -> dict  # uses requests against https://data.rcsb.org/rest/v1/core/entry/{id}
│   └── _verify_pdb_has_peptide(pdb_id: str, expected_seq: str) -> bool
├── Conversion & filtering
│   ├── _kd_to_pkd(kd_nM: float) -> float
│   ├── _ki_to_pkd_estimate(ki_nM: float) -> float  # Cheng-Prusoff approximation
│   └── _filter_pkd_range(pkd: float) -> bool
├── Manifest builder
│   └── _build_expanded_csv(filtered: pd.DataFrame, existing: pd.DataFrame, out: Path)
└── CLI entry point
```

### 5.5 Key functions

```python
def _is_peptide_smiles(smiles: str) -> bool:
    """Heuristic: SMILES looks like a peptide if it contains ≥3 amide bonds
    (N[C@@H]) AND total length 10–300 chars AND no rings outside Phe/Tyr/Trp/His/Pro.
    Use RDKit (already in env): parse mol, count substructure matches for
    amide bond pattern [N;H1,H2][C;H1](=[O]) — peptides have len-1 such bonds."""

def _smiles_to_sequence(smiles: str) -> str | None:
    """Convert peptide SMILES to 1-letter sequence via RDKit substructure matching
    against the 20 amino acid templates. Return None if not a clean peptide.
    For phospho residues: detect P(=O)(O)O on Thr/Ser/Tyr scaffolds and
    encode as [TPO]/[SEP]/[PTR]."""

def _kd_to_pkd(kd_nM: float) -> float:
    """pKd = -log10(Kd_nM × 1e-9). Returns NaN for non-positive Kd."""

def _ki_to_pkd_estimate(ki_nM: float) -> float:
    """For Ki measurements where Kd is unavailable: treat Ki as Kd estimate.
    Cheng-Prusoff for IC50→Ki is not applied here (would need substrate Km).
    Return -log10(Ki_nM × 1e-9). Mark these rows with source='bindingdb_ki_converted'."""
```

### 5.6 Filter pipeline

```
Raw BindingDB (~2.5M rows)
  │
  ├─ Drop rows without PDB ID
  │
  ├─ Drop rows without Kd (or use Ki if --use-ki flag enabled)
  │
  ├─ Drop rows where ligand SMILES is not peptide-like (_is_peptide_smiles)
  │
  ├─ Convert peptide SMILES → 1-letter sequence; drop rows where conversion fails
  │
  ├─ Filter peptide length 5 ≤ len ≤ 30
  │
  ├─ Convert Kd → pKd; filter to 3 ≤ pKd ≤ 12
  │
  ├─ Cross-check PDB ID exists at RCSB (cache results in datasets/cache/pdb_metadata.json)
  │
  ├─ Drop rows where PDB ID is in PepSet (preserve PepSet as test set, don't leak into training)
  │
  ├─ Deduplicate by (pdb_id, peptide_sequence) keeping the highest-confidence Kd
  │
  └─ Output rows → data/training_complexes_expanded.csv
```

### 5.7 Caveats Sonnet should be aware of

- BindingDB's `Ligand SMILES` column for peptides may have non-standard stereochemistry or be encoded as a single linear string with no AA breaks. Some entries have explicit `[*]` placeholders for N/C terminal capping
- "PDB ID(s) of Target Chain" can be a comma-separated list — split and take the first ID
- Many entries have Kd from different references (literature curation can disagree); the column `Curation/DataSource` indicates this. Prefer entries where `Curation/DataSource` is "BindingDB"
- Some peptides are cyclic / disulfide-bridged — SMILES will have ring closures. Drop these (out of scope for HybriDock-Pep v1)
- BindingDB has affinity for the same ligand–target pair from multiple labs. Aggregate by median pKd if multiple rows exist for the same (pdb_id, sequence) — log spread; if max−min > 1.5 log units, exclude with `excluded_reason="affinity_spread"`

### 5.8 Performance

- First run: download ~2GB, parse ~9GB TSV → 10–20 min on broadband + SSD
- Cache the parsed/filtered DataFrame as Parquet in `datasets/cache/bindingdb_filtered.parquet`
- Subsequent runs: load Parquet, run filter pipeline → 1–2 min
- RCSB metadata calls: batch 100 IDs per query via `https://data.rcsb.org/graphql` GraphQL endpoint (much faster than per-ID REST). Skeleton query in §5.9.

### 5.9 RCSB GraphQL batch query template

```graphql
query batchMeta($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    polymer_entities {
      entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    }
  }
}
```

POST to `https://data.rcsb.org/graphql` with `{"query": <above>, "variables": {"ids": ["1A0N","2VZG",...]}}`. Batch size 100 IDs per request.

---

## 6. Item 6 detailed implementation notes

### 6.1 Files to modify

- `src/hybridock_pep/prep/ligand.py`:
  - Add `_has_phospho_residues(pdb_path: Path) -> bool` at module level (~10 lines, just checks for `TPO`/`SEP`/`PTR` substrings in ATOM/HETATM residue name columns)
  - Branch `_prepare_single_ligand` based on detection
  - On phospho path: call into `phospho._prepare_phospho_ligand(pdb_path, output_dir, pose_idx)`
- `src/hybridock_pep/prep/phospho.py` (new, ~150 lines):
  - Module-level constant `PHOSPHO_RESIDUES = {"TPO", "SEP", "PTR"}`
  - Module-level dict `RESIDUE_TEMPLATES` loaded once at import from `src/hybridock_pep/prep/data/phospho_templates.json`
  - `_prepare_phospho_ligand(pdb_path, output_dir, pose_idx) -> Path | PoseFailure`
  - Uses `meeko.PolymerTopology` or `meeko.MoleculePreparation` with custom polymer extensions
  - Writes PDBQT directly; no babel call
- `src/hybridock_pep/prep/data/phospho_templates.json` (new, ~50 lines): the JSON residue templates per §3.4
- `src/hybridock_pep/prep/receptor.py`:
  - Currently strips ALL HETATM — must be modified to **keep** phospho residues if they appear as HETATM (some PDBs encode TPO/SEP/PTR as HETATM). Add a `PRESERVE_HETATM_RESNAMES` constant = `{"TPO","SEP","PTR"}` and special-case those.

### 6.2 Test plan — `tests/test_phospho_residues.py`

```python
class TestPhosphoDetection:
    def test_detects_tpo_in_pdb(self):
        # fixture: plk1_3rq7/pose_000.pdb (will need to create)
        assert _has_phospho_residues(plk1_pose_path) is True

    def test_detects_no_phospho(self):
        # existing fixture: pdz_1jq8/pose_000.pdb
        assert _has_phospho_residues(pdz_pose_path) is False

class TestPhosphoLigandPrep:
    def test_tpo_ligand_prep_round_trip(self, tmp_path):
        # input: synthetic PDB with one TPO residue
        # output: valid PDBQT with P atom present, charge < -1.0 on phosphate oxygens
        ...

    def test_phospho_peptide_pdbqt_passes_vina_parse(self, tmp_path):
        # Generate PDBQT for a TPO-containing peptide; load into Vina; verify no error
        ...

@pytest.mark.slow
class TestPhosphoScoring:
    def test_shp2_ptr_recovers_electrostatic_signal(self, tmp_path):
        # Score 4JMG SHP2 peptide V[PTR]ENVGLM through full pipeline
        # Expect AD4 < -7.0 (vs ~-4.0 if phospho is dropped)
        ...
```

### 6.3 SHP2 fixture creation

The existing `scripts/extract_pepset_fixtures.py` is the template. Add a new target:

```python
("shp2_4jmg", "4jmg", "SH2 / phospho-Tyr (SHP2 N-SH2)"),
```

Then run with `--include-hetatm-resnames TPO,SEP,PTR` (new CLI flag) so the script keeps phospho residues in `pose_000.pdb`. The 4JMG structure must be obtained from `datasets/cases/SHP2/` or downloaded directly via the Item 1 script.

### 6.4 Validation milestone

Before declaring Item 6 done, run the existing benchmark and confirm:
- All existing tests still pass (`pytest -m "not slow"` and `pytest -m slow`)
- The new SHP2 fixture passes the 4 standard pepset-style assertions (pipeline completes, Vina < 0, AD4 < 0, no AD4 anomaly)
- The SHP2 AD4 score is at least 2.5 kcal/mol more negative than scoring the same peptide with PTR→TYR substitution (demonstrates the phospho parametrisation is doing real work)

---

## 7. Dependencies to add

Edit `envs/score-env.yml` — add one line under `- pip:`:

```yaml
      - "rcsbsearch>=2.3"
```

Everything else (`pandas`, `requests`, `gemmi`, `biopython`, `meeko`, `rdkit`) is already present. Confirm with `conda run -n score-env pip list | grep -iE 'rcsb|meeko|rdkit'` before starting.

If `rcsbsearch` is not on conda-forge yet, this stays in the pip section and is fine.

---

## 8. Execution order (do not parallelise items)

1. **Add `rcsbsearch` to env file and update env**:
   ```
   conda env update -n score-env -f envs/score-env.yml
   ```
2. **Implement `fetch_pdb_complexes.py`** — Item 1 mode first, validate manifest has ≥ 800 rows
3. **Run PPII mode** — validate manifest has ≥ 150 rows passing the filter
4. **Implement `bindingdb_calibration_join.py`** — independent of items 1/2 but reuses RCSB infrastructure helpers; refactor shared helpers into `scripts/_pdb_utils.py` if duplication exceeds ~30 lines
5. **Run BindingDB join** — validate `training_complexes_expanded.csv` has ≥ 100 rows
6. **Implement Item 6** — phospho residue parametrisation
7. **Run full test suite** — `pytest && pytest -m slow` — all green before commit
8. **Single commit** with all three items:
   ```
   feat(data): expand training data + phospho residue support

   - Item 1: fetch_pdb_complexes.py --mode recent (datasets/pdb_2024_2026/)
   - Item 2: fetch_pdb_complexes.py --mode ppii (datasets/ppii_enriched/)
   - Item 3: bindingdb_calibration_join.py (data/training_complexes_expanded.csv)
   - Item 6: phospho residue parametrisation (TPO, SEP, PTR)
   ```

---

## 9. Out of scope (do NOT do in this round)

- AlphaFold3 synthetic structure generation (Tier 2 item 4 from §earlier discussion)
- MD ensemble augmentation (Tier 2 item 5)
- RAPiDock fine-tuning (Tier 3 item 7)
- Re-running `calibrate_alpha.py` against the expanded training set — that's the next chunk of work, separate from this plan
- Cyclic / disulfide-bridged peptides
- Re-doing the existing benchmark with the new data — only validate that nothing regressed

If during execution Sonnet finds that one of the in-scope items is materially blocked (e.g. BindingDB schema has changed, Meeko 0.7 doesn't expose `PolymerTopology` as described), it should stop and surface the blocker rather than work around it silently.

---

## 10. Reference files in this repo

- `scripts/extract_pepset_fixtures.py` — pattern for fixture extraction scripts (stdlib + numpy only)
- `scripts/score_family_benchmark.py` — pattern for benchmark runner scripts
- `src/hybridock_pep/scoring/entropy.py` — example of calibration data loading (see `load_calibration`)
- `src/hybridock_pep/prep/ligand.py` — current ligand prep (the file to extend in Item 6)
- `data/training_complexes.csv` — current 2-row training file (the file to expand)
- `CLAUDE.md` §4 — coding conventions (type hints everywhere, Google docstrings, ruff/black, line length 100)
- `CLAUDE.md` §6 — git conventions (Conventional Commits, no Co-Authored-By unless instructed)

---

*This plan was written by Opus 4.7 for execution by Sonnet 4.6. Specs are normative; deviation should be flagged before implementation.*
