# HybriDock-Pep Dataset Validation Report

Generated: 2026-05-23

---

## Summary

PepSet IDs loaded: 21


## pdb_2024_2026 (Recent Complexes)
- Manifest rows: 1026
- Included (no exclusion): 646
- Exclusion reasons:
  - `chain_count`: 321
  - `peptide_length`: 42
  - `download_failed`: 17
- Structure files on disk: 1009
- Included IDs with files: 646
- ✅ Structure file spot-check: first 20 OK
- ✅ No PepSet leakage

## ppii_enriched (PPII-Enriched, relaxed filter)
- Manifest rows: 337
- Included (no exclusion): 74
- Exclusion reasons:
  - `chain_count`: 116
  - `fails_ppii_filter`: 112
  - `peptide_length`: 22
  - `download_failed`: 13
- Structure files on disk: 324
- Included IDs with files: 74
- ✅ Structure file spot-check: first 20 OK
- ✅ No PepSet leakage
- PPII filter passed: 74
  - mean fraction: 0.47
  - mean consec_pro: 1.7

## Core PDB Structures (raw_pdbs)
- Files: 30
- ✅ All 30 files valid
- IDs: ['1A0N', '1DDV', '1EJ4', '1G73', '1JQ8', '1JW6', '1L2Z', '1NRL', '1PMX', '1PRM', '1T2D', '1YFN', '1YWI', '2CNY', '2FLU', '2HWN', '2KHH', '2KOH', '2OY2', '2VWF', '2VZG', '3BEJ', '3DAB', '3EG6', '3EQS', '3EQY', '3SHB', '3TWR', '4GQ6', '4JMG']

## PepSet (Held-out Test Set)
- pepset_ids.txt: 21 IDs
- IDs: ['1A0N', '1EJ4', '1G73', '1JQ8', '1JW6', '1PMX', '1PRM', '1YFN', '1YWI', '2CNY', '2FLU', '2KHH', '2VWF', '2VZG', '3BEJ', '3DAB', '3EG6', '3EQS', '3EQY', '3SHB', '3TWR']
- test_complexes.csv: 10 entries
- Rows with sequence: 10/10
- pKd valid: True
- ⚠️ NOTE: Only 10 test complexes vs plan's 185. Full PepSet (RefPepDB)
  not yet built. Population-level r claims must use n=10 until PepSet is expanded.

## Training Complexes (Calibration)
- Rows: 6
- Columns: ['pdb_id', 'peptide_sequence', 'experimental_pkd']
- pKd range valid: True (4.1–8.7)
- Rows with sequence: 6/6
- PDB IDs: ['2hwn', '1nrl', '1l2z', '1ddv', '1a0n', '1ywi']

### Expanded CSV (training_complexes_expanded.csv)
- Rows: 42
- Rows with sequence: 3/42
- **⚠️ WARNING**: 39 rows have no peptide sequence
- **ROOT CAUSE**: BindingDB join captured small-molecule inhibitor PDB entries
  (1A85/1FU4/etc. are HIV protease small molecule structures, not peptide complexes)
- **FIX needed**: Re-run bindingdb_calibration_join.py with --amide-min 4 and RCSB polymer verification
- **True peptide rows available**: 3/42 (2OY2, 1YCR, 5HI3 + partially)
- pKd range: 3.3–10.9
- Sources: {'bindingdb_ki_converted': 25, 'bindingdb_kd': 15, 'manual': 2}

## Calibration State
- α (alpha): 0.1
- β (beta): 0.0
- γ (gamma): 0.2
- n_complexes: 6
- Pearson r: 0.8597
- RMSE: 1.7348 kcal/mol
- 🚨 **ALERT**: α=0.1 at lower bound — calibration is invalid for production use
- Calibrated at: 2026-05-22T00:16:22.090275+00:00

## BindingDB Cache
- File: bindingdb_all.zip (581 MB)
- ZIP valid: yes
- Contents: ['BindingDB_All.tsv']
- Uncompressed: 8.8 GB
- ✅ BindingDB All Data ready for processing
- ⚠️ True peptide entries with PDB+affinity: ~84 (scan result)
- **NOTE**: Only ~84 of 3.17M rows are genuine peptide-protein pairs
  with PDB structures. Plan's 200-row target will need supplementation
  from PepBDB, AffinityBench, or manual curation.

---

## Quick Stats Table

| Dataset | Manifest | Included | On Disk | Missing | PepSet Leak |
|---------|----------|----------|---------|---------|-------------|
| pdb_2024_2026 (Recent Complexes) | 1026 | 646 | 1009 | 0 | ✅ |
| ppii_enriched (PPII-Enriched, relaxed filter) | 337 | 74 | 324 | 0 | ✅ |

---

## Next Actions (Priority Order)

### Before Tuesday (do now on Mac)
1. ✅ BindingDB zip downloaded → `datasets/cache/bindingdb_all.zip` (581 MB, 8.81 GB uncompressed)
2. ✅ CIF retry complete → 252 more pdb_2024_2026 structures recovered (646 total included)
3. ✅ PPII filter relaxed → 29→74 included structures
4. ✅ PepSet IDs file created → `datasets/pepset/pepset_ids.txt` (21 IDs)
5. ⚠️ **Fix bindingdb_calibration_join.py** → see "BUG: BindingDB join captures small molecules" below
6. 🔲 Run `bindingdb_calibration_join.py` (needs score-env + rdkit)
7. 🔲 Verify PepSet structures for all 10 test complexes

### Tuesday on RTX Machine
1. **First**: Copy `datasets/cache/bindingdb_all.zip` to Linux machine
2. **Pre-flight**: Run §2 checklist from accuracy_improvement_plan.md
3. **Data prep**: Install rdkit in score-env if missing: `pip install rdkit`
4. **Run**: `conda run -n score-env python scripts/bindingdb_calibration_join.py`
   - With the 8.81 GB database, expect ~15 min for first-run processing
5. **Tier 0.1**: Run finetune_and_compare.sh (3 hrs GPU)
6. **Tier 0.4**: Production recalibration

---

## BUG: BindingDB Join Captures Small-Molecule Structures

**Problem**: `training_complexes_expanded.csv` has 39/42 rows with empty peptide_sequence.
These are protein-small molecule complexes (HIV protease inhibitors, etc.), NOT peptide complexes.

**Root cause**: The SMILES→sequence parser in `_smiles_to_sequence()` correctly fails to parse
small molecule SMILES as amino acid sequences. But the script doesn't exclude these rows —
it keeps them with empty sequences, which then fail validation gates.

**Fix** (add to `bindingdb_calibration_join.py`):
```python
# After SMILES→sequence attempt, add:
if not peptide_sequence or len(peptide_sequence) < 5:
    # Verify via RCSB that the PDB has a short polymer chain
    peptide_sequence = _fetch_peptide_seq_from_rcsb(pdb_id)

# Add function:
def _fetch_peptide_seq_from_rcsb(pdb_id: str) -> str | None:
    # Query RCSB GraphQL for polymer entities with len 5-30
    # Return sequence if found, None otherwise
    ...

# After fix: expected rows with valid sequence ~84 (from BindingDB scan)
# Strategy: supplement with PepBDB or manual entries to reach 100+
```

**BindingDB actual peptide coverage**: Only ~84 of 3.17M rows are genuine
peptide-protein pairs with PDB structures AND affinity data.
Plan's 200-row target requires supplementation from:
- PepBDB (https://huanglab.phys.hust.edu.cn/pepbdb/)
- AffinityBench peptide subset
- Manual curation from literature

---

## File Inventory

| File | Status | Notes |
|------|--------|-------|
| datasets/cache/bindingdb_all.zip | ✅ 581 MB | 8.81 GB uncompressed, ready for processing |
| datasets/pdb_2024_2026/manifest.csv | ✅ 1026 rows | 646 included (was 394) |
| datasets/pdb_2024_2026/structures/ | ✅ ~1009 files | Mix of .pdb.gz and CIF-converted |
| datasets/ppii_enriched/manifest.csv | ✅ 337 rows | 74 included (was 29, relaxed filter) |
| datasets/ppii_enriched/structures/ | ✅ 324 files | |
| datasets/raw_pdbs/ | ✅ 31 files | Core training + test complex PDBs |
| datasets/pepset/pepset_ids.txt | ✅ 21 IDs | Created this session |
| data/training_complexes.csv | ✅ 6 rows | All have sequences + pKd |
| data/training_complexes_expanded.csv | ⚠️ 42 rows | Only 3/42 have valid sequences |
| data/calibration.json | ⚠️ | α=0.10 at lower bound |
| datasets/training_expanded_structures/ | ✅ ~40 files | Downloaded by fetch_expanded_sequences.py |
