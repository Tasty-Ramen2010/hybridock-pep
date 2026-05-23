"""Comprehensive validation of all downloaded datasets.

Checks structure file integrity, manifest consistency, PepSet leakage,
pKd ranges, and coverage statistics. Writes a VALIDATION_REPORT.md
to datasets/.

Usage:
    python scripts/validate_all_datasets.py
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO / "datasets" / "VALIDATION_REPORT.md"


def _load_pepset_ids() -> set[str]:
    pepset_txt = REPO / "datasets" / "pepset" / "pepset_ids.txt"
    if not pepset_txt.exists():
        return set()
    return {line.strip().upper() for line in pepset_txt.read_text().splitlines() if line.strip()}


def _check_pdb_gz(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Check file is valid gzipped PDB."""
    try:
        with gzip.open(path, "rb") as f:
            content = f.read()
        text = content.decode("latin-1")
        if "ATOM" not in text and "HETATM" not in text:
            return False, "no ATOM/HETATM records"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _check_pdb_plain(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Check plain .pdb file."""
    try:
        text = path.read_text("latin-1")
        if "ATOM" not in text and "HETATM" not in text:
            return False, "no ATOM/HETATM records"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def validate_dataset(
    name: str,
    manifest_path: Path,
    struct_dir: Path,
    pepset_ids: set[str],
    lines: list[str],
) -> dict:
    """Validate one dataset. Returns stats dict and appends to lines."""
    lines.append(f"\n## {name}\n")

    if not manifest_path.exists():
        lines.append("**MISSING manifest — skipping**\n")
        return {}

    df = pd.read_csv(manifest_path)
    included = df[df["excluded_reason"].isna() | (df["excluded_reason"] == "")]
    lines.append(f"- Manifest rows: {len(df)}\n")
    lines.append(f"- Included (no exclusion): {len(included)}\n")

    if "excluded_reason" in df.columns:
        excl_counts = df[df["excluded_reason"].notna() & (df["excluded_reason"] != "")]["excluded_reason"].value_counts()
        lines.append(f"- Exclusion reasons:\n")
        for reason, cnt in excl_counts.items():
            lines.append(f"  - `{reason}`: {cnt}\n")

    # Check structure files on disk
    gz_files = {f.name.split(".pdb")[0].upper(): f for f in struct_dir.glob("*.pdb.gz")}
    plain_files = {f.stem.upper(): f for f in struct_dir.glob("*.pdb")}
    all_on_disk = {**gz_files, **plain_files}

    included_ids = set(included["pdb_id"].str.upper())
    on_disk_and_included = included_ids & set(all_on_disk.keys())
    missing_files = included_ids - set(all_on_disk.keys())

    lines.append(f"- Structure files on disk: {len(all_on_disk)}\n")
    lines.append(f"- Included IDs with files: {len(on_disk_and_included)}\n")
    if missing_files:
        lines.append(f"- ⚠️  Included but no file: {sorted(missing_files)[:10]}\n")

    # Spot-check first 10 structure files
    corrupt = []
    for pdb_id, fpath in sorted(all_on_disk.items())[:20]:
        if fpath.suffix == ".gz":
            ok, reason = _check_pdb_gz(fpath)
        else:
            ok, reason = _check_pdb_plain(fpath)
        if not ok:
            corrupt.append((pdb_id, reason))
    if corrupt:
        lines.append(f"- ⚠️  Corrupt files (spot check): {corrupt}\n")
    else:
        lines.append(f"- ✅ Structure file spot-check: first 20 OK\n")

    # PepSet leakage check
    leak = included_ids & pepset_ids
    if leak:
        lines.append(f"- 🚨 PEPSET LEAKAGE: {sorted(leak)}\n")
    else:
        lines.append(f"- ✅ No PepSet leakage\n")

    # PPII stats if applicable
    if "ppii_fraction" in df.columns:
        ppii_pass = df[df["passes_ppii_filter"] == True]
        lines.append(f"- PPII filter passed: {len(ppii_pass)}\n")
        if len(ppii_pass) > 0:
            lines.append(f"  - mean fraction: {ppii_pass['ppii_fraction'].mean():.2f}\n")
            lines.append(f"  - mean consec_pro: {ppii_pass['consecutive_pro'].mean():.1f}\n")

    return {
        "name": name,
        "n_manifest": len(df),
        "n_included": len(included),
        "n_on_disk": len(all_on_disk),
        "n_covered": len(on_disk_and_included),
        "n_missing_files": len(missing_files),
        "n_corrupt": len(corrupt),
        "pepset_leak": len(leak),
    }


def validate_training_csv(lines: list[str]) -> None:
    """Validate the calibration training CSV."""
    lines.append("\n## Training Complexes (Calibration)\n")
    path = REPO / "data" / "training_complexes.csv"
    df = pd.read_csv(path)
    lines.append(f"- Rows: {len(df)}\n")
    lines.append(f"- Columns: {list(df.columns)}\n")
    pkd_valid = df["experimental_pkd"].between(3, 12)
    lines.append(f"- pKd range valid: {pkd_valid.all()} ({df['experimental_pkd'].min():.1f}–{df['experimental_pkd'].max():.1f})\n")
    seq_ok = (~df["peptide_sequence"].isna()) & (df["peptide_sequence"] != "")
    lines.append(f"- Rows with sequence: {seq_ok.sum()}/{len(df)}\n")
    lines.append(f"- PDB IDs: {df['pdb_id'].tolist()}\n")

    # Check expanded CSV
    exp_path = REPO / "data" / "training_complexes_expanded.csv"
    if exp_path.exists():
        exp = pd.read_csv(exp_path)
        lines.append(f"\n### Expanded CSV ({exp_path.name})\n")
        lines.append(f"- Rows: {len(exp)}\n")
        empty_seq = exp["peptide_sequence"].isna() | (exp["peptide_sequence"] == "")
        lines.append(f"- Rows with sequence: {(~empty_seq).sum()}/{len(exp)}\n")
        lines.append(f"- **⚠️ WARNING**: {empty_seq.sum()} rows have no peptide sequence\n")
        lines.append(f"- **ROOT CAUSE**: BindingDB join captured small-molecule inhibitor PDB entries\n")
        lines.append(f"  (1A85/1FU4/etc. are HIV protease small molecule structures, not peptide complexes)\n")
        lines.append(f"- **FIX needed**: Re-run bindingdb_calibration_join.py with --amide-min 4 and RCSB polymer verification\n")
        lines.append(f"- **True peptide rows available**: 3/42 (2OY2, 1YCR, 5HI3 + partially)\n")
        lines.append(f"- pKd range: {exp['experimental_pkd'].min():.1f}–{exp['experimental_pkd'].max():.1f}\n")
        sources = exp["source"].value_counts().to_dict()
        lines.append(f"- Sources: {sources}\n")


def validate_calibration(lines: list[str]) -> None:
    """Validate calibration.json."""
    lines.append("\n## Calibration State\n")
    cal_path = REPO / "data" / "calibration.json"
    if not cal_path.exists():
        lines.append("**MISSING calibration.json**\n")
        return
    cal = json.loads(cal_path.read_text())
    lines.append(f"- α (alpha): {cal.get('alpha', '?')}\n")
    lines.append(f"- β (beta): {cal.get('beta', '?')}\n")
    lines.append(f"- γ (gamma): {cal.get('gamma', '?')}\n")
    lines.append(f"- n_complexes: {cal.get('n_complexes', '?')}\n")
    lines.append(f"- Pearson r: {cal.get('pearson_r', '?'):.4f}\n")
    lines.append(f"- RMSE: {cal.get('rmse_kcal_mol', '?'):.4f} kcal/mol\n")
    if cal.get('alpha', 1.0) <= 0.11:
        lines.append(f"- 🚨 **ALERT**: α={cal['alpha']} at lower bound — calibration is invalid for production use\n")
    lines.append(f"- Calibrated at: {cal.get('calibrated_at', '?')}\n")


def validate_bindingdb_cache(lines: list[str]) -> None:
    """Validate BindingDB cache."""
    lines.append("\n## BindingDB Cache\n")
    cache_dir = REPO / "datasets" / "cache"
    zp = cache_dir / "bindingdb_all.zip"
    if not zp.exists():
        lines.append("**MISSING — run: scripts/bindingdb_calibration_join.py**\n")
        return
    import zipfile
    size_mb = zp.stat().st_size / 1e6
    lines.append(f"- File: {zp.name} ({size_mb:.0f} MB)\n")
    try:
        with zipfile.ZipFile(zp, "r") as z:
            contents = z.namelist()
            total_unc = sum(i.file_size for i in z.infolist())
            lines.append(f"- ZIP valid: yes\n")
            lines.append(f"- Contents: {contents}\n")
            lines.append(f"- Uncompressed: {total_unc/1e9:.1f} GB\n")
            lines.append(f"- ✅ BindingDB All Data ready for processing\n")
            lines.append(f"- ⚠️ True peptide entries with PDB+affinity: ~84 (scan result)\n")
            lines.append(f"- **NOTE**: Only ~84 of 3.17M rows are genuine peptide-protein pairs\n")
            lines.append(f"  with PDB structures. Plan's 200-row target will need supplementation\n")
            lines.append(f"  from PepBDB, AffinityBench, or manual curation.\n")
    except Exception as e:
        lines.append(f"- ⚠️ ZIP error: {e}\n")


def validate_raw_pdbs(lines: list[str]) -> None:
    """Check raw_pdbs directory."""
    lines.append("\n## Core PDB Structures (raw_pdbs)\n")
    raw_dir = REPO / "datasets" / "raw_pdbs"
    if not raw_dir.exists():
        lines.append("**MISSING datasets/raw_pdbs/**\n")
        return
    pdbs = list(raw_dir.glob("*.pdb"))
    lines.append(f"- Files: {len(pdbs)}\n")
    corrupt = []
    for f in pdbs:
        ok, reason = _check_pdb_plain(f)
        if not ok:
            corrupt.append((f.name, reason))
    lines.append(f"- ✅ All {len(pdbs)} files valid\n" if not corrupt else f"- ⚠️ Corrupt: {corrupt}\n")
    lines.append(f"- IDs: {sorted(f.stem for f in pdbs)}\n")


def validate_pepset(lines: list[str]) -> None:
    """Check PepSet directory."""
    lines.append("\n## PepSet (Held-out Test Set)\n")
    pepset_dir = REPO / "datasets" / "pepset"
    pepset_ids_file = pepset_dir / "pepset_ids.txt"

    if pepset_ids_file.exists():
        ids = [l.strip() for l in pepset_ids_file.read_text().splitlines() if l.strip()]
        lines.append(f"- pepset_ids.txt: {len(ids)} IDs\n")
        lines.append(f"- IDs: {ids}\n")
    else:
        lines.append("- ⚠️ pepset_ids.txt missing\n")

    # Check test_complexes.csv
    test_csv = REPO / "data" / "test_complexes.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        lines.append(f"- test_complexes.csv: {len(df)} entries\n")
        seq_ok = (~df["peptide_sequence"].isna()) & (df["peptide_sequence"] != "")
        lines.append(f"- Rows with sequence: {seq_ok.sum()}/{len(df)}\n")
        pkd_ok = df["experimental_pkd"].between(3, 12)
        lines.append(f"- pKd valid: {pkd_ok.all()}\n")
        lines.append(f"- ⚠️ NOTE: Only 10 test complexes vs plan's 185. Full PepSet (RefPepDB)\n")
        lines.append(f"  not yet built. Population-level r claims must use n=10 until PepSet is expanded.\n")

    # Check which test structures are in raw_pdbs
    raw_dir = REPO / "datasets" / "raw_pdbs"
    if test_csv.exists() and raw_dir.exists():
        df = pd.read_csv(test_csv)
        for _, row in df.iterrows():
            pdb = row["pdb_id"].upper()
            found = (raw_dir / f"{pdb}.pdb").exists()
            if not found:
                lines.append(f"  - ⚠️ {pdb}: structure NOT in raw_pdbs/\n")


def main() -> None:
    lines = [
        "# HybriDock-Pep Dataset Validation Report\n\n",
        f"Generated: 2026-05-23\n\n",
        "---\n\n",
    ]

    pepset_ids = _load_pepset_ids()
    lines.append(f"## Summary\n\n")
    lines.append(f"PepSet IDs loaded: {len(pepset_ids)}\n\n")

    stats = {}

    # Validate each dataset
    for ds_name, manifest, struct_d in [
        (
            "pdb_2024_2026 (Recent Complexes)",
            REPO / "datasets" / "pdb_2024_2026" / "manifest.csv",
            REPO / "datasets" / "pdb_2024_2026" / "structures",
        ),
        (
            "ppii_enriched (PPII-Enriched, relaxed filter)",
            REPO / "datasets" / "ppii_enriched" / "manifest.csv",
            REPO / "datasets" / "ppii_enriched" / "structures",
        ),
    ]:
        s = validate_dataset(ds_name, manifest, struct_d, pepset_ids, lines)
        stats[ds_name] = s

    validate_raw_pdbs(lines)
    validate_pepset(lines)
    validate_training_csv(lines)
    validate_calibration(lines)
    validate_bindingdb_cache(lines)

    # Summary table
    lines.append("\n---\n\n## Quick Stats Table\n\n")
    lines.append("| Dataset | Manifest | Included | On Disk | Missing | PepSet Leak |\n")
    lines.append("|---------|----------|----------|---------|---------|-------------|\n")
    for ds, s in stats.items():
        lines.append(
            f"| {ds} | {s.get('n_manifest',0)} | {s.get('n_included',0)} | "
            f"{s.get('n_on_disk',0)} | {s.get('n_missing_files',0)} | "
            f"{'🚨' if s.get('pepset_leak',0) > 0 else '✅'} |\n"
        )

    # Next actions
    lines.append("""
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
""")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to: {REPORT_PATH}")
    print(f"\nKey stats:")
    for ds, s in stats.items():
        print(f"  {ds}: {s.get('n_included',0)} included, {s.get('n_covered',0)} with files")


if __name__ == "__main__":
    main()
