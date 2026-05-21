"""Build inference_input.csv for the PepSet benchmark.

Produces datasets/pepset/inference_input.csv with columns:
  complex_name, protein_description, peptide_description

protein_description  = absolute path to {id}_rec_unbound_pocket.pdb (apo pocket)
peptide_description  = sequence string from {id}_peptide_sequence
                       (sequence, not crystal PDB, so RAPiDock poses from scratch)

The crystal reference for RMSD is {id}_pep_ref.pdb — kept separate from inference.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PEPSET = REPO / "datasets" / "pepset"
OUT = PEPSET / "inference_input.csv"

rows = []
skipped = []

for entry_dir in sorted(PEPSET.iterdir()):
    if not entry_dir.is_dir():
        continue
    pdb_id = entry_dir.name
    pocket = entry_dir / f"{pdb_id}_rec_unbound_pocket.pdb"
    ref_pep = entry_dir / f"{pdb_id}_pep_ref.pdb"
    seq_file = entry_dir / f"{pdb_id}_peptide_sequence"

    if not (pocket.exists() and ref_pep.exists() and seq_file.exists()):
        skipped.append(pdb_id)
        continue
    if pocket.stat().st_size == 0 or ref_pep.stat().st_size == 0:
        skipped.append(pdb_id)
        continue

    seq = seq_file.read_text().strip()
    if not seq:
        skipped.append(pdb_id)
        continue

    rows.append({
        "complex_name": pdb_id,
        "protein_description": str(pocket.resolve()),
        "peptide_description": seq,          # sequence string, not PDB path
        "crystal_ref": str(ref_pep.resolve()),  # for RMSD only, not passed to inference.py
    })

with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["complex_name", "protein_description", "peptide_description", "crystal_ref"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} entries to {OUT}")
if skipped:
    print(f"Skipped {len(skipped)}: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
