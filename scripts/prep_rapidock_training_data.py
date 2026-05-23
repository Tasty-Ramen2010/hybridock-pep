"""Prepare training data for RAPiDock last-layer fine-tuning.

Converts our downloaded PDB structures into the RAPiDock training format:
  datasets/training_formatted/{id}/
    {id}_peptide.pdb           — crystal peptide coordinates
    {id}_protein_pocket.pdb    — receptor pocket (20 Å around peptide)
    {id}_peptide_sequence      — plain-text AA sequence

Also builds:
  datasets/training_formatted/training_data.csv   — CSV for train_lastlayer.py
  datasets/training_formatted/val_data.csv        — 10% held-out validation set

Sources (in priority order, deduped by PDB ID):
  1. datasets/RefPepDB-RecentSet/   — 523 pre-formatted complexes (original training data)
  2. datasets/pdb_2024_2026/        — newly fetched recent complexes (post-2023)
  3. datasets/ppii_enriched/        — PPII-enriched complexes (any date)

Usage:
    conda run --no-capture-output -n score-env \\
        python scripts/prep_rapidock_training_data.py [--max-workers 8]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.NeighborSearch import NeighborSearch

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
REFPEPDB_DIR = REPO / "datasets" / "RefPepDB-RecentSet"
RECENT_DIR = REPO / "datasets" / "pdb_2024_2026"
PPII_DIR = REPO / "datasets" / "ppii_enriched"
OUT_DIR = REPO / "datasets" / "training_formatted"

POCKET_THRESHOLD = 20.0   # Å — far cutoff for pocket
POCKET_KEEP = 5.0         # Å — near cutoff (always keep nearby chains)
VAL_FRACTION = 0.10


class _ResidueSelect(Select):
    def __init__(self, residues: set) -> None:
        self.residues = residues

    def accept_residue(self, residue) -> int:  # type: ignore[override]
        return 1 if residue in self.residues else 0


def _write_chain(structure, chain_ids: list[str], out_path: Path) -> None:
    """Write a subset of chains from a structure to a PDB file."""
    io_obj = PDBIO()
    io_obj.set_structure(structure)

    class ChainSelect(Select):
        def accept_chain(self, chain) -> int:
            return 1 if chain.id in chain_ids else 0

    io_obj.save(str(out_path), ChainSelect())


def _write_pocket(
    structure, peptide_chain_id: str, receptor_chain_ids: list[str], out_path: Path
) -> None:
    """Write receptor pocket residues within POCKET_THRESHOLD of any peptide atom."""
    # Collect peptide coordinates
    pep_coords = []
    for model in structure:
        for chain in model:
            if chain.id == peptide_chain_id:
                pep_coords.extend(atom.get_coord() for atom in chain.get_atoms())
        break

    if not pep_coords:
        _log.warning("No peptide coords for pocket extraction")
        return

    # Build NeighborSearch on receptor atoms only
    rec_atoms = []
    for model in structure:
        for chain in model:
            if chain.id in receptor_chain_ids:
                rec_atoms.extend(chain.get_atoms())
        break

    ns = NeighborSearch(rec_atoms)

    pocket_residues: set = set()
    for coord in pep_coords:
        for atom in ns.search(coord, POCKET_THRESHOLD, level="A"):
            pocket_residues.add(atom.get_parent())

    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(str(out_path), _ResidueSelect(pocket_residues))


def _read_pdb_gz(path: Path) -> str:
    """Read a .pdb or .pdb.gz file and return text."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as fh:
            return fh.read()
    return path.read_text(errors="replace")


def _get_sequence_from_chain(structure, chain_id: str) -> str:
    """Return one-letter AA sequence for a chain (standard residues only)."""
    from Bio.PDB.Polypeptide import PPBuilder
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                peptides = ppb.build_peptides(chain, aa_only=False)
                return "".join(str(pp.get_sequence()) for pp in peptides)
    return ""


def prep_new_complex(
    pdb_id: str,
    pdb_gz_path: Path,
    pep_chain: str,
    rec_chain: str,
) -> tuple[str, bool, str]:
    """Process one newly-fetched complex. Returns (pdb_id, success, reason)."""
    out_dir = OUT_DIR / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    peptide_out = out_dir / f"{pdb_id}_peptide.pdb"
    pocket_out = out_dir / f"{pdb_id}_protein_pocket.pdb"
    seq_out = out_dir / f"{pdb_id}_peptide_sequence"

    # Skip if already done
    if peptide_out.exists() and pocket_out.exists() and seq_out.exists():
        if peptide_out.stat().st_size > 0 and pocket_out.stat().st_size > 0:
            return pdb_id, True, "cached"

    try:
        pdb_text = _read_pdb_gz(pdb_gz_path)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))
    except Exception as exc:
        return pdb_id, False, f"parse_error: {exc}"

    # Determine all receptor chains (everything except the peptide chain)
    all_chains = [ch.id for model in structure for ch in model]
    rec_chains = [c for c in all_chains if c != pep_chain]

    if not rec_chains:
        return pdb_id, False, "no_receptor_chains"

    try:
        _write_chain(structure, [pep_chain], peptide_out)
    except Exception as exc:
        return pdb_id, False, f"peptide_write_error: {exc}"

    try:
        _write_pocket(structure, pep_chain, rec_chains, pocket_out)
    except Exception as exc:
        return pdb_id, False, f"pocket_write_error: {exc}"

    if pocket_out.stat().st_size == 0:
        return pdb_id, False, "empty_pocket"

    seq = _get_sequence_from_chain(structure, pep_chain)
    if not seq:
        return pdb_id, False, "empty_sequence"

    seq_out.write_text(seq)
    return pdb_id, True, "ok"


def collect_refpepdb_entries() -> list[dict]:
    """Return CSV rows for all pre-formatted RefPepDB entries."""
    rows = []
    if not REFPEPDB_DIR.exists():
        _log.warning(
            "RefPepDB-RecentSet not found at %s — proceeding with new entries only. "
            "Expected count will drop from ~925 to ~%d. To restore RefPepDB-RecentSet, "
            "clone from https://github.com/DMCB-GIST/RefPepDB and place under "
            "datasets/RefPepDB-RecentSet/",
            REFPEPDB_DIR, 925 - 523,
        )
        return rows
    for entry_dir in sorted(REFPEPDB_DIR.iterdir()):
        if not entry_dir.is_dir():
            continue
        pdb_id = entry_dir.name.upper()
        pocket = entry_dir / f"{entry_dir.name}_protein_pocket.pdb"
        peptide = entry_dir / f"{entry_dir.name}_peptide.pdb"
        seq_file = entry_dir / f"{entry_dir.name}_peptide_sequence"
        if pocket.exists() and peptide.exists() and seq_file.exists():
            if pocket.stat().st_size > 0 and peptide.stat().st_size > 0:
                rows.append({
                    "complex_name": pdb_id,
                    "protein_description": str(pocket),
                    "peptide_description": str(peptide),
                    "source": "refpepdb",
                })
    return rows


def collect_new_entries(
    manifest_csv: Path,
    struct_dir: Path,
    source_label: str,
    known_ids: set[str],
    max_workers: int,
) -> list[dict]:
    """Process new complexes from a manifest and return CSV rows."""
    df = pd.read_csv(manifest_csv)
    # Only use passing entries with available structures
    passing = df[df["excluded_reason"].isna()].copy()
    passing = passing[passing["pdb_id"].str.upper().apply(lambda i: i not in known_ids)]

    _log.info("%s: %d entries to process", source_label, len(passing))

    tasks = []
    for _, row in passing.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        gz_path = struct_dir / f"{row['pdb_id']}.pdb.gz"
        if not gz_path.exists():
            _log.debug("Skipping %s — structure file missing", pdb_id)
            continue
        pep_chain = str(row.get("peptide_chain", "")).strip()
        rec_chain = str(row.get("receptor_chain", "")).strip()
        if not pep_chain or pep_chain == "nan":
            _log.debug("Skipping %s — no peptide chain in manifest", pdb_id)
            continue
        tasks.append((pdb_id, gz_path, pep_chain, rec_chain))

    rows: list[dict] = []
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(prep_new_complex, pdb_id, gz_path, pep_chain, rec_chain): pdb_id
            for pdb_id, gz_path, pep_chain, rec_chain in tasks
        }
        for fut in as_completed(futures):
            pdb_id, success, reason = fut.result()
            if success:
                ok += 1
                out_dir = OUT_DIR / pdb_id
                rows.append({
                    "complex_name": pdb_id,
                    "protein_description": str(out_dir / f"{pdb_id}_protein_pocket.pdb"),
                    "peptide_description": str(out_dir / f"{pdb_id}_peptide.pdb"),
                    "source": source_label,
                })
            else:
                fail += 1
                _log.debug("%s failed: %s", pdb_id, reason)

    _log.info("%s: %d succeeded, %d failed", source_label, ok, fail)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # 1. RefPepDB (pre-formatted, no conversion needed)
    _log.info("Collecting RefPepDB-RecentSet entries...")
    refpepdb_rows = collect_refpepdb_entries()
    _log.info("RefPepDB: %d entries", len(refpepdb_rows))
    known_ids = {r["complex_name"] for r in refpepdb_rows}

    # 2. Recent 2024-2026 complexes
    _log.info("Processing recent 2024-2026 complexes...")
    recent_rows = collect_new_entries(
        RECENT_DIR / "manifest.csv",
        RECENT_DIR / "structures",
        "recent_2024_2026",
        known_ids,
        args.max_workers,
    )
    known_ids |= {r["complex_name"] for r in recent_rows}

    # 3. PPII-enriched (skip duplicates already in RefPepDB or recent)
    _log.info("Processing PPII-enriched complexes...")
    ppii_rows = collect_new_entries(
        PPII_DIR / "manifest.csv",
        PPII_DIR / "structures",
        "ppii_enriched",
        known_ids,
        args.max_workers,
    )

    all_rows = refpepdb_rows + recent_rows + ppii_rows
    _log.info("Total training complexes: %d", len(all_rows))

    # Shuffle and split train/val
    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_fraction))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    _log.info("Train: %d  Val: %d", len(train_rows), len(val_rows))

    fieldnames = ["complex_name", "protein_description", "peptide_description", "source"]
    for rows, name in [(train_rows, "training_data.csv"), (val_rows, "val_data.csv")]:
        out_csv = OUT_DIR / name
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in fieldnames})
        _log.info("Wrote %s (%d rows)", out_csv, len(rows))

    # Summary by source
    import collections
    src_counts = collections.Counter(r["source"] for r in all_rows)
    _log.info("Source breakdown: %s", dict(src_counts))


if __name__ == "__main__":
    main()
