"""Preprocess PepPC + PepPC-F datasets into RAPiDock training format.

Converts the DiffPepDock training data (PepPC / PepPC-F) into the split format
expected by train_lastlayer.py:
    datasets/training_formatted_peppc/{complex_id}/
        {complex_id}_peptide.pdb           (crystal peptide, one chain)
        {complex_id}_protein_pocket.pdb    (receptor pocket ≤20 Å from peptide)
        {complex_id}_peptide_sequence      (one-letter AA sequence, plain text)

Output CSVs (same schema as training_formatted/training_data.csv):
    datasets/training_formatted_peppc/peppc_train.csv
    datasets/training_formatted_peppc/peppcf_train.csv
    datasets/training_formatted_peppc/combined_train.csv   (merged with existing)
    datasets/training_formatted_peppc/combined_val.csv

PepPC filename pattern:
    nat_raw_data_final/{lig_chain}_{pdb_id}_{res}_fixed.pdb

PepPC-F filename pattern:
    frag_raw_data_final/{all_chains}_{pdb_id}_{res}_fixed_{start}_{end}_{lig_chain}{len}_{ss}.pdb

Usage (run from project root):
    python scripts/prep_peppc_training_data.py [--max-workers 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import random
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# BioPython
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.NeighborSearch import NeighborSearch
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
NAT_DIR    = REPO / "datasets" / "nat_raw_data_final"
FRAG_DIR   = REPO / "datasets" / "frag_raw_data_final"
OUT_DIR    = REPO / "datasets" / "training_formatted_peppc"
EXIST_CSV  = REPO / "datasets" / "training_formatted" / "training_data.csv"
EXIST_VAL  = REPO / "datasets" / "training_formatted" / "val_data.csv"
PEPPC_TAR  = REPO / "PepPC_raw_data.tar.gz"
PEPPCF_TAR = REPO / "PepPC-F_raw_data.tar.gz"
PEPPC_TRAIN_CSV  = REPO / "third_party" / "DiffPepDock" / "datasets" / "PepPC_before_202201.csv"
PEPPCF_CSV       = REPO / "third_party" / "DiffPepDock" / "datasets" / "PepPC-F_dataset.csv"

POCKET_THRESHOLD = 20.0     # Å far cutoff for pocket selection
MIN_PEP_LEN      = 4        # minimum peptide residues
MAX_PEP_LEN      = 35       # maximum peptide residues
VAL_FRACTION     = 0.10

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine → Met
}


# ── BioPython selectors ───────────────────────────────────────────────────────

class ChainSelect(Select):
    """Keep only atoms from the given chain."""
    def __init__(self, chain_ids: list[str]) -> None:
        self._ids = set(chain_ids)
    def accept_chain(self, chain) -> int:
        return 1 if chain.id in self._ids else 0


class ChainResidueRangeSelect(Select):
    """Keep only ATOM/HETATM records of chain_id within [res_start, res_end]."""
    def __init__(self, chain_id: str, res_start: int, res_end: int) -> None:
        self._chain = chain_id
        self._start = res_start
        self._end   = res_end

    def accept_chain(self, chain) -> int:
        return 1 if chain.id == self._chain else 0

    def accept_residue(self, residue) -> int:
        rid = residue.get_id()[1]   # sequence number (int)
        return 1 if self._start <= rid <= self._end else 0


class ResidueSubsetSelect(Select):
    """Keep only the given Bio.PDB Residue objects."""
    def __init__(self, residues: set) -> None:
        self._residues = residues

    def accept_residue(self, residue) -> int:
        return 1 if residue in self._residues else 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sequence(structure, chain_id: str,
                  res_start: Optional[int] = None,
                  res_end: Optional[int] = None) -> str:
    """One-letter AA sequence for chain_id, optionally filtered by residue range."""
    seq_chars = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain.get_residues():
                hetflag, seqid, icode = res.get_id()
                if hetflag.strip() not in ("", "MSE"):
                    continue   # skip HETATM (water, ligand) — but keep MSE
                if res_start is not None and seqid < res_start:
                    continue
                if res_end is not None and seqid > res_end:
                    continue
                letter = THREE_TO_ONE.get(res.get_resname().strip(), "X")
                seq_chars.append(letter)
        break
    return "".join(seq_chars)


def _extract_pocket(structure, pep_atoms: list, rec_chain_ids: list[str]) -> set:
    """Return pocket residues within POCKET_THRESHOLD Å of any peptide atom."""
    rec_atoms = []
    for model in structure:
        for chain in model:
            if chain.id in rec_chain_ids:
                rec_atoms.extend(chain.get_atoms())
        break
    if not rec_atoms:
        return set()
    ns = NeighborSearch(rec_atoms)
    pocket: set = set()
    for coord in pep_atoms:
        for atom in ns.search(coord, POCKET_THRESHOLD, level="A"):
            pocket.add(atom.get_parent())
    return pocket


def _write_pdb(structure, selector: Select, out_path: Path) -> int:
    """Write structure with selector; return file size in bytes."""
    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(str(out_path), selector)
    return out_path.stat().st_size


# ── PepPC processor ───────────────────────────────────────────────────────────

_PEPPC_RE = re.compile(
    r"^([A-Za-z]+)_([0-9A-Z]{4})_([\d.]+)_fixed\.pdb$", re.IGNORECASE
)


def _process_peppc(pdb_file: Path, out_base: Path) -> tuple[str, bool, str, dict]:
    """Process one PepPC PDB file.

    DiffPepDock's OpenMM preprocessing ALWAYS renames the original ligand chain
    to chain A and the receptor to chain B (and C, D... for multi-chain receptors).
    The filename prefix is the ORIGINAL PDB chain letter (before renaming) and is
    used only for complex_id uniqueness — NOT for chain lookup in the file.

    Returns (complex_id, success, reason, csv_row_dict).
    """
    m = _PEPPC_RE.match(pdb_file.name)
    if not m:
        return ("?", False, f"filename_mismatch: {pdb_file.name}", {})

    orig_chain = m.group(1).upper()   # original chain letter (for complex_id only)
    pdb_id     = m.group(2).upper()
    # After DiffPepDock preprocessing, the ligand is always chain A.
    lig_chain  = "A"
    complex_id = f"peppc_{pdb_id}_{orig_chain}"

    out_dir  = out_base / complex_id
    pep_out  = out_dir / f"{complex_id}_peptide.pdb"
    poc_out  = out_dir / f"{complex_id}_protein_pocket.pdb"
    seq_out  = out_dir / f"{complex_id}_peptide_sequence"

    # Cache — skip if already done
    if pep_out.exists() and poc_out.exists() and seq_out.exists():
        if pep_out.stat().st_size > 0 and poc_out.stat().st_size > 0:
            existing_seq = seq_out.read_text().strip()
            if MIN_PEP_LEN <= len(existing_seq) <= MAX_PEP_LEN:
                row = {
                    "complex_name": complex_id,
                    "protein_description": str(poc_out.resolve()),
                    "peptide_description": str(pep_out.resolve()),
                    "source": "peppc",
                }
                return (complex_id, True, "cached", row)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(complex_id, str(pdb_file))
    except Exception as exc:
        return (complex_id, False, f"parse_error: {exc}", {})

    # Identify all chains in model 0
    all_chain_ids = [ch.id for model in structure for ch in model]
    if not all_chain_ids:
        return (complex_id, False, "no_chains", {})
    if lig_chain not in all_chain_ids:
        return (complex_id, False, f"lig_chain_{lig_chain}_missing_in_pdb", {})
    rec_chain_ids = [c for c in all_chain_ids if c != lig_chain]
    if not rec_chain_ids:
        return (complex_id, False, "no_receptor_chains", {})

    # Sequence check
    seq = _get_sequence(structure, lig_chain)
    if not seq:
        return (complex_id, False, "empty_sequence", {})
    if len(seq) < MIN_PEP_LEN:
        return (complex_id, False, f"too_short_{len(seq)}", {})
    if len(seq) > MAX_PEP_LEN:
        return (complex_id, False, f"too_long_{len(seq)}", {})

    # Collect peptide atom coordinates for pocket extraction
    pep_coords = []
    for model in structure:
        for chain in model:
            if chain.id == lig_chain:
                pep_coords.extend(a.get_coord().tolist() for a in chain.get_atoms())
        break
    if not pep_coords:
        return (complex_id, False, "empty_peptide_atoms", {})

    # Extract receptor pocket
    pocket_residues = _extract_pocket(structure, pep_coords, rec_chain_ids)
    if not pocket_residues:
        return (complex_id, False, "empty_pocket", {})

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _write_pdb(structure, ChainSelect([lig_chain]), pep_out)
    except Exception as exc:
        return (complex_id, False, f"pep_write_error: {exc}", {})
    if pep_out.stat().st_size == 0:
        return (complex_id, False, "empty_peptide_pdb", {})

    try:
        _write_pdb(structure, ResidueSubsetSelect(pocket_residues), poc_out)
    except Exception as exc:
        return (complex_id, False, f"pocket_write_error: {exc}", {})
    if poc_out.stat().st_size == 0:
        return (complex_id, False, "empty_pocket_pdb", {})

    seq_out.write_text(seq)

    row = {
        "complex_name": complex_id,
        "protein_description": str(poc_out.resolve()),
        "peptide_description": str(pep_out.resolve()),
        "source": "peppc",
    }
    return (complex_id, True, "ok", row)


# ── PepPC-F processor ─────────────────────────────────────────────────────────

_PEPPCF_RE = re.compile(
    r"^([A-Za-z]+)_([0-9A-Z]{4})_([\d.]+)_fixed_(\d+)_(\d+)_([A-Za-z])(\d+)_(loop|helix)\.pdb$",
    re.IGNORECASE,
)


def _process_peppcf(pdb_file: Path, out_base: Path) -> tuple[str, bool, str, dict]:
    """Process one PepPC-F PDB file.

    Ligand = chain lig_chain, residues start_res..end_res.
    Receptor = remaining chain(s), pocket within 20 Å of ligand.

    Returns (complex_id, success, reason, csv_row_dict).
    """
    m = _PEPPCF_RE.match(pdb_file.name)
    if not m:
        return ("?", False, f"filename_mismatch: {pdb_file.name}", {})

    all_chains_str = m.group(1).upper()
    pdb_id         = m.group(2).upper()
    start_res      = int(m.group(4))
    end_res        = int(m.group(5))
    lig_chain      = m.group(6).upper()

    complex_id = f"peppcf_{pdb_id}_{lig_chain}_{start_res}_{end_res}"

    out_dir  = out_base / complex_id
    pep_out  = out_dir / f"{complex_id}_peptide.pdb"
    poc_out  = out_dir / f"{complex_id}_protein_pocket.pdb"
    seq_out  = out_dir / f"{complex_id}_peptide_sequence"

    # Cache check
    if pep_out.exists() and poc_out.exists() and seq_out.exists():
        if pep_out.stat().st_size > 0 and poc_out.stat().st_size > 0:
            existing_seq = seq_out.read_text().strip()
            if MIN_PEP_LEN <= len(existing_seq) <= MAX_PEP_LEN:
                row = {
                    "complex_name": complex_id,
                    "protein_description": str(poc_out.resolve()),
                    "peptide_description": str(pep_out.resolve()),
                    "source": "peppcf",
                }
                return (complex_id, True, "cached", row)

    # Receptor chains = everything in all_chains_str except the ligand chain
    rec_chain_ids = [c for c in all_chains_str if c != lig_chain]
    if not rec_chain_ids:
        # Fallback: all other chains in PDB
        rec_chain_ids = None  # resolved below

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(complex_id, str(pdb_file))
    except Exception as exc:
        return (complex_id, False, f"parse_error: {exc}", {})

    all_chain_ids = [ch.id for model in structure for ch in model]
    if lig_chain not in all_chain_ids:
        return (complex_id, False, f"lig_chain_{lig_chain}_missing", {})

    if rec_chain_ids is None:
        rec_chain_ids = [c for c in all_chain_ids if c != lig_chain]
    if not rec_chain_ids:
        return (complex_id, False, "no_receptor_chains", {})

    # Collect peptide residue atoms (within start_res..end_res of lig_chain)
    pep_coords = []
    pep_residues = set()
    for model in structure:
        for chain in model:
            if chain.id != lig_chain:
                continue
            for res in chain.get_residues():
                hetflag, seqid, icode = res.get_id()
                if hetflag.strip() not in ("", "MSE"):
                    continue
                if start_res <= seqid <= end_res:
                    pep_residues.add(res)
                    pep_coords.extend(a.get_coord().tolist() for a in res.get_atoms())
        break

    if not pep_coords:
        return (complex_id, False, "empty_pep_coords_in_range", {})

    seq = _get_sequence(structure, lig_chain, start_res, end_res)
    if not seq:
        return (complex_id, False, "empty_sequence", {})
    if len(seq) < MIN_PEP_LEN:
        return (complex_id, False, f"too_short_{len(seq)}", {})
    if len(seq) > MAX_PEP_LEN:
        return (complex_id, False, f"too_long_{len(seq)}", {})

    pocket_residues = _extract_pocket(structure, pep_coords, rec_chain_ids)
    if not pocket_residues:
        return (complex_id, False, "empty_pocket", {})

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write peptide (only the fragment residues from lig_chain)
    try:
        _write_pdb(structure, ResidueSubsetSelect(pep_residues), pep_out)
    except Exception as exc:
        return (complex_id, False, f"pep_write_error: {exc}", {})
    if pep_out.stat().st_size == 0:
        return (complex_id, False, "empty_pep_pdb", {})

    # Write receptor pocket
    try:
        _write_pdb(structure, ResidueSubsetSelect(pocket_residues), poc_out)
    except Exception as exc:
        return (complex_id, False, f"pocket_write_error: {exc}", {})
    if poc_out.stat().st_size == 0:
        return (complex_id, False, "empty_pocket_pdb", {})

    seq_out.write_text(seq)

    row = {
        "complex_name": complex_id,
        "protein_description": str(poc_out.resolve()),
        "peptide_description": str(pep_out.resolve()),
        "source": "peppcf",
    }
    return (complex_id, True, "ok", row)


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_tar(tar_path: Path, dest_dir: Path) -> int:
    """Extract a tar archive to dest_dir. Returns number of files extracted."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.glob("*.pdb"))
    if existing:
        log.info("  %s already exists with %d PDB files — skipping extraction",
                 dest_dir, len(existing))
        return len(existing)
    log.info("Extracting %s → %s", tar_path.name, dest_dir)
    with tarfile.open(str(tar_path)) as tf:
        # Extract only PDB files, strip directory prefix
        members = [m for m in tf.getmembers() if m.name.endswith(".pdb")]
        count = 0
        for member in members:
            member.name = Path(member.name).name  # strip subdirectory
            tf.extract(member, path=str(dest_dir))
            count += 1
    log.info("  Extracted %d PDB files", count)
    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prep PepPC + PepPC-F for RAPiDock fine-tuning")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Parallel workers for processing (default 16)")
    p.add_argument("--dry-run", action="store_true",
                   help="Process only first 20 complexes from each dataset, then exit")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip tar extraction if directories already exist")
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                   help="Fraction held out for validation (default 0.10)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_parallel(files: list[Path], processor, out_base: Path,
                  max_workers: int) -> tuple[list[dict], dict]:
    """Run processor(pdb_path, out_base) in parallel. Returns (csv_rows, stats)."""
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    n_ok = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(processor, f, out_base): f for f in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                cid, ok, reason, row = fut.result()
            except Exception as exc:
                reasons[f"worker_exception"] = reasons.get("worker_exception", 0) + 1
                log.debug("Worker exception: %s", exc)
                continue
            if ok:
                rows.append(row)
                n_ok += 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
            if done % 500 == 0:
                log.info("  Progress: %d/%d done (%d ok)", done, len(files), n_ok)

    reasons["ok"] = n_ok
    return rows, reasons


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract tarballs ────────────────────────────────────────────
    if not args.skip_extract:
        if PEPPC_TAR.exists():
            _extract_tar(PEPPC_TAR, NAT_DIR)
        else:
            log.warning("PepPC tar not found: %s", PEPPC_TAR)
        if PEPPCF_TAR.exists():
            _extract_tar(PEPPCF_TAR, FRAG_DIR)
        else:
            log.warning("PepPC-F tar not found: %s", PEPPCF_TAR)

    # ── Step 2: Build PepPC file list (training split only) ─────────────────
    log.info("Building PepPC file list from training split CSV...")
    peppc_train_pdb_ids: set[str] = set()
    if PEPPC_TRAIN_CSV.exists():
        df_train = pd.read_csv(PEPPC_TRAIN_CSV)
        peppc_train_pdb_ids = set(df_train["PDB ID"].str.upper())
        log.info("  PepPC training split: %d PDB IDs", len(peppc_train_pdb_ids))
    else:
        log.warning("PepPC_before_202201.csv not found — using ALL PepPC complexes")

    peppc_files: list[Path] = []
    if NAT_DIR.exists():
        for f in sorted(NAT_DIR.glob("*.pdb")):
            m = _PEPPC_RE.match(f.name)
            if not m:
                continue
            pdb_id = m.group(2).upper()
            if peppc_train_pdb_ids and pdb_id not in peppc_train_pdb_ids:
                continue   # exclude test split
            peppc_files.append(f)
    log.info("PepPC files to process: %d", len(peppc_files))

    # ── Step 3: Build PepPC-F file list (all complexes) ─────────────────────
    peppcf_files: list[Path] = []
    if FRAG_DIR.exists():
        peppcf_files = sorted(FRAG_DIR.glob("*.pdb"))
    log.info("PepPC-F files to process: %d", len(peppcf_files))

    # ── Dry-run: truncate to 20 each ─────────────────────────────────────────
    if args.dry_run:
        peppc_files  = peppc_files[:20]
        peppcf_files = peppcf_files[:20]
        log.info("DRY RUN: processing %d PepPC + %d PepPC-F", len(peppc_files), len(peppcf_files))

    # ── Step 4: Process PepPC ────────────────────────────────────────────────
    log.info("Processing PepPC complexes (workers=%d)...", args.max_workers)
    peppc_rows, peppc_stats = _run_parallel(
        peppc_files, _process_peppc, OUT_DIR, args.max_workers
    )
    log.info("PepPC results: %s", peppc_stats)

    peppc_csv = OUT_DIR / "peppc_train.csv"
    _write_csv(peppc_rows, peppc_csv)
    log.info("Wrote %d PepPC rows → %s", len(peppc_rows), peppc_csv)

    # ── Step 5: Process PepPC-F ──────────────────────────────────────────────
    log.info("Processing PepPC-F complexes (workers=%d)...", args.max_workers)
    peppcf_rows, peppcf_stats = _run_parallel(
        peppcf_files, _process_peppcf, OUT_DIR, args.max_workers
    )
    log.info("PepPC-F results: %s", peppcf_stats)

    peppcf_csv = OUT_DIR / "peppcf_train.csv"
    _write_csv(peppcf_rows, peppcf_csv)
    log.info("Wrote %d PepPC-F rows → %s", len(peppcf_rows), peppcf_csv)

    # ── Step 6: Merge with existing training data ────────────────────────────
    all_rows: list[dict] = []

    # Existing training data (deduped by complex_name)
    existing_ids: set[str] = set()
    for csv_path in [EXIST_CSV, EXIST_VAL]:
        if csv_path.exists():
            df_ex = pd.read_csv(csv_path)
            for _, row in df_ex.iterrows():
                cid = str(row["complex_name"])
                if cid not in existing_ids:
                    existing_ids.add(cid)
                    r = row.to_dict()
                    if "source" not in r or str(r.get("source", "nan")) == "nan":
                        r["source"] = "refpepdb"
                    all_rows.append(r)

    # Add new PepPC + PepPC-F (deduped)
    for row in peppc_rows + peppcf_rows:
        if row["complex_name"] not in existing_ids:
            existing_ids.add(row["complex_name"])
            all_rows.append(row)

    log.info("Combined total: %d complexes", len(all_rows))

    # ── Step 7: Train / val split ────────────────────────────────────────────
    # Separate out existing val rows (keep them in val)
    existing_val_ids: set[str] = set()
    if EXIST_VAL.exists():
        df_val = pd.read_csv(EXIST_VAL)
        existing_val_ids = set(df_val["complex_name"].astype(str))

    val_rows:   list[dict] = []
    train_rows: list[dict] = []

    for row in all_rows:
        cid = row["complex_name"]
        if cid in existing_val_ids:
            val_rows.append(row)
        else:
            train_rows.append(row)

    # Add val fraction from new PepPC + PepPC-F rows
    new_rows = [r for r in train_rows
                if r.get("source") in ("peppc", "peppcf")]
    n_new_val = int(len(new_rows) * args.val_fraction)
    random.shuffle(new_rows)
    new_val  = new_rows[:n_new_val]
    new_val_ids = {r["complex_name"] for r in new_val}

    final_val:   list[dict] = val_rows + new_val
    final_train: list[dict] = [r for r in train_rows if r["complex_name"] not in new_val_ids]

    combined_train = OUT_DIR / "combined_train.csv"
    combined_val   = OUT_DIR / "combined_val.csv"
    _write_csv(final_train, combined_train)
    _write_csv(final_val,   combined_val)

    log.info("Final split: %d train / %d val", len(final_train), len(final_val))
    log.info("Train CSV → %s", combined_train)
    log.info("Val CSV   → %s", combined_val)

    # ── Step 8: Sanity checks ────────────────────────────────────────────────
    log.info("\n── Sanity checks ────────────────────────────────────────────")
    _sanity_check(combined_train, n_sample=10)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_ok = peppc_stats.get("ok", 0) + peppcf_stats.get("ok", 0)
    total_attempted = len(peppc_files) + len(peppcf_files)
    log.info("\n══ SUMMARY ══")
    log.info("  PepPC:   %d / %d processed", peppc_stats.get("ok", 0), len(peppc_files))
    log.info("  PepPC-F: %d / %d processed", peppcf_stats.get("ok", 0), len(peppcf_files))
    log.info("  Total success rate: %.1f%%", 100 * total_ok / max(total_attempted, 1))
    log.info("  Train: %d  Val: %d", len(final_train), len(final_val))
    if total_ok / max(total_attempted, 1) < 0.70:
        log.warning("⚠ Success rate below 70%% — check failures above before training")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        log.warning("No rows to write to %s", path)
        return
    fields = ["complex_name", "protein_description", "peptide_description", "source"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _sanity_check(csv_path: Path, n_sample: int = 10) -> None:
    """Verify that n_sample rows in the CSV have valid file paths."""
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return
    df = pd.read_csv(csv_path)
    rows = df.sample(min(n_sample, len(df)), random_state=42)
    n_ok = 0
    for _, row in rows.iterrows():
        p = Path(row["protein_description"])
        l = Path(row["peptide_description"])
        if p.exists() and l.exists() and p.stat().st_size > 0 and l.stat().st_size > 0:
            n_ok += 1
        else:
            log.warning("  MISSING: %s (prot=%s pep=%s)", row["complex_name"], p.exists(), l.exists())
    log.info("  Sanity check %d/%d sampled rows have valid files", n_ok, len(rows))


if __name__ == "__main__":
    main()
