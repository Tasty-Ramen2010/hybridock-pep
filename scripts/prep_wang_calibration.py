#!/usr/bin/env python3
"""prep_wang_calibration.py — Build calibration dataset from Wang et al. PpI benchmark.

Pipeline
--------
1.  Parse SM_TableS1.xls (Wang et al. PpI benchmark) with quality filters:
      • X-ray structure only
      • Direct Kd measurement (ITC only by default; --all-kd-methods widens to SPR/FP/etc.)
      • Peptide 5–25 AA (from spreadsheet sequence column)
      • Resolution ≤ 3.0 Å
      • Exclude Table S2 entries by default (those become the held-out test set)
2.  Download PDB files from RCSB to datasets/raw_pdbs/ (8-thread parallel, idempotent).
3.  For each PDB, identify the peptide chain by sequence matching:
      • Extract per-chain one-letter sequences from ATOM records
      • Match against the known peptide sequence via SequenceMatcher (threshold 0.70)
      • Prefer chains within ±4 residues of the expected peptide length
      • All other chains go into the receptor PDB
4.  Split PDB into:
      datasets/wang_pepset/{PDB_ID}/{PDB_ID}_rec_ref.pdb
      datasets/wang_pepset/{PDB_ID}/{PDB_ID}_pep_ref.pdb
5.  Write outputs:
      data/training_complexes_wang.csv       — train entries (pdb_id, peptide_sequence, pkd)
      data/test_complexes_wang.csv           — held-out test (S2 entries, same schema)
      runs/wang_prep_report.csv              — per-entry status with chain-matching details

After this script:
    conda run --no-capture-output -n score-env python scripts/score_crystal_poses.py \\
        --training-csv data/training_complexes_wang.csv \\
        --pepset-dir datasets/wang_pepset \\
        --output data/training_scores_wang.json

    conda run --no-capture-output -n score-env python scripts/calibrate_alpha.py \\
        --training-csv data/training_complexes_wang.csv \\
        --scores-json data/training_scores_wang.json \\
        --output data/calibration_wang.json

Usage
-----
    conda run --no-capture-output -n score-env \\
        python scripts/prep_wang_calibration.py

    # Include all Kd measurement methods (SPR, FP, NMR titration, etc.) not just ITC:
    python scripts/prep_wang_calibration.py --all-kd-methods

    # Also split and write S2 entries as held-out test set:
    python scripts/prep_wang_calibration.py --split-test-set

    # Dry-run: parse + print report, skip download and PDB splitting:
    python scripts/prep_wang_calibration.py --dry-run

    # Re-split already-downloaded PDBs (skips re-download):
    python scripts/prep_wang_calibration.py --skip-download
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import logging
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

# ── input files ────────────────────────────────────────────────────────────────
S1_XLS = REPO / "SM_TableS1.xls"
S2_XLS = REPO / "SM_TableS2.xls"

# ── output paths ───────────────────────────────────────────────────────────────
RAW_PDBS_DIR = REPO / "datasets" / "raw_pdbs"
PEPSET_DIR = REPO / "datasets" / "wang_pepset"
TRAIN_CSV = REPO / "data" / "training_complexes_wang.csv"
TEST_CSV = REPO / "data" / "test_complexes_wang.csv"
REPORT_CSV = REPO / "runs" / "wang_prep_report.csv"

# ── quality filter constants ───────────────────────────────────────────────────
STRUCT_METHOD_REQUIRED = "X-RAY"
MIN_PEPTIDE_LEN = 5
MAX_PEPTIDE_LEN = 25
MAX_RESOLUTION_ANG = 3.0
ITC_METHODS_ONLY = {"ITC", "Titration Microcalorimetry"}
ALL_DIRECT_KD_METHODS = {
    "ITC", "Titration Microcalorimetry",
    "SPR", "Biolayer interferometry real-time kinetic analysis ",
    "Bio-layer Interferometry (BLI)",
    "Fluorescence polarization", "FP",
    "Fluorescence anisotropy", "Fluorescence spectroscopy",
    "Fluorescence", "Fluorescence titration",
    "Biosensor binding isotherm", "ESI-MS",
}
# NMR titration excluded: reports chemical-shift–derived values, not direct Kd

# ── chain matching constants ───────────────────────────────────────────────────
SEQ_MATCH_THRESHOLD = 0.70   # minimum SequenceMatcher ratio to call a match
LEN_TOLERANCE = 4            # ±residues vs spreadsheet sequence length
DOWNLOAD_WORKERS = 8
DOWNLOAD_SLEEP = 0.12        # seconds between RCSB requests per thread

# ── standard and modified residue translation ──────────────────────────────────
AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common modified/alternate residues
    "MSE": "M",                   # selenomethionine
    "HSD": "H", "HSE": "H", "HSP": "H",  # HIS protonation states (CHARMM)
    "HIE": "H", "HID": "H", "HIP": "H",  # HIS protonation states (AMBER)
    "CSO": "C", "CSD": "C", "CSS": "C", "CME": "C", "OCS": "C",
    "TPO": "T", "SEP": "S", "PTR": "Y",  # phosphorylated
    "MLY": "K", "M3L": "K", "LLP": "K", "KCX": "K",  # modified Lys
    "NEP": "H", "CGU": "E",
    "SEC": "U", "PYL": "O",
    "GLZ": "G", "NLE": "L",       # norleucine, aminomethylene glycine
}


# ── PDB reading ────────────────────────────────────────────────────────────────

def _read_pdb_file(path: Path) -> str:
    """Read PDB file (plain or .gz)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="ignore") as fh:
            return fh.read()
    return path.read_text(errors="ignore")


def _find_raw_pdb(pdb_id: str) -> Path | None:
    """Find downloaded PDB file for pdb_id in RAW_PDBS_DIR."""
    upper = pdb_id.upper()
    lower = pdb_id.lower()
    for name in (f"{upper}.pdb", f"{lower}.pdb", f"{upper}.pdb.gz", f"{lower}.pdb.gz"):
        p = RAW_PDBS_DIR / name
        if p.exists():
            return p
    return None


# ── sequence extraction ────────────────────────────────────────────────────────

def _chain_sequences(pdb_text: str) -> dict[str, str]:
    """Extract per-chain one-letter sequences from ATOM records only.

    Uses ordered unique (resnum, icode) per chain. Modified residues are mapped
    via AA3TO1; unknown three-letter codes → 'X'.

    Args:
        pdb_text: Raw PDB file content.

    Returns:
        Dict mapping chain ID → one-letter sequence string.
    """
    chain_residues: dict[str, OrderedDict[tuple[int, str], str]] = {}

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 27:
            continue
        chain = line[21:22]
        res3 = line[17:20].strip()
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26:27].strip()
        aa1 = AA3TO1.get(res3)
        if aa1 is None:
            aa1 = "X" if res3 else None
        if aa1 is None:
            continue
        if chain not in chain_residues:
            chain_residues[chain] = OrderedDict()
        key = (resnum, icode)
        if key not in chain_residues[chain]:
            chain_residues[chain][key] = aa1

    return {ch: "".join(seq.values()) for ch, seq in chain_residues.items()}


def _match_peptide_chain(
    chain_seqs: dict[str, str],
    target_seq: str,
    len_tolerance: int = LEN_TOLERANCE,
    sim_threshold: float = SEQ_MATCH_THRESHOLD,
) -> tuple[str | None, float, str]:
    """Identify which PDB chain corresponds to the peptide sequence.

    Strategy:
    1. Score every chain against target_seq with SequenceMatcher.
    2. Among chains within ±len_tolerance residues of target length, pick highest score.
    3. Fall back to best across all chains if none in length range.
    4. Return None if best score < sim_threshold.

    Args:
        chain_seqs: Dict of chain_id → one-letter sequence.
        target_seq: Expected peptide sequence from spreadsheet.
        len_tolerance: Allowed residue-count difference.
        sim_threshold: Minimum SequenceMatcher ratio to accept a match.

    Returns:
        (best_chain_id, best_score, note_string)
        best_chain_id is None if score below threshold.
    """
    target_len = len(target_seq)
    all_scores: dict[str, float] = {}

    for ch, seq in chain_seqs.items():
        # Use ratio() for full sequence comparison; quick_ratio() for pre-filter
        score = SequenceMatcher(None, target_seq, seq).ratio()
        all_scores[ch] = score

    if not all_scores:
        return None, 0.0, "no ATOM chains in PDB"

    # Primary: chains in length range
    in_range = {ch: s for ch, s in all_scores.items()
                if abs(len(chain_seqs[ch]) - target_len) <= len_tolerance}

    if in_range:
        best_ch = max(in_range, key=in_range.__getitem__)
        best_score = in_range[best_ch]
        note = (f"len-range match: chain {best_ch} ({len(chain_seqs[best_ch])} res, "
                f"sim={best_score:.3f})")
    else:
        # Fall back: any chain
        best_ch = max(all_scores, key=all_scores.__getitem__)
        best_score = all_scores[best_ch]
        note = (f"fallback (no len-range candidate): chain {best_ch} "
                f"({len(chain_seqs[best_ch])} res, sim={best_score:.3f})")

    # Ambiguity check: is there a close second in the length range?
    second_scores = {ch: s for ch, s in in_range.items() if ch != best_ch}
    if second_scores:
        second_best = max(second_scores.values())
        if abs(best_score - second_best) < 0.05:
            note += f"; AMBIGUOUS (2nd={second_best:.3f})"

    if best_score < sim_threshold:
        return None, best_score, note + " [BELOW THRESHOLD]"

    return best_ch, best_score, note


# ── PDB splitting ──────────────────────────────────────────────────────────────

def _split_pdb(
    pdb_text: str,
    pep_chain: str,
    rec_chains: list[str],
) -> tuple[str, str]:
    """Split PDB text into separate receptor and peptide PDB strings.

    Receptor: HEADER/REMARK/CRYST1 + ATOM records for rec_chains.
    Peptide: ATOM records for pep_chain.
    HETATM and HOH are excluded from both (prepare_receptor/babel handle H-add).

    Args:
        pdb_text: Full PDB file content.
        pep_chain: Chain ID for the peptide.
        rec_chains: Chain IDs for the receptor (all non-pep chains).

    Returns:
        Tuple of (receptor_pdb_text, peptide_pdb_text).
    """
    rec_chains_set = set(rec_chains)
    header_lines: list[str] = []
    rec_atom_lines: list[str] = []
    pep_atom_lines: list[str] = []

    for raw_line in pdb_text.splitlines(keepends=True):
        line = raw_line
        tag = line[:6]

        # Header records go into both files
        if tag in ("HEADER", "TITLE ", "COMPND", "SOURCE", "REMARK", "CRYST1",
                   "ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2", "SCALE3",
                   "MTRIX1", "MTRIX2", "MTRIX3"):
            header_lines.append(line)
            continue

        if tag in ("ATOM  ", "TER   "):
            if len(line) > 21:
                ch = line[21:22]
                if ch == pep_chain and tag == "ATOM  ":
                    pep_atom_lines.append(line)
                elif ch in rec_chains_set:
                    rec_atom_lines.append(line)
            continue
        # HETATM and ANISOU: skip (cleaner input for prepare_receptor)

    rec_pdb = "".join(header_lines) + "".join(rec_atom_lines) + "END\n"
    pep_pdb = "".join(header_lines) + "".join(pep_atom_lines) + "END\n"
    return rec_pdb, pep_pdb


# ── download helper ────────────────────────────────────────────────────────────

def _download_pdb(pdb_id: str) -> tuple[str, str]:
    """Download one PDB from RCSB. Returns (pdb_id, 'ok N KB'/'skip'/'FAIL: reason')."""
    out = RAW_PDBS_DIR / f"{pdb_id.upper()}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return pdb_id, "skip"
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, out)
        time.sleep(DOWNLOAD_SLEEP)
        sz = out.stat().st_size
        if sz < 300:
            out.unlink(missing_ok=True)
            return pdb_id, f"FAIL: tiny file ({sz} bytes)"
        return pdb_id, f"ok ({sz // 1024} KB)"
    except urllib.error.HTTPError as exc:
        return pdb_id, f"FAIL: HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return pdb_id, f"FAIL: {exc}"


def _download_all(pdb_ids: list[str]) -> dict[str, str]:
    """Download all PDB IDs in parallel. Returns {pdb_id: status}."""
    RAW_PDBS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    already = sum(1 for p in pdb_ids if _find_raw_pdb(p) is not None)
    _log.info("PDBs to download: %d  already on disk: %d  fetching: %d",
              len(pdb_ids), already, len(pdb_ids) - already)

    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_pdb, pid): pid for pid in pdb_ids}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            pid, status = fut.result()
            done += 1
            results[pid] = status
            if not status.startswith("skip"):
                _log.info("[%d/%d] %s: %s", done, len(pdb_ids), pid, status)
    return results


# ── Wang XLS parsing ───────────────────────────────────────────────────────────

def _parse_wang_xls(path: Path) -> list[dict[str, Any]]:
    """Parse a Wang et al. SM table XLS. Returns list of row dicts with clean column names.

    Handles the two-level merged header by reading header=2 and fixing the
    duplicate 'Method' and 'Length (AAs)' column names.

    Args:
        path: Path to SM_TableS1.xls or SM_TableS2.xls.

    Returns:
        List of dicts with keys: pdb_id, struct_method, resolution, classification,
        protein, protein_len, sequence, peptide_len, parent, ph, temp, buffer,
        binding_method, kd_um, pkd, reference.
    """
    import pandas as pd

    df = pd.read_excel(path, sheet_name="Sheet1", engine="xlrd", header=2)
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

    # Fix duplicate column names (Method appears twice, Length (AAs) twice)
    cols = df.columns.tolist()
    cols[1] = "Struct_Method"
    cols[7] = "Peptide_Length"
    cols[12] = "Binding_Method"
    df.columns = cols

    rows = []
    for _, row in df.iterrows():
        pdb_id = str(row.get("PDB", "")).strip().upper()
        if not pdb_id or len(pdb_id) != 4:
            continue
        rows.append({
            "pdb_id": pdb_id,
            "struct_method": str(row.get("Struct_Method", "")).strip().upper(),
            "resolution": row.get("Resolution (Å)"),
            "classification": str(row.get("Classification", "")).strip(),
            "protein": str(row.get("Protein", "")).strip(),
            "protein_len": row.get("Length (AAs)"),
            "sequence": str(row.get("Sequence", "")).strip(),
            "peptide_len": row.get("Peptide_Length"),
            "parent": str(row.get("Parent", "")).strip(),
            "ph": row.get("pH"),
            "temp": row.get("Temperature (°C)"),
            "binding_method": str(row.get("Binding_Method", "")).strip(),
            "kd_um": row.get("Kd（μM）"),
            "pkd": row.get("pKd"),
            "reference": str(row.get("Reference", "")).strip(),
        })
    return rows


def _apply_quality_filters(
    rows: list[dict[str, Any]],
    s2_pdbs: set[str],
    *,
    itc_only: bool,
    include_s2: bool,
) -> tuple[list[dict], list[dict]]:
    """Apply quality filters. Returns (train_rows, test_rows).

    Train rows: S1-only (exclude S2) entries passing all filters.
    Test rows: S2 entries passing all filters (only populated if include_s2=True).

    Args:
        rows: Parsed S1 rows.
        s2_pdbs: Set of PDB IDs from S2 (held-out test).
        itc_only: If True, require ITC binding method; else allow all direct-Kd methods.
        include_s2: If True, also populate test_rows for splitting.

    Returns:
        (train_rows, test_rows) — each entry has all original keys.
    """
    import pandas as pd
    accepted_methods = ITC_METHODS_ONLY if itc_only else ALL_DIRECT_KD_METHODS
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    rejected = 0

    for r in rows:
        # ── filter: X-ray only
        if r["struct_method"] != STRUCT_METHOD_REQUIRED:
            rejected += 1
            continue

        # ── filter: valid numeric Kd and pKd
        try:
            kd = float(str(r["kd_um"]).replace(",", "."))
            pkd = float(str(r["pkd"]).replace(",", "."))
        except (ValueError, TypeError):
            rejected += 1
            continue
        if kd <= 0 or pkd <= 0:
            rejected += 1
            continue

        # ── filter: binding method
        if r["binding_method"] not in accepted_methods:
            rejected += 1
            continue

        # ── filter: peptide sequence validity (only standard + X)
        seq = r["sequence"].upper().replace(" ", "")
        if not seq or not all(c in "ACDEFGHIKLMNPQRSTVWXY" for c in seq):
            rejected += 1
            continue

        # ── filter: peptide length from sequence (not spreadsheet column)
        seq_len = len(seq)
        if not (MIN_PEPTIDE_LEN <= seq_len <= MAX_PEPTIDE_LEN):
            rejected += 1
            continue

        # ── filter: resolution
        try:
            res = float(str(r["resolution"]).replace(",", "."))
        except (ValueError, TypeError):
            rejected += 1
            continue
        if res > MAX_RESOLUTION_ANG or res <= 0:
            rejected += 1
            continue

        r["sequence"] = seq          # normalised
        r["kd_um"] = kd
        r["pkd"] = pkd
        r["resolution"] = res

        if r["pdb_id"] in s2_pdbs:
            if include_s2:
                test_rows.append(r)
        else:
            train_rows.append(r)

    _log.info("Quality filter: %d passed (train=%d, test=%d), %d rejected",
              len(train_rows) + len(test_rows), len(train_rows), len(test_rows), rejected)
    return train_rows, test_rows


# ── per-entry processing ───────────────────────────────────────────────────────

def _process_entry(
    row: dict[str, Any],
    *,
    dry_run: bool,
    split_test_set: bool,
    is_test: bool,
) -> dict[str, Any]:
    """Process one calibration entry: find PDB, match chains, split, write files.

    Args:
        row: Parsed + quality-filtered row dict.
        dry_run: If True, skip PDB splitting (only report).
        split_test_set: If True, write PDB splits for test entries too.
        is_test: True if this is an S2 (held-out test) entry.

    Returns:
        Dict with verdict, chains, similarity, notes for the report CSV.
    """
    pdb_id = row["pdb_id"]
    target_seq = row["sequence"]
    pkd = row["pkd"]
    result: dict[str, Any] = {
        "pdb_id": pdb_id,
        "sequence": target_seq,
        "pkd": pkd,
        "kd_um": row["kd_um"],
        "resolution": row["resolution"],
        "binding_method": row["binding_method"],
        "classification": row["classification"],
        "is_test_set": is_test,
        "pdb_found": False,
        "n_chains": 0,
        "chain_sizes": "",
        "pep_chain": "",
        "rec_chains": "",
        "seq_match_score": 0.0,
        "match_note": "",
        "files_written": False,
        "verdict": "RED",
        "reason": "",
    }

    # ── find PDB ──────────────────────────────────────────────────────────────
    pdb_path = _find_raw_pdb(pdb_id)
    if pdb_path is None:
        result["reason"] = "PDB file not found in datasets/raw_pdbs/"
        return result

    result["pdb_found"] = True
    pdb_text = _read_pdb_file(pdb_path)

    # ── extract chain sequences ───────────────────────────────────────────────
    chain_seqs = _chain_sequences(pdb_text)
    if not chain_seqs:
        result["reason"] = "no ATOM records found in PDB"
        return result

    result["n_chains"] = len(chain_seqs)
    result["chain_sizes"] = " ".join(
        f"{ch}:{len(seq)}" for ch, seq in sorted(chain_seqs.items())
    )

    # ── single-chain PDB — cannot split ──────────────────────────────────────
    if len(chain_seqs) == 1:
        result["reason"] = "only one ATOM chain — cannot separate receptor from peptide"
        return result

    # ── match peptide chain ───────────────────────────────────────────────────
    pep_chain, sim_score, match_note = _match_peptide_chain(chain_seqs, target_seq)
    result["seq_match_score"] = round(sim_score, 3)
    result["match_note"] = match_note

    if pep_chain is None:
        result["reason"] = (
            f"no chain matches target sequence (best sim={sim_score:.3f} < {SEQ_MATCH_THRESHOLD})"
        )
        return result

    result["pep_chain"] = pep_chain
    rec_chains = sorted(ch for ch in chain_seqs if ch != pep_chain)
    result["rec_chains"] = " ".join(rec_chains)

    # ── validate receptor has substance ──────────────────────────────────────
    rec_total_res = sum(len(chain_seqs[ch]) for ch in rec_chains)
    if rec_total_res < 20:
        result["reason"] = (
            f"receptor chains ({'+'.join(rec_chains)}) have only {rec_total_res} "
            f"ATOM residues — likely misidentified peptide chain"
        )
        return result

    # ── validate peptide chain has correct residue count ─────────────────────
    pep_actual_len = len(chain_seqs[pep_chain])
    if abs(pep_actual_len - len(target_seq)) > LEN_TOLERANCE + 2:
        # FLAG, not RED — still usable but worth flagging
        result["verdict"] = "FLAG"
        result["reason"] = (
            f"length mismatch: expected {len(target_seq)} aa, chain {pep_chain} "
            f"has {pep_actual_len} residues (delta={abs(pep_actual_len - len(target_seq))})"
        )
    else:
        result["verdict"] = "PASS"

    # Ambiguity in match note → downgrade to FLAG if currently PASS
    if "AMBIGUOUS" in match_note and result["verdict"] == "PASS":
        result["verdict"] = "FLAG"
        result["reason"] = "ambiguous chain match: " + match_note

    # ── split and write PDB files ─────────────────────────────────────────────
    should_write = not dry_run and (not is_test or split_test_set)
    if should_write:
        entry_dir = PEPSET_DIR / pdb_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        rec_pdb, pep_pdb = _split_pdb(pdb_text, pep_chain, rec_chains)

        rec_out = entry_dir / f"{pdb_id}_rec_ref.pdb"
        pep_out = entry_dir / f"{pdb_id}_pep_ref.pdb"

        # Validate outputs have content
        if len([l for l in rec_pdb.splitlines() if l.startswith("ATOM")]) < 5:
            result["verdict"] = "RED"
            result["reason"] = "receptor PDB has < 5 ATOM lines after split — aborting write"
            return result
        if len([l for l in pep_pdb.splitlines() if l.startswith("ATOM")]) < 3:
            result["verdict"] = "RED"
            result["reason"] = "peptide PDB has < 3 ATOM lines after split — aborting write"
            return result

        rec_out.write_text(rec_pdb)
        pep_out.write_text(pep_pdb)
        result["files_written"] = True
        _log.debug("[%s] Wrote %s and %s", pdb_id, rec_out.name, pep_out.name)

    return result


# ── report + CSV writing ───────────────────────────────────────────────────────

def _write_report(report_rows: list[dict[str, Any]]) -> None:
    """Write per-entry screening report to REPORT_CSV."""
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id", "verdict", "is_test_set",
        "sequence", "pkd", "kd_um", "resolution",
        "binding_method", "classification",
        "pdb_found", "n_chains", "chain_sizes",
        "pep_chain", "rec_chains",
        "seq_match_score", "match_note",
        "files_written", "reason",
    ]
    with REPORT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)
    _log.info("Report written to %s", REPORT_CSV)


def _write_training_csv(
    report_rows: list[dict[str, Any]],
    *,
    train_csv: Path,
    test_csv: Path,
) -> tuple[int, int]:
    """Write training and test CSVs from PASS/FLAG entries.

    Only entries with verdict PASS or FLAG AND files_written=True are included.

    Args:
        report_rows: Full list of per-entry result dicts.
        train_csv: Output path for training entries (S1 non-S2).
        test_csv: Output path for test entries (S2).

    Returns:
        (n_train, n_test) counts written.
    """
    train_rows = [
        r for r in report_rows
        if r["verdict"] in ("PASS", "FLAG")
        and r["files_written"]
        and not r["is_test_set"]
    ]
    test_rows = [
        r for r in report_rows
        if r["verdict"] in ("PASS", "FLAG")
        and r["files_written"]
        and r["is_test_set"]
    ]

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    with train_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdb_id", "peptide_sequence", "experimental_pkd"])
        writer.writeheader()
        for r in train_rows:
            writer.writerow({
                "pdb_id": r["pdb_id"],
                "peptide_sequence": r["sequence"],
                "experimental_pkd": r["pkd"],
            })

    test_csv.parent.mkdir(parents=True, exist_ok=True)
    with test_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdb_id", "peptide_sequence", "experimental_pkd"])
        writer.writeheader()
        for r in test_rows:
            writer.writerow({
                "pdb_id": r["pdb_id"],
                "peptide_sequence": r["sequence"],
                "experimental_pkd": r["pkd"],
            })

    _log.info("Training CSV: %d entries → %s", len(train_rows), train_csv)
    _log.info("Test CSV: %d entries → %s", len(test_rows), test_csv)
    return len(train_rows), len(test_rows)


# ── main ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prep_wang_calibration",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--s1-xls",
        type=Path,
        default=S1_XLS,
        metavar="PATH",
        help=f"Path to SM_TableS1.xls. Default: {S1_XLS}",
    )
    p.add_argument(
        "--s2-xls",
        type=Path,
        default=S2_XLS,
        metavar="PATH",
        help=f"Path to SM_TableS2.xls (held-out test IDs). Default: {S2_XLS}",
    )
    p.add_argument(
        "--all-kd-methods",
        action="store_true",
        default=False,
        help=(
            "Accept all direct-Kd measurement methods (SPR, FP, fluorescence...), "
            "not just ITC. ITC only by default."
        ),
    )
    p.add_argument(
        "--split-test-set",
        action="store_true",
        default=False,
        help="Also split and write PDB files for S2 (held-out test) entries.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip RCSB download step (use already-downloaded PDBs only).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse + filter + report only. Skip download and PDB splitting.",
    )
    p.add_argument(
        "--pepset-dir",
        type=Path,
        default=PEPSET_DIR,
        metavar="PATH",
        help=f"Output pepset directory. Default: {PEPSET_DIR}",
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=TRAIN_CSV,
        metavar="PATH",
        help=f"Output training CSV. Default: {TRAIN_CSV}",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        metavar="PATH",
        help=f"Output test CSV. Default: {TEST_CSV}",
    )
    p.add_argument(
        "--sim-threshold",
        type=float,
        default=SEQ_MATCH_THRESHOLD,
        metavar="FLOAT",
        help=f"Minimum SequenceMatcher ratio for chain identity. Default: {SEQ_MATCH_THRESHOLD}",
    )
    p.add_argument("-v", "--verbose", action="store_true", default=False)
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the Wang et al. calibration dataset preparation pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # ── patch globals for CLI overrides ──────────────────────────────────────
    global SEQ_MATCH_THRESHOLD
    SEQ_MATCH_THRESHOLD = args.sim_threshold

    # ── 1. Parse Wang S1 ──────────────────────────────────────────────────────
    _log.info("Parsing S1: %s", args.s1_xls)
    if not args.s1_xls.exists():
        _log.error("SM_TableS1.xls not found at %s", args.s1_xls)
        sys.exit(1)
    s1_rows = _parse_wang_xls(args.s1_xls)
    _log.info("S1 raw entries: %d", len(s1_rows))

    # ── 2. Parse Wang S2 (held-out test IDs) ────────────────────────────────
    if args.s2_xls.exists():
        _log.info("Parsing S2 (held-out test IDs): %s", args.s2_xls)
        s2_rows = _parse_wang_xls(args.s2_xls)
        s2_pdbs = {r["pdb_id"] for r in s2_rows}
        _log.info("S2 entries (excluded from training): %d PDB IDs", len(s2_pdbs))
    else:
        _log.warning("S2 XLS not found at %s — no test set separation", args.s2_xls)
        s2_rows = []
        s2_pdbs = set()

    # ── 3. Quality filter ─────────────────────────────────────────────────────
    itc_only = not args.all_kd_methods
    method_desc = "ITC only" if itc_only else "all direct-Kd methods"
    _log.info(
        "Filters: X-ray, %s, peptide %d–%d AA, resolution ≤ %.1f Å",
        method_desc, MIN_PEPTIDE_LEN, MAX_PEPTIDE_LEN, MAX_RESOLUTION_ANG,
    )
    train_rows, test_rows = _apply_quality_filters(
        s1_rows,
        s2_pdbs,
        itc_only=itc_only,
        include_s2=args.split_test_set,
    )
    all_rows_to_process = [(r, False) for r in train_rows] + [(r, True) for r in test_rows]

    if not all_rows_to_process:
        _log.error("No entries passed quality filters — check XLS paths and filters")
        sys.exit(1)

    # ── 4. Download PDB files ─────────────────────────────────────────────────
    if not args.dry_run and not args.skip_download:
        all_pdb_ids = [r["pdb_id"] for r, _ in all_rows_to_process]
        _log.info("Downloading %d PDB files from RCSB...", len(all_pdb_ids))
        dl_results = _download_all(all_pdb_ids)
        fails = [pid for pid, s in dl_results.items() if s.startswith("FAIL")]
        if fails:
            _log.warning("Download failed for %d PDB IDs: %s", len(fails), fails)
    else:
        if args.dry_run:
            _log.info("[dry-run] Skipping downloads")
        else:
            _log.info("[skip-download] Using already-downloaded PDBs only")

    # ── 5. Process entries (chain matching + splitting) ───────────────────────
    _log.info("Processing %d entries (train=%d, test=%d)...",
              len(all_rows_to_process), len(train_rows), len(test_rows))

    report_rows: list[dict[str, Any]] = []
    for row, is_test in all_rows_to_process:
        pdb_id = row["pdb_id"]
        _log.info("[%s] Processing (test=%s)...", pdb_id, is_test)
        try:
            result = _process_entry(
                row,
                dry_run=args.dry_run,
                split_test_set=args.split_test_set,
                is_test=is_test,
            )
        except Exception as exc:  # noqa: BLE001 — catch-all to avoid partial run failures
            _log.exception("[%s] Unhandled error: %s", pdb_id, exc)
            result = {
                "pdb_id": pdb_id,
                "sequence": row["sequence"],
                "pkd": row["pkd"],
                "kd_um": row["kd_um"],
                "resolution": row["resolution"],
                "binding_method": row["binding_method"],
                "classification": row["classification"],
                "is_test_set": is_test,
                "pdb_found": False,
                "n_chains": 0,
                "chain_sizes": "",
                "pep_chain": "",
                "rec_chains": "",
                "seq_match_score": 0.0,
                "match_note": "",
                "files_written": False,
                "verdict": "RED",
                "reason": f"EXCEPTION: {exc}",
            }
        report_rows.append(result)

    # ── 6. Write outputs ──────────────────────────────────────────────────────
    _write_report(report_rows)

    if not args.dry_run:
        n_train, n_test = _write_training_csv(
            report_rows,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
        )
    else:
        _log.info("[dry-run] Skipping CSV output writes")
        n_train = sum(1 for r in report_rows
                      if r["verdict"] in ("PASS", "FLAG") and not r["is_test_set"])
        n_test = sum(1 for r in report_rows
                     if r["verdict"] in ("PASS", "FLAG") and r["is_test_set"])

    # ── 7. Summary ────────────────────────────────────────────────────────────
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in report_rows)
    train_report = [r for r in report_rows if not r["is_test_set"]]
    test_report = [r for r in report_rows if r["is_test_set"]]

    print("\n" + "=" * 65)
    print("WANG CALIBRATION DATASET PREP — SUMMARY")
    print("=" * 65)
    print(f"S1 entries parsed:           {len(s1_rows)}")
    print(f"Held-out S2 IDs:             {len(s2_pdbs)}")
    print(f"Entries processed (train):   {len(train_report)}")
    print(f"Entries processed (test):    {len(test_report)}")
    print()
    print(f"  PASS : {verdicts['PASS']:3d}")
    print(f"  FLAG : {verdicts['FLAG']:3d}  (usable, inspect manually)")
    print(f"  RED  : {verdicts['RED']:3d}  (excluded)")
    print()
    print(f"Training CSV entries ready:  {n_train}  → {args.train_csv}")
    print(f"Test CSV entries ready:      {n_test}  → {args.test_csv}")
    print(f"Report:                      {REPORT_CSV}")

    if args.dry_run:
        print("\n[dry-run mode — no files written except report]")
    else:
        print()
        print("Next step (scoring):")
        print(f"  conda run --no-capture-output -n score-env \\")
        print(f"      python scripts/score_crystal_poses.py \\")
        print(f"          --training-csv {args.train_csv} \\")
        print(f"          --pepset-dir {args.pepset_dir or PEPSET_DIR} \\")
        print(f"          --output data/training_scores_wang.json")
        print()
        print("Then calibrate:")
        print(f"  conda run --no-capture-output -n score-env \\")
        print(f"      python scripts/calibrate_alpha.py \\")
        print(f"          --training-csv {args.train_csv} \\")
        print(f"          --scores-json data/training_scores_wang.json \\")
        print(f"          --output data/calibration_wang.json")

    # RED entries breakdown
    reds = [r for r in report_rows if r["verdict"] == "RED"]
    if reds:
        print(f"\n── RED entries ({len(reds)}) {'─' * 40}")
        for r in sorted(reds, key=lambda x: x["pdb_id"]):
            print(f"  {r['pdb_id']:5s}  {r.get('reason', '')[:80]}")

    # FLAG entries breakdown
    flags = [r for r in report_rows if r["verdict"] == "FLAG"]
    if flags:
        print(f"\n── FLAG entries ({len(flags)}) {'─' * 39}")
        for r in sorted(flags, key=lambda x: x["pdb_id"]):
            print(f"  {r['pdb_id']:5s}  {r.get('reason', '')[:80]}")

    print("=" * 65)


if __name__ == "__main__":
    main()
