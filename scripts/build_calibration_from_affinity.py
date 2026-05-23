"""Build an expanded calibration CSV from all downloaded structures + affinity data.

Combines:
  - data/rcsb_binding_affinity.csv         (RCSB affinity, already fetched)
  - data/affinity_supplement.csv           (PDBe + ChEMBL + REMARK)
  - data/training_complexes.csv            (original 6 gold-standard entries)
  - data/training_complexes_expanded.csv   (BindingDB join — partial)

Cross-references each PDB ID against all downloaded structure datasets to:
  1. Verify the structure file exists on disk
  2. Extract the peptide sequence directly from the PDB file (no SMILES needed)
  3. Verify pKd is in physical range [3, 12]
  4. Check for PepSet leakage

Writes data/training_complexes_full.csv — the comprehensive calibration set.

Usage:
    python scripts/build_calibration_from_affinity.py
"""
from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

PKD_MIN, PKD_MAX = 3.0, 12.0
PEP_LEN_MIN, PEP_LEN_MAX = 5, 30
REC_LEN_MIN = 50


def _load_pepset() -> set[str]:
    f = DATA_DIR / "pepset_ids.txt"
    if f.exists():
        return {l.strip().upper() for l in f.read_text().splitlines() if l.strip()}
    return set()


def _find_structure(pdb_id: str) -> Path | None:
    """Search all dataset directories for a structure file."""
    search_dirs = [
        REPO / "datasets" / ds
        for ds in [
            "raw_pdbs", "pdb_2024_2026/structures", "ppii_enriched/structures",
            "pdb_2019_2023/structures", "pdb_2010_2018/structures", "pdb_pre2010/structures",
            "family_targeted/structures", "ppii_extended/structures",
            "training_expanded_structures",
        ]
    ]
    uid = pdb_id.upper()
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in [f"{uid}.pdb.gz", f"{uid}.pdb", f"{uid.lower()}.pdb"]:
            p = d / pattern
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def _extract_chains_from_pdb(pdb_path: Path) -> list[tuple[str, str]]:
    """Extract (chain_id, sequence) from a PDB or PDB.gz file.
    Returns list of (chain_id, one_letter_seq) for all chains.
    """
    try:
        if str(pdb_path).endswith(".gz"):
            with gzip.open(pdb_path, "rb") as f:
                text = f.read().decode("latin-1")
        else:
            text = pdb_path.read_text("latin-1")
    except Exception as exc:
        _log.debug("Cannot read %s: %s", pdb_path, exc)
        return []

    # 3-letter → 1-letter AA mapping
    AA3 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        # Non-standard: map to common equivalents
        "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "HIE": "H",
        "HID": "H", "HIP": "H", "CYX": "C", "CYM": "C",
        "TPO": "T", "SEP": "S", "PTR": "Y",  # phosphorylated
        "MLY": "K", "ACE": "X", "NME": "X",
    }

    chains: dict[str, dict[int, str]] = {}  # chain → {resnum: aa}

    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 27:
            continue
        try:
            chain = line[21]
            resnum_str = line[22:26].strip()
            resnum = int(resnum_str) if resnum_str else 0
            resname = line[17:20].strip()
        except (ValueError, IndexError):
            continue
        aa = AA3.get(resname)
        if aa and aa != "X":
            if chain not in chains:
                chains[chain] = {}
            chains[chain][resnum] = aa

    return [(ch, "".join(chains[ch][k] for k in sorted(chains[ch])))
            for ch in sorted(chains)]


def _classify_chains(
    chains: list[tuple[str, str]],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (pep_chain, pep_seq, rec_chain, rec_seq) or Nones if ambiguous.

    Handles:
    - Simple heterodimer: 1 peptide + 1+ receptor
    - Symmetric dimer:    2 peptides + 2 receptors → pick longest of each
    """
    peptide_candidates = [(ch, seq) for ch, seq in chains
                          if PEP_LEN_MIN <= len(seq) <= PEP_LEN_MAX]
    receptor_candidates = [(ch, seq) for ch, seq in chains
                           if len(seq) >= REC_LEN_MIN]

    if not peptide_candidates or not receptor_candidates:
        return None, None, None, None

    # Exactly 1 peptide → simple case
    if len(peptide_candidates) == 1:
        pep_chain, pep_seq = peptide_candidates[0]
        # Pick the longest receptor
        rec_chain, rec_seq = max(receptor_candidates, key=lambda x: len(x[1]))
        return pep_chain, pep_seq, rec_chain, rec_seq

    # 2 peptides + 2 receptors → symmetric homodimer; pick longest of each
    if len(peptide_candidates) == 2 and len(receptor_candidates) >= 2:
        pep_chain, pep_seq = max(peptide_candidates, key=lambda x: len(x[1]))
        rec_chain, rec_seq = max(receptor_candidates, key=lambda x: len(x[1]))
        return pep_chain, pep_seq, rec_chain, rec_seq

    # Multiple peptides but only 1 receptor — pick longest peptide
    if len(peptide_candidates) > 1 and len(receptor_candidates) == 1:
        pep_chain, pep_seq = max(peptide_candidates, key=lambda x: len(x[1]))
        rec_chain, rec_seq = receptor_candidates[0]
        return pep_chain, pep_seq, rec_chain, rec_seq

    return None, None, None, None


def main() -> None:
    pepset = _load_pepset()

    # ---------------------------------------------------------------
    # 1. Load all affinity sources
    # ---------------------------------------------------------------
    frames: list[pd.DataFrame] = []

    # Original 6 training complexes (gold standard)
    orig = pd.read_csv(DATA_DIR / "training_complexes.csv")
    orig["source"] = "manual"
    orig["affinity_type"] = "Kd"
    frames.append(orig)

    # RCSB affinity
    rcsb_path = DATA_DIR / "rcsb_binding_affinity.csv"
    if rcsb_path.exists():
        rcsb = pd.read_csv(rcsb_path)
        rcsb = rcsb.rename(columns={"value": "kd_nM"})
        if "source" not in rcsb.columns:
            rcsb["source"] = "rcsb"
        else:
            rcsb["source"] = rcsb["source"].fillna("rcsb")
        frames.append(rcsb)

    # Affinity supplement (PDBe + ChEMBL + REMARK)
    supp_path = DATA_DIR / "affinity_supplement.csv"
    if supp_path.exists():
        supp = pd.read_csv(supp_path)
        supp = supp.rename(columns={"value": "kd_nM"})
        if "source" not in supp.columns:
            supp["source"] = "supplement"
        else:
            supp["source"] = supp["source"].fillna("supplement")
        frames.append(supp)

    # Bulk RCSB affinity (2689 records for 294 PDB IDs, from all manifests)
    bulk_path = DATA_DIR / "rcsb_binding_affinity_bulk.csv"
    if bulk_path.exists():
        bulk = pd.read_csv(bulk_path)
        bulk = bulk.rename(columns={"value": "kd_nM"})
        if "source" not in bulk.columns:
            bulk["source"] = "rcsb_bulk"
        else:
            bulk["source"] = bulk["source"].fillna("rcsb_bulk")
        frames.append(bulk)

    # BindingDB expanded (the few rows with sequences)
    exp_path = DATA_DIR / "training_complexes_expanded.csv"
    if exp_path.exists():
        exp = pd.read_csv(exp_path)
        # Only rows with valid sequences
        valid_seq = exp[
            (~exp["peptide_sequence"].isna()) &
            (exp["peptide_sequence"] != "") &
            (exp["peptide_sequence"].str.len() >= PEP_LEN_MIN)
        ].copy()
        if not valid_seq.empty:
            valid_seq["source"] = valid_seq.get("source", "bindingdb")
            valid_seq["affinity_type"] = "Kd"
            frames.append(valid_seq)

    if not frames:
        _log.error("No affinity sources found")
        return

    combined = pd.concat(frames, ignore_index=True)
    _log.info("Combined affinity records: %d", len(combined))

    # ---------------------------------------------------------------
    # 2. Normalize columns
    # ---------------------------------------------------------------
    if "pdb_id" not in combined.columns:
        _log.error("No pdb_id column")
        return

    combined["pdb_id"] = combined["pdb_id"].str.upper()

    # pKd column — use experimental_pkd if present, else compute from kd_nM
    if "experimental_pkd" not in combined.columns:
        combined["experimental_pkd"] = None
    if "kd_nM" in combined.columns:
        mask = combined["experimental_pkd"].isna() & combined["kd_nM"].notna()
        combined.loc[mask, "experimental_pkd"] = (
            -combined.loc[mask, "kd_nM"].apply(
                lambda x: None if x <= 0 else __import__("math").log10(x * 1e-9)
            ) * -1
        )

    # Filter to valid pKd range
    combined = combined.dropna(subset=["experimental_pkd"])
    combined["experimental_pkd"] = pd.to_numeric(combined["experimental_pkd"], errors="coerce")
    combined = combined[combined["experimental_pkd"].between(PKD_MIN, PKD_MAX)]
    _log.info("After pKd filter: %d records", len(combined))

    # ---------------------------------------------------------------
    # 3. Deduplicate by pdb_id, keeping highest-priority affinity type
    # ---------------------------------------------------------------
    priority = {"manual": 0, "Kd": 1, "KD": 1, "Ki": 2, "KI": 2,
                "IC50": 3, "REMARK": 4, "chembl": 5, "pdbe": 6}
    combined["_priority"] = combined.get("affinity_type", "unknown").map(priority).fillna(99)
    combined = combined.sort_values(["pdb_id", "_priority"])
    best_per_pdb = combined.groupby("pdb_id").first().reset_index()
    _log.info("Unique PDB IDs with affinity: %d", len(best_per_pdb))

    # ---------------------------------------------------------------
    # 4. Find structures + extract sequences
    # ---------------------------------------------------------------
    rows_out: list[dict] = []
    not_found: list[str] = []
    no_pep: list[str] = []

    for _, row in best_per_pdb.iterrows():
        pdb_id = row["pdb_id"].upper()

        # PepSet leakage guard
        if pdb_id in pepset:
            _log.debug("Skipping %s (PepSet)", pdb_id)
            continue

        # Check if we already have the sequence
        existing_seq = str(row.get("peptide_sequence", "") or "")
        if existing_seq and len(existing_seq) >= PEP_LEN_MIN:
            rows_out.append({
                "pdb_id": pdb_id,
                "peptide_sequence": existing_seq,
                "experimental_pkd": float(row["experimental_pkd"]),
                "affinity_type": str(row.get("affinity_type", "Kd")),
                "source": str(row.get("source", "unknown")),
                "receptor_chain": str(row.get("receptor_chain", "")),
                "family_hint": str(row.get("family_hint", "")),
            })
            continue

        # Find the structure file
        struct_path = _find_structure(pdb_id)
        if not struct_path:
            not_found.append(pdb_id)
            continue

        # Extract chains from PDB
        chains = _extract_chains_from_pdb(struct_path)
        pep_chain, pep_seq, rec_chain, rec_seq = _classify_chains(chains)
        if not pep_seq:
            no_pep.append(pdb_id)
            _log.debug("%s: could not classify peptide chain (%d chains)", pdb_id, len(chains))
            continue

        rows_out.append({
            "pdb_id": pdb_id,
            "peptide_sequence": pep_seq,
            "experimental_pkd": float(row["experimental_pkd"]),
            "affinity_type": str(row.get("affinity_type", "Kd")),
            "source": str(row.get("source", "unknown")),
            "receptor_chain": rec_chain or "",
            "family_hint": str(row.get("family_hint", "")),
        })
        _log.info("  %s: pep_chain=%s seq=%s pKd=%.2f",
                  pdb_id, pep_chain, pep_seq[:20], float(row["experimental_pkd"]))

    # ---------------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------------
    if rows_out:
        df_out = pd.DataFrame(rows_out)
        out_path = DATA_DIR / "training_complexes_full.csv"
        df_out.to_csv(out_path, index=False)
        _log.info("Saved %d rows → %s", len(df_out), out_path)

        print(f"\n=== Calibration Set Summary ===")
        print(f"Total rows:           {len(df_out)}")
        print(f"With peptide sequence: {(~df_out['peptide_sequence'].isna()).sum()}")
        print(f"pKd range:            {df_out['experimental_pkd'].min():.1f}–{df_out['experimental_pkd'].max():.1f}")
        print(f"Source breakdown:     {df_out['source'].value_counts().to_dict()}")
        print(f"Affinity types:       {df_out['affinity_type'].value_counts().to_dict()}")
        print(f"\nNot found on disk ({len(not_found)}): {not_found[:15]}")
        print(f"No peptide chain ({len(no_pep)}): {no_pep[:15]}")
    else:
        _log.warning("No calibration rows produced")


if __name__ == "__main__":
    main()
