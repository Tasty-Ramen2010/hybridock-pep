"""Fetch peptide sequences from RCSB for rows missing them in training_complexes_expanded.csv.

The BindingDB join found PDB IDs with experimental Kd but couldn't parse
peptide sequences from SMILES (a known limitation). This script queries
RCSB GraphQL to get the polymer sequences for those PDB entries, identifies
which chain is the short one (5–30 aa, the peptide), and fills in the
peptide_sequence column.

Also downloads the actual structure files to datasets/training_expanded_structures/.

Usage:
    python scripts/fetch_expanded_sequences.py
"""
from __future__ import annotations

import gzip
import io
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser, PDBIO

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
STRUCT_DIR = REPO / "datasets" / "training_expanded_structures"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
RCSB_PDB = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"

SHORT_MIN, SHORT_MAX = 5, 30


GRAPHQL_QUERY = """
query GetPolymerSequences($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      entity_poly {
        pdbx_seq_one_letter_code_can
        type
      }
      rcsb_polymer_entity {
        pdbx_description
      }
      entity_poly_seq {
        num
      }
    }
  }
}
"""


def _query_rcsb(pdb_id: str) -> dict | None:
    """Query RCSB GraphQL for polymer chain sequences."""
    try:
        r = requests.post(
            RCSB_GRAPHQL,
            json={"query": GRAPHQL_QUERY, "variables": {"id": pdb_id.upper()}},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        _log.warning("GraphQL query failed for %s: %s", pdb_id, exc)
        return None


def _best_peptide_seq(entities: list[dict]) -> str | None:
    """Pick the short polymer chain that looks like a peptide."""
    candidates = []
    for ent in entities:
        ep = ent.get("entity_poly") or {}
        seq = ep.get("pdbx_seq_one_letter_code_can", "") or ""
        # Strip whitespace and newlines
        seq = re.sub(r"\s+", "", seq)
        # Filter: only polypeptide L or D; not DNA/RNA
        ptype = ep.get("type", "")
        if "polypeptide" not in ptype.lower():
            continue
        n = len(seq)
        if SHORT_MIN <= n <= SHORT_MAX:
            candidates.append((n, seq))
    if not candidates:
        return None
    # Return shortest candidate (most likely to be the peptide)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _download_structure(pdb_id: str, out_dir: Path) -> bool:
    """Download structure as .pdb.gz. Try PDB format first, then CIF→PDB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}.pdb.gz"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True  # already downloaded

    # Try PDB.gz first
    try:
        r = requests.get(RCSB_PDB.format(pdb_id=pdb_id), timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            out_path.write_bytes(r.content)
            _log.info("Downloaded (pdb)  %s  %.1f KB", pdb_id, len(r.content) / 1024)
            return True
    except Exception:
        pass

    # Fall back to CIF.gz → PDB
    try:
        r = requests.get(RCSB_CIF.format(pdb_id=pdb_id), timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            cif_text = gzip.decompress(r.content).decode("latin-1")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, io.StringIO(cif_text))
            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            buf = io.StringIO()
            pdb_io.save(buf)
            pdb_gz = gzip.compress(buf.getvalue().encode("latin-1"))
            out_path.write_bytes(pdb_gz)
            _log.info("Downloaded (cif→pdb)  %s  %.1f KB", pdb_id, len(pdb_gz) / 1024)
            return True
    except Exception as exc:
        _log.warning("Download failed for %s: %s", pdb_id, exc)

    return False


def main() -> None:
    csv_path = DATA_DIR / "training_complexes_expanded.csv"
    df = pd.read_csv(csv_path)
    _log.info("Loaded %d rows from %s", len(df), csv_path.name)

    empty_mask = df["peptide_sequence"].isna() | (df["peptide_sequence"] == "")
    _log.info("Rows with empty peptide_sequence: %d", empty_mask.sum())

    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    filled = 0
    failed_seq = []
    failed_dl = []

    for idx, row in df[empty_mask].iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        _log.info("Querying RCSB for %s…", pdb_id)
        data = _query_rcsb(pdb_id)
        seq = None
        if data:
            entities = (
                (data.get("data") or {})
                .get("entry") or {}
            ).get("polymer_entities") or []
            seq = _best_peptide_seq(entities)

        if seq:
            df.at[idx, "peptide_sequence"] = seq
            _log.info("  %s → %s", pdb_id, seq)
            filled += 1
        else:
            _log.warning("  %s: no peptide sequence found in RCSB", pdb_id)
            failed_seq.append(pdb_id)

        # Download structure regardless of sequence
        ok = _download_structure(pdb_id, STRUCT_DIR)
        if not ok:
            failed_dl.append(pdb_id)

        time.sleep(0.15)  # gentle on RCSB

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    _log.info("Updated CSV saved: %s", csv_path)

    print(f"\n=== Results ===")
    print(f"Sequences filled in: {filled} / {empty_mask.sum()}")
    print(f"Sequences still missing: {len(failed_seq)}: {failed_seq}")
    print(f"Download failures: {len(failed_dl)}: {failed_dl}")
    print(f"Total rows with sequence: {(~(df['peptide_sequence'].isna() | (df['peptide_sequence'] == ''))).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
