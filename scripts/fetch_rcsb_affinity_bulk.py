"""Bulk-query RCSB GraphQL for binding affinity data across all downloaded structures.

Queries all PDB IDs from our manifests for rcsb_binding_affinity data.
Extends data/rcsb_binding_affinity.csv with many more entries.

Usage:
    python scripts/fetch_rcsb_affinity_bulk.py
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
DATASETS_DIR = REPO / "datasets"

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
PKD_MIN, PKD_MAX = 3.0, 12.0
BATCH_SIZE = 50   # GraphQL IDs per request


def _all_manifest_ids() -> set[str]:
    """Collect all PDB IDs from all dataset manifests."""
    ids: set[str] = set()

    # CSV manifests
    for m in DATASETS_DIR.glob("*/manifest.csv"):
        try:
            df = pd.read_csv(m)
            if "pdb_id" in df.columns:
                ids.update(df["pdb_id"].str.upper().tolist())
        except Exception:
            pass

    # Supplement with raw_pdbs .pdb files
    raw_dir = DATASETS_DIR / "raw_pdbs"
    if raw_dir.exists():
        for f in raw_dir.glob("*.pdb"):
            ids.add(f.stem.upper())

    # Original training + test CSVs
    for csv_f in [DATA_DIR / "training_complexes.csv", DATA_DIR / "test_complexes.csv"]:
        if csv_f.exists():
            df = pd.read_csv(csv_f)
            if "pdb_id" in df.columns:
                ids.update(df["pdb_id"].str.upper().tolist())

    return ids


def _to_pkd(val: float, unit: str) -> float | None:
    unit = unit.strip().upper().replace("Μ", "U")  # unicode mu → u
    multipliers = {
        "NM": 1.0, "NANOMOLAR": 1.0, "NANOMOL/L": 1.0,
        "UM": 1000.0, "MICROMOLAR": 1000.0, "MICROMOL/L": 1000.0,
        "MM": 1e6, "MILLIMOLAR": 1e6,
        "PM": 0.001, "PICOMOLAR": 0.001,
        "M": 1e9,
    }
    factor = multipliers.get(unit)
    if factor is None:
        return None
    v_nM = val * factor
    if v_nM <= 0:
        return None
    return round(-math.log10(v_nM * 1e-9), 3)


_AFFINITY_GQL = """
query AffinityBulk($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_binding_affinity {
      comp_id
      type
      value
      unit
      reference_sequence_identity
    }
  }
}
"""


def _query_batch(pdb_ids: list[str]) -> list[dict]:
    """Query RCSB GraphQL for binding affinity on a batch of PDB IDs."""
    records = []
    try:
        resp = requests.post(
            RCSB_GRAPHQL,
            json={"query": _AFFINITY_GQL, "variables": {"ids": pdb_ids}},
            timeout=30,
        )
        if resp.status_code != 200:
            _log.warning("GraphQL HTTP %d for batch %s...", resp.status_code, pdb_ids[:3])
            return records
        data = resp.json()
        entries = (data.get("data") or {}).get("entries") or []
        for entry in entries:
            pdb_id = entry.get("rcsb_id", "")
            affinities = entry.get("rcsb_binding_affinity") or []
            for aff in affinities:
                atype = (aff.get("type") or "").strip()
                raw_val = aff.get("value")
                unit = (aff.get("unit") or "nM").strip()
                if raw_val is None:
                    continue
                try:
                    val = float(raw_val)
                except (ValueError, TypeError):
                    continue
                pkd = _to_pkd(val, unit)
                if pkd is None or not (PKD_MIN <= pkd <= PKD_MAX):
                    continue
                records.append({
                    "pdb_id": pdb_id.upper(),
                    "affinity_type": atype,
                    "value": val,
                    "unit": unit,
                    "experimental_pkd": pkd,
                    "comp_id": aff.get("comp_id", ""),
                    "ref_seq_identity": aff.get("reference_sequence_identity"),
                    "source": "rcsb_bulk",
                })
    except Exception as exc:
        _log.warning("GraphQL batch error: %s", exc)
    return records


def main() -> None:
    # Load existing RCSB affinity to avoid duplication
    existing_path = DATA_DIR / "rcsb_binding_affinity.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_ids = set(existing["pdb_id"].str.upper().tolist())
        _log.info("Existing RCSB affinity: %d records for %d PDB IDs",
                  len(existing), len(existing_ids))
    else:
        existing = pd.DataFrame()
        existing_ids = set()

    all_ids = _all_manifest_ids()
    _log.info("Total unique PDB IDs in manifests: %d", len(all_ids))

    # We'll query ALL IDs (RCSB GraphQL is fast, ~50/request)
    query_ids = sorted(all_ids)
    total_batches = (len(query_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    _log.info("Querying %d IDs in %d batches of %d", len(query_ids), total_batches, BATCH_SIZE)

    all_new_records: list[dict] = []
    hits = 0

    for i in range(0, len(query_ids), BATCH_SIZE):
        batch = query_ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        records = _query_batch(batch)
        if records:
            hits += len(records)
            all_new_records.extend(records)
            pdb_hits = {r["pdb_id"] for r in records}
            _log.info("Batch %d/%d: %d affinity records for %s",
                      batch_num, total_batches, len(records), sorted(pdb_hits)[:5])
        else:
            if batch_num % 20 == 0:
                _log.info("Batch %d/%d: no hits (processed so far: %d)",
                          batch_num, total_batches, hits)
        time.sleep(0.05)  # polite rate limiting

    _log.info("Total new records found: %d", len(all_new_records))

    if all_new_records:
        new_df = pd.DataFrame(all_new_records)

        # Merge with existing
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.dropna(subset=["experimental_pkd"])
        combined = combined[combined["experimental_pkd"].between(PKD_MIN, PKD_MAX)]

        # Save extended affinity file
        out_path = DATA_DIR / "rcsb_binding_affinity_bulk.csv"
        combined.to_csv(out_path, index=False)
        _log.info("Saved bulk affinity: %d records for %d unique PDB IDs → %s",
                  len(combined), combined["pdb_id"].nunique(), out_path)

        # Also save just the new records
        new_only_path = DATA_DIR / "rcsb_affinity_new.csv"
        new_df.to_csv(new_only_path, index=False)

        print(f"\n=== Bulk Affinity Query Results ===")
        print(f"PDB IDs queried:       {len(query_ids)}")
        print(f"New records found:     {len(new_df)}")
        print(f"New unique PDB IDs:    {new_df['pdb_id'].nunique()}")
        print(f"Affinity types:        {new_df['affinity_type'].value_counts().to_dict()}")
        print(f"pKd range (new):       {new_df['experimental_pkd'].min():.1f}–{new_df['experimental_pkd'].max():.1f}")
        print(f"\nTop 20 by pKd:")
        top = new_df.sort_values("experimental_pkd", ascending=False).drop_duplicates("pdb_id")
        print(top[["pdb_id","affinity_type","value","unit","experimental_pkd"]].head(20).to_string())
        print(f"\nSaved to: {out_path}")
    else:
        _log.info("No new affinity records found")
        print("No new affinity records found.")


if __name__ == "__main__":
    main()
