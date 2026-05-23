"""Extended PDB + affinity data acquisition — maximum coverage.

Fetches four independent data streams in parallel:

  1. Historical peptide-protein complexes (2010-2023) — fills the gap between
     the existing ppii_enriched set (all dates, small) and the pdb_2024_2026 set.
     Splits into pdb_2019_2023/ and pdb_2010_2018/ for manageability.

  2. Family-targeted motif queries (SH3/WW/PDZ/BCL2/MDM2, all time) —
     uses RCSB seqmotif service to find structures with canonical binding
     motifs in the short chain. Outputs to datasets/family_targeted/.

  3. RCSB binding affinity extraction — queries the RCSB Data API for
     rcsb_binding_affinity records in ALL structures we have on disk.
     Writes data/rcsb_binding_affinity.csv.

  4. Extended PPII (Pre-2024 PP-containing peptides) — all structures with
     proline-proline in the short chain, all dates, deduplicated against
     existing ppii_enriched set. Outputs to datasets/ppii_extended/.

Usage:
    python scripts/fetch_all_extended.py [--stream 1|2|3|4|all]

Outputs:
    datasets/pdb_2019_2023/manifest.csv + structures/
    datasets/pdb_2010_2018/manifest.csv + structures/
    datasets/family_targeted/manifest.csv + structures/
    datasets/ppii_extended/manifest.csv + structures/
    data/rcsb_binding_affinity.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, PPBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RCSB_SEARCH   = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL  = "https://data.rcsb.org/graphql"
RCSB_PDB_GZ   = "https://files.rcsb.org/download/{}.pdb.gz"
RCSB_CIF_GZ   = "https://files.rcsb.org/download/{}.cif.gz"

RESOLUTION    = 2.5
SHORT_MIN, SHORT_MAX = 5, 30
LONG_MIN      = 50
PAGE_SIZE     = 250
GRAPHQL_BATCH = 100
MAX_WORKERS   = 8

PEPSET_IDS_FILE = REPO / "data" / "pepset_ids.txt"

# Family binding motifs (PROSITE-style, applied to short chain)
FAMILY_MOTIFS = {
    "sh3":         ("PXXP",  "SH3 domain ligands"),
    "ww_class1":   ("PPxY",  "WW class I (PPxY)"),
    "ww_class2":   ("PPxP",  "WW class II (PPxP)"),
    "pdz":         ("GLGF",  "PDZ carboxylate-binding loop"),
    "bcl2":        ("LXXXD", "BCL-2 / BH3 hydrophobic core"),
    "mdm2":        ("FXXXW", "MDM2 / p53 interaction motif"),
}

# Date ranges for historical fetch
PERIODS = [
    ("pdb_2019_2023", "2019-01-01", "2023-08-31"),
    ("pdb_2010_2018", "2010-01-01", "2018-12-31"),
    ("pdb_pre2010",   None,          "2009-12-31"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pepset() -> set[str]:
    if PEPSET_IDS_FILE.exists():
        return {l.strip().upper() for l in PEPSET_IDS_FILE.read_text().splitlines() if l.strip()}
    # Fallback: inline known PepSet IDs
    return {
        "1A0N","1EJ4","1G73","1JQ8","1JW6","1PMX","1PRM","1YFN","1YWI",
        "2CNY","2FLU","2KHH","2VWF","2VZG","3BEJ","3DAB","3EG6","3EQS","3EQY","3SHB","3TWR",
    }


def _load_existing_ids(dataset_dir: Path) -> set[str]:
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        return set()
    df = pd.read_csv(manifest)
    return set(df["pdb_id"].str.upper())


def _node(attribute: str, operator: str, value: Any) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def _search_all(payload_base: dict, label: str = "") -> list[str]:
    """Paginate RCSB Search v2, return all entry IDs."""
    all_ids: list[str] = []
    start = 0
    total: int | None = None

    while total is None or start < total:
        payload = {
            **payload_base,
            "request_options": {"paginate": {"start": start, "rows": PAGE_SIZE}},
        }
        try:
            resp = requests.post(RCSB_SEARCH, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            _log.warning("[%s] RCSB search failed: %s", label, exc)
            time.sleep(5)
            break

        if not resp.text.strip():
            _log.debug("[%s] Empty response at start=%d", label, start)
            break

        data = resp.json()
        if total is None:
            total = data.get("total_count", 0)
            _log.info("[%s] total_count=%d", label, total)

        batch = [r["identifier"] for r in data.get("result_set", [])]
        all_ids.extend(batch)
        start += PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.2)

    return all_ids


def _query_period(since: str | None, until: str | None) -> list[str]:
    nodes = [
        _node("rcsb_entry_info.resolution_combined", "less_or_equal", RESOLUTION),
        _node("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
        _node("rcsb_entry_info.polymer_monomer_count_minimum", "greater_or_equal", SHORT_MIN),
        _node("rcsb_entry_info.polymer_monomer_count_minimum", "less_or_equal", SHORT_MAX),
        _node("rcsb_entry_info.polymer_monomer_count_maximum", "greater_or_equal", LONG_MIN),
        _node("rcsb_entry_info.experimental_method", "exact_match", "X-ray"),
    ]
    if since:
        nodes.append(_node("rcsb_accession_info.initial_release_date", "greater_or_equal", since))
    if until:
        nodes.append(_node("rcsb_accession_info.initial_release_date", "less_or_equal", until))
    payload = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
    }
    label = f"{since or 'start'}–{until or 'now'}"
    return _search_all(payload, label=label)


def _query_motif(motif: str, family_label: str) -> list[str]:
    """Seqmotif + attribute filter for family-targeted queries."""
    attr_group = {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            _node("rcsb_entry_info.resolution_combined", "less_or_equal", RESOLUTION),
            _node("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
            _node("rcsb_entry_info.polymer_monomer_count_minimum", "greater_or_equal", SHORT_MIN),
            _node("rcsb_entry_info.polymer_monomer_count_minimum", "less_or_equal", SHORT_MAX),
            _node("rcsb_entry_info.polymer_monomer_count_maximum", "greater_or_equal", LONG_MIN),
            _node("rcsb_entry_info.experimental_method", "exact_match", "X-ray"),
        ],
    }
    seq_node = {
        "type": "terminal",
        "service": "seqmotif",
        "parameters": {
            "value": motif,
            "pattern_type": "prosite",
            "sequence_type": "protein",
        },
    }
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [attr_group, seq_node],
        },
        "return_type": "entry",
    }
    ids = _search_all(payload, label=family_label)
    if not ids:
        _log.warning("[%s] seqmotif returned 0, falling back to attr-only", family_label)
        payload["query"] = attr_group
        ids = _search_all(payload, label=family_label)
    return ids


# ---------------------------------------------------------------------------
# Metadata fetch
# ---------------------------------------------------------------------------

_META_QUERY = """
{
  entries(entry_ids: %s) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    rcsb_entry_info { experimental_method resolution_combined }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
      entity_poly {
        pdbx_seq_one_letter_code_can
        rcsb_sample_sequence_length
        rcsb_non_std_monomers
      }
    }
  }
}
"""


def _fetch_meta(pdb_ids: list[str]) -> list[dict]:
    id_list = json.dumps(pdb_ids)
    query = _META_QUERY % id_list
    try:
        resp = requests.post(RCSB_GRAPHQL, json={"query": query}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("entries", [])
    except Exception as exc:
        _log.warning("GraphQL batch failed: %s", exc)
        return []


def _parse_entry(entry: dict) -> dict | None:
    """Extract peptide + receptor from a GraphQL entry. Returns row dict or None."""
    pdb_id = entry.get("rcsb_id", "")
    entities = entry.get("polymer_entities") or []

    # Classify chains by length
    peptide_entity = None
    receptor_entity = None
    for ent in entities:
        ep = ent.get("entity_poly") or {}
        seq = (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
        n = len(seq)
        chains = (ent.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
        nonstd = ep.get("rcsb_non_std_monomers") or ""
        if SHORT_MIN <= n <= SHORT_MAX:
            peptide_entity = {"seq": seq, "len": n, "chains": chains, "nonstd": nonstd}
        elif n >= LONG_MIN:
            receptor_entity = {"seq": seq, "len": n, "chains": chains}

    if not peptide_entity or not receptor_entity:
        return None

    info = entry.get("rcsb_entry_info") or {}
    acc = entry.get("rcsb_accession_info") or {}
    pep_chain = (peptide_entity["chains"] or ["?"])[0]
    rec_chain = (receptor_entity["chains"] or ["?"])[0]

    return {
        "pdb_id": pdb_id,
        "peptide_chain": pep_chain,
        "peptide_seq": peptide_entity["seq"],
        "peptide_len": peptide_entity["len"],
        "peptide_nonstd": peptide_entity["nonstd"],
        "receptor_chain": rec_chain,
        "receptor_len": receptor_entity["len"],
        "resolution_A": info.get("resolution_combined", ""),
        "method": info.get("experimental_method", ""),
        "deposition_date": acc.get("initial_release_date", ""),
        "excluded_reason": "",
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(pdb_id: str, out_dir: Path) -> bool:
    out_path = out_dir / f"{pdb_id}.pdb.gz"
    if out_path.exists() and out_path.stat().st_size > 500:
        return True  # already have it

    # Try PDB.gz first
    try:
        r = requests.get(RCSB_PDB_GZ.format(pdb_id), timeout=60)
        if r.status_code == 200 and len(r.content) > 500:
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass

    # Fall back to CIF → PDB conversion
    try:
        r = requests.get(RCSB_CIF_GZ.format(pdb_id), timeout=60)
        if r.status_code == 200 and len(r.content) > 500:
            cif_text = gzip.decompress(r.content).decode("latin-1")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, io.StringIO(cif_text))
            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            buf = io.StringIO()
            pdb_io.save(buf)
            pdb_gz = gzip.compress(buf.getvalue().encode("latin-1"))
            out_path.write_bytes(pdb_gz)
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    name: str,
    pdb_ids: list[str],
    pepset_ids: set[str],
    extra_col: str | None = None,
    extra_val: str | None = None,
) -> None:
    out_dir = REPO / "datasets" / name
    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    # Load existing manifest (idempotent)
    existing_rows: dict[str, dict] = {}
    if manifest_path.exists():
        for _, row in pd.read_csv(manifest_path).iterrows():
            existing_rows[row["pdb_id"].upper()] = row.to_dict()

    # Filter: remove PepSet, remove already-known entries
    new_ids = [
        pid for pid in pdb_ids
        if pid.upper() not in pepset_ids and pid.upper() not in existing_rows
    ]
    _log.info("[%s] %d new IDs to process (from %d total, %d existing, %d pepset filtered)",
              name, len(new_ids), len(pdb_ids),
              len(existing_rows), sum(1 for pid in pdb_ids if pid.upper() in pepset_ids))

    if not new_ids:
        _log.info("[%s] nothing new to fetch", name)
        return

    # Fetch metadata in batches
    rows: list[dict] = list(existing_rows.values())
    for i in range(0, len(new_ids), GRAPHQL_BATCH):
        batch = new_ids[i : i + GRAPHQL_BATCH]
        entries = _fetch_meta(batch)
        for entry in entries:
            row = _parse_entry(entry)
            pid = (entry.get("rcsb_id") or "").upper()
            if row is None:
                existing_rows[pid] = {
                    "pdb_id": pid,
                    "peptide_chain": "", "peptide_seq": "", "peptide_len": 0,
                    "peptide_nonstd": "", "receptor_chain": "", "receptor_len": 0,
                    "resolution_A": "", "method": "", "deposition_date": "",
                    "excluded_reason": "chain_count",
                }
                rows.append(existing_rows[pid])
            else:
                if extra_col:
                    row[extra_col] = extra_val
                existing_rows[pid] = row
                rows.append(row)
        # IDs not returned by GraphQL → mark as chain_count issue
        returned = {e.get("rcsb_id", "").upper() for e in entries}
        for pid in batch:
            if pid.upper() not in returned and pid.upper() not in existing_rows:
                existing_rows[pid.upper()] = {
                    "pdb_id": pid.upper(), "excluded_reason": "chain_count",
                    "peptide_chain": "", "peptide_seq": "", "peptide_len": 0,
                    "peptide_nonstd": "", "receptor_chain": "", "receptor_len": 0,
                    "resolution_A": "", "method": "", "deposition_date": "",
                }
                rows.append(existing_rows[pid.upper()])
        time.sleep(0.3)

    # Build DataFrame
    df = pd.DataFrame(rows).drop_duplicates(subset=["pdb_id"])

    # Download structures for included entries
    to_download = df[df["excluded_reason"] == ""]["pdb_id"].tolist()
    _log.info("[%s] downloading %d structures", name, len(to_download))

    success = fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download, pid, struct_dir): pid for pid in to_download}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                success += 1
            else:
                fail += 1
                df.loc[df["pdb_id"] == pid, "excluded_reason"] = "download_failed"

    _log.info("[%s] downloads: %d OK, %d failed", name, success, fail)

    # Save manifest
    df.to_csv(manifest_path, index=False)
    included = (df["excluded_reason"] == "").sum()
    _log.info("[%s] manifest saved: %d/%d included", name, included, len(df))


# ---------------------------------------------------------------------------
# Stream 3: RCSB binding affinity extraction
# ---------------------------------------------------------------------------

_AFFINITY_QUERY = """
{
  entries(entry_ids: %s) {
    rcsb_id
    rcsb_binding_affinity {
      type
      value
      unit
      link
    }
  }
}
"""

def fetch_rcsb_binding_affinity() -> None:
    """Query RCSB for binding_affinity records across all local structure files."""
    # Collect all PDB IDs we have on disk
    all_ids: set[str] = set()
    for ds in ["pdb_2024_2026", "ppii_enriched", "raw_pdbs",
               "pdb_2019_2023", "pdb_2010_2018", "family_targeted", "ppii_extended"]:
        d = REPO / "datasets" / ds
        if d.is_dir():
            sdir = d / "structures"
            if sdir.is_dir():
                for f in sdir.glob("*.pdb.gz"):
                    all_ids.add(f.name.split(".")[0].upper())
            else:
                for f in d.glob("*.pdb"):
                    all_ids.add(f.stem.upper())

    pepset = _load_pepset()
    query_ids = sorted(all_ids - pepset)
    _log.info("Querying RCSB binding_affinity for %d structures", len(query_ids))

    records: list[dict] = []
    for i in range(0, len(query_ids), GRAPHQL_BATCH):
        batch = query_ids[i : i + GRAPHQL_BATCH]
        id_list = json.dumps(batch)
        query = _AFFINITY_QUERY % id_list
        try:
            resp = requests.post(RCSB_GRAPHQL, json={"query": query}, timeout=60)
            resp.raise_for_status()
            entries = resp.json().get("data", {}).get("entries", [])
            for entry in entries:
                pdb_id = entry.get("rcsb_id", "")
                aff_list = entry.get("rcsb_binding_affinity") or []
                for aff in aff_list:
                    aff_type = aff.get("type", "")
                    if aff_type.lower() in ("kd", "ki", "ic50", "deltag"):
                        records.append({
                            "pdb_id": pdb_id,
                            "affinity_type": aff_type,
                            "value": aff.get("value"),
                            "unit": aff.get("unit"),
                            "link": aff.get("link", ""),
                        })
        except Exception as exc:
            _log.warning("RCSB affinity query failed for batch %d: %s", i, exc)
        time.sleep(0.2)

    if records:
        df = pd.DataFrame(records)
        # Convert to pKd (nM → pKd = -log10(Kd/1e9))
        def to_pkd(row: pd.Series) -> float | None:
            v = row.get("value")
            unit = str(row.get("unit", "")).lower()
            atype = str(row.get("affinity_type", "")).lower()
            if v is None:
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            if unit in ("nm", "nanomolar", "nanomol/l"):
                v_nM = v
            elif unit in ("um", "micromolar", "micromol/l", "µm"):
                v_nM = v * 1000
            elif unit in ("pm", "picomolar", "picomol/l"):
                v_nM = v / 1000
            elif unit in ("mm", "millimolar"):
                v_nM = v * 1e6
            else:
                return None
            if v_nM <= 0:
                return None
            return round(-math.log10(v_nM * 1e-9), 3)

        df["experimental_pkd"] = df.apply(to_pkd, axis=1)
        out = REPO / "data" / "rcsb_binding_affinity.csv"
        df.to_csv(out, index=False)
        pkd_valid = df["experimental_pkd"].dropna()
        _log.info("Saved %d affinity records (%d with valid pKd) → %s",
                  len(df), len(pkd_valid), out)
        # Show summary
        valid = df[df["experimental_pkd"].notna()]
        if not valid.empty:
            _log.info("pKd range: %.1f – %.1f", valid["experimental_pkd"].min(),
                      valid["experimental_pkd"].max())
            _log.info("Affinity types: %s", valid["affinity_type"].value_counts().to_dict())
    else:
        _log.info("No binding_affinity records found in RCSB for the queried structures")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stream",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help=(
            "1=historical periods, 2=family motifs, "
            "3=RCSB affinity, 4=extended PPII"
        ),
    )
    args = ap.parse_args()
    streams = {"1", "2", "3", "4"} if args.stream == "all" else {args.stream}

    pepset = _load_pepset()

    # ------------------------------------------------------------------
    # Stream 1: Historical peptide-protein complexes
    # ------------------------------------------------------------------
    if "1" in streams:
        _log.info("=== STREAM 1: Historical peptide-protein complexes ===")
        for ds_name, since, until in PERIODS:
            _log.info("Fetching %s (%s – %s)", ds_name, since or "start", until or "now")
            ids = _query_period(since, until)
            _log.info("Got %d candidate IDs", len(ids))
            run_dataset(ds_name, ids, pepset)

    # ------------------------------------------------------------------
    # Stream 2: Family-targeted motif queries
    # ------------------------------------------------------------------
    if "2" in streams:
        _log.info("=== STREAM 2: Family-targeted motif queries ===")
        all_family_ids: list[str] = []
        family_map: dict[str, str] = {}  # id → family_hint

        for fam_key, (motif, label) in FAMILY_MOTIFS.items():
            _log.info("Querying [%s] motif=%s", fam_key, motif)
            ids = _query_motif(motif, label)
            _log.info("[%s] got %d candidate IDs", fam_key, len(ids))
            for pid in ids:
                if pid.upper() not in family_map:
                    family_map[pid.upper()] = fam_key
                    all_family_ids.append(pid)

        _log.info("Total unique family-targeted IDs: %d", len(all_family_ids))
        run_dataset("family_targeted", all_family_ids, pepset,
                    extra_col="family_hint", extra_val="motif_derived")

    # ------------------------------------------------------------------
    # Stream 3: RCSB binding affinity extraction
    # ------------------------------------------------------------------
    if "3" in streams:
        _log.info("=== STREAM 3: RCSB binding affinity extraction ===")
        fetch_rcsb_binding_affinity()

    # ------------------------------------------------------------------
    # Stream 4: Extended PPII (PP-containing, all dates, not in ppii_enriched)
    # ------------------------------------------------------------------
    if "4" in streams:
        _log.info("=== STREAM 4: Extended PPII (all-time PP motif) ===")
        existing_ppii = _load_existing_ids(REPO / "datasets" / "ppii_enriched")
        ids = _query_motif("PP", "extended PPII (PP motif, all dates)")
        _log.info("Extended PPII: %d candidate IDs", len(ids))
        new_ids = [pid for pid in ids if pid.upper() not in existing_ppii]
        _log.info("After dedup vs existing ppii_enriched: %d new IDs", len(new_ids))
        run_dataset("ppii_extended", new_ids, pepset)


if __name__ == "__main__":
    main()
