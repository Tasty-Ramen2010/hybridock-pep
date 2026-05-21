"""Fetch protein–peptide complexes from the RCSB PDB.

Two modes:
  --mode recent  : Item 1 — complexes deposited after 2023-09-01
  --mode ppii    : Item 2 — PPII-enriched complexes (any deposition date)
  --mode both    : run both (default)

Outputs:
  datasets/pdb_2024_2026/manifest.csv           (recent mode)
  datasets/pdb_2024_2026/structures/{id}.pdb.gz
  datasets/ppii_enriched/manifest.csv           (ppii mode)
  datasets/ppii_enriched/structures/{id}.pdb.gz

Idempotent: skips already-downloaded structures; only refreshes manifests.

Usage:
    conda run --no-capture-output -n score-env \\
        python scripts/fetch_pdb_complexes.py --mode both --max-workers 4

Requires: biopython, requests, pandas (all in score-env).
RCSB API: Search v2 (https://search.rcsb.org/rcsbsearch/v2/query),
          Data GraphQL (https://data.rcsb.org/graphql).
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from Bio.PDB import PDBParser, PPBuilder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent

RECENT_DIR = REPO / "datasets" / "pdb_2024_2026"
PPII_DIR = REPO / "datasets" / "ppii_enriched"
PEPSET_DIR = REPO / "datasets" / "pepset"
REFPEPDB_DIR = REPO / "datasets" / "RefPepDB-RecentSet"

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"

RECENT_SINCE = "2023-09-01"
RESOLUTION_CUTOFF = 2.5
SHORT_MIN, SHORT_MAX = 5, 30
LONG_MIN = 50
PPII_SHORT_MAX = 25

PPII_PHI_MIN, PPII_PHI_MAX = -90.0, -20.0
PPII_PSI_MIN, PPII_PSI_MAX = 110.0, 180.0
PPII_PSI_WRAP_MIN, PPII_PSI_WRAP_MAX = -180.0, -170.0
PPII_FRACTION_THRESHOLD = 0.30
PPII_MIN_CONSEC_PRO = 2

GRAPHQL_BATCH_SIZE = 100
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 60
PAGE_SIZE = 100

# sequence motifs → family hint
_FAMILY_MOTIFS: list[tuple[str, str]] = [
    ("GLGF", "PDZ"),
    ("GYGF", "PDZ"),
    ("FLVRES", "SH2"),
    ("WPF", "bromodomain"),
    ("PPII", "SH3"),
]

_MHC_LEN_MIN, _MHC_LEN_MAX = 260, 320

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCSB Search API v2
# ---------------------------------------------------------------------------

def _node(attribute: str, operator: str, value: Any) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def _search_all(payload_base: dict) -> list[str]:
    """Paginate through RCSB Search API v2 and collect all result IDs."""
    all_ids: list[str] = []
    start = 0
    total: int | None = None

    while total is None or start < total:
        payload = {**payload_base, "request_options": {"paginate": {"start": start, "rows": PAGE_SIZE}}}
        try:
            resp = requests.post(RCSB_SEARCH, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            _log.warning("RCSB search request failed: %s", exc)
            break

        if not resp.text.strip():
            _log.warning("RCSB returned empty response body — treating as 0 results")
            break

        data = resp.json()
        if total is None:
            total = data.get("total_count", 0)
            _log.info("RCSB query total_count=%d", total)

        batch = [r["identifier"] for r in data.get("result_set", [])]
        all_ids.extend(batch)
        start += PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break

    return all_ids


def _query_recent_complexes() -> list[str]:
    """Return PDB IDs for protein–peptide complexes deposited after RECENT_SINCE."""
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                _node("rcsb_entry_info.resolution_combined", "less_or_equal", RESOLUTION_CUTOFF),
                _node("rcsb_accession_info.initial_release_date", "greater_or_equal", RECENT_SINCE),
                _node("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
                _node("rcsb_entry_info.polymer_monomer_count_minimum", "greater_or_equal", SHORT_MIN),
                _node("rcsb_entry_info.polymer_monomer_count_minimum", "less_or_equal", SHORT_MAX),
                _node("rcsb_entry_info.polymer_monomer_count_maximum", "greater_or_equal", LONG_MIN),
            ],
        },
        "return_type": "entry",
    }
    return _search_all(payload)


def _query_ppii_candidates() -> list[str]:
    """Return PDB IDs whose short chain has length 5–25 and sequence contains 'PP'.

    RCSB v2 requires seqmotif and attribute services to be siblings in a top-level
    group, not flattened together — mixing them in one group returns an empty body.
    Post-filter (Ramachandran) is applied locally after download.
    """
    attribute_group = {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            _node("rcsb_entry_info.resolution_combined", "less_or_equal", RESOLUTION_CUTOFF),
            _node("rcsb_entry_info.polymer_entity_count_protein", "equals", 2),
            _node("rcsb_entry_info.polymer_monomer_count_minimum", "greater_or_equal", SHORT_MIN),
            _node("rcsb_entry_info.polymer_monomer_count_minimum", "less_or_equal", PPII_SHORT_MAX),
            _node("rcsb_entry_info.polymer_monomer_count_maximum", "greater_or_equal", LONG_MIN),
            _node("rcsb_entry_info.experimental_method", "exact_match", "X-ray"),
        ],
    }
    seqmotif_node = {
        "type": "terminal",
        "service": "seqmotif",
        "parameters": {
            "value": "PP",
            "pattern_type": "prosite",
            "sequence_type": "protein",
        },
    }
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [attribute_group, seqmotif_node],
        },
        "return_type": "entry",
    }
    ids = _search_all(payload)
    if not ids:
        # seqmotif service unavailable or returns nothing — fall back to attribute-only
        _log.warning("PPII seqmotif query returned 0 — retrying with attributes only")
        payload["query"] = attribute_group
        ids = _search_all(payload)
    return ids


# ---------------------------------------------------------------------------
# RCSB GraphQL batch metadata
# ---------------------------------------------------------------------------

_GRAPHQL_QUERY = """
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


def _fetch_metadata_batch(pdb_ids: list[str]) -> list[dict]:
    """Fetch entry metadata for up to GRAPHQL_BATCH_SIZE IDs via RCSB GraphQL."""
    id_list = json.dumps(pdb_ids)
    query = _GRAPHQL_QUERY % id_list
    try:
        resp = requests.post(RCSB_GRAPHQL, json={"query": query}, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("entries", [])
    except Exception as exc:
        _log.warning("GraphQL batch failed for %d IDs: %s", len(pdb_ids), exc)
        return []


def _fetch_metadata_all(pdb_ids: list[str]) -> dict[str, dict]:
    """Return {pdb_id: entry_metadata} for all IDs, batched."""
    results: dict[str, dict] = {}
    batches = [pdb_ids[i : i + GRAPHQL_BATCH_SIZE] for i in range(0, len(pdb_ids), GRAPHQL_BATCH_SIZE)]
    for i, batch in enumerate(batches):
        _log.info("Fetching metadata batch %d/%d (%d IDs)", i + 1, len(batches), len(batch))
        entries = _fetch_metadata_batch(batch)
        for entry in entries:
            results[entry["rcsb_id"]] = entry
    return results


# ---------------------------------------------------------------------------
# PDB chain parsing
# ---------------------------------------------------------------------------

def _parse_chain_info(pdb_path: Path) -> dict[str, dict]:
    """Parse chain info from a local PDB(.gz) file using Biopython.

    Returns {chain_id: {length, sequence, is_protein, nonstd_residues}}.
    """
    _NONSTD_1L = {
        "MSE": "M", "HYP": "P", "SEP": "S", "TPO": "T", "PTR": "Y",
        "CSO": "C", "KCX": "K", "MLY": "M",
    }
    _STD_3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "HID": "H", "HIE": "H", "HIP": "H",
    }

    opener = gzip.open if str(pdb_path).endswith(".gz") else open
    try:
        with opener(pdb_path, "rt", errors="replace") as fh:
            pdb_text = fh.read()
    except Exception as exc:
        _log.debug("Failed to read %s: %s", pdb_path, exc)
        return {}

    chains: dict[str, dict] = {}
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 26:
            continue
        chain_id = line[21]
        resname = line[17:20].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue

        if chain_id not in chains:
            chains[chain_id] = {"residues": {}, "nonstd_residues": set()}

        if resname in _STD_3_TO_1 or resname in _NONSTD_1L:
            chains[chain_id]["residues"][resseq] = resname
            if resname in _NONSTD_1L and resname not in ("HID", "HIE", "HIP"):
                chains[chain_id]["nonstd_residues"].add(resname)

    result: dict[str, dict] = {}
    for chain_id, info in chains.items():
        seq = ""
        for _, resname in sorted(info["residues"].items()):
            aa = _STD_3_TO_1.get(resname) or _NONSTD_1L.get(resname, "X")
            seq += aa
        n = len(info["residues"])
        if n == 0:
            continue
        result[chain_id] = {
            "length": n,
            "sequence": seq,
            "is_protein": True,
            "nonstd_residues": sorted(info["nonstd_residues"]),
        }
    return result


def _classify_complex(chains: dict[str, dict]) -> tuple[str | None, str | None, str]:
    """Identify peptide and receptor chain IDs.

    Returns (peptide_chain, receptor_chain, excluded_reason).
    excluded_reason is '' on success.
    """
    protein_chains = [(cid, info) for cid, info in chains.items() if info["is_protein"]]
    if len(protein_chains) != 2:
        return None, None, "chain_count"

    # Sort by length; peptide is the shorter
    protein_chains.sort(key=lambda x: x[1]["length"])
    pep_id, pep_info = protein_chains[0]
    rec_id, rec_info = protein_chains[1]

    if not (SHORT_MIN <= pep_info["length"] <= SHORT_MAX):
        return None, None, "peptide_length"
    if rec_info["length"] < LONG_MIN:
        return None, None, "receptor_too_short"

    return pep_id, rec_id, ""


def _classify_family(receptor_seq: str, receptor_len: int) -> str:
    seq_upper = receptor_seq.upper()
    if _MHC_LEN_MIN <= receptor_len <= _MHC_LEN_MAX and "GSHSMR" in seq_upper:
        return "MHC-class-I"
    for motif, family in _FAMILY_MOTIFS:
        if motif in seq_upper:
            return family
    return "unclassified"


# ---------------------------------------------------------------------------
# PPII Ramachandran check
# ---------------------------------------------------------------------------

def _is_ppii_phi_psi(phi: float | None, psi: float | None) -> bool:
    if phi is None or psi is None:
        return False
    # Bio.PDB returns angles in radians; thresholds are in degrees
    phi_deg = math.degrees(phi)
    psi_deg = math.degrees(psi)
    phi_ok = PPII_PHI_MIN <= phi_deg <= PPII_PHI_MAX
    psi_ok = (PPII_PSI_MIN <= psi_deg <= PPII_PSI_MAX) or (PPII_PSI_WRAP_MIN <= psi_deg <= PPII_PSI_WRAP_MAX)
    return phi_ok and psi_ok


def _compute_ramachandran_ppii(
    pdb_path: Path, chain_id: str
) -> tuple[int, float, int]:
    """Compute PPII residue count, fraction, and max consecutive Pro for a chain.

    Returns (n_ppii, ppii_fraction, max_consec_pro).
    """
    opener = gzip.open if str(pdb_path).endswith(".gz") else open
    try:
        with opener(pdb_path, "rt", errors="replace") as fh:
            pdb_text = fh.read()
    except Exception:
        return 0, 0.0, 0

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", io.StringIO(pdb_text))
    except Exception:
        return 0, 0.0, 0

    # Find the requested chain in the first model
    target_chain = None
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                target_chain = chain
                break
        if target_chain:
            break

    if target_chain is None:
        return 0, 0.0, 0

    ppbuilder = PPBuilder()
    polypeptides = ppbuilder.build_peptides(target_chain, aa_only=False)
    if not polypeptides:
        return 0, 0.0, 0

    n_ppii = 0
    n_total = 0  # residues with computable phi AND psi (excludes first and last)

    # Consecutive Pro tracking over entire chain
    seq = "".join(str(pp.get_sequence()) for pp in polypeptides)
    max_consec_pro = 0
    cur = 0
    for aa in seq:
        if aa == "P":
            cur += 1
            max_consec_pro = max(max_consec_pro, cur)
        else:
            cur = 0

    for pp in polypeptides:
        phi_psi = pp.get_phi_psi_list()
        # skip first and last (undefined phi/psi)
        for phi, psi in phi_psi[1:-1]:
            n_total += 1
            if _is_ppii_phi_psi(phi, psi):
                n_ppii += 1

    frac = n_ppii / n_total if n_total > 0 else 0.0
    return n_ppii, frac, max_consec_pro


# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------

def _download_pdb_gz(pdb_id: str, dest: Path) -> bool:
    """Download {pdb_id}.pdb.gz to dest. Skip if dest exists and non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        return True

    url = RCSB_DOWNLOAD.format(pdb_id=pdb_id)
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:
            wait = 2 ** attempt
            _log.debug("Download %s attempt %d failed: %s (retry in %ds)", pdb_id, attempt, exc, wait)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(wait)

    _log.warning("Failed to download %s after %d attempts", pdb_id, DOWNLOAD_RETRIES)
    return False


def _download_batch(
    pdb_ids: list[str], dest_dir: Path, max_workers: int
) -> dict[str, bool]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    def _task(pdb_id: str) -> tuple[str, bool]:
        dest = dest_dir / f"{pdb_id}.pdb.gz"
        return pdb_id, _download_pdb_gz(pdb_id, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = {exc.submit(_task, pid): pid for pid in pdb_ids}
        done = 0
        for future in as_completed(futures):
            pid, ok = future.result()
            results[pid] = ok
            done += 1
            if done % 50 == 0:
                _log.info("Downloaded %d/%d", done, len(pdb_ids))
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _load_known_ids(base_dir: Path) -> set[str]:
    """Return set of PDB IDs found as directory names under base_dir."""
    if not base_dir.exists():
        return set()
    return {p.name.upper() for p in base_dir.iterdir() if p.is_dir() and len(p.name) == 4}


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

_RECENT_COLS = [
    "pdb_id", "peptide_chain", "peptide_seq", "peptide_len", "peptide_nonstd",
    "receptor_chain", "receptor_len", "receptor_seq_md5",
    "resolution_A", "method", "deposition_date",
    "family_hint", "excluded_reason",
]

_PPII_EXTRA_COLS = ["ppii_residues", "ppii_fraction", "consecutive_pro", "passes_ppii_filter"]


def _build_row(
    pdb_id: str,
    meta: dict,
    chains: dict[str, dict],
    pep_chain: str | None,
    rec_chain: str | None,
    excluded_reason: str,
    ppii_stats: tuple[int, float, int] | None = None,
) -> dict:
    """Construct a manifest row dict."""
    resolution = (meta.get("rcsb_entry_info") or {}).get("resolution_combined") or ""
    method = (meta.get("rcsb_entry_info") or {}).get("experimental_method") or ""
    dep_date = (meta.get("rcsb_accession_info") or {}).get("initial_release_date") or ""

    if pep_chain and rec_chain and chains:
        pep = chains.get(pep_chain, {})
        rec = chains.get(rec_chain, {})
        pep_seq = pep.get("sequence", "")
        rec_seq = rec.get("sequence", "")
        row = {
            "pdb_id": pdb_id.upper(),
            "peptide_chain": pep_chain,
            "peptide_seq": pep_seq,
            "peptide_len": pep.get("length", 0),
            "peptide_nonstd": ",".join(pep.get("nonstd_residues", [])),
            "receptor_chain": rec_chain,
            "receptor_len": rec.get("length", 0),
            "receptor_seq_md5": hashlib.md5(rec_seq.encode()).hexdigest(),
            "resolution_A": resolution,
            "method": method,
            "deposition_date": dep_date,
            "family_hint": _classify_family(rec_seq, rec.get("length", 0)),
            "excluded_reason": excluded_reason,
        }
    else:
        row = {
            "pdb_id": pdb_id.upper(),
            "peptide_chain": "", "peptide_seq": "", "peptide_len": 0,
            "peptide_nonstd": "", "receptor_chain": "", "receptor_len": 0,
            "receptor_seq_md5": "", "resolution_A": resolution,
            "method": method, "deposition_date": dep_date,
            "family_hint": "", "excluded_reason": excluded_reason,
        }

    if ppii_stats is not None:
        n_ppii, frac, consec = ppii_stats
        passes = frac >= PPII_FRACTION_THRESHOLD and consec >= PPII_MIN_CONSEC_PRO
        row.update({
            "ppii_residues": n_ppii,
            "ppii_fraction": round(frac, 3),
            "consecutive_pro": consec,
            "passes_ppii_filter": passes,
        })

    return row


# ---------------------------------------------------------------------------
# Mode: recent
# ---------------------------------------------------------------------------

def run_recent(max_workers: int) -> None:
    _log.info("=== Mode: recent (deposited >= %s, resolution <= %.1f) ===", RECENT_SINCE, RESOLUTION_CUTOFF)
    struct_dir = RECENT_DIR / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    known_pepset = _load_known_ids(PEPSET_DIR)
    known_refpepdb = _load_known_ids(REFPEPDB_DIR)
    _log.info("Known PepSet IDs: %d, RefPepDB IDs: %d", len(known_pepset), len(known_refpepdb))

    _log.info("Querying RCSB Search API...")
    candidate_ids = _query_recent_complexes()
    _log.info("Candidates from RCSB query: %d", len(candidate_ids))

    _log.info("Fetching metadata via GraphQL...")
    meta_by_id = _fetch_metadata_all(candidate_ids)

    _log.info("Downloading PDB files...")
    download_status = _download_batch(candidate_ids, struct_dir, max_workers)

    rows: list[dict] = []
    for pdb_id in candidate_ids:
        pdb_upper = pdb_id.upper()
        excluded = ""
        chains: dict[str, dict] = {}
        pep_chain = rec_chain = None

        if pdb_upper in known_pepset:
            excluded = "duplicate_pepset"
        elif pdb_upper in known_refpepdb:
            excluded = "duplicate_refpepdb"
        elif not download_status.get(pdb_id, False):
            excluded = "download_failed"
        else:
            pdb_path = struct_dir / f"{pdb_id}.pdb.gz"
            chains = _parse_chain_info(pdb_path)
            if not chains:
                excluded = "parse_failed"
            else:
                pep_chain, rec_chain, excluded = _classify_complex(chains)

        meta = meta_by_id.get(pdb_id, meta_by_id.get(pdb_upper, {}))
        row = _build_row(pdb_id, meta, chains, pep_chain, rec_chain, excluded)
        rows.append(row)

    manifest_path = RECENT_DIR / "manifest.csv"
    df = pd.DataFrame(rows, columns=_RECENT_COLS)
    df.to_csv(manifest_path, index=False)

    included = df[df["excluded_reason"] == ""]
    _log.info("Recent manifest: %d total, %d included, written to %s", len(df), len(included), manifest_path)


# ---------------------------------------------------------------------------
# Mode: ppii
# ---------------------------------------------------------------------------

def run_ppii(max_workers: int) -> None:
    _log.info("=== Mode: ppii (Ramachandran-validated PPII complexes) ===")
    struct_dir = PPII_DIR / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    known_pepset = _load_known_ids(PEPSET_DIR)
    known_refpepdb = _load_known_ids(REFPEPDB_DIR)

    _log.info("Querying RCSB for PP-containing peptide chains...")
    candidate_ids = _query_ppii_candidates()
    _log.info("PPII candidates: %d", len(candidate_ids))

    _log.info("Fetching metadata...")
    meta_by_id = _fetch_metadata_all(candidate_ids)

    _log.info("Downloading PDB files...")
    download_status = _download_batch(candidate_ids, struct_dir, max_workers)

    rows: list[dict] = []
    for pdb_id in candidate_ids:
        pdb_upper = pdb_id.upper()
        excluded = ""
        chains: dict[str, dict] = {}
        pep_chain = rec_chain = None
        ppii_stats: tuple[int, float, int] | None = (0, 0.0, 0)

        if pdb_upper in known_pepset:
            excluded = "duplicate_pepset"
        elif pdb_upper in known_refpepdb:
            excluded = "duplicate_refpepdb"
        elif not download_status.get(pdb_id, False):
            excluded = "download_failed"
        else:
            pdb_path = struct_dir / f"{pdb_id}.pdb.gz"
            chains = _parse_chain_info(pdb_path)
            if not chains:
                excluded = "parse_failed"
            else:
                pep_chain, rec_chain, excluded = _classify_complex(chains)
                if not excluded and pep_chain:
                    ppii_stats = _compute_ramachandran_ppii(pdb_path, pep_chain)
                    n_ppii, frac, consec = ppii_stats
                    # Exclude if peptide does not pass PPII threshold
                    if not (frac >= PPII_FRACTION_THRESHOLD and consec >= PPII_MIN_CONSEC_PRO):
                        excluded = "fails_ppii_filter"

        meta = meta_by_id.get(pdb_id, meta_by_id.get(pdb_upper, {}))
        row = _build_row(pdb_id, meta, chains, pep_chain, rec_chain, excluded, ppii_stats)
        rows.append(row)

    manifest_path = PPII_DIR / "manifest.csv"
    df = pd.DataFrame(rows, columns=_RECENT_COLS + _PPII_EXTRA_COLS)
    df.to_csv(manifest_path, index=False)

    included = df[df["excluded_reason"] == ""]
    _log.info("PPII manifest: %d total, %d pass filter, written to %s", len(df), len(included), manifest_path)


# ---------------------------------------------------------------------------
# .gitignore update
# ---------------------------------------------------------------------------

def _ensure_gitignore() -> None:
    gitignore = REPO / ".gitignore"
    entries = ["datasets/pdb_2024_2026/", "datasets/ppii_enriched/", "datasets/cache/"]
    if not gitignore.exists():
        gitignore.write_text("\n".join(entries) + "\n")
        return

    text = gitignore.read_text()
    additions = [e for e in entries if e not in text]
    if additions:
        with gitignore.open("a") as fh:
            fh.write("\n# Data augmentation outputs\n")
            for a in additions:
                fh.write(a + "\n")
        _log.info("Added to .gitignore: %s", additions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch protein–peptide complexes from RCSB PDB.")
    parser.add_argument("--mode", choices=["recent", "ppii", "both"], default="both")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    _ensure_gitignore()

    if args.mode in ("recent", "both"):
        run_recent(args.max_workers)

    if args.mode in ("ppii", "both"):
        run_ppii(args.max_workers)


if __name__ == "__main__":
    main()
