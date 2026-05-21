"""Expand data/training_complexes.csv using BindingDB Kd measurements.

Downloads the BindingDB All Data TSV, filters to peptide–protein pairs that
have a matching PDB structure, converts Kd → pKd, and writes an expanded
calibration CSV that calibrate_alpha.py can consume.

Outputs:
  data/training_complexes_expanded.csv  (≥100 rows, backward-compat schema)
  datasets/cache/bindingdb_filtered.parquet  (intermediate cache)

Usage:
    conda run --no-capture-output -n score-env \\
        python scripts/bindingdb_calibration_join.py [--use-ki] [--force-download]

Notes:
  - BindingDB column names are unstable across releases; the script
    searches for column names by keyword rather than exact match.
  - Existing rows from training_complexes.csv are preserved with source="manual".
  - PepSet PDB IDs are excluded from the output (keep PepSet as test set).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "datasets" / "cache"
DATA_DIR = REPO / "data"
PEPSET_DIR = REPO / "datasets" / "pepset"

BINDINGDB_BROWSE_URL = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
BINDINGDB_FALLBACK_URL = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2D_202504_tsv.zip"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"

PKD_MIN, PKD_MAX = 3.0, 12.0
PEPTIDE_LEN_MIN, PEPTIDE_LEN_MAX = 5, 30
AMIDE_BOND_MIN = 3
AFFINITY_SPREAD_MAX = 1.5  # log units — exclude rows with this spread across sources
GRAPHQL_BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# 3-letter code for 20 standard AAs + common variants
_STD_AA_SMARTS = "[N;H1,H2][C;H1](~[C;H0](=O))"
_AMIDE_PATTERN = Chem.MolFromSmarts("[N;!$(NC=S)][C](=O)")


# ---------------------------------------------------------------------------
# BindingDB download & cache
# ---------------------------------------------------------------------------

def _discover_bindingdb_url() -> str:
    """Scrape BindingDB download page for the latest All Data TSV zip URL."""
    try:
        resp = requests.get(BINDINGDB_BROWSE_URL, timeout=20)
        resp.raise_for_status()
        matches = re.findall(r'href="([^"]*BindingDB_All[^"]*tsv\.zip)"', resp.text, re.IGNORECASE)
        if matches:
            url = matches[0]
            if not url.startswith("http"):
                url = "https://www.bindingdb.org" + url
            _log.info("Discovered BindingDB URL: %s", url)
            return url
    except Exception as exc:
        _log.warning("Could not scrape BindingDB download page: %s", exc)
    _log.info("Using fallback BindingDB URL: %s", BINDINGDB_FALLBACK_URL)
    return BINDINGDB_FALLBACK_URL


def _download_and_extract(url: str, cache_dir: Path) -> Path:
    """Download and extract BindingDB zip; return path to the TSV file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "bindingdb_all.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        _log.info("Downloading BindingDB (~2 GB compressed)... this will take several minutes")
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        _log.info("  %.1f%%", pct) if downloaded % (100 * 1024 * 1024) == 0 else None
        _log.info("Download complete: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    # Extract TSV
    _log.info("Extracting TSV from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if not tsv_names:
            raise RuntimeError(f"No TSV found in {zip_path}")
        tsv_name = tsv_names[0]
        tsv_path = cache_dir / tsv_name
        if not tsv_path.exists():
            zf.extract(tsv_name, cache_dir)
        return tsv_path


def _find_column(columns: list[str], keyword: str) -> str | None:
    """Find a column name containing keyword (case-insensitive)."""
    kw = keyword.lower()
    matches = [c for c in columns if kw in c.lower()]
    return matches[0] if matches else None


def _load_bindingdb_chunks(tsv_path: Path, use_ki: bool) -> pd.DataFrame:
    """Load BindingDB TSV in chunks and extract relevant columns.

    Column names vary by release; we discover them by keyword match.
    """
    _log.info("Loading BindingDB TSV (may take several minutes for ~9 GB file)...")

    # First read just the header to find column names
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline()
    columns = header_line.rstrip("\n").split("\t")
    _log.info("BindingDB columns (%d): %s...", len(columns), columns[:10])

    # Discover needed columns
    col_smiles = _find_column(columns, "ligand smiles") or _find_column(columns, "smiles")
    col_name = _find_column(columns, "ligand name") or _find_column(columns, "compound name")
    col_pdb = _find_column(columns, "pdb id") or _find_column(columns, "pdb_id")
    col_kd = _find_column(columns, "kd (nm)") or _find_column(columns, "kd(nm)")
    col_ki = _find_column(columns, "ki (nm)") or _find_column(columns, "ki(nm)") if use_ki else None
    col_target = _find_column(columns, "target name") or _find_column(columns, "target")
    col_source = _find_column(columns, "curation") or _find_column(columns, "datasource")

    required = {"smiles": col_smiles, "pdb": col_pdb, "kd": col_kd}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Required BindingDB columns not found: {missing}. "
                           f"Available columns: {columns[:30]}")

    keep_cols = [c for c in [col_smiles, col_name, col_pdb, col_kd, col_ki, col_target, col_source] if c]
    _log.info("Using columns: %s", keep_cols)

    chunks = []
    chunk_size = 500_000
    reader = pd.read_csv(tsv_path, sep="\t", usecols=keep_cols, chunksize=chunk_size,
                         encoding="utf-8", encoding_errors="replace", low_memory=False)
    loaded = 0
    for chunk in reader:
        # Drop rows without PDB ID or Kd
        chunk = chunk.dropna(subset=[col_pdb])
        if col_kd:
            has_affinity = chunk[col_kd].notna()
            if use_ki and col_ki:
                has_affinity = has_affinity | chunk[col_ki].notna()
            chunk = chunk[has_affinity]
        if len(chunk) > 0:
            chunks.append(chunk)
        loaded += chunk_size
        if loaded % 2_000_000 == 0:
            _log.info("  Loaded %dM rows so far...", loaded // 1_000_000)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=keep_cols)
    _log.info("BindingDB raw (with PDB + affinity): %d rows", len(df))

    # Standardize column names for downstream processing
    rename = {}
    if col_smiles: rename[col_smiles] = "smiles"
    if col_name: rename[col_name] = "ligand_name"
    if col_pdb: rename[col_pdb] = "pdb_ids_raw"
    if col_kd: rename[col_kd] = "kd_nm_raw"
    if col_ki: rename[col_ki] = "ki_nm_raw"
    if col_target: rename[col_target] = "target_name"
    if col_source: rename[col_source] = "data_source"
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Peptide SMILES detection
# ---------------------------------------------------------------------------

def _parse_affinity(val: str | float | None) -> float | None:
    """Parse an affinity value that may be a string like '>1000', '<0.1', or a float."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    s = re.sub(r"[><~≥≤]", "", s).strip().split()[0]
    try:
        return float(s)
    except ValueError:
        return None


def _kd_to_pkd(kd_nm: float) -> float:
    """pKd = -log10(Kd_M) = -log10(Kd_nM × 1e-9) = 9 - log10(Kd_nM)."""
    if kd_nm <= 0:
        return float("nan")
    return 9.0 - math.log10(kd_nm)


def _is_peptide_smiles(smiles: str | None) -> bool:
    """Return True if SMILES looks like a linear peptide.

    Heuristic: ≥3 amide bonds, no large rings (except Phe/Tyr/Trp/His/Pro aromatics).
    """
    if not smiles or len(smiles) < 10 or len(smiles) > 500:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
    except Exception:
        return False

    amide_matches = mol.GetSubstructMatches(_AMIDE_PATTERN)
    if len(amide_matches) < AMIDE_BOND_MIN:
        return False

    # Count ring atoms outside allowed aromatic rings
    ring_info = mol.GetRingInfo()
    non_aromatic_large_rings = [r for r in ring_info.AtomRings() if len(r) > 6]
    if non_aromatic_large_rings:
        return False

    return True


def _smiles_to_rough_length(smiles: str) -> int | None:
    """Estimate peptide length from SMILES by counting backbone amide bonds + 1."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        matches = mol.GetSubstructMatches(_AMIDE_PATTERN)
        return len(matches) + 1
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDB cross-check via RCSB GraphQL
# ---------------------------------------------------------------------------

_META_QUERY = """
{
  entries(entry_ids: %s) {
    rcsb_id
    polymer_entities {
      entity_poly {
        rcsb_sample_sequence_length
        pdbx_seq_one_letter_code_can
      }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    }
  }
}
"""


def _batch_fetch_pdb_meta(pdb_ids: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    batches = [pdb_ids[i : i + GRAPHQL_BATCH_SIZE] for i in range(0, len(pdb_ids), GRAPHQL_BATCH_SIZE)]
    for batch in batches:
        id_list = json.dumps(batch)
        query = _META_QUERY % id_list
        try:
            resp = requests.post(RCSB_GRAPHQL, json={"query": query}, timeout=30)
            resp.raise_for_status()
            for entry in resp.json().get("data", {}).get("entries", []) or []:
                results[entry["rcsb_id"]] = entry
        except Exception as exc:
            _log.warning("GraphQL batch failed: %s", exc)
    return results


def _extract_receptor_chain(meta: dict) -> tuple[str | None, str | None]:
    """Return (receptor_chain, receptor_seq) from entry metadata.

    Finds the longest polymer entity chain as the receptor.
    """
    best_chain, best_seq, best_len = None, None, 0
    for entity in meta.get("polymer_entities") or []:
        ep = entity.get("entity_poly") or {}
        length = ep.get("rcsb_sample_sequence_length", 0)
        if length > best_len:
            chains = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
            if chains:
                best_chain = chains[0]
                best_seq = ep.get("pdbx_seq_one_letter_code_can", "")
                best_len = length
    return best_chain, best_seq


# ---------------------------------------------------------------------------
# Load existing PepSet IDs to exclude from calibration training
# ---------------------------------------------------------------------------

def _load_pepset_ids() -> set[str]:
    if not PEPSET_DIR.exists():
        return set()
    return {p.name.upper() for p in PEPSET_DIR.iterdir() if p.is_dir() and len(p.name) == 4}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_expanded_calibration(use_ki: bool, force_download: bool) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = CACHE_DIR / "bindingdb_filtered.parquet"

    # Load existing calibration rows
    existing_csv = DATA_DIR / "training_complexes.csv"
    if existing_csv.exists():
        existing = pd.read_csv(existing_csv)
        existing["source"] = "manual"
        existing["kd_nM"] = float("nan")
        existing["receptor_chain"] = ""
        existing["family_hint"] = ""
        _log.info("Loaded %d existing calibration rows", len(existing))
    else:
        existing = pd.DataFrame(columns=["pdb_id", "peptide_sequence", "experimental_pkd"])
        existing["source"] = "manual"
        existing["kd_nM"] = float("nan")
        existing["receptor_chain"] = ""
        existing["family_hint"] = ""

    known_manual_pdbs = set(existing["pdb_id"].str.upper().tolist())
    pepset_ids = _load_pepset_ids()
    _log.info("PepSet IDs to exclude: %d", len(pepset_ids))

    # --- Load / filter BindingDB ---
    if parquet_path.exists() and not force_download:
        _log.info("Loading cached BindingDB parquet: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        url = _discover_bindingdb_url()
        tsv_path = _download_and_extract(url, CACHE_DIR)
        df = _load_bindingdb_chunks(tsv_path, use_ki)
        df.to_parquet(parquet_path, index=False)
        _log.info("Saved filtered BindingDB parquet: %s", parquet_path)

    _log.info("Filtering peptide SMILES...")
    df["is_peptide"] = df["smiles"].apply(_is_peptide_smiles)
    df = df[df["is_peptide"]].copy()
    _log.info("After peptide filter: %d rows", len(df))

    # Estimate peptide length
    df["pep_len"] = df["smiles"].apply(_smiles_to_rough_length)
    df = df[df["pep_len"].between(PEPTIDE_LEN_MIN, PEPTIDE_LEN_MAX, inclusive="both")].copy()
    _log.info("After length filter (%d–%d): %d rows", PEPTIDE_LEN_MIN, PEPTIDE_LEN_MAX, len(df))

    # Parse affinities
    df["kd_nm"] = df["kd_nm_raw"].apply(_parse_affinity)
    has_kd = df["kd_nm"].notna()
    if use_ki and "ki_nm_raw" in df.columns:
        df["ki_nm"] = df["ki_nm_raw"].apply(_parse_affinity)
        has_ki = df["ki_nm"].notna() & ~has_kd
        df.loc[has_ki, "kd_nm"] = df.loc[has_ki, "ki_nm"]
        df.loc[has_ki, "affinity_type"] = "ki_converted"
    df.loc[has_kd, "affinity_type"] = "kd"
    df = df[df["kd_nm"].notna() & (df["kd_nm"] > 0)].copy()

    df["pkd"] = df["kd_nm"].apply(_kd_to_pkd)
    df = df[df["pkd"].between(PKD_MIN, PKD_MAX, inclusive="both")].copy()
    _log.info("After pKd filter (%.0f–%.0f): %d rows", PKD_MIN, PKD_MAX, len(df))

    # Extract first PDB ID from potentially comma-separated list
    def _first_pdb(raw: str) -> str:
        parts = re.split(r"[,;\s]+", str(raw).strip())
        for p in parts:
            p = p.strip().upper()
            if re.match(r"^[0-9][A-Z0-9]{3}$", p):
                return p
        return ""

    df["pdb_id"] = df["pdb_ids_raw"].apply(_first_pdb)
    df = df[df["pdb_id"] != ""].copy()
    df = df[~df["pdb_id"].isin(pepset_ids)].copy()
    df = df[~df["pdb_id"].isin(known_manual_pdbs)].copy()
    _log.info("After PDB filter (removing PepSet + manual): %d rows", len(df))

    # Deduplicate: for same (pdb_id, pep_len), keep median pKd
    # Flag entries with high spread as unreliable
    grp = df.groupby("pdb_id")["pkd"]
    spread = grp.transform(lambda x: x.max() - x.min())
    df["pkd_spread"] = spread
    high_spread = df["pkd_spread"] > AFFINITY_SPREAD_MAX
    if high_spread.sum() > 0:
        _log.info("Excluding %d rows with pKd spread > %.1f", high_spread.sum(), AFFINITY_SPREAD_MAX)
    df = df[~high_spread].copy()

    # Keep median pKd per PDB
    median_pkd = df.groupby("pdb_id")["pkd"].median()
    median_kd = df.groupby("pdb_id")["kd_nm"].median()
    aff_type = df.groupby("pdb_id")["affinity_type"].first()
    unique_pdbs = median_pkd.index.tolist()
    _log.info("Unique PDB IDs after dedup: %d", len(unique_pdbs))

    # Cross-check PDB IDs at RCSB and get receptor chain info
    _log.info("Fetching receptor chain info from RCSB...")
    meta_by_id = _batch_fetch_pdb_meta(unique_pdbs)
    valid_pdbs = set(meta_by_id.keys())
    _log.info("Valid RCSB PDB IDs: %d / %d", len(valid_pdbs), len(unique_pdbs))

    # Build output rows
    new_rows = []
    for pdb_id in unique_pdbs:
        if pdb_id not in valid_pdbs:
            continue
        rec_chain, rec_seq = _extract_receptor_chain(meta_by_id[pdb_id])
        pkd = float(median_pkd[pdb_id])
        kd = float(median_kd[pdb_id])
        source_type = str(aff_type[pdb_id])
        source = "bindingdb_kd" if source_type == "kd" else "bindingdb_ki_converted"

        # Get peptide sequence (best-effort from SMILES; BindingDB ligand_name sometimes has it)
        name_rows = df[df["pdb_id"] == pdb_id]
        pep_seq = ""
        if "ligand_name" in name_rows.columns:
            candidate = name_rows["ligand_name"].iloc[0] if len(name_rows) > 0 else ""
            # If ligand name looks like a 1-letter sequence, use it
            if candidate and re.match(r"^[ACDEFGHIKLMNPQRSTVWY]{5,30}$", str(candidate).strip().upper()):
                pep_seq = str(candidate).strip().upper()

        new_rows.append({
            "pdb_id": pdb_id,
            "peptide_sequence": pep_seq,
            "experimental_pkd": round(pkd, 3),
            "kd_nM": round(kd, 2),
            "source": source,
            "receptor_chain": rec_chain or "",
            "family_hint": "",
        })

    new_df = pd.DataFrame(new_rows)
    _log.info("New BindingDB-sourced rows: %d", len(new_df))

    # Combine with existing manual rows
    out_cols = ["pdb_id", "peptide_sequence", "experimental_pkd", "kd_nM", "source", "receptor_chain", "family_hint"]
    combined = pd.concat([
        existing[["pdb_id", "peptide_sequence", "experimental_pkd", "kd_nM", "source", "receptor_chain", "family_hint"]],
        new_df[out_cols],
    ], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pdb_id", "peptide_sequence"], keep="first")

    out_path = DATA_DIR / "training_complexes_expanded.csv"
    combined.to_csv(out_path, index=False)
    _log.info("Wrote %d rows to %s", len(combined), out_path)

    if len(combined) < 100:
        _log.warning(
            "Only %d rows — below target of 100. BindingDB may have changed format "
            "or fewer peptide–PDB pairs are available. Check SMILES filter.",
            len(combined),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Expand calibration training set via BindingDB.")
    parser.add_argument("--use-ki", action="store_true",
                        help="Also use Ki measurements where Kd is unavailable")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download BindingDB even if cache exists")
    args = parser.parse_args()
    build_expanded_calibration(args.use_ki, args.force_download)


if __name__ == "__main__":
    main()
