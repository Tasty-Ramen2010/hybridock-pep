"""Retry downloading PDB entries that failed in .pdb format, using .cif.gz instead.

RCSB deprecated .pdb format for structures deposited after ~2024.
This script finds all entries marked download_failed in the manifest,
downloads them as .cif.gz, converts to minimal .pdb.gz using BioPython,
and updates the manifest.

Usage:
    python scripts/retry_failed_as_cif.py --dataset pdb_2024_2026
    python scripts/retry_failed_as_cif.py --dataset ppii_enriched
    python scripts/retry_failed_as_cif.py  # runs both
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def _fetch_cif_gz(pdb_id: str) -> bytes | None:
    """Download CIF.gz for a PDB ID. Returns raw bytes or None on failure."""
    url = RCSB_CIF.format(pdb_id=pdb_id)
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        _log.warning("Failed to fetch %s: %s", pdb_id, exc)
        return None


def _cif_gz_to_pdb_gz(cif_gz_bytes: bytes, pdb_id: str) -> bytes | None:
    """Convert CIF.gz bytes to PDB.gz bytes using BioPython.

    Returns None if parsing fails (e.g., very large structure,
    unusual format).
    """
    try:
        # Decompress CIF
        cif_text = gzip.decompress(cif_gz_bytes).decode("latin-1")
        # Parse with BioPython MMCIF parser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, io.StringIO(cif_text))
        if not list(structure.get_models()):
            _log.warning("%s: no models in structure", pdb_id)
            return None
        # Write PDB to string buffer
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        buf = io.StringIO()
        pdb_io.save(buf)
        pdb_text = buf.getvalue()
        # Compress
        return gzip.compress(pdb_text.encode("latin-1"))
    except Exception as exc:
        _log.warning("%s: CIF→PDB conversion failed: %s", pdb_id, exc)
        return None


def retry_dataset(dataset_name: str) -> None:
    """Retry failed downloads for one dataset."""
    dataset_dir = REPO / "datasets" / dataset_name
    manifest_path = dataset_dir / "manifest.csv"
    structures_dir = dataset_dir / "structures"

    if not manifest_path.exists():
        _log.error("Manifest not found: %s", manifest_path)
        return

    df = pd.read_csv(manifest_path)
    # Normalise excluded_reason — NaN means included
    fail_mask = df["excluded_reason"] == "download_failed"
    failures = df[fail_mask]
    _log.info(
        "Dataset %s: %d entries marked download_failed",
        dataset_name,
        len(failures),
    )
    if failures.empty:
        _log.info("Nothing to retry for %s", dataset_name)
        return

    success_ids: list[str] = []
    still_failed_ids: list[str] = []

    def _process(pdb_id: str) -> tuple[str, bool]:
        out_path = structures_dir / f"{pdb_id}.pdb.gz"
        cif_gz = _fetch_cif_gz(pdb_id)
        if cif_gz is None:
            return pdb_id, False
        pdb_gz = _cif_gz_to_pdb_gz(cif_gz, pdb_id)
        if pdb_gz is None:
            return pdb_id, False
        out_path.write_bytes(pdb_gz)
        _log.info("OK  %s → %s (%.1f KB)", pdb_id, out_path.name, len(pdb_gz) / 1024)
        return pdb_id, True

    ids_to_retry = failures["pdb_id"].tolist()
    _log.info("Retrying %d entries with CIF.gz…", len(ids_to_retry))

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_process, pid): pid for pid in ids_to_retry}
        for fut in as_completed(futures):
            pid, ok = fut.result()
            if ok:
                success_ids.append(pid)
            else:
                still_failed_ids.append(pid)

    # Update manifest: clear excluded_reason for successes
    success_set = set(success_ids)
    mask_success = fail_mask & df["pdb_id"].isin(success_set)
    df.loc[mask_success, "excluded_reason"] = ""
    df.to_csv(manifest_path, index=False)

    _log.info(
        "Retry complete: %d recovered, %d still failing",
        len(success_ids),
        len(still_failed_ids),
    )
    if still_failed_ids:
        _log.info("Still failing: %s", still_failed_ids[:10])

    # Print summary
    included_now = (df["excluded_reason"].isna() | (df["excluded_reason"] == "")).sum()
    print(f"\n{dataset_name}: {included_now} entries now included in manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["pdb_2024_2026", "ppii_enriched", "both"],
        default="both",
        help="Which dataset to retry",
    )
    args = parser.parse_args()

    datasets = (
        ["pdb_2024_2026", "ppii_enriched"]
        if args.dataset == "both"
        else [args.dataset]
    )
    for ds in datasets:
        retry_dataset(ds)


if __name__ == "__main__":
    main()
