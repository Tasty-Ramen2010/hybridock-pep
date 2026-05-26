"""Re-download all structure files from RCSB using the committed manifest CSVs.

Run this on a new machine (e.g. Linux RTX) after git clone to reconstruct
the 1.5 GB structure directories that are not committed to git.

The manifests (datasets/*/manifest.csv) record which PDB IDs are included
and their download status from the original Mac session.  This script:
  1. Reads each manifest
  2. Identifies rows where status == 'downloaded' (or 'included')
  3. Downloads the PDB.gz from RCSB if not already present on disk
  4. Verifies file size > 500 bytes (rejects empty/404 responses)
  5. Prints a summary table per dataset

Usage:
    python scripts/download_from_manifests.py                       # all datasets
    python scripts/download_from_manifests.py --datasets pdb_2024_2026 family_targeted
    python scripts/download_from_manifests.py --workers 8           # parallel downloads
    python scripts/download_from_manifests.py --dry-run             # just show counts

Expected time:
    ~2–4 hours for all 8,732 structures at default 4 workers (limited by RCSB rate limit)
    ~1–2 hours at 8 workers (RCSB allows ~5–10 req/s per IP)

RCSB PDB.gz URL: https://files.rcsb.org/download/{PDBID}.pdb.gz
"""
from __future__ import annotations

import argparse
import logging
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

# Dataset name → structures subdirectory
DATASETS: dict[str, str] = {
    "pdb_2024_2026":   "datasets/pdb_2024_2026/structures",
    "ppii_enriched":   "datasets/ppii_enriched/structures",
    "ppii_extended":   "datasets/ppii_extended/structures",
    "pdb_2019_2023":   "datasets/pdb_2019_2023/structures",
    "pdb_2010_2018":   "datasets/pdb_2010_2018/structures",
    "pdb_pre2010":     "datasets/pdb_pre2010/structures",
    "family_targeted": "datasets/family_targeted/structures",
}

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"
MIN_SIZE = 500  # bytes — anything smaller is a 404 or empty response
RETRY_SLEEP = 5.0  # seconds between retries


def _download_one(pdb_id: str, out_path: Path, max_retries: int = 3) -> tuple[str, str]:
    """Download one PDB.gz. Returns (pdb_id, status) where status in
    {ok, skip, fail}.
    """
    if out_path.exists() and out_path.stat().st_size > MIN_SIZE:
        return pdb_id, "skip"

    url = RCSB_URL.format(pdb_id=pdb_id.lower())
    for attempt in range(1, max_retries + 1):
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < MIN_SIZE:
                tmp.unlink(missing_ok=True)
                return pdb_id, "fail"
            tmp.rename(out_path)
            return pdb_id, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pdb_id, "fail"
            if attempt < max_retries:
                time.sleep(RETRY_SLEEP * attempt)
        except Exception:
            if attempt < max_retries:
                time.sleep(RETRY_SLEEP * attempt)

    return pdb_id, "fail"


def download_dataset(name: str, manifest_path: Path, struct_dir: Path,
                     workers: int = 4, dry_run: bool = False) -> dict:
    """Download all 'downloaded'/'included' entries from one manifest."""
    if not manifest_path.exists():
        _log.warning("Manifest not found: %s — skipping %s", manifest_path, name)
        return {"name": name, "total": 0, "ok": 0, "skip": 0, "fail": 0}

    df = pd.read_csv(manifest_path)

    # Identify the PDB-ID column
    id_col = next((c for c in df.columns if c.lower() in ("pdb_id", "pdbid", "id")), None)
    if id_col is None:
        _log.error("Cannot find pdb_id column in %s (columns: %s)", manifest_path, list(df.columns))
        return {"name": name, "total": 0, "ok": 0, "skip": 0, "fail": 0}

    # Identify rows that should be downloaded
    status_col = next((c for c in df.columns if c.lower() == "status"), None)
    if status_col:
        targets = df[df[status_col].str.lower().isin(
            {"downloaded", "included", "ok", "success", "yes", "true"}
        )][id_col].dropna().str.upper().tolist()
    else:
        targets = df[id_col].dropna().str.upper().tolist()

    # Also include_col if present (ppii_enriched uses 'included' bool)
    inc_col = next((c for c in df.columns if c.lower() in ("included", "include")), None)
    if inc_col and inc_col != status_col:
        inc_mask = df[inc_col].astype(str).str.lower().isin({"true", "yes", "1"})
        targets = list(set(targets) | set(df.loc[inc_mask, id_col].dropna().str.upper().tolist()))

    n_total = len(targets)
    already = sum(1 for pid in targets
                  if (struct_dir / f"{pid}.pdb.gz").exists()
                  and (struct_dir / f"{pid}.pdb.gz").stat().st_size > MIN_SIZE)

    _log.info("[%s] %d targets, %d already on disk, %d to download",
              name, n_total, already, n_total - already)

    if dry_run:
        return {"name": name, "total": n_total, "ok": 0, "skip": already, "fail": 0}

    results = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_download_one, pid, struct_dir / f"{pid}.pdb.gz"): pid
            for pid in targets
        }
        for i, fut in enumerate(as_completed(futs), 1):
            pid, status = fut.result()
            results[status] = results.get(status, 0) + 1
            if status == "fail":
                _log.warning("  FAIL: %s", pid)
            if i % 100 == 0 or i == n_total:
                _log.info("  [%s] %d/%d — ok=%d skip=%d fail=%d",
                          name, i, n_total, results["ok"], results["skip"], results["fail"])

    results["total"] = n_total
    results["name"] = name
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Which datasets to download (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without downloading")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  HybriDock-Pep structure downloader")
    print(f"  Datasets: {args.datasets}")
    print(f"  Workers: {args.workers}   Dry-run: {args.dry_run}")
    print(f"{'=' * 70}\n")

    summaries = []
    for ds_name in args.datasets:
        struct_dir = REPO / DATASETS[ds_name]
        manifest = REPO / "datasets" / ds_name / "manifest.csv"
        result = download_dataset(ds_name, manifest, struct_dir,
                                  workers=args.workers, dry_run=args.dry_run)
        summaries.append(result)

    print(f"\n{'=' * 70}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':30}  {'Total':6}  {'Downloaded':10}  {'Skipped':8}  {'Failed':6}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*6}")
    total_ok = total_skip = total_fail = total_n = 0
    for s in summaries:
        print(f"  {s['name']:30}  {s['total']:6d}  {s.get('ok', 0):10d}  "
              f"{s.get('skip', 0):8d}  {s.get('fail', 0):6d}")
        total_n += s["total"]
        total_ok += s.get("ok", 0)
        total_skip += s.get("skip", 0)
        total_fail += s.get("fail", 0)
    print(f"  {'TOTAL':30}  {total_n:6d}  {total_ok:10d}  {total_skip:8d}  {total_fail:6d}")

    if total_fail:
        print(f"\n  ⚠ {total_fail} downloads failed — these PDB IDs may have been retracted.")
        print(f"    Re-run to retry, or exclude them in the manifest.")
    else:
        print(f"\n  ✓ All structures present.")


if __name__ == "__main__":
    main()
