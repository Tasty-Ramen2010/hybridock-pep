#!/usr/bin/env python3
"""Fast parallel downloader for the 284 calibration PDB files.

Downloads from RCSB to datasets/raw_pdbs/. Idempotent — skips existing files.
Uses 8 parallel threads for speed.
"""
from __future__ import annotations
import concurrent.futures
import time
import sys
from pathlib import Path
import urllib.request
import urllib.error

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "datasets" / "raw_pdbs"
CSV = REPO / "data" / "training_complexes_full.csv"
WORKERS = 8
SLEEP_PER_THREAD = 0.1  # seconds between requests per thread


def download_pdb(pid: str) -> tuple[str, str]:
    """Return (pid, 'ok'/'skip'/'FAIL: reason')."""
    outpath = OUTDIR / f"{pid}.pdb"
    if outpath.exists() and outpath.stat().st_size > 500:
        return pid, "skip"

    url = f"https://files.rcsb.org/download/{pid}.pdb"
    try:
        urllib.request.urlretrieve(url, outpath)
        time.sleep(SLEEP_PER_THREAD)
        size = outpath.stat().st_size
        if size < 200:
            # Likely an error page or empty response
            outpath.unlink(missing_ok=True)
            return pid, f"FAIL: tiny file ({size} bytes)"
        return pid, f"ok ({size//1024} KB)"
    except urllib.error.HTTPError as e:
        return pid, f"FAIL: HTTP {e.code}"
    except Exception as e:
        return pid, f"FAIL: {e}"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import csv
    pids = []
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pids.append(row["pdb_id"].upper())
    pids = sorted(set(pids))
    print(f"PDB IDs to download: {len(pids)}", flush=True)

    # Count how many already exist
    existing = sum(1 for p in pids if (OUTDIR / f"{p}.pdb").exists())
    print(f"Already on disk: {existing}", flush=True)
    to_fetch = len(pids) - existing
    print(f"Fetching: {to_fetch}", flush=True)
    if to_fetch == 0:
        print("All present — nothing to download.")
        return

    ok, skip, fail = 0, 0, []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_pdb, pid): pid for pid in pids}
        for fut in concurrent.futures.as_completed(futures):
            pid, status = fut.result()
            done += 1
            if status == "skip":
                skip += 1
            elif status.startswith("ok"):
                ok += 1
                print(f"  [{done}/{len(pids)}] {pid}: {status}", flush=True)
            else:
                fail.append((pid, status))
                print(f"  [{done}/{len(pids)}] {pid}: {status}", flush=True)

    print(f"\nDone: {ok} downloaded, {skip} skipped, {len(fail)} failed")
    if fail:
        print("Failed:")
        for pid, reason in fail:
            print(f"  {pid}: {reason}")


if __name__ == "__main__":
    main()
