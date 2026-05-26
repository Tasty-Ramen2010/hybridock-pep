"""Relax the PPII filter threshold in ppii_enriched/manifest.csv.

The original filter (ppii_fraction >= 0.30 AND consecutive_pro >= 2) is too
strict: only 29 structures pass. Based on dataset analysis (obs 744), the
relaxed threshold (ppii_fraction >= 0.20 AND consecutive_pro >= 1) captures
74 structures that are already downloaded and structurally valid.

This script updates the passes_ppii_filter and excluded_reason columns in the
manifest without re-downloading anything. Structures that pass the relaxed
filter AND have a file on disk are marked as included.

Threshold change:
  OLD: ppii_fraction >= 0.30 AND consecutive_pro >= 2  → 29 pass
  NEW: ppii_fraction >= 0.20 AND consecutive_pro >= 1  → 74 pass (all on disk)

Run:
    python scripts/relax_ppii_filter.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "datasets" / "ppii_enriched" / "manifest.csv"
STRUCT_DIR = REPO / "datasets" / "ppii_enriched" / "structures"

# Relaxed thresholds
NEW_FRAC_THRESH = 0.20
NEW_CONSEC_THRESH = 1


def main(dry_run: bool = False) -> None:
    df = pd.read_csv(MANIFEST)
    _log.info("Loaded manifest: %d rows", len(df))

    # Current pass count
    old_pass = df["passes_ppii_filter"].sum()
    _log.info("Currently passing filter: %d", old_pass)

    # Structures actually on disk
    on_disk = set()
    for f in STRUCT_DIR.glob("*.pdb.gz"):
        on_disk.add(f.name.split(".")[0].upper())
    for f in STRUCT_DIR.glob("*.pdb"):
        on_disk.add(f.stem.upper())
    _log.info("Structure files on disk: %d", len(on_disk))

    # Apply new filter
    new_pass_mask = (
        (df["ppii_fraction"] >= NEW_FRAC_THRESH)
        & (df["consecutive_pro"] >= NEW_CONSEC_THRESH)
        & (df["pdb_id"].str.upper().isin(on_disk))
    )
    _log.info(
        "New filter (frac>=%.2f AND consec>=%d AND on disk): %d pass",
        NEW_FRAC_THRESH,
        NEW_CONSEC_THRESH,
        new_pass_mask.sum(),
    )

    # Update columns
    df["passes_ppii_filter"] = new_pass_mask
    # Re-derive excluded_reason:
    # - If chain_count/peptide_length/download_failed for non-structural reasons → keep those
    # - If previously fails_ppii_filter → recheck
    # - If now passes → clear to ""
    for idx, row in df.iterrows():
        reason = str(row.get("excluded_reason", "") or "")
        if reason in ("chain_count", "peptide_length", "download_failed"):
            # Keep these structural exclusions
            df.at[idx, "passes_ppii_filter"] = False
        elif new_pass_mask.iloc[idx]:
            df.at[idx, "excluded_reason"] = ""  # included
        else:
            df.at[idx, "excluded_reason"] = "fails_ppii_filter"

    new_pass = (df["excluded_reason"] == "").sum()
    _log.info("After update: %d included, %d excluded", new_pass, len(df) - new_pass)

    # Show the extra structures that are now included
    old_pass_ids = set(df[df["passes_ppii_filter"].astype(bool) & (df["ppii_fraction"] >= 0.30) & (df["consecutive_pro"] >= 2)]["pdb_id"])
    new_pass_ids = set(df[df["excluded_reason"] == ""]["pdb_id"])
    newly_included = new_pass_ids - old_pass_ids
    _log.info("Newly included structures (%d):", len(newly_included))
    for pid in sorted(newly_included)[:20]:
        row = df[df["pdb_id"] == pid].iloc[0]
        _log.info(
            "  %s  frac=%.2f  consec=%d  family=%s  seq=%s",
            pid,
            row["ppii_fraction"],
            row["consecutive_pro"],
            row.get("family_hint", "?"),
            str(row.get("peptide_seq", ""))[:20],
        )

    if dry_run:
        _log.info("DRY RUN — manifest not saved")
    else:
        df.to_csv(MANIFEST, index=False)
        _log.info("Manifest saved: %s", MANIFEST)
        print(f"\nPPII filter relaxed: {old_pass} → {new_pass} included structures")
        print(f"Newly added: {len(newly_included)} structures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
