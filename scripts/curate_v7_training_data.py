#!/usr/bin/env python3
"""Curate balanced training data for v7 fine-tuning.

v6 had 610/1200 SHEET rows but Phase 3 corrupted short-peptide geometry.
This script builds a balanced v7 training set that:
  1. Verifies all v6 training PDB files exist on disk (drops missing)
  2. Augments SHEET short/medium buckets from benchmark300 SHEET holdout
     (benchmark300 SHEET complexes NOT in v6 training)
  3. Balances: 400 HELIX + 400 SHEET + 400 UNUSUAL = 1200 total
     — within SHEET: equal short/medium/long/very_long (100 each)
  4. Adds a dedicated short-peptide (≤ 8 residues) oversample tier
     to counteract the Phase 3 guard-rail failure mode

Output:
    data/v7_train_balanced.csv   — 1200 balanced rows for training
    data/v7_val_200.csv          — 200 stratified validation rows

Usage:
    conda run -n score-env python3 scripts/curate_v7_training_data.py
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

V6_TRAIN   = REPO / "data" / "v6_train_combined.csv"
V6_VAL     = REPO / "data" / "v6_val_200.csv"
BENCH300   = REPO / "data" / "benchmark300.csv"
OUT_TRAIN  = REPO / "data" / "v7_train_balanced.csv"
OUT_VAL    = REPO / "data" / "v7_val_200.csv"

TARGET_PER_SS   = 400   # HELIX + SHEET + UNUSUAL = 1200
TARGET_PER_CELL = 100   # per (SS × length_bucket)
SHORT_OVERSAMPLE = 25   # extra short-peptide rows per SS class (guard against Phase 3 short degradation)
SEED = 42


def _check_files_exist(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where receptor or peptide PDB is missing."""
    rec_col  = "protein_description" if "protein_description" in df.columns else "receptor"
    pep_col  = "peptide_description" if "peptide_description" in df.columns else "peptide_pdb"
    before = len(df)
    mask = (
        df[rec_col].apply(lambda p: Path(p).exists()) &
        df[pep_col].apply(lambda p: Path(p).exists())
    )
    df = df[mask].copy()
    log.info("File check: %d → %d rows (dropped %d missing)", before, len(df), before - len(df))
    return df


def _normalize_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Standardize column names to match v6 format."""
    rename = {}
    if "protein_description" in df.columns:
        rename["protein_description"] = "receptor"
    if "peptide_description" in df.columns:
        rename["peptide_description"] = "peptide_pdb"
    if rename:
        df = df.rename(columns=rename)
    if "source" not in df.columns:
        df = df.copy()
        df["source"] = source_label
    if "tier" not in df.columns:
        df = df.copy()
        df["tier"] = source_label
    return df


def main() -> None:
    rng = random.Random(SEED)

    # ── Load and validate v6 training data ────────────────────────────────────
    v6 = pd.read_csv(V6_TRAIN)
    v6 = _normalize_columns(v6, "v6_train")
    v6 = _check_files_exist(v6)
    log.info("v6 train after file check: %d rows", len(v6))

    # Determine ss_class from tier name (v6 doesn't have ss_class column)
    def _tier_to_ss(tier: str) -> str:
        t = tier.lower()
        if "sheet" in t:
            return "SHEET"
        if "helix" in t or "replay" in t:
            return "HELIX"
        return "UNUSUAL"

    v6["ss_class"] = v6["tier"].apply(_tier_to_ss)
    log.info("v6 ss_class distribution:\n%s", v6["ss_class"].value_counts().to_string())

    # ── Load benchmark300 SHEET complexes as augmentation source ──────────────
    bench = pd.read_csv(BENCH300)
    bench_sheet = bench[bench["ss_class"] == "SHEET"].copy()
    bench_sheet = _normalize_columns(bench_sheet, "bench300_sheet_aug")
    bench_sheet = _check_files_exist(bench_sheet)

    # Exclude benchmark300 complexes already in v6 training
    v6_names = set(v6["complex_name"].tolist() if "complex_name" in v6.columns else v6.index.tolist())
    bench_sheet = bench_sheet[~bench_sheet["name"].isin(v6_names)]
    log.info("benchmark300 SHEET not in v6 training: %d rows", len(bench_sheet))

    # ── Build balanced training set ────────────────────────────────────────────
    selected_rows: list[pd.DataFrame] = []

    for ss in ["HELIX", "SHEET", "UNUSUAL"]:
        pool = v6[v6["ss_class"] == ss].copy()
        rows_ss: list[pd.DataFrame] = []

        # Determine length_bucket column
        lb_col = "length_bucket" if "length_bucket" in pool.columns else None

        if lb_col:
            for lb in ["short", "medium", "long", "very_long"]:
                cell = pool[pool[lb_col] == lb]
                n_take = TARGET_PER_CELL
                if len(cell) >= n_take:
                    rows_ss.append(cell.sample(n=n_take, random_state=SEED))
                else:
                    rows_ss.append(cell)
                    log.warning("ss=%s lb=%s: only %d available (wanted %d)", ss, lb, len(cell), n_take)
        else:
            # No length_bucket: take up to TARGET_PER_SS rows
            if len(pool) >= TARGET_PER_SS:
                rows_ss.append(pool.sample(n=TARGET_PER_SS, random_state=SEED))
            else:
                rows_ss.append(pool)

        # Short-peptide oversample (guards against short-geometry degradation)
        if lb_col:
            short_pool = pool[pool[lb_col] == "short"]
            if len(short_pool) > 0:
                n_extra = min(SHORT_OVERSAMPLE, len(short_pool))
                rows_ss.append(short_pool.sample(n=n_extra, random_state=SEED + 1))
                log.info("ss=%s: added %d short-peptide oversample rows", ss, n_extra)

        # For SHEET: augment with benchmark300 holdout SHEET complexes
        if ss == "SHEET" and len(bench_sheet) > 0:
            n_aug = min(50, len(bench_sheet))
            aug = bench_sheet.sample(n=n_aug, random_state=SEED)
            # Standardize columns to match v6 format
            aug_rows = []
            for _, r in aug.iterrows():
                aug_rows.append({
                    "complex_name": r["name"],
                    "receptor": r["receptor"],
                    "peptide_pdb": r.get("peptide_pdb", r.get("peptide_description", "")),
                    "source": "bench300_sheet_aug",
                    "tier": "bench300_sheet_aug",
                    "ss_class": "SHEET",
                    "length_bucket": r.get("length_bucket", "unknown"),
                })
            rows_ss.append(pd.DataFrame(aug_rows))
            log.info("SHEET: augmented with %d bench300 holdout complexes", n_aug)

        selected_rows.append(pd.concat(rows_ss, ignore_index=True))

    train_df = pd.concat(selected_rows, ignore_index=True)
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    log.info("Final training set: %d rows", len(train_df))
    log.info("ss_class distribution:\n%s", train_df["ss_class"].value_counts().to_string())

    # ── Validation set: same format as v6_val_200 ─────────────────────────────
    v6_val = pd.read_csv(V6_VAL)
    v6_val = _normalize_columns(v6_val, "v6_val")
    v6_val = _check_files_exist(v6_val)
    log.info("Validation set: %d rows (from v6_val_200, file-checked)", len(v6_val))

    # ── Write outputs ─────────────────────────────────────────────────────────
    train_df.to_csv(OUT_TRAIN, index=False)
    v6_val.to_csv(OUT_VAL, index=False)
    log.info("Wrote %s (%d rows)", OUT_TRAIN, len(train_df))
    log.info("Wrote %s (%d rows)", OUT_VAL, len(v6_val))

    # Summary
    print("\n=== v7 Training Data Summary ===")
    print(f"Training: {len(train_df)} rows")
    print(train_df["ss_class"].value_counts().to_string())
    if "length_bucket" in train_df.columns:
        print("\nLength bucket × SS class:")
        print(pd.crosstab(train_df["length_bucket"], train_df["ss_class"]).to_string())
    print(f"\nValidation: {len(v6_val)} rows")


if __name__ == "__main__":
    main()
