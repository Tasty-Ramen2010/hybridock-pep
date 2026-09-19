#!/usr/bin/env python
"""Split the designed-complex set and build an oversampled finetune CSV.

The experiment: does adding de novo designed protein/peptide complexes to the finetune mix
teach the model the geometry it fails on (9CCE, and by extension Coventry's 18)?

BE HONEST ABOUT THE ODDS.  We have 41 usable designed complexes against 6,009 existing
training complexes.  Oversampling cannot create information that is not there: 29 unique
structures repeated N times is still 29 structures, and the main thing it buys is a good
chance of memorising them.  This is run as a cheap, properly-controlled experiment, not
because it is expected to work.

CONTROLS THAT MAKE THE RESULT READABLE:
  * A held-out designed TEST split that is never trained on, so "it improved" can be
    checked on designed complexes the model has not seen.  Judging on 9CCE alone (n=1)
    would prove nothing.
  * 7dng and 8gjg are forced into TRAIN, never TEST -- they are already in the existing
    finetune set, so scoring them as held-out would be measuring leakage.
  * 9CCE is excluded upstream by the leakage guard and stays a pure external check.
  * The validation CSV is left untouched so val loss stays comparable to earlier runs.

Usage: build_denovo_finetune_csv.py [--oversample 20] [--test 12] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
ALL = ROOT / "data/denovo_pep_all.csv"
BASE_TRAIN = ROOT / "data/longft_train_scope25_authorsP.csv"
ALREADY_TRAINED = {"7dng", "8gjg"}      # present in the existing finetune set
COLS = ["complex_name", "protein_description", "peptide_description", "source",
        "pep_len", "ss_class", "length_bucket"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oversample", type=int, default=20)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(ALL)))
    # Split by PDB ID, not by row. Several entries contribute more than one complex
    # (different peptide chains of the same structure); shuffling rows would put the same
    # protein on both sides of the split and the "held-out" number would be leakage.
    by_pdb: dict[str, list] = {}
    for r in rows:
        by_pdb.setdefault(r["pdb"], []).append(r)
    ids = sorted(p for p in by_pdb if p not in ALREADY_TRAINED)
    random.Random(a.seed).shuffle(ids)
    test_ids, train_ids = set(ids[:a.test]), set(ids[a.test:]) | ALREADY_TRAINED
    test = [r for p in test_ids for r in by_pdb[p]]
    train_new = [r for p in train_ids if p in by_pdb for r in by_pdb[p]]
    assert not (test_ids & train_ids), "pdb on both sides of the split"

    base = list(csv.DictReader(open(BASE_TRAIN)))
    out_rows = list(base)
    for _ in range(a.oversample):
        for r in train_new:
            out_rows.append({k: r[k] for k in COLS})

    out_train = ROOT / "data/longft_train_denovo.csv"
    with out_train.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in COLS})

    out_test = ROOT / "data/denovo_pep_test.csv"
    with out_test.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in test:
            w.writerow(r)

    n_new = len(train_new) * a.oversample
    print(f"designed complexes usable      : {len(rows)} rows / {len(by_pdb)} PDB ids")
    print(f"  held out as TEST (never seen): {len(test)} rows / {len(test_ids)} pdb"
          f"  -> {out_test.name}")
    print(f"  in TRAIN                     : {len(train_new)} rows / "
          f"{len([p for p in train_ids if p in by_pdb])} pdb"
          f"  (incl. forced {sorted(ALREADY_TRAINED)})")
    print(f"  oversampled x{a.oversample}             : {n_new} rows")
    print(f"base finetune rows             : {len(base)}")
    print(f"TOTAL train rows               : {len(out_rows)}  "
          f"({100 * n_new / len(out_rows):.1f}% designed)")
    print(f"wrote {out_train}")
    print(f"TEST pdbs: {[r['pdb'] for r in test]}")


if __name__ == "__main__":
    main()
