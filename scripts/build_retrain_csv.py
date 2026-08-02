#!/usr/bin/env python3
"""build_retrain_csv.py — Assemble the long-peptide-targeted retrain set.

Target (from true baseline): fix LONG/VERY-LONG placement (>=13mer, currently 0/22 Hit@2A).
- TARGET data: Propedia long/vlong (data/propedia_train.csv, 2224 real complexes >=13mer).
- REPLAY anchor: real short/medium complexes from RefPepDB-RecentSet (anti-forgetting; the
  model is already good there and we must not regress it).
- LEAKAGE GUARD: exclude the 52 benchmark_real complexes by id AND by peptide sequence.
- SPLIT: 90/10 train/val with NO peptide sequence shared across the split.

Output: data/retrain_train.csv, data/retrain_val.csv (cols: complex_name, protein_description,
peptide_description, source).

Usage: python scripts/build_retrain_csv.py [--replay 800]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_benchmark150 import get_seq_from_pdb, length_bucket  # noqa: E402

RECENTSET = ROOT / "datasets" / "RefPepDB-RecentSet"


def load_exclude() -> tuple[set[str], set[str]]:
    ids, seqs = set(), set()
    for r in csv.DictReader(open(ROOT / "data" / "benchmark_real.csv")):
        ids.add(r["name"].lower())
        seqs.add(r["seq"].upper())
    return ids, seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=int, default=800, help="real short/med replay complexes")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ex_ids, ex_seqs = load_exclude()
    print(f"[retrain] excluding {len(ex_ids)} benchmark ids / {len(ex_seqs)} seqs")

    rows: list[dict] = []
    seen_seq: set[str] = set()

    # 1) TARGET: Propedia long/vlong
    n_prop = 0
    for r in csv.DictReader(open(ROOT / "data" / "propedia_train.csv")):
        pep = r["peptide_description"]
        seq = get_seq_from_pdb(Path(pep)) or ""
        seq = seq.upper()
        pid = r["complex_name"].split("_")[1].lower() if "_" in r["complex_name"] else ""
        if pid in ex_ids or seq in ex_seqs or not seq:
            continue
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        rows.append({**{k: r[k] for k in ("complex_name", "protein_description",
                                          "peptide_description")}, "source": "propedia_long"})
        n_prop += 1

    # 2) REPLAY: real short/medium from RecentSet (not benchmark)
    replay_pool = []
    for d in sorted(RECENTSET.iterdir()):
        if not d.is_dir() or d.name.lower() in ex_ids:
            continue
        pep = d / f"{d.name}_peptide.pdb"
        rec = d / f"{d.name}_protein_pocket.pdb"
        if not pep.exists() or not rec.exists():
            continue
        seq = (get_seq_from_pdb(pep) or "").upper()
        if not seq or seq in ex_seqs or seq in seen_seq:
            continue
        if length_bucket(len(seq)) in ("short", "medium"):
            replay_pool.append((d.name, str(rec.resolve()), str(pep.resolve()), seq))
    rng.shuffle(replay_pool)
    n_replay = 0
    for name, rec, pep, seq in replay_pool[: args.replay]:
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        rows.append({"complex_name": name, "protein_description": rec,
                     "peptide_description": pep, "source": "recentset_replay"})
        n_replay += 1

    # 3) split by sequence (rows already unique seq)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val, train = rows[:n_val], rows[n_val:]

    fields = ["complex_name", "protein_description", "peptide_description", "source"]
    for name, data in [("retrain_train.csv", train), ("retrain_val.csv", val)]:
        with open(ROOT / "data" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    print(f"[retrain] propedia_long={n_prop}  recentset_replay={n_replay}")
    print(f"[retrain] TRAIN={len(train)}  VAL={len(val)}  (unique seqs, benchmark-excluded)")
    print(f"[retrain] wrote data/retrain_train.csv + data/retrain_val.csv")


if __name__ == "__main__":
    main()
