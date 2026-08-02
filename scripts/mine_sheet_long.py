#!/usr/bin/env python3
"""mine_sheet_long.py — Mine beta-sheet + very-long peptides from on-disk data.

The formatted training set (datasets/training_formatted_peppc, PepPC/PepPC-F
fragments) is 71% loop / 29% helix / 0% beta-sheet. But we already have ~1046
experimental crystal complexes on disk in datasets/RefPepDB-RecentSet/ that were
never formatted into training. This script SS-classifies those peptides to see how
many beta-sheet and very-long (>=20-mer) examples we can recover from current data.

HARD dedup against every data/benchmark*.csv (by complex name AND peptide sequence)
so we never train on what we evaluate on (the leakage that inflated past numbers).

Reuses classify_ss / get_seq_from_pdb / length_bucket from build_benchmark150.py.

Usage:
    python3 scripts/mine_sheet_long.py [--out data/mined_sheet_long.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from build_benchmark150 import classify_ss, get_seq_from_pdb, length_bucket  # noqa: E402

RECENTSET = ROOT / "datasets" / "RefPepDB-RecentSet"


def load_benchmark_blocklist() -> tuple[set[str], set[str]]:
    """Return (names, peptide_seqs) that appear in ANY data/benchmark*.csv."""
    names: set[str] = set()
    seqs: set[str] = set()
    for p in glob.glob(str(ROOT / "data" / "benchmark*.csv")):
        try:
            rows = list(csv.DictReader(open(p)))
        except OSError:
            continue
        for r in rows:
            for k in ("name", "complex_name"):
                if r.get(k):
                    names.add(r[k].lower())
            if r.get("seq"):
                seqs.add(r["seq"].upper())
    return names, seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "mined_sheet_long.csv")
    ap.add_argument("--min-len", type=int, default=4)
    args = ap.parse_args()

    bench_names, bench_seqs = load_benchmark_blocklist()
    print(f"[mine] benchmark blocklist: {len(bench_names)} names, {len(bench_seqs)} seqs")

    dirs = sorted(d for d in RECENTSET.iterdir() if d.is_dir())
    print(f"[mine] scanning {len(dirs)} RefPepDB-RecentSet complexes")

    by_ss: dict[str, int] = defaultdict(int)
    by_len: dict[str, int] = defaultdict(int)
    candidates: list[dict] = []
    skipped = leaked = 0

    for d in dirs:
        pep = d / f"{d.name}_peptide.pdb"
        rec = d / f"{d.name}_protein_pocket.pdb"
        if not pep.exists() or not rec.exists():
            skipped += 1
            continue
        seq = get_seq_from_pdb(pep)
        if not seq or len(seq) < args.min_len:
            skipped += 1
            continue
        if d.name.lower() in bench_names or seq.upper() in bench_seqs:
            leaked += 1
            continue

        ss, fH, fE, fP = classify_ss(pep, seq)
        lb = length_bucket(len(seq))
        by_ss[ss] += 1
        by_len[lb] += 1

        # keep the gap-filling targets: beta-sheet OR very-long
        if ss == "SHEET" or lb == "very_long":
            candidates.append({
                "name": d.name, "receptor": str(rec), "peptide_pdb": str(pep),
                "seq": seq, "pep_len": len(seq), "ss_class": ss,
                "frac_H": round(fH, 3), "frac_E": round(fE, 3), "frac_P": round(fP, 3),
                "length_bucket": lb,
            })

    print(f"\n[mine] scanned ok, skipped={skipped}, dropped_for_leakage={leaked}")
    print("SS distribution (RecentSet, leakage-filtered):")
    for k in ("HELIX", "SHEET", "UNUSUAL"):
        print(f"  {k:8s}: {by_ss[k]}")
    print("Length distribution:")
    for k in ("short", "medium", "long", "very_long"):
        print(f"  {k:10s}: {by_len[k]}")

    n_sheet = sum(1 for c in candidates if c["ss_class"] == "SHEET")
    n_vlong = sum(1 for c in candidates if c["length_bucket"] == "very_long")
    print(f"\n[mine] GAP-FILL candidates: {len(candidates)} unique "
          f"(SHEET={n_sheet}, very_long={n_vlong}, overlap counted once)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if candidates:
        fields = ["name", "receptor", "peptide_pdb", "seq", "pep_len",
                  "ss_class", "frac_H", "frac_E", "frac_P", "length_bucket"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(candidates)
        print(f"[mine] wrote {len(candidates)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
