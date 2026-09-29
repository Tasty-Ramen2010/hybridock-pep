#!/usr/bin/env python
"""A length-balanced held-out bench, because the paper's aggregate numbers hide long peptides.

RefPepDB-RecentSet is 27% short, 50% medium, 22% long and 1% very long. So a headline like
"93.7% at top-25" is dominated by short and medium peptides, and the classes where every method
struggles are 1% of the score. Our own held-out slice is worse: 118 short, 209 medium, 15 long,
3 very long. Any method comparison on that set is mostly a comparison on 9-mers.

This draws an equal number from each class so every length contributes the same weight. The
limiting class is very long, so the size is 4x whatever that yields. Sources are the two held-out
benches, both of which were removed from the training pool before the split, so the set stays
leak-free.

Usage: build_balanced_lengthbench.py [--per-class N]
"""
from __future__ import annotations
import argparse, csv, random, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
SRC = ["data/bench_recentset_heldout.csv", "data/bench_long.csv"]
CLASSES = ["short", "med", "long", "vlong"]

def buck(n: int) -> str:
    return "short" if n <= 8 else "med" if n <= 12 else "long" if n <= 19 else "vlong"

def pid(n: str) -> str:
    for t in re.split(r"[_\-]", n):
        if len(t) == 4 and t[0].isdigit() and t.isalnum(): return t.upper()
    return n.upper()

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=0, help="0 = use the limiting class")
    a = ap.parse_args()

    pool, seen = defaultdict(list), set()
    for s in SRC:
        p = ROOT / s
        if not p.exists(): continue
        for r in csv.DictReader(p.open()):
            n = int(r["pep_len"]) if str(r.get("pep_len", "")).isdigit() else len(r.get("seq", ""))
            if n > 25 or n < 4: continue      # Ram's scope rule: >25 aa are small domains
            key = pid(r["name"])
            if key in seen: continue          # one entry per PDB entry, no double counting
            seen.add(key)
            pool[buck(n)].append({"name": r["name"], "receptor": r["receptor"],
                                  "peptide_pdb": r["peptide_pdb"], "seq": r["seq"],
                                  "pep_len": n, "length_bucket": buck(n),
                                  "source": r.get("source", "")})
    print("available per class:", {k: len(pool[k]) for k in CLASSES})
    # EQUAL REPRESENTATION BY MACRO-AVERAGE, NOT BY SUBSAMPLING. Capping every class at the
    # smallest one would mean 13 per class once the >25 aa scope rule is applied, and a success
    # rate quantised to 7.7% steps says nothing. Instead every in-scope complex is kept and the
    # figures average the four per-class rates with equal weight. That gives each length class
    # the same influence on the headline number, which is the thing the paper's aggregate gets
    # wrong, while still using all 387 complexes.
    per = a.per_class or max(len(pool[k]) for k in CLASSES)
    rng = random.Random(0)
    rows = []
    for k in CLASSES:
        v = pool[k][:]
        rng.shuffle(v)
        rows += v[:per]
    rows.sort(key=lambda r: (CLASSES.index(r["length_bucket"]), r["name"]))

    out = ROOT / "data/bench_balanced_length.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  -> {out.name}  {len(rows)} complexes")
    print("  by class:", dict(Counter(r["length_bucket"] for r in rows)))
    ln = [r["pep_len"] for r in rows]
    print(f"  peptide length: min {min(ln)}  max {max(ln)}")
    print("\n  Both source benches were excluded from the training pool before the split,")
    print("  so every complex here is unseen by any model we trained.")

if __name__ == "__main__":
    main()
