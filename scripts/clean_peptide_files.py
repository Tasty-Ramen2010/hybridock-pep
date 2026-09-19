#!/usr/bin/env python3
"""Strip non-peptide junk from peptide PDBs before they reach training.

Why: get() extracts the peptide SEQUENCE from the peptide PDB via MDAnalysis residues.
Waters and other HETATM groups become residues -- unmapped ones turn into '[HOH]'-style
tokens that PeptideBuilder then expands, so the training graph ends up with more residues
than the real peptide (8BWE: pep_len 18, file 21 residues, graph 33). Measured Sep-12:
95.5% of recent_2024_2026 and 95.8% of ppii_enriched files are affected.

Keeps only residues that have a complete backbone (N, CA, C, O) -- the same test
get_ori_peptide_feature_mda applies when deciding which residues are amino acids -- and
drops hydrogens. Writes <stem>_clean.pdb beside the original and emits a new pool CSV.
Never overwrites source data.
"""
from __future__ import annotations
import csv, os, sys, collections

BB = {"N", "CA", "C", "O"}

def clean(src: str, dst: str) -> tuple[int, int]:
    res: dict = collections.OrderedDict()
    for l in open(src):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        if l[76:78].strip() == "H" or l[12:16].strip().startswith("H"):
            continue
        res.setdefault((l[21], l[22:27]), []).append(l)
    keep = [(k, v) for k, v in res.items()
            if BB <= {x[12:16].strip() for x in v}]
    with open(dst, "w") as fh:
        for _, v in keep:
            fh.writelines(v)
        fh.write("END\n")
    return len(res), len(keep)

def main() -> None:
    pool, out_csv = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(pool)))
    stats = collections.Counter(); changed = 0
    for r in rows:
        src = r["peptide_description"]
        dst = src.replace(".pdb", "_clean.pdb")
        try:
            n_all, n_keep = clean(src, dst)
        except Exception as exc:
            stats[f"fail:{type(exc).__name__}"] += 1
            continue
        if n_keep == 0:                     # never hand training an empty peptide
            os.remove(dst); stats["empty-after-clean"] += 1
            continue
        if n_keep != n_all:
            changed += 1; stats[f"{r['source']}:trimmed"] += 1
        r["peptide_description"] = dst
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv}: {len(rows)} rows, {changed} files trimmed")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
