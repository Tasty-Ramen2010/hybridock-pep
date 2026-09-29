#!/usr/bin/env python
"""Filter the training corpus by INTERFACE QUALITY instead of growing it.

WHY THIS IS THE OPPOSITE OF WHAT WE KEPT DOING. Six corpus-enrichment attempts have now failed,
and the most recent is the clearest: the 17,232-complex full corpus scores a median 3.03 A on
the length-balanced bench against 2.01 A for the curated 4,245-complex set. More data made it
worse, every time.

WHAT THE FAILURES LOOK LIKE. The 76 bench complexes that NO method solves -- not ours, not
RAPiDock, and not Boltz-2, which is 466x bigger and was trained on these exact structures --
share a signature: burial per peptide residue 2.87 against 3.81 for everything else, receptors
190 residues against 250, peptides 11.6 against 9.7, and 14% from propedia against 6%. Shallow,
weakly-determined interfaces. A peptide with almost no burial is either a crystal-packing
artifact or a genuinely floppy binder, and in both cases the "true" pose is barely determined by
the structure, so training on it teaches the model to denoise toward an answer that was never
there.

WHAT IS FILTERED, and each is a defect, not a preference:
  burial     mean receptor residues within 8 A per peptide residue, in the bottom --drop-frac
             of its OWN length class. A flat floor cannot be used: burial per residue falls with
             peptide length, so a single threshold keeps 84% of short peptides and 40% of very
             long ones, which strips out exactly the class the model is worst at. Ranking within
             class removes the worst example relative to its peers instead.
  breaks     consecutive peptide Ca-Ca above 4.2 A: a chain break in the ground truth itself
  contacts   zero receptor residues within 8 A: there is no interface to learn at all. 8 A,
             not 5 A: burial already measures DEPTH, so a second depth-like criterion at 5 A
             would double-penalise shallow-but-real surface binders (it dropped 2,073 complexes
             that had already passed the burial floor).

The thresholds are NOT fitted to the benchmark. --min-burial defaults to 2.0, which is below
the 2.87 mean of the universally-missed set, so it removes the clearly-degenerate tail rather
than everything that looks like it. Any threshold tuned against the bench would make the
resulting training run unreportable.

Usage: filter_corpus_quality.py --in data/fullcorpus_train.csv --out data/qual_train.csv
"""
from __future__ import annotations

import argparse, csv, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def ca(p: str) -> np.ndarray:
    seen, xyz = set(), []
    try:
        fh = open(p, errors="ignore")
    except OSError:
        return np.zeros((0, 3))
    for l in fh:
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    fh.close()
    return np.array(xyz) if xyz else np.zeros((0, 3))


def measure(row: dict) -> dict:
    rec, pep = row["protein_description"], row["peptide_description"]
    if not (os.path.exists(rec) and os.path.exists(pep)):
        return {"name": row["complex_name"], "ok": False, "why": "missing_file"}
    P, R = ca(pep), ca(rec)
    if len(P) < 3 or len(R) < 3:
        return {"name": row["complex_name"], "ok": False, "why": "unparseable"}
    D = np.linalg.norm(R[:, None] - P[None], axis=2)
    return {"name": row["complex_name"], "ok": True,
            "burial": float((D < 8.0).sum()) / len(P),
            "breaks": int((np.linalg.norm(np.diff(P, axis=0), axis=1) > 4.2).sum()),
            "contacts": int((D.min(1) < 8.0).sum()),
            "contacts5": int((D.min(1) < 5.0).sum()),
            "npep": len(P), "bucket": row.get("length_bucket", "?")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-frac", type=float, default=0.20,
                    help="fraction of each length class to drop by lowest burial")
    ap.add_argument("--max-breaks", type=int, default=0)
    ap.add_argument("--min-contacts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp)))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        stats = list(ex.map(measure, rows, chunksize=32))

    import collections
    # burial cut is per length class, at the --drop-frac quantile of that class
    by_cls = collections.defaultdict(list)
    for s_ in stats:
        if s_["ok"]:
            by_cls[s_["bucket"]].append(s_["burial"])
    cut = {k: float(np.quantile(v, a.drop_frac)) for k, v in by_cls.items() if v}

    keep, drop = [], collections.Counter()
    for r, s_ in zip(rows, stats):
        if not s_["ok"]:
            drop[s_["why"]] += 1
            continue
        if s_["burial"] < cut.get(s_["bucket"], 0.0):
            drop["burial"] += 1
        elif s_["breaks"] > a.max_breaks:
            drop["breaks"] += 1
        elif s_["contacts"] < a.min_contacts:
            drop["contacts"] += 1
        else:
            keep.append(r)
    print("   per-class burial cut: " + "  ".join(f"{k}<{v:.2f}" for k, v in sorted(cut.items())))

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)

    good = [s for s in stats if s["ok"]]
    print(f"{a.inp}: {len(rows)} in -> {len(keep)} kept ({100*len(keep)/len(rows):.0f}%)")
    for k, v in drop.items():
        if v:
            print(f"   dropped {v:6d}  {k}")
    import collections
    kept_names = {r["complex_name"] for r in keep}
    bin_all = collections.Counter(s["bucket"] for s in good)
    bin_keep = collections.Counter(s["bucket"] for s in good if s["name"] in kept_names)
    print("   retention by length class (a filter that eats the long peptides is a bad filter):")
    for k in ("short", "med", "long", "vlong"):
        if bin_all.get(k):
            print(f"     {k:<6} {bin_keep.get(k,0):5d}/{bin_all[k]:5d} = {100*bin_keep.get(k,0)/bin_all[k]:3.0f}%")
    if good:
        b = np.array([s["burial"] for s in good])
        print(f"   burial per residue, all complexes: median {np.median(b):.2f}  "
              f"p10 {np.percentile(b,10):.2f}")


if __name__ == "__main__":
    main()
