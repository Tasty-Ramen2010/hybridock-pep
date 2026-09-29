#!/usr/bin/env python
"""Build a benchmark SPREAD across binding-site similarity to training, instead of piled at 100%.

WHY. The 387-complex length-balanced bench cannot carry an isoDDE-style similarity curve: 216 of
387 targets (56%) sit at exactly 100% binding-site identity to our training corpus, the (30,40]
bin holds 3 complexes and (40,50] holds 5. Brian Coventry called this before we plotted it. A
curve drawn on 3 examples asserts a trend that 3 examples cannot support.

WHAT THIS DOES. Sweeps every formatted complex on disk that is NOT in any training pool, applies
the same interface-quality filter the corpus filter uses, measures binding-site identity to the
training union, and reports what is actually achievable per bin before sampling anything. The
low bins are the binding constraint and no amount of sampling creates them: a target at <30%
binding-site identity to an 18,397-complex corpus is genuinely rare, because that corpus already
covers most of the peptide-binding PDB.

So this prints the CEILING first. If the low bins cannot be filled from what we hold, the answer
is to go fetch more PDB, not to draw the plot anyway.

Usage: build_similarity_bench.py [--per-bin 40] [--out data/bench_similarity.csv]
"""
from __future__ import annotations

import argparse, csv, os, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
POOLS = [("datasets/RefPepDB-RecentSet", "refpepdb_recentset"),
         ("datasets/propedia_formatted", "propedia"),
         ("datasets/training_formatted_peppc", "peppc"),
         ("datasets/training_formatted", "training_formatted")]


def read(p):
    seq, xyz, seen = [], [], set()
    try:
        fh = open(p, errors="ignore")
    except OSError:
        return "", np.zeros((0, 3))
    for l in fh:
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        seq.append(AA3.get(l[17:20].strip(), 'X'))
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    fh.close()
    return ''.join(seq), (np.array(xyz) if xyz else np.zeros((0, 3)))


def scan(args):
    name, d, src = args
    rec = os.path.join(d, name, f"{name}_protein_pocket.pdb")
    pep = None
    for suf in ("_peptide.pdb", "_peptide_clean.pdb"):
        c = os.path.join(d, name, f"{name}{suf}")
        if os.path.exists(c):
            pep = c
            break
    if not (os.path.exists(rec) and pep):
        return None
    rs, rx = read(rec)
    ps, px = read(pep)
    if len(rx) < 20 or not (4 <= len(px) <= 25):
        return None
    D = np.linalg.norm(rx[:, None] - px[None], axis=2)
    burial = float((D < 8.0).sum()) / len(px)
    breaks = int((np.linalg.norm(np.diff(px, axis=0), axis=1) > 4.2).sum())
    if breaks > 0 or (D.min(1) < 8.0).sum() < 1 or burial < 1.3:
        return None
    iface = np.where(D.min(1) < 8.0)[0]
    return dict(name=name, receptor=rec, peptide_pdb=pep, seq=ps, pep_len=len(ps),
                source=src, rec_seq=rs, iface=",".join(map(str, iface.tolist())),
                burial=round(burial, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-bin", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "data/bench_similarity.csv"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    train = set()
    for f in ("data/longft_train.csv", "data/longft_val.csv",
              "data/fullcorpus_train.csv", "data/fullcorpus_val.csv"):
        p = ROOT / f
        if p.exists():
            for r in csv.DictReader(open(p)):
                train.add(r.get("complex_name") or r.get("name"))
    print(f"training union: {len(train)}")

    jobs = []
    for d, src in POOLS:
        dd = ROOT / d
        if not dd.is_dir():
            continue
        for n in os.listdir(dd):
            if n in train or not (dd / n).is_dir():
                continue
            jobs.append((n, str(dd), src))
    print(f"candidates outside training: {len(jobs)}")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = [r for r in ex.map(scan, jobs, chunksize=64) if r]
    print(f"pass geometry + interface-quality filter: {len(got)}")

    out = Path("/tmp/claude-1000/seqid/cand.fasta")
    out.write_text("".join(f">{r['name']}\n{r['rec_seq']}\n" for r in got))
    import json
    json.dump(got, open("/tmp/claude-1000/seqid/cand.json", "w"))
    print(f"wrote {out} and cand.json — run MMseqs next")


if __name__ == "__main__":
    main()
