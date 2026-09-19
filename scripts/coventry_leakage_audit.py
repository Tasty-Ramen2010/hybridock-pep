#!/usr/bin/env python
"""Leakage audit for the Coventry challenge.

Coventry's exact words: "Hopefully you didn't find these sequences when you were
looking for your training data."  This answers that, before any prediction is made.

Two separate questions, audited separately, because the molecules play different
roles and a peptide must not be compared against a whole protein:

  PEPTIDE SIDE   is any of the 18 target peptides present among the peptides we
                 trained or benchmarked on?  Compared against every peptide in the
                 RAPiDock finetune train/val sets (read out of the peptide PDBs),
                 the held-out RecentSet and balanced benchmarks, and every
                 sequence-bearing CSV under data/ (the scorer's PDBbind / PPIKB /
                 SKEMPI pools).

  RECEPTOR SIDE  is any of the 18 designed binders present among the receptors we
                 trained on?  Compared against receptor sequences read out of the
                 training receptor pocket PDBs.

SCORING.  Identity is normalised by the LONGER of the two sequences, so a generic
3-mer sitting inside a 260-aa design scores 3/260, not 100%.  A naive local-identity
score normalised by the shorter sequence reports such fragments as exact hits and is
worthless here.  We also report the longest common substring, which is the quantity
that actually matters for "did the model already see this motif".

Usage: coventry_leakage_audit.py [workers]
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
IDENT_FLAG = 0.60          # the same 60% cutoff our scorer's held-out split uses
MIN_LCS = 6                # shorter shared runs are generic, not evidence of leakage
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def seq_from_pdb(path: str) -> str:
    """One-letter sequence of a PDB, in residue order, all chains concatenated."""
    out, seen = [], set()
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                    key = (line[21], line[22:27])
                    if key not in seen:
                        seen.add(key)
                        out.append(AA3.get(line[17:20].strip().upper(), "X"))
    except OSError:
        return ""
    return "".join(out)


def compare(q: str, s: str) -> tuple[float, int]:
    """(identity normalised by the longer sequence, longest common substring length).

    The normalisation is the whole point: it asks "are these the same molecule",
    not "does this short thing appear somewhere in that long thing".
    """
    if not q or not s:
        return 0.0, 0
    lo, hi = (q, s) if len(q) <= len(s) else (s, q)
    best = 0
    for off in range(len(hi) - len(lo) + 1):
        m = sum(1 for i, c in enumerate(lo) if hi[off + i] == c)
        best = max(best, m)
    ident = best / max(len(q), len(s))
    match = SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
    return ident, match.size


QUERIES: list[tuple[str, str]] = []


def _score_one(args: tuple[str, str, str]) -> list[tuple]:
    src, name, seq = args
    hits = []
    for qname, qseq in QUERIES:
        if not seq:
            continue
        ident, lcs = compare(qseq, seq)
        if seq == qseq:
            hits.append((src, name, qname, 1.0, lcs, "EXACT"))
        elif ident >= IDENT_FLAG or lcs >= MIN_LCS:
            hits.append((src, name, qname, ident, lcs, "FLAG"))
    return hits


def load_pool(workers: int) -> tuple[list, list]:
    """(peptide pool, receptor pool) as (source, name, sequence) triples."""
    peps: list[tuple[str, str, str]] = []
    recs: list[tuple[str, str, str]] = []

    pep_jobs, rec_jobs = [], []
    for label, path in [("finetune_train", "data/longft_train_scope25_authorsP.csv"),
                        ("finetune_val", "data/longft_val_scope25_authorsP.csv")]:
        f = ROOT / path
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            pep_jobs.append((label, r["complex_name"], r["peptide_description"]))
            rec_jobs.append((label, r["complex_name"], r["protein_description"]))
    print(f"peptide PDBs: {len(pep_jobs)}   receptor PDBs: {len(rec_jobs)}")

    with ProcessPoolExecutor(workers) as ex:
        ps = list(ex.map(seq_from_pdb, [j[2] for j in pep_jobs], chunksize=64))
        rs = list(ex.map(seq_from_pdb, [j[2] for j in rec_jobs], chunksize=64))
    peps += [(a, b, s) for (a, b, _), s in zip(pep_jobs, ps)]
    recs += [(a, b, s) for (a, b, _), s in zip(rec_jobs, rs)]
    print(f"  read {sum(1 for _,_,s in peps if s)} peptide / "
          f"{sum(1 for _,_,s in recs if s)} receptor sequences")

    for label, path in [("bench_recentset", "data/bench_recentset_heldout.csv"),
                        ("bench_balanced", "data/bench_balanced_post2020.csv")]:
        f = ROOT / path
        if f.exists():
            rd = list(csv.DictReader(open(f)))
            if rd and "seq" in rd[0]:
                peps += [(label, r["name"], r["seq"].strip().upper()) for r in rd]
                print(f"  {label}: {len(rd)}")

    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.name.startswith("coventry_") or "longft_" in f.name:
            continue
        try:
            rd = csv.DictReader(open(f, errors="replace"))
            cols = rd.fieldnames or []
        except OSError:
            continue
        scol = next((c for c in cols if c and c.lower() in
                     ("peptide", "seq", "sequence", "peptide_seq", "pep_seq")), None)
        if not scol:
            continue
        ncol = next((c for c in cols if c and c.lower() in
                     ("pdb", "name", "complex", "complex_name", "id")), scol)
        n = 0
        for r in rd:
            s = (r.get(scol) or "").strip().upper()
            if s and set(s) <= set("ACDEFGHIKLMNPQRSTVWYXUO") and len(s) >= 4:
                peps.append((f"csv:{f.name}", r.get(ncol) or "?", s)); n += 1
        if n:
            print(f"  csv:{f.name}: {n}")
    return peps, recs


def report(title: str, pool: list, queries: list[tuple[str, str]], workers: int) -> None:
    global QUERIES
    QUERIES = queries
    with ProcessPoolExecutor(workers) as ex:
        allhits = list(ex.map(_score_one, pool, chunksize=256))
    hits = [h for hs in allhits for h in hs]

    print("\n" + "=" * 86)
    print(f"{title}   ({len(pool)} sequences audited, {len(queries)} queries)")
    print("=" * 86)
    if not hits:
        print(f"CLEAN: no exact match, nothing at or above {IDENT_FLAG:.0%} "
              f"length-normalised identity, and no shared run of {MIN_LCS}+ residues.")
    else:
        print(f"{len(hits)} flagged (identity normalised by the longer sequence):")
        for h in sorted(hits, key=lambda x: (-x[3], -x[4]))[:25]:
            print(f"   {h[5]:6s} ident {h[3]:5.1%}  LCS {h[4]:3d}  {h[2]:18s} <-> {h[0]}/{h[1]}")

    print(f"\nclosest audited sequence per query:")
    for qname, qseq in queries:
        best, blcs, bn, bs = 0.0, 0, "", ""
        for _, name, s in pool:
            if not s:
                continue
            i, l = compare(qseq, s)
            if (i, l) > (best, blcs):
                best, blcs, bn, bs = i, l, name, s
        print(f"  {qname:22s} len {len(qseq):3d}  best ident {best:5.1%}  "
              f"longest shared run {blcs:2d}  vs {bn} ({bs[:30]})")


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tgt = list(csv.DictReader(open(ROOT / "data/coventry_targets.csv")))
    bnd = list(csv.DictReader(open(ROOT / "data/coventry_binders.csv")))
    peps, recs = load_pool(workers)

    report("PEPTIDE SIDE - the 18 target peptides vs everything we trained/benchmarked on",
           peps, [(f"target:{r['name']}", r["sequence"]) for r in tgt], workers)
    report("RECEPTOR SIDE - the 18 designed binders vs our training receptors",
           recs, [(f"binder:{r['name']}", r["sequence"]) for r in bnd], workers)


if __name__ == "__main__":
    main()
