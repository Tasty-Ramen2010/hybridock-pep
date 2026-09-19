#!/usr/bin/env python
"""Where do we fall apart on the Coventry grid, and does anything predict it?

Takes our per-peptide result (rank of the correct binder among 18) and correlates it
against every property of the peptide, the binder, the experiment and our own pipeline
that could plausibly explain a miss. The point is to find out whether our failures are
structured -- concentrated on a chemistry or a class we could name and fix -- or whether
they are scattered, which would mean the problem is upstream in pose generation and not
specific to any peptide type.

Peptide classes (assigned from the sequences in Table S3A, not from the results):
  repeat     a single dipeptide repeated                      n1 n2 n3 n4 n7
  chimeric   two or more such motifs concatenated             pc2 pc11 pc12 pc17 pc18 pc21 pc26
  wordlike   an arbitrary sequence (lab members' names)       pc28 pc34 pc35 pc43 pc44 pc46

Usage: coventry_failure_correlation.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
CLASS = {**{p: "repeat" for p in ("n1", "n2", "n3", "n4", "n7")},
         **{p: "chimeric" for p in ("pc2", "pc11", "pc12", "pc17", "pc18", "pc21", "pc26")},
         **{p: "wordlike" for p in ("pc28", "pc34", "pc35", "pc43", "pc44", "pc46")}}
BLI = {"n1": 0.5, "n2": 18, "n3": 72, "n4": 350, "n7": 5, "pc2": 84, "pc11": 50,
       "pc12": 36, "pc18": 10, "pc17": 45, "pc21": 27, "pc26": 200, "pc28": 9,
       "pc34": 180, "pc35": 90, "pc43": 9, "pc44": 35, "pc46": 37}
HYD = set("AVLIMFWYC")
ARO = set("FWY")
POS = set("KR")
NEG = set("DE")


def spearman(a, b):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def entropy(seq):
    c = Counter(seq)
    n = len(seq)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def main() -> None:
    tg = {r["name"]: r for r in csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    bd = {r["grid_label"][:-1]: r for r in
          csv.DictReader(open(ROOT / "data/coventry_binders.csv"))}
    fold = json.loads((ROOT / "logs/coventry_fold.json").read_text())
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    S, NC = {}, {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            p, b = d["name"].split("__")
            S[(p, b[:-1])] = d["best_iface"]
            NC[(p, b[:-1])] = d["n_clean"]
    rm = {p: st.mean([S[(p, b)] for b in ORDER]) for p in ORDER}
    cm = {b: st.mean([S[(p, b)] for p in ORDER]) for b in ORDER}
    g = st.mean(S.values())
    W = {k: S[k] - rm[k[0]] - cm[k[1]] + g for k in S}

    rows = []
    for p in ORDER:
        seq = tg[p]["sequence"]
        b = bd[p]
        rank = sorted(ORDER, key=lambda x: W[(p, x)]).index(p) + 1
        rows.append({
            "pep": p, "class": CLASS[p], "rank": rank,
            "rank_raw": sorted(ORDER, key=lambda x: S[(p, x)]).index(p) + 1,
            "len": len(seq),
            "charge": sum(1 for c in seq if c in POS) - sum(1 for c in seq if c in NEG),
            "abs_charge": abs(sum(1 for c in seq if c in POS)
                              - sum(1 for c in seq if c in NEG)),
            "f_hyd": sum(1 for c in seq if c in HYD) / len(seq),
            "f_aro": sum(1 for c in seq if c in ARO) / len(seq),
            "f_pro": seq.count("P") / len(seq),
            "f_gly": seq.count("G") / len(seq),
            "entropy": entropy(seq),
            "n_distinct": len(set(seq)),
            "binder_len": len(b["sequence"]),
            "plddt": fold.get(b["name"], {}).get("plddt", float("nan")),
            "kd_nanobit_nM": float(truth[(p, p)]["kd_M"]) * 1e9,
            "kd_bli_nM": BLI[p],
            "n_clean": NC[(p, p)],
            "cog_score": S[(p, p)],
        })

    print("=" * 96)
    print("WHERE WE FALL APART - per peptide (rank of the CORRECT binder among 18, "
          "two-way normalised)")
    print("=" * 96)
    hdr = (f"{'pep':>6}{'class':>10}{'rank':>6}{'len':>5}{'chg':>5}{'f_hyd':>7}"
           f"{'f_aro':>7}{'entropy':>9}{'Kd nM':>8}{'n_clean':>9}{'cog E':>8}")
    print(hdr)
    for r in sorted(rows, key=lambda x: x["rank"]):
        print(f"{r['pep']:>6}{r['class']:>10}{r['rank']:>6}{r['len']:>5}{r['charge']:>5}"
              f"{r['f_hyd']:>7.2f}{r['f_aro']:>7.2f}{r['entropy']:>9.2f}"
              f"{r['kd_nanobit_nM']:>8.1f}{r['n_clean']:>9}{r['cog_score']:>8.1f}")

    print(f"\n{'=' * 96}\nCORRELATION with our rank (positive = that property makes us WORSE)")
    print("=" * 96)
    feats = ["len", "charge", "abs_charge", "f_hyd", "f_aro", "f_pro", "f_gly",
             "entropy", "n_distinct", "binder_len", "plddt", "kd_nanobit_nM",
             "kd_bli_nM", "n_clean", "cog_score"]
    rk = [r["rank"] for r in rows]
    out = []
    for f in feats:
        v = [r[f] for r in rows]
        if any(x != x for x in v):
            keep = [(a, b) for a, b in zip(v, rk) if a == a]
            s = spearman([a for a, _ in keep], [b for _, b in keep])
        else:
            s = spearman(v, rk)
        out.append((abs(s), s, f))
    for a, s, f in sorted(out, reverse=True):
        bar = "#" * int(a * 30)
        note = "  <-- strongest" if a == out and False else ""
        print(f"  {f:<16}{s:+.3f}  {bar}{note}")

    print(f"\n{'=' * 96}\nBY PEPTIDE CLASS\n{'=' * 96}")
    for c in ("repeat", "chimeric", "wordlike"):
        sub = [r for r in rows if r["class"] == c]
        if not sub:
            continue
        rr = [r["rank"] for r in sub]
        print(f"  {c:<10} n={len(sub)}  mean rank {st.mean(rr):5.1f}  "
              f"top3 {sum(1 for x in rr if x <= 3)}/{len(sub)}  "
              f"worst {max(rr)}  members: {[r['pep'] for r in sub]}")
    print(f"  {'chance':<10}        mean rank   9.5")


if __name__ == "__main__":
    main()
