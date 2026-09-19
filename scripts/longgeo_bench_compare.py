#!/usr/bin/env python
"""longgeo vs longft_tanh on the 345 leak-free RecentSet complexes — paired, with the controls.

WHY PAIRED AND NOT TWO MEANS. The 345 complexes differ enormously in difficulty: a 6-mer in a
deep pocket and a 20-mer on a shallow groove are not interchangeable samples, so the spread
between complexes is far larger than the effect we are looking for. Comparing two summary means
buries a real difference in that spread. Both arms ran the SAME complexes through the SAME
inference tree, so the honest test is per-complex: how often does longgeo beat tanh on the same
structure, and by how much.

WHAT COUNTS AS A RESULT, DECIDED BEFORE LOOKING. A two-sided sign test on wins, plus a Wilcoxon
signed-rank on the RMSD differences. The solenoid finetune won 34 of 60 (p = 0.257) and was
called null; the same bar applies here, and the same bar applied to the designed-data run
(p = 0.167). Three ways of buying pose accuracy with corpus composition have now been tested and
the rule is that val loss does not get a vote -- particularly here.

THE VALIDATION SET CANNOT DECIDE THIS ONE AT ALL. longgeo selected on longft_val_scope25_authorsP,
which is 97.4% contaminated: 113 of its 116 PDB ids appear in longgeo's own training rows. Its
val_loss of 0.3568 is a memorisation score. The RecentSet held-out set used here is 0% leaked
against the same training rows, checked by PDB id.

Also reports the buckets that longgeo was actually built to improve -- long peptides and
high-enclosure grooves -- because a null overall with a win in the target class is a different
result from a null everywhere, and the pool was cut specifically for those.

Usage: longgeo_bench_compare.py
"""
from __future__ import annotations

import csv
import re
import statistics as st
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
LINE = re.compile(r"^\[([0-9a-zA-Z]{4})\] partial=\S+ n=(\d+) dt=\S+ "
                  r"best_direct=([\d.]+)A spread=([\d.]+)A nearnative=(\d+)/(\d+)")


def parse(*logs: str) -> dict[str, dict]:
    """Per-complex results; later files win, so a resume log overrides the interrupted run."""
    out: dict[str, dict] = {}
    for lg in logs:
        p = ROOT / "logs" / lg
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            m = LINE.match(line.strip())
            if m:
                out[m.group(1).lower()] = {
                    "rmsd": float(m.group(3)), "spread": float(m.group(4)),
                    "nn": int(m.group(5)), "n": int(m.group(6))}
    return out


def signtest(wins: int, n: int) -> float:
    """Exact two-sided binomial p at q=0.5, without scipy."""
    from math import comb
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(d: list[float]) -> float:
    """Normal-approximation signed-rank p, ties averaged. n is large here so this is fine."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n < 10:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sd == 0:
        return float("nan")
    z = (wp - mu) / sd
    from math import erfc
    return float(erfc(abs(z) / (2 ** 0.5)))


def summarise(name: str, rs: list[float]) -> None:
    print(f"  {name:<26}n={len(rs):<5}mean {st.mean(rs):5.2f}A  median {st.median(rs):5.2f}A  "
          f"<=2A {sum(r <= 2 for r in rs):3d}  <=5A {sum(r <= 5 for r in rs):3d}")


def main() -> None:
    base = parse("recentset_hybridock_ft.log", "recentset_hybridock_ft_resume.log")
    new = parse("recentset_longgeo.log")
    shared = sorted(set(base) & set(new))
    print("longgeo (epoch 8, final) vs longft_tanh epoch010 — RecentSet held-out, 0% leaked.\n")
    print(f"  baseline complexes {len(base)} · longgeo complexes {len(new)} · "
          f"paired on {len(shared)}")
    if not shared:
        print("\n  longgeo run has produced no complexes yet.")
        return
    if len(new) < len(base):
        print(f"  (longgeo still running — {len(base) - len(new)} to go; numbers below are "
              f"the paired subset so far)")
    print()

    b = [base[c]["rmsd"] for c in shared]
    g = [new[c]["rmsd"] for c in shared]
    summarise("longft_tanh ep010", b)
    summarise("longgeo final", g)

    diff = [gi - bi for gi, bi in zip(g, b)]          # negative = longgeo better
    wins = sum(1 for d in diff if d < 0)
    ties = sum(1 for d in diff if d == 0)
    dec = len(diff) - ties
    print(f"\n  longgeo better on {wins}/{dec} decided complexes "
          f"({100 * wins / max(dec, 1):.0f}%), {ties} exact ties")
    print(f"  sign test p = {signtest(wins, dec):.4f}   "
          f"Wilcoxon signed-rank p = {wilcoxon(diff):.4f}")
    print(f"  mean ΔRMSD {st.mean(diff):+.3f}A   median ΔRMSD {st.median(diff):+.3f}A "
          f"(negative favours longgeo)")
    print(f"  near-native poses per complex: tanh {st.mean(base[c]['nn'] for c in shared):.1f} "
          f"vs longgeo {st.mean(new[c]['nn'] for c in shared):.1f} of 24")

    meta = {}
    bench = ROOT / "data/bench_recentset_heldout.csv"
    if bench.exists():
        for r in csv.DictReader(open(bench)):
            meta[r["name"].lower()] = r

    print("\nBY LENGTH — longgeo was cut FOR the long buckets, so this is where it must show.\n")
    print(f"  {'bucket':<14}{'n':>5}{'tanh':>9}{'longgeo':>10}{'Δ':>9}{'wins':>10}")
    for bucket in ("short", "medium", "long", "very_long"):
        ids = [c for c in shared if meta.get(c, {}).get("length_bucket") == bucket]
        if len(ids) < 5:
            continue
        bb = [base[c]["rmsd"] for c in ids]
        gg = [new[c]["rmsd"] for c in ids]
        w = sum(1 for c in ids if new[c]["rmsd"] < base[c]["rmsd"])
        print(f"  {bucket:<14}{len(ids):>5}{st.mean(bb):>8.2f}A{st.mean(gg):>9.2f}A"
              f"{st.mean(gg) - st.mean(bb):>+8.2f}A{w:>6}/{len(ids)}")

    print("\n  Long/very-long only (the target class):")
    ids = [c for c in shared
           if meta.get(c, {}).get("length_bucket") in ("long", "very_long")]
    if len(ids) >= 10:
        d2 = [new[c]["rmsd"] - base[c]["rmsd"] for c in ids]
        w = sum(1 for x in d2 if x < 0)
        print(f"    {w}/{len(ids)} wins, sign p = {signtest(w, len(ids)):.4f}, "
              f"mean Δ {st.mean(d2):+.3f}A")
    else:
        print(f"    only {len(ids)} paired so far")


if __name__ == "__main__":
    main()
