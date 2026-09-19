#!/usr/bin/env python
"""Per-peptide performance bars for the Coventry grid.

A  where each peptide's cognate binder ranks among the 18, raw vs two-way interaction.
   Rank 1 is perfect, 9.5 is chance. Sorted by the two-way rank.
B  how far the paper's OWN two instruments disagree on that peptide's cognate Kd
   (nanoBiT Table S1 vs Octet/BLI Table S4).

Panels A and B are deliberately drawn on the same peptide order so the reader can check
for themselves that they are UNRELATED: Spearman(disagreement, our rank) = -0.10. We do
not get to blame our misses on their experimental noise -- our three worst rows (pc2,
pc46, pc44) are all peptides where their two methods agree to within 1.4x.

Usage: coventry_bars.py [out.png]
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_bars.png"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
# Octet/BLI Kd in nM from Table S4 p.49; flag marks a censored value.
BLI = {"n1": (0.5, "<"), "n2": (18, ""), "n3": (72, ""), "n4": (350, ">"), "n7": (5, ""),
       "pc2": (84, ""), "pc11": (50, ""), "pc12": (36, ""), "pc18": (10, ""),
       "pc17": (45, ""), "pc21": (27, ""), "pc26": (200, ">"), "pc28": (9, ""),
       "pc34": (180, ""), "pc35": (90, ""), "pc43": (9, ""), "pc44": (35, ""),
       "pc46": (37, "")}


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    S = {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            p, b = r["name"].split("__")
            if r.get("best_iface") is not None:
                S[(p, b[:-1])] = r["best_iface"]
    rm = {p: st.mean([S[(p, b)] for b in ORDER]) for p in ORDER}
    cm = {b: st.mean([S[(p, b)] for p in ORDER]) for b in ORDER}
    g = st.mean(S.values())
    W = {k: S[k] - rm[k[0]] - cm[k[1]] + g for k in S}

    rows = []
    for p in ORDER:
        rr = sorted(ORDER, key=lambda x: S[(p, x)]).index(p) + 1
        rw = sorted(ORDER, key=lambda x: W[(p, x)]).index(p) + 1
        nb = float(truth[(p, p)]["kd_M"]) * 1e9
        b, flag = BLI[p]
        rows.append((p, rr, rw, max(nb, b) / min(nb, b), flag))
    rows.sort(key=lambda r: r[2])
    labs = [r[0] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.4), height_ratios=[2.1, 1],
                           sharex=True)
    w = 0.38
    a = ax[0]
    a.bar(x - w / 2, [r[1] for r in rows], w, color="#b9c6d4", label="raw score")
    a.bar(x + w / 2, [r[2] for r in rows], w, color="#1f5c8b",
          label="two-way interaction (specificity)")
    a.axhline(9.5, color="#c1121f", ls="--", lw=1.2, label="chance (9.5 of 18)")
    a.axhline(1, color="#2a9d3f", ls=":", lw=1.2, label="perfect (rank 1)")
    a.set_ylabel("rank of the CORRECT binder\namong 18  (lower is better)", fontsize=9)
    a.set_ylim(0, 19)
    a.invert_yaxis()
    a.legend(fontsize=8, loc="lower right", framealpha=0.95)
    a.set_title("A   Did we pick each peptide's real binder?", fontsize=10.5, loc="left")
    a.grid(axis="y", color="#e5e5e5", lw=0.6)
    a.set_axisbelow(True)

    b = ax[1]
    cols = ["#d9a441" if r[4] == "" else "#dddddd" for r in rows]
    b.bar(x, [r[3] for r in rows], 0.6, color=cols)
    b.axhline(1, color="#888888", lw=0.8)
    b.set_ylabel("their own two methods\ndisagree by (fold)", fontsize=9)
    b.set_xticks(x)
    b.set_xticklabels(labs, rotation=45, fontsize=8.5)
    b.set_title("B   nanoBiT (Table S1) vs Octet/BLI (Table S4) on the SAME cognate pair"
                "   — grey = censored value", fontsize=10.5, loc="left")
    b.grid(axis="y", color="#e5e5e5", lw=0.6)
    b.set_axisbelow(True)

    fig.suptitle("Coventry grid, all 324 cells.  Panels A and B are UNRELATED: "
                 "Spearman(disagreement, our rank) = -0.10 — we cannot blame our misses "
                 "on their experimental noise.", fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    good = [r[0] for r in rows if r[2] <= 3]
    bad = [r[0] for r in rows if r[2] >= 14]
    print(f"  cognate in top 3 (two-way): {len(good)}/18 -> {good}")
    print(f"  worse than rank 14        : {len(bad)}/18 -> {bad}")


if __name__ == "__main__":
    main()
