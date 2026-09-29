#!/usr/bin/env python
"""Success rate against binding-site similarity to the training data, isoDDE-style.

THE X-AXIS IS A PROPERTY OF THE TARGET, NOT OF ANY METHOD. Each tool here has a different
training set -- ours, RAPiDock's, ESMFold's (PDB pre-2020-05), Boltz-2's (pre-2023-06) -- and
ADCP has none at all, it is physics. A per-method axis would make the curves incomparable. So
the axis is: how similar is this target's binding site to the PDB structures that were available
for training, using the union of our 18,214-complex corpus as the proxy for "what the PDB
contains". That is one number per target, and every arm is scored on the same one.

Similarity is sequence identity at the 8 A interface positions ONLY. Whole-chain identity is the
wrong measure for docking and the difference is not academic: the pocket crops here are a median
185 residues of which a median 17 touch the peptide, so a whole-chain number is 91% scaffold.
MMseqs2 aligns the pockets, interface positions are mapped through that alignment, and identity
is counted at those positions.

METRIC IS TOP-1 for every arm, because that is the only thing the co-folding arms can report:
ESMFold and Boltz-2 produce one pose. The diffusion arms get 24 and ADCP its own ~20, each
ranked by its own ranker (consensus for ours, affinity for ADCP), then the top one is taken.

TWO BINNINGS, because the fixed bins are honest about the shape and dishonest about the error.
  a  the isoDDE bins. (30,40] holds 3 complexes here and (40,50] holds 5, so those points mean
     nothing; they carry 95% Wilson intervals and are drawn hollow.
  b  merged bins chosen so every one holds at least 20 complexes. Equal-count QUANTILE bins are
     impossible here: 216 of 387 targets sit at exactly 100% binding-site identity, so the upper
     quantile edges all tie at 100 and three bins come out empty. Merging adjacent fixed bins is
     the honest way to guarantee n without inventing resolution the data does not have.

Usage: similarity_curve.py [--out docs/figures]
"""
from __future__ import annotations

import argparse, csv, json, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/igem/unknown_software")
THR = 5.0
ARMS = [("hybridock_ft",  "HybriDock-Pep",      "#e64980", "o", 2.2, "-"),
        ("qual_v1",       "HDP quality-filt.",  "#0ca678", "s", 1.5, "-"),
        ("rapidock_og",   "RAPiDock (base)",    "#1f6feb", "s", 1.7, "-"),
        ("fullcorpus_v1", "HDP full-corpus",    "#b07d2b", "^", 1.3, "-"),
        ("adcp",          "ADCP",               "#2f9e44", "v", 1.3, "-"),
        ("esmfold",       "ESMFold",            "#7048e8", "D", 1.3, "--"),
        ("boltz2",        "Boltz-2 (leaked)",   "#adb5bd", "d", 1.3, ":")]


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    a = ap.parse_args()

    sim = pickle.load(open("/tmp/claude-1000/seqid/iface_ident_union.pkl", "rb"))
    c = np.load(ROOT / "logs/balanced_figures_cache.npz", allow_pickle=True)
    R = {k: c[k].item() for k in c.files}
    order = {}
    for arm, *_ in ARMS:
        f = ROOT / f"logs/consensus_{arm}.jsonl"
        if f.exists():
            order[arm] = {json.loads(l)["name"]: json.loads(l)["order"]
                          for l in f.read_text().splitlines() if l.strip()}
    rows = list(csv.DictReader(open(ROOT / "data/bench_balanced_length.csv")))

    def top1(arm, n):
        v = R.get(arm, {}).get(n)
        if not v:
            return None
        o = order.get(arm, {}).get(n)
        seq = [v[i] for i in o] if o else v            # consensus order, else the arm's own
        seq = [x for x in seq if np.isfinite(x)]
        return seq[0] if seq else None

    names = [r["name"] for r in rows]
    s = np.array([100 * sim.get(n, 0.0) for n in names])

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))

    def draw(ax, groups, labels, title, note, min_n=10):
        for arm, lab, col, mk, lw, ls in ARMS:
            xs, ys, los, his, ns = [], [], [], [], []
            for i, g in enumerate(groups):
                have = [n for n in g if top1(arm, n) is not None]
                if not have:
                    xs.append(i); ys.append(np.nan); los.append(0); his.append(0); ns.append(0)
                    continue
                k = sum(top1(arm, n) <= THR for n in have)
                lo, hi = wilson(k, len(have))
                xs.append(i); ys.append(100 * k / len(have))
                los.append(100 * k / len(have) - 100 * lo); his.append(100 * hi - 100 * k / len(have))
                ns.append(len(have))
            xs, ys = np.array(xs, float), np.array(ys, float)
            ns_a = np.array(ns)
            # a line drawn through an n=3 point asserts a trend that three complexes cannot
            # support, so the line is broken there and the point is left as a hollow marker
            yline = np.where(ns_a >= min_n, ys, np.nan)
            ax.plot(xs, yline, color=col, lw=lw, ls=ls, zorder=3 if lw > 1.6 else 2, label=lab)
            ax.errorbar(xs, ys, yerr=[los, his], color=col, lw=0, marker=mk, ms=5, capsize=2,
                        elinewidth=.7, zorder=3 if lw > 1.6 else 2,
                        markerfacecolor=[col if k >= min_n else "white" for k in ns_a][0]
                        if len(set(ns_a >= min_n)) == 1 else "white")
            for xi, yi, ni in zip(xs, ys, ns_a):
                if ni >= min_n:
                    ax.plot([xi], [yi], marker=mk, ms=5, color=col, markerfacecolor=col, zorder=4)
        for i, g in enumerate(groups):
            ax.text(i, 103, f"n={len(g)}", ha="center", fontsize=7,
                    color="#868e96" if len(g) >= 10 else "#e03131",
                    fontweight="normal" if len(g) >= 10 else "bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 108); ax.set_ylabel(f"top-1 success rate, ≤{THR:g} Å (%)")
        ax.set_xlabel("binding-site identity to the training data (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=.25, lw=.5)
        ax.text(.02, .02, note, transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(fc="white", ec="#ced4da", lw=.6))

    EDGES = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
    g1 = [[n for n, v in zip(names, s) if (v <= hi if lo == 0 else lo < v <= hi)]
          for lo, hi in zip(EDGES, EDGES[1:])]
    l1 = [f"({lo:g}, {min(hi,100):g}]" for lo, hi in zip(EDGES, EDGES[1:])]
    draw(axes[0], g1, l1, "a   isoDDE-style fixed bins",
         "red n = fewer than 10 complexes; those points are noise.\nbars are 95% Wilson intervals")

    E2 = [0, 30, 60, 80, 90, 100.01]
    g2 = [[n for n, v in zip(names, s) if (v <= hi if lo == 0 else lo < v <= hi)]
          for lo, hi in zip(E2, E2[1:])]
    l2 = [f"({lo:g}, {min(hi,100):g}]" for lo, hi in zip(E2, E2[1:])]
    draw(axes[1], g2, l2, "b   merged bins — every point has n ≥ 20",
         "same data and same metric, bins merged\nuntil each holds at least 20 complexes", min_n=0)
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)

    fig.tight_layout()
    out = Path(a.out) / "similarity_success_curve.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
