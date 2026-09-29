#!/usr/bin/env python
"""Success rate against binding-site similarity, on the bench built for that question.

The length-balanced bench could not carry this plot: 216 of 387 targets sat at exactly 100%
binding-site identity and two bins held 3 and 5 complexes. This bench was constructed instead --
every formatted complex outside training, interface-quality filtered, PDB-id filtered against
every training pool, one complex per 90%-identity cluster -- so all nine bins are populated.

SPLIT BY WHAT THE COMPLEX ACTUALLY IS. 158 of the 405 come from the peppc pool, and those are
not peptide complexes: peppcf_<PDB>_<chain>_<start>_<end> is a window excised from a larger
partner chain at a protein-protein interface, so the fragment's conformation is set by residues
that are not in the file. Every arm scores far worse on them (ours 35.4% against 71.5% on real
peptide complexes), and averaging the two gave a headline of 57.4% that describes neither task.
They are in the bench because the low-identity bins could only be filled from material unlike
our peptide training corpus -- which is exactly the material that is not peptides. So: two
panels, never one average.

Metric is top-1 at <=5 A for every arm, because the co-folding arms produce one pose. The
diffusion arms are ranked by consensus. Bars are 95% Wilson intervals; the bins are thin, so
read the shape and not small gaps between arms.

Usage: simbench_curve.py [--out docs/figures]
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/igem/unknown_software")
THR = 5.0
ARMS = [("hybridock_ft",  "HybriDock-Pep",     "#e64980", "o", 2.2, "-"),
        ("qual_v1",       "HDP quality-filt.", "#0ca678", "s", 1.5, "-"),
        ("rapidock_og",   "RAPiDock (base)",   "#1f6feb", "s", 1.7, "-"),
        ("fullcorpus_v1", "HDP full-corpus",   "#b07d2b", "^", 1.3, "-"),
        ("adcp",          "ADCP",              "#2f9e44", "v", 1.3, "-"),
        ("esmfold",       "ESMFold",           "#7048e8", "D", 1.3, "--"),
        ("boltz2",        "Boltz-2 (leaked)",  "#868e96", "d", 1.3, ":")]


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    args = ap.parse_args()

    cache = np.load(ROOT / "logs/simbench_cache.npz", allow_pickle=True)
    R = {k: cache[k].item() for k in cache.files}
    order = {}
    for arm, *_ in ARMS:
        f = ROOT / f"logs/consensus_sim_{arm}.jsonl"
        if f.exists():
            order[arm] = {json.loads(l)["name"]: json.loads(l)["order"]
                          for l in f.read_text().splitlines() if l.strip()}

    allrows = list(csv.DictReader(open(ROOT / "data/bench_similarity.csv")))
    bins = sorted({r["ident_bin"] for r in allrows}, key=lambda x: float(x.split(",")[0][1:]))
    panels = [("real peptide complexes", [r for r in allrows if r["source"] != "peppc"]),
              ("PPI-interface fragments (peppc)", [r for r in allrows if r["source"] == "peppc"])]

    def top1(arm, name):
        v = R.get(arm, {}).get(name)
        if not v:
            return None
        o = order.get(arm, {}).get(name)
        seq = [x for x in ([v[i] for i in o] if o else v) if np.isfinite(x)]
        return seq[0] if seq else None

    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9), sharey=True)

    for panel, (ptitle, rows) in enumerate(panels):
        ax = axes[panel]
        groups = [[r["name"] for r in rows if r["ident_bin"] == b] for b in bins]
        for arm, lab, col, mk, lw, ls in ARMS:
            xs, ys, lo_e, hi_e, cov = [], [], [], [], 0
            for i, g in enumerate(groups):
                have = [n for n in g if top1(arm, n) is not None]
                if len(have) < 5:
                    continue
                cov += len(have)
                k = sum(top1(arm, n) <= THR for n in have)
                p = k / len(have)
                lo, hi = wilson(k, len(have))
                xs.append(i); ys.append(100 * p)
                lo_e.append(100 * (p - lo)); hi_e.append(100 * (hi - p))
            if not xs:
                continue
            over = 100 * np.mean([top1(arm, r["name"]) <= THR for r in rows
                                  if top1(arm, r["name"]) is not None])
            ax.errorbar(xs, ys, yerr=[lo_e, hi_e], color=col, lw=lw, ls=ls, marker=mk, ms=6,
                        capsize=2.5, elinewidth=.8, zorder=3 if lw > 1.6 else 2,
                        label=f"{lab}  ({over:.0f}% overall)")
        for i, g in enumerate(groups):
            ax.text(i, 101.5, f"n={len(g)}", ha="center", fontsize=7.5, color="#868e96")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=40, ha="right", fontsize=8.5)
        ax.set_xlabel("binding-site identity to the training data (%)")
        if panel == 0:
            ax.set_ylabel(f"top-1 success rate, ≤{THR:g} Å (%)")
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.set_title(f"{'ab'[panel]}   {ptitle}   (n={len(rows)})", loc="left",
                     fontweight="bold", fontsize=11)
        ax.set_ylim(0, 107)
        ax.grid(alpha=.25, lw=.5)
        ax.text(.99, .02, "95% Wilson intervals; bins are thin,\nread the shape not small gaps",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(fc="white", ec="#ced4da", lw=.6))

    fig.tight_layout()
    out = Path(args.out) / "simbench_similarity_curve.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
