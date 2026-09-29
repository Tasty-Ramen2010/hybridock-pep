#!/usr/bin/env python
"""Does the finetune's advantage survive when the bench receptor is NOT a homolog of training?

WHY THIS EXISTS. Our held-out claim was built on exact complex name and PDB id, which is an
IDENTITY check, not a HOMOLOGY check. Brian Coventry pointed out that the field stratifies by
sequence identity instead, and he was right: on this bench, 234 of 387 receptors (60.5%) have a
training receptor above 90% aligned-region identity to them, and 41.1% are still above 90% on
the stricter coverage-weighted measure. "Zero overlap" was true and almost meaningless.

WHICH IDENTITY. Ram's correction, and it is the right one: whole-chain identity is the wrong
measure for a docking model. Measured here, the "receptor" we align is a 20 A pocket crop of a
median 185 residues, of which a median of 17 -- NINE PERCENT -- are within 8 A of the peptide.
Aligning all 185 makes the number 91% scaffold.

So the primary stratifier is BINDING-SITE identity: MMseqs2 aligns the pocket sequences, then
each bench complex's 8 A interface positions are mapped through that alignment and scored for
identity at those positions only. Whole-pocket identity is kept as the secondary curve because
it is what the field usually reports and it shows how much the two differ.

The bins are the ones Brian named: 90 / 70 / 50 / 30%.

Usage: identity_stratified.py [--out docs/figures]
"""
from __future__ import annotations

import argparse, collections, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("/home/igem/unknown_software")
HITS = Path("/tmp/claude-1000/seqid/rec_hits.tsv")
BLUE, GREY = "#1f6feb", "#6e7781"
BINS = [(0.9, 1.01, ">90%"), (0.7, 0.9, "70-90%"), (0.5, 0.7, "50-70%"),
        (0.3, 0.5, "30-50%"), (0.0, 0.3, "<30%")]


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    a = ap.parse_args()

    longft = set()
    for f in ("data/longft_train.csv", "data/longft_val.csv"):
        for r in csv.DictReader(open(ROOT / f)):
            longft.add(r.get("complex_name") or r.get("name"))

    import pickle
    cov = collections.defaultdict(float, pickle.load(
        open("/tmp/claude-1000/seqid/iface_ident.pkl", "rb")))      # binding site, the primary
    aln = collections.defaultdict(float)                             # whole pocket, secondary
    for l in Path("/tmp/claude-1000/seqid/rec_hits.tsv").read_text().splitlines():
        q, t, fid, alnlen, qlen, tlen, ev, bits = l.split("\t")
        if t in longft:
            aln[q] = max(aln[q], float(fid) * int(alnlen) / int(qlen))

    c = np.load(ROOT / "logs/balanced_figures_cache.npz", allow_pickle=True)
    R = {k: c[k].item() for k in c.files}
    order = {arm: {json.loads(l)["name"]: json.loads(l)["order"]
                   for l in (ROOT / f"logs/consensus_{arm}.jsonl").read_text().splitlines() if l.strip()}
             for arm in ("hybridock_ft", "rapidock_og")}
    rows = list(csv.DictReader(open(ROOT / "data/bench_balanced_length.csv")))
    bk = {r["name"]: ("long" if r["length_bucket"] in ("long", "vlong") else r["length_bucket"])
          for r in rows}
    THR = {"short": 2.0, "med": 2.0, "long": 5.0}

    def best(arm, n, N=24):
        v = R[arm].get(n)
        if not v:
            return None
        o = order[arm].get(n)
        vv = [x for x in ([v[i] for i in o][:N] if o else v[:N]) if np.isfinite(x)]
        return min(vv) if vv else None

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.1))

    # --- panel a: how homologous the bench actually is
    A = ax[0]
    for lab, d, col, ls in (("whole-pocket identity (185 res)", aln, GREY, "--"),
                            ("BINDING-SITE identity (17 res)", cov, BLUE, "-")):
        xs = np.linspace(0, 1, 101)
        ys = [100 * np.mean([d.get(r["name"], 0.0) > x for r in rows]) for x in xs]
        A.plot(xs * 100, ys, color=col, lw=1.8, ls=ls, label=lab)
    for t in (0.9, 0.7, 0.5, 0.3):
        A.axvline(t * 100, color="#adb5bd", lw=.7, ls=":")
    A.set_xlabel("max identity to any training complex (%)")
    A.set_ylabel("% of the 387 bench complexes above it")
    A.set_title("a   the bench is NOT homology-free", loc="left", fontweight="bold")
    A.legend(fontsize=7.5, frameon=False); A.grid(alpha=.25, lw=.5); A.set_ylim(0, 102)
    A.text(.03, .06, f"{100*np.mean([cov.get(r['name'],0)>0.9 for r in rows]):.0f}% have a training complex with a\n"
                     f"binding site >90% identical to theirs",
           transform=A.transAxes, fontsize=7.5, bbox=dict(fc="white", ec="#ced4da", lw=.6))

    # --- panels b, c: does the advantage survive
    for k, (N, title) in enumerate(((24, "b   top-25"), (1, "c   top-1"))):
        B = ax[k + 1]
        labs, us, ra, ns = [], [], [], []
        for lo, hi, lab in BINS:
            sel = [r["name"] for r in rows if lo <= cov.get(r["name"], 0.0) < hi
                   and best("hybridock_ft", r["name"]) is not None]
            if len(sel) < 5:
                continue
            labs.append(f"{lab}\nn={len(sel)}"); ns.append(sel)
            us.append(np.mean([best("hybridock_ft", n, N) <= THR[bk[n]] for n in sel]))
            ra.append(np.mean([best("rapidock_og", n, N) <= THR[bk[n]] for n in sel]))
        x = np.arange(len(labs)); w = 0.38
        for off, vals, lab, col in ((-w/2, us, "HybriDock-Pep", BLUE), (w/2, ra, "RAPiDock", GREY)):
            lo_e = [100*v - 100*wilson(round(v*len(s)), len(s))[0] for v, s in zip(vals, ns)]
            hi_e = [100*wilson(round(v*len(s)), len(s))[1] - 100*v for v, s in zip(vals, ns)]
            B.bar(x + off, [100*v for v in vals], w, color=col, label=lab,
                  yerr=[lo_e, hi_e], capsize=2.5, error_kw=dict(lw=.8, ecolor="#343a40"))
        for xi, u, f, sel in zip(x, us, ra, ns):
            d = 100 * (u - f)
            B.text(xi, 3, f"{d:+.0f}", ha="center", fontsize=8, fontweight="bold",
                   color=BLUE if d > 2 else "#868e96")
        B.set_xticks(x); B.set_xticklabels(labs, fontsize=7.5)
        B.set_xlabel("binding-site identity to training", fontsize=8)
        B.set_ylabel("success (%)"); B.set_ylim(0, 100)
        B.set_title(f"{title}  — the gain lives only at >90%", loc="left", fontweight="bold")
        B.grid(alpha=.25, axis="y", lw=.5)
        if k == 0:
            B.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.tight_layout()
    out = Path(a.out) / "identity_stratified.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
