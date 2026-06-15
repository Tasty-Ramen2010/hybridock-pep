"""E223 — THE receptor-propensity test: can a rich RECEPTOR representation predict its binding propensity
(the hidden variable)? Our 22 pocket-means gave r=0.049. PPI describes the receptor with full ProtDCal
(220 descriptors). Test on PPIKB's 617 receptors (>=5 peptides each, labeled affinity).

  PART A: receptor features → receptor-MEAN affinity (leave-receptor-cluster-out). Can we ID binding strength?
          representations: composition · full-receptor ProtDCal(220) · +physchem · all.
  PART B: per-complex ΔG with receptor-rich features added — does absolute Kd climb past peptide-only?
          (grouped by receptor cluster, no leak.)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
SN = list(_SCALES.keys())
AA = "ACDEFGHIKLMNPQRSTVWY"


def comp(seq):
    n = max(len(seq), 1)
    return [seq.count(a) / n for a in AA]


def physchem(seq):
    n = max(len(seq), 1)
    return [float(len(seq)),
            (sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)) / n,
            sum(c in "FWY" for c in seq) / n,
            float(np.mean([_SCALES["kd"].get(c, 0) for c in seq])),
            float(np.mean([_SCALES["vol"].get(c, 0) for c in seq]))]


def R(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def cluster_receptors(seqs, thr=0.5):
    """quick greedy clustering of receptor sequences by k-mer Jaccard (homology grouping for no-leak CV)."""
    def kmers(s):
        return set(s[i:i + 4] for i in range(len(s) - 3))
    ks = [kmers(s) for s in seqs]
    lab = [-1] * len(seqs); reps = []
    for i in range(len(seqs)):
        for ri, rk in reps:
            inter = len(ks[i] & rk)
            if inter / max(min(len(ks[i]), len(rk)), 1) >= thr:
                lab[i] = lab[ri]; break
        else:
            lab[i] = len(reps); reps.append((i, ks[i]))
    return np.array(lab)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_main_clean.jsonl")]
    fam = defaultdict(list)
    for r in rows:
        fam[r["protein_seq"]].append(r)
    recs = [(k, v) for k, v in fam.items() if len(v) >= 5 and len(k) >= 20 and set(k) <= set(AA)]
    print(f"=== receptor-propensity: {len(recs)} receptors (>=5 peptides each) ===", flush=True)
    rseq = [k for k, _ in recs]
    ymean = np.array([np.mean([x["y"] for x in v]) for _, v in recs])
    rgrp = cluster_receptors(rseq, 0.5)
    print(f"  receptor homology clusters: {len(set(rgrp))} (group-CV avoids homolog leak)", flush=True)

    reps = {
        "composition(20)": np.array([comp(s) for s in rseq]),
        "receptor-ProtDCal(220)": np.nan_to_num([_protdcal_descriptors(s) for s in rseq]),
        "physchem(5)": np.array([physchem(s) for s in rseq]),
    }
    reps["ALL"] = np.hstack([reps["composition(20)"], reps["receptor-ProtDCal(220)"], reps["physchem(5)"]])
    print("\n=== PART A: receptor features → receptor-MEAN affinity (leave-homolog-cluster-out CV) ===")
    print(f"  target: receptor-mean ΔG, std={ymean.std():.2f}")
    for nm, Xr in reps.items():
        pred = np.full(len(recs), np.nan)
        for tr, te in GroupKFold(5).split(Xr, ymean, rgrp):
            pred[te] = e202._hgb().fit(Xr[tr], ymean[tr]).predict(Xr[te])
        print(f"  {nm:<26} r={R(pred, ymean):+.3f}  (vs our pocket-means 0.049)")

    # PART B: per-complex ΔG, peptide-only vs peptide+receptor-rich
    print("\n=== PART B: per-complex ΔG with receptor-rich features (leave-receptor-cluster-out) ===", flush=True)
    Xpc, ypc, gpc = [], [], []
    rep_pd = {s: _protdcal_descriptors(s) for s in rseq}
    rep_comp = {s: comp(s) for s in rseq}
    for ci, (k, v) in enumerate(recs):
        for x in v:
            pep = x["seq"]
            base = _protdcal_descriptors(pep) + [float(len(pep))]
            Xpc.append((base, rep_pd[k] + rep_comp[k])); ypc.append(x["y"]); gpc.append(rgrp[ci])
    ypc = np.array(ypc); gpc = np.array(gpc)
    Xpep = np.nan_to_num([a for a, _ in Xpc]); Xrec = np.nan_to_num([b for _, b in Xpc])
    for nm, Xuse in [("peptide-only", Xpep), ("peptide + receptor-rich", np.hstack([Xpep, Xrec]))]:
        pred = np.full(len(ypc), np.nan)
        for tr, te in GroupKFold(5).split(Xuse, ypc, gpc):
            pred[te] = e202._hgb().fit(Xuse[tr], ypc[tr]).predict(Xuse[te])
        sl = np.polyfit(pred[~np.isnan(pred)], ypc[~np.isnan(pred)], 1)[0]
        print(f"  {nm:<26} per-complex r={R(pred, ypc):+.3f}  shrink-slope={sl:.2f}  (n={len(ypc)})")
    print("\n  → if receptor-rich LIFTS per-complex r, the receptor representation captures the hidden variable")


if __name__ == "__main__":
    main()
