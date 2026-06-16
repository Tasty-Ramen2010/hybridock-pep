"""E245 — does POSE-AWARE GIST displaced-water predict per-complex affinity, and does it ADD to the
existing scorer? Honest test: clustered CV (group by receptor 5-mer family, no near-duplicate leakage).

  (1) single-feature Pearson r of each displaced-water descriptor vs y (per-complex Kd)
  (2) apo-pocket descriptors (the dead baseline) on the SAME complexes — head-to-head
  (3) does disp_total ADD to the production scorer? compare clustered-CV r of [base feats] vs [base+disp]

Run: python3 scripts/e245_pose_gist_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DISP = ["disp_total", "disp_unhappy", "disp_happy", "disp_max", "disp_per_pepatom"]
APO = ["gist_dG_pocket", "gist_unhappy_dG"]


def rr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 3 else np.nan


def kmers(s, k=5):
    s = s.replace("/", "")
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def cluster(seqs, thr=0.3):
    n = len(seqs); ks = [kmers(s) for s in seqs]; par = list(range(n))
    f = lambda x: x if par[x] == x else f(par[x])
    for i in range(n):
        for j in range(i + 1, n):
            u = len(ks[i] | ks[j]) or 1
            if len(ks[i] & ks[j]) / u >= thr:
                par[f(i)] = f(j)
    lab = {}; return np.array([lab.setdefault(f(i), len(lab)) for i in range(n)])


def grouped_cv(X, y, g):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    pred = np.empty(len(y))
    for gid in np.unique(g):
        te = g == gid; tr = ~te
        if tr.sum() < 4:
            pred[te] = y[tr].mean() if tr.sum() else y.mean(); continue
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=2.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return rr(pred, y)


def main():
    rows = []
    for c in sorted(ROOT.glob("data/e244_pose_gist*.jsonl")):
        rows += [json.loads(l) for l in c.read_text().splitlines() if l.strip()]
    seen = {}
    for r in rows:
        seen.setdefault(r["pdb"], r)
    rows = list(seen.values())
    if len(rows) < 8:
        print(f"only {len(rows)} complexes — need >=8"); return
    y = np.array([r["y"] for r in rows])
    g = cluster([r["seq"] for r in rows])
    print(f"=== E245 pose-GIST eval (n={len(rows)} complexes, {len(np.unique(g))} families, "
          f"Kd std {y.std():.2f}) ===\n")
    print("[1] POSE-AWARE displaced-water, single-feature r vs per-complex Kd:")
    for f in DISP:
        print(f"    {f:<16} r={rr([r.get(f) for r in rows], y):+.3f}")
    print("\n[2] APO-pocket (the dead baseline), same complexes:")
    for f in APO:
        print(f"    {f:<16} r={rr([r.get(f) for r in rows], y):+.3f}")
    # [3] does disp ADD? base = length only (cheap proxy) vs base+disp_total, clustered CV
    print("\n[3] clustered-CV (leave-family-out): does disp_total ADD over length-alone?")
    L = np.array([[r["L"]] for r in rows], float)
    Ld = np.array([[r["L"], r.get("disp_total", 0) or 0] for r in rows], float)
    print(f"    length only            r={grouped_cv(L, y, g):+.3f}")
    print(f"    length + disp_total    r={grouped_cv(Ld, y, g):+.3f}")
    Xall = np.array([[r["L"]] + [r.get(f, 0) or 0 for f in DISP] for r in rows], float)
    print(f"    length + all disp      r={grouped_cv(Xall, y, g):+.3f}")


if __name__ == "__main__":
    main()
