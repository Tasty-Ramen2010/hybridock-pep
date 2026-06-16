"""E243 — does GIST free-energy (E242) beat RISM density (E230 max_g) at predicting the receptor baseline?
The decisive test of the RISM post-mortem axis-1 fix. Per-dataset (PDBbind / PPIKB) + pooled:
GIST single-feature + multivariate LOO-Ridge vs y_mean, head-to-head with RISM max_g on the SAME receptors.

Run: python3 scripts/e243_gist_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GIST = ROOT / "data" / "e242_gist.jsonl"
RISM_C = ["data/e230_rism.jsonl", "data/e230_t100_rism.jsonl", "data/e230_rism_all.jsonl",
          "data/e240_ppikb_rism.jsonl"]
GFEATS = ["gist_dG_pocket", "gist_unhappy_dG", "gist_happy_dG", "gist_dEww", "gist_mTdS",
          "gist_max_vox_dG", "gist_Esw"]


def rr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 3 else np.nan


def loo(rows, feats):
    y = np.array([r["y_mean"] for r in rows])
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    pred = np.empty(len(rows))
    for i in range(len(rows)):
        tr = np.arange(len(rows)) != i
        sc = StandardScaler().fit(X[tr])
        pred[i] = Ridge(alpha=2.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i+1]))[0]
    return rr(pred, y)


def main():
    gist = [json.loads(l) for l in GIST.read_text().splitlines() if l.strip()]
    # RISM max_g by pdb
    maxg = {}
    for c in RISM_C:
        p = ROOT / c
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l); maxg.setdefault(d["rep_pdb"], d.get("max_g"))
    ppikb_ids = {r["peptides"][0]["pdb"]
                 for r in json.load(open(ROOT / "data/e240_ppikb_manifest.json"))["receptors"]}
    for r in gist:
        r["rism_max_g"] = maxg.get(r["rep_pdb"], np.nan)
        r["dset"] = "PPIKB" if r["rep_pdb"] in ppikb_ids else "PDBbind"

    print(f"=== E243 GIST eval (n={len(gist)} done) ===")
    for name in ("PDBbind", "PPIKB", "ALL"):
        rows = gist if name == "ALL" else [r for r in gist if r["dset"] == name]
        if len(rows) < 6:
            print(f"\n[{name}] n={len(rows)} — too few"); continue
        y = [r["y_mean"] for r in rows]
        print(f"\n[{name}] n={len(rows)}  baseline std={np.std(y):.2f}")
        for f in GFEATS:
            print(f"    {f:<16} r={rr([r.get(f) for r in rows], y):+.3f}")
        print(f"    {'GIST multivariate LOO':<16} r={loo(rows, GFEATS):+.3f}")
        print(f"    {'-- RISM max_g (same n)':<16} r={rr([r['rism_max_g'] for r in rows], y):+.3f}")


if __name__ == "__main__":
    main()
