"""E201 — deepest forensic audit of OUR model + PPI feature forensics. Four parts:
  A. Feature-class ablation per band: is our crystal model secretly just a SEQUENCE model? (geometry noise?)
  B. Redundancy audit: effective rank of our 240 features, near-duplicate pairs, what the GBT actually uses.
  C. PPI-pred forensics: what is PPI's prediction REALLY tracking on long/vlong/neutral? (corr with our feats)
  D. Bug checks: NaN/constant features, label sign, size confound, train/test leakage.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
       "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]


def r_(p, y):
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]) if ok.sum() > 4 else float("nan")


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        rows.append({"pid": pid, "seq": r["seq"], "y": float(r["y"]), "L": r["length"], "q": abs(pq), "pn": float(r["poc_net"]),
                     "sd": SD(r["seq"]), "pkf": [float(np.mean([SCALES[s2].get(c, 0) for c in ps])) for s2 in SN],
                     "geo": [float(r.get(k, 0)) for k in GEO], "ps": ps})
    y = np.array([r["y"] for r in rows]); L = np.array([r["L"] for r in rows]); q = np.array([r["q"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    n = len(rows)
    SEQ = np.nan_to_num([r["sd"] for r in rows]); POC = np.nan_to_num([r["pkf"] for r in rows])
    GEOX = np.nan_to_num([r["geo"] for r in rows])
    CHG = np.nan_to_num([[r["q"], r["q"] * r["pn"], float(r["L"])] for r in rows])
    print(f"crystal-925: n={n}, SEQ={SEQ.shape[1]} POC={POC.shape[1]} GEO={GEOX.shape[1]}\n", flush=True)

    def cv(X, mask=None):
        pred = np.full(n, np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    # ---- A. feature-class ablation per band ----
    classes = {"SEQ(220)": SEQ, "POCKET(22)": POC, "GEO(16)": GEOX,
               "SEQ+POC": np.hstack([SEQ, POC]), "SEQ+POC+GEO": np.hstack([SEQ, POC, GEOX]),
               "ALL+chg": np.hstack([SEQ, POC, GEOX, CHG])}
    preds = {k: cv(X) for k, X in classes.items()}
    print("=== A. feature-class ablation — what drives our model per band? ===")
    bands = {"ALL": np.ones(n, bool), "neutral": q <= 1, "charged": q >= 2,
             "long13-16": (L >= 13) & (L <= 16), "vlong>=17": L >= 17}
    print(f"  {'band':<12}{'n':>5}" + "".join(f"{k:>13}" for k in classes))
    for bn, mk in bands.items():
        print(f"  {bn:<12}{int(mk.sum()):>5}" + "".join(f"{r_(preds[k][mk], y[mk]):>13.3f}" for k in classes))
    print("  (if SEQ alone ≈ ALL, geometry is NOT helping crystal = our 'physics' is noise here)\n")

    # ---- B. redundancy ----
    ALL = np.hstack([SEQ, POC, GEOX])
    Xs = (ALL - ALL.mean(0)) / (ALL.std(0) + 1e-9)
    sv = np.linalg.svd(np.nan_to_num(Xs), compute_uv=False)
    evr = (sv ** 2) / (sv ** 2).sum()
    eff95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
    print(f"=== B. redundancy: {ALL.shape[1]} features → {eff95} PCs explain 95% variance "
          f"(effective dim, samples={n}) ===")
    # geometry internal redundancy
    G = np.nan_to_num(GEOX); C = np.corrcoef(G.T)
    pairs = [(abs(C[i, j]), GEO[i], GEO[j]) for i in range(len(GEO)) for j in range(i + 1, len(GEO))]
    pairs.sort(reverse=True)
    print("  most-collinear geometry pairs:", ", ".join(f"{a}~{b}({c:.2f})" for c, a, b in pairs[:4]))

    # ---- D. bug checks ----
    print("\n=== D. bug / sanity checks ===")
    allnames = ([f"sd{i}" for i in range(SEQ.shape[1])] + [f"poc:{s}" for s in SN] + [f"geo:{g}" for g in GEO])
    allX = np.hstack([SEQ, POC, GEOX])
    nconst = sum(np.std(allX[:, j]) < 1e-9 for j in range(allX.shape[1]))
    nnan = int(np.isnan(np.hstack([[r["sd"] for r in rows]]).astype(float)).sum())
    print(f"  constant features: {nconst}/{allX.shape[1]}  | NaN in seq-desc: {nnan}")
    print(f"  label y: min={y.min():.1f} max={y.max():.1f} mean={y.mean():.2f} (neg=strong; sign OK if mostly <0)")
    # size confound: does prediction track length more than truth does?
    base = preds["ALL+chg"]
    print(f"  SIZE CONFOUND: corr(pred, length)={r_(base, L.astype(float)):+.3f} vs corr(y, length)={r_(y, L.astype(float)):+.3f}")
    print(f"                 corr(pred, n_contacts/poc_n)={r_(base, GEOX[:,0]):+.3f} vs corr(y, poc_n)={r_(y, GEOX[:,0]):+.3f}")
    # leakage check: any T100 pdb in training? (should be 0 for held-out evals elsewhere)
    man = {m['pdb'].lower() for m in json.loads((ROOT/'data/biolip/t100_peptide_manifest.json').read_text())}
    ours = {r['pid'] for r in rows}
    print(f"  T100∩our-925 (these must be HELD OUT in T100 evals): {len(man & ours)}")

    # ---- C. PPI-pred forensics on long/vlong/neutral (T100) ----
    print("\n=== C. what is PPI's prediction REALLY tracking? (T100 slices) ===")
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    have = {r["pid"]: r for r in rows}
    t = []
    for pid, m in {mm['pdb'].lower(): mm for mm in json.loads((ROOT/'data/biolip/t100_peptide_manifest.json').read_text())}.items():
        d = have.get(pid)
        if d is None and pid in cache:
            c = cache[pid]; ps = e158.pocket_seq(pid) or c["seq"]
            d = {"seq": c["seq"], "L": len(c["seq"]), "q": abs(sum(ch in POS for ch in c["seq"]) - sum(ch in NEG for ch in c["seq"])),
                 "sd": SD(c["seq"]), "pkf": [float(np.mean([SCALES[s2].get(ch, 0) for ch in ps])) for s2 in SN],
                 "geo": [float(c.get(k, 0)) for k in GEO], "pn": float(c.get("poc_net", 0))}
        if d is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        t.append({**d, "y": float(m["dg_exp"]), "ship": ship})
    tL = np.array([x["L"] for x in t]); tq = np.array([x["q"] for x in t])
    ty = np.array([x["y"] for x in t]); tship = np.array([x["ship"] for x in t])
    # candidate simple features
    feats = {"length": tL.astype(float),
             "pep_hyd": np.array([np.mean([SCALES["kd"].get(c, 0) for c in x["seq"]]) for x in t]),
             "pep_vol": np.array([np.mean([SCALES["vol"].get(c, 0) for c in x["seq"]]) for x in t]),
             "pep_helix_prop": np.array([np.mean([SCALES["helix"].get(c, 0) for c in x["seq"]]) for x in t]),
             "pocket_hyd": np.array([x["pkf"][SN.index("hopp")] for x in t]),
             "bsa_hyd": np.array([x["geo"][GEO.index("bsa_hyd")] for x in t]),
             "mj_contact": np.array([x["geo"][GEO.index("mj_contact")] for x in t])}
    for bn, mk in [("long13-16", (tL >= 13) & (tL <= 16)), ("vlong>=17", tL >= 17), ("neutral", tq <= 1)]:
        if mk.sum() < 5:
            continue
        print(f"  [{bn} n={mk.sum()}] PPI r_truth={r_(tship[mk], ty[mk]):+.3f}")
        ranked = sorted(((abs(r_(v[mk], ty[mk])), r_(v[mk], ty[mk]), r_(tship[mk], v[mk]), k) for k, v in feats.items()), reverse=True)
        for _, rt, rp, k in ranked[:4]:
            print(f"      {k:<16} corr(feat,truth)={rt:+.3f}  corr(feat,PPI_pred)={rp:+.3f}")


if __name__ == "__main__":
    main()
