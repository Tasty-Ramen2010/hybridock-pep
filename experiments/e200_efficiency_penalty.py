"""E200 — Ram's two ideas:
  (A) STRUCTURED-FRACTION reward: (helix+sheet)/1 as a feature (pre-organised peptides pay less entropy → bind
      stronger). Test additive.
  (B) HYDROPHOBIC-INCOMPATIBILITY EFFICIENCY PENALTY (the big idea): when pocket & peptide don't match
      hydrophobically, the WHOLE interaction is less efficient → scale DOWN |ΔG| (−8 → −6). Multiplicative
      efficiency, vs the additive-feature version.

GATE FIRST: does the signed residual (pred − y) correlate with hydrophobic mismatch? (over-predict when
mismatch high?). If residual ⊥ mismatch, the efficiency idea has no signal. Crystal-925, OOF clustered-CV.
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
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
       "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]
ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}
compl = {json.loads(l)["pdb"].lower(): json.loads(l)["compl"]
         for l in open(ROOT / "data/e199_compl.jsonl") if json.loads(l).get("compl")}


def met(p, y, mask=None):
    if mask is not None:
        p, y = p[mask], y[mask]
    ok = ~(np.isnan(p) | np.isnan(y))
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def main():
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        ps = e158.pocket_seq(pid)
        if ps is None or pid not in compl:
            continue
        s = ss.get(pid, {})
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        pep_hyd = float(np.mean([SCALES["kd"].get(c, 0) for c in r["seq"]]))
        pock_hyd = float(np.mean([SCALES["kd"].get(c, 0) for c in ps]))
        rows.append({"seq": r["seq"], "y": float(r["y"]), "q": abs(pq), "L": r["length"], "pn": float(r["poc_net"]),
                     "geo": [float(r.get(k, 0)) for k in GEO], "ps": ps,
                     "pkf": [float(np.mean([SCALES[s2].get(c, 0) for c in ps])) for s2 in SN],
                     "helix": float(s.get("helix", 0)), "sheet": float(s.get("sheet", 0)),
                     "compl": compl[pid], "pep_hyd": pep_hyd, "pock_hyd": pock_hyd})
    y = np.array([r["y"] for r in rows]); q = np.array([r["q"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)
    print(f"crystal-925 with all: n={len(rows)}\n", flush=True)

    def base_feat(r, extra=()):
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        f = SD(r["seq"]) + r["pkf"] + r["geo"] + [pq * r["pn"], abs(pq) * abs(r["pn"]), abs(pq + r["pn"]), float(len(r["seq"]))]
        return f + list(extra)

    def oof(featfn):
        X = np.nan_to_num([featfn(r) for r in rows]); pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    base = oof(lambda r: base_feat(r))
    resid = base - y  # <0 = over-predict (pred more negative than truth)

    # mismatch scores
    M = {
        "iface_hyd_mismatch": np.array([r["compl"][1] for r in rows]),
        "global|pep-pock|hyd": np.array([abs(r["pep_hyd"] - r["pock_hyd"]) for r in rows]),
        "directional(pock-pep)+": np.array([max(0.0, r["pock_hyd"] - r["pep_hyd"]) for r in rows]),
        "iface_hyd_compl(neg=bad)": np.array([-r["compl"][0] for r in rows]),
    }
    print("=== GATE: does signed residual (pred−y, <0=overpredict) correlate with mismatch? ===")
    for k, v in M.items():
        rr = np.corrcoef(v, resid)[0, 1]
        rn = np.corrcoef(v[q <= 1], resid[q <= 1])[0, 1]
        print(f"  corr(residual, {k:<24}) all={rr:+.3f}  neutral={rn:+.3f}")
    print("  (want NEGATIVE: high mismatch → over-predict → residual<0 → efficiency penalty would correct)\n")

    # (A) structured fraction
    structfrac = oof(lambda r: base_feat(r, [r["helix"] + r["sheet"], (r["helix"] + r["sheet"]) * len(r["seq"])]))
    rb, eb = met(base, y); rs, es = met(structfrac, y)
    print(f"=== (A) +structured-fraction feature: overall r {rb:+.3f}→{rs:+.3f}  RMSE {eb:.2f}→{es:.2f}", flush=True)
    for nm, mk in [("neutral", q <= 1), ("charged", q >= 2)]:
        print(f"     {nm}: r {met(base,y,mk)[0]:+.3f}→{met(structfrac,y,mk)[0]:+.3f}")

    # (B) efficiency multiplier — pick the mismatch with best gate corr; grid λ on OOF (optimistic upper bound)
    Mbest = M["global|pep-pock|hyd"]
    Mn = (Mbest - Mbest.mean()) / (Mbest.std() + 1e-9)
    Mexcess = np.maximum(0.0, Mn)  # only penalise above-average mismatch
    print(f"\n=== (B) efficiency multiplier pred×(1−λ·mismatch_excess) — grid λ (OOF optimistic) ===")
    best = (met(base, y)[0], 0.0)
    for lam in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3):
        corr = base * (1 - lam * Mexcess)  # less negative when mismatch high
        r_, e_ = met(corr, y); rn_, en_ = met(corr, y, q <= 1)
        flag = "  ← best" if r_ > best[0] else ""
        print(f"   λ={lam:<4}: overall r={r_:+.3f} RMSE={e_:.2f} | neutral r={rn_:+.3f} RMSE={en_:.2f}{flag}")
        if r_ > best[0]:
            best = (r_, lam)
    # also additive residual-model form
    print(f"\n  (additive form tested separately via +mismatch feature in e199 = +0.01 neutral)")


if __name__ == "__main__":
    main()
