"""E274 — the two clean anchoring numbers Ram asked for:
  (A) SAME protein, DIFFERENT peptides  -> tests c(P) transfer with b(R) cancelled exactly = WORKING case
  (B) SAME peptide, DIFFERENT proteins  -> tests b(R) transfer with c(P) cancelled exactly = THE WALL

PPIKB, leave-one-out, OOF absolute S (GroupKFold by receptor) for the relative term.
(A): for each receptor with >=2 distinct peptides, anchor each peptide to the OTHER same-receptor peptides.
(B): for each peptide measured on >=2 distinct receptors, anchor each (P,R) to (P,R_other). Also report the
     raw |Δy| spread of the SAME peptide across receptors = the pure cross-receptor wall magnitude.
Run: OMP_NUM_THREADS=1 python experiments/e274_two_clean_cases.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
        continue
    recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
L = max(len(r["x"]) for r in recs); recs = [r for r in recs if len(r["x"]) == L]
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
rec = [r["rec"] for r in recs]; pep = [r["pep"] for r in recs]
print(f"PPIKB Kd/Ki rows {len(recs)}", flush=True)

# OOF absolute by receptor
grp = np.array([hash(s) % (10 ** 9) for s in rec])
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0
                                          ).fit(X[tr], y[tr]).predict(X[te])
print(f"OOF absolute r={pearsonr(y, S)[0]:+.3f} MAE={np.mean(np.abs(y-S)):.2f}\n", flush=True)

SIG = np.median(np.linalg.norm((X[::5] - X.mean(0))[:, None] - (X[::5] - X.mean(0))[None], axis=2)) or 1.0


def anchor(qi, refs):
    refs = np.array(refs)
    d = np.linalg.norm(X[refs] - X[qi], axis=1)
    lw = -d ** 2 / (2 * SIG ** 2); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    return float(np.sum(w * (y[refs] + S[qi] - S[refs])))


def rmae(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return (pearsonr(t, p)[0], float(np.mean(np.abs(t - p))), len(t))


# (A) same protein, different peptides
by_rec = defaultdict(list)
for i, s in enumerate(rec):
    by_rec[s].append(i)
At, Ap, Aa = [], [], []
for s, idxs in by_rec.items():
    if len({pep[i] for i in idxs}) < 2:
        continue
    for i in idxs:
        others = [j for j in idxs if pep[j] != pep[i]]
        if others and np.isfinite(S[i]):
            At.append(y[i]); Ap.append(anchor(i, others)); Aa.append(S[i])
rA, mA, nA = rmae(Ap, At); rAa, mAa, _ = rmae(Aa, At)

# (B) same peptide, different proteins
by_pep = defaultdict(list)
for i, s in enumerate(pep):
    by_pep[s].append(i)
Bt, Bp, Ba, dyspread = [], [], [], []
for s, idxs in by_pep.items():
    recset = {rec[i] for i in idxs}
    if len(recset) < 2:
        continue
    ys = [y[i] for i in idxs]
    dyspread.append(max(ys) - min(ys))
    for i in idxs:
        others = [j for j in idxs if rec[j] != rec[i]]   # same peptide, DIFFERENT receptor
        if others and np.isfinite(S[i]):
            Bt.append(y[i]); Bp.append(anchor(i, others)); Ba.append(S[i])
if len(Bt) > 3:
    rB, mB, nB = rmae(Bp, Bt); rBa, mBa, _ = rmae(Ba, Bt)
else:
    rB = mB = nB = rBa = mBa = float("nan")

print("=== (A) SAME protein, DIFFERENT peptides  [b(R) cancels -> WORKING case] ===")
print(f"  n={nA}  ABSOLUTE r={rAa:+.3f} MAE={mAa:.2f}  ->  ANCHORED r={rA:+.3f} MAE={mA:.2f}")
print("\n=== (B) SAME peptide, DIFFERENT proteins  [c(P) cancels -> pure b(R) wall] ===")
print(f"  peptides on >=2 receptors: {len(dyspread)}  | anchored queries n={nB}")
print(f"  raw |Δy| of SAME peptide across receptors: mean={np.mean(dyspread):.2f} "
      f"max={np.max(dyspread):.2f} kcal/mol (= the cross-receptor wall, c(P) removed)")
if np.isfinite(rB):
    print(f"  ABSOLUTE r={rBa:+.3f} MAE={mBa:.2f}  ->  ANCHORED r={rB:+.3f} MAE={mB:.2f}")
json.dump(dict(A=dict(n=nA, abs_r=float(rAa), abs_mae=float(mAa), anc_r=float(rA), anc_mae=float(mA)),
               B=dict(n_pep=len(dyspread), dy_mean=float(np.mean(dyspread)) if dyspread else None,
                      dy_max=float(np.max(dyspread)) if dyspread else None,
                      n=int(nB) if np.isfinite(rB) else 0,
                      abs_r=float(rBa) if np.isfinite(rB) else None,
                      anc_r=float(rB) if np.isfinite(rB) else None,
                      anc_mae=float(mB) if np.isfinite(rB) else None)),
          open("data/e274_two_cases.json", "w"), indent=1)
print("\nsaved data/e274_two_cases.json")
