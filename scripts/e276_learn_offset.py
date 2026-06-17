"""E276 — the one untested gap: can we directly LEARN b(R) (not just test pairwise similarity)?

e271 tested whether static SIMILARITY metrics predict offset transfer (best corr +0.084). But a learned
nonlinear regressor on the richest receptor descriptors is a different, stronger test — maybe b(R) is a
smooth function of pocket structure that no hand-crafted similarity captures. If a leave-receptor-out GBT
can predict OOF b(R) from pocket-3D + composition with r>~0.3, cross-receptor calibration becomes possible
(predict the offset, add it back). If r~0, the offset is genuinely unlearnable from static structure =
final nail, and the same-receptor anchor is the only route.

Target: OOF b(R) per multi-peptide receptor. Features: pocket ProtDCal-3D (mean), pocket composition,
pocket-seq length, pocket net charge, receptor size. Leave-one-receptor-out GBT + Ridge.
Run: OMP_NUM_THREADS=1 python scripts/e276_learn_offset.py
"""
from __future__ import annotations
import json, importlib.util, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneOut
from scipy.stats import pearsonr

spec = importlib.util.spec_from_file_location("e158", "scripts/e158_overfit_failure_analysis.py")
e158 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(e158)
except Exception:
    pass
KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
      "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
      "N": -3.5, "K": -3.9, "R": -4.5}
AROM = set("FWY"); POS = set("KR"); NEG = set("DE"); POL = set("STNQHY")
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v
def comp(s):
    n = max(len(s), 1)
    return [sum(c in POS for c in s) / n, sum(c in NEG for c in s) / n, sum(c in AROM for c in s) / n,
            sum(c in POL for c in s) / n, np.mean([KD.get(c, 0) for c in s]) if s else 0.0, len(s)]

recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD") or not r.get("desc3d"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
        continue
    recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])], "pk": pk,
                 "pocket": e158.pocket_seq(r["pdb"].lower())})
L = max(len(r["x"]) for r in recs); recs = [r for r in recs if len(r["x"]) == L]
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs]); rec = [r["rec"] for r in recs]
grp = np.array([hash(s) % (10 ** 9) for s in rec])
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
resid = y - S
by = defaultdict(list)
for i, s in enumerate(rec):
    by[s].append(i)
# per-receptor OOF b(R) + receptor descriptor features
PK = np.array([r["pk"] for r in recs]); PKz = (PK - PK.mean(0)) / (PK.std(0) + 1e-9)
bR, RX = [], []
for s, idxs in by.items():
    if len({recs[i]["pep"] for i in idxs}) < 2:
        continue
    bR.append(float(np.mean([resid[i] for i in idxs])))
    pkmean = PKz[idxs].mean(0)
    ps = next((recs[i]["pocket"] for i in idxs if recs[i]["pocket"]), "")
    RX.append(list(pkmean) + comp(ps))
bR = np.array(bR); RX = np.nan_to_num(np.array(RX))
print(f"multi-peptide receptors: {len(bR)} | b(R) std={bR.std():.2f} kcal/mol | feat dim {RX.shape[1]}",
      flush=True)

# leave-one-receptor-out: predict b(R)
loo = LeaveOneOut()
for name, mk in [("GBT", lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                  learning_rate=0.05, l2_regularization=2.0, random_state=0)),
                 ("Ridge", lambda: Ridge(alpha=10.0))]:
    pred = np.zeros(len(bR))
    for tr, te in loo.split(RX):
        pred[te] = mk().fit(RX[tr], bR[tr]).predict(RX[te])
    r = pearsonr(bR, pred)[0]
    mae = np.mean(np.abs(bR - pred))
    # baseline: predict the global mean offset
    base_mae = np.mean(np.abs(bR - bR.mean()))
    print(f"  LEARN b(R) [{name}]: r={r:+.3f} MAE={mae:.2f} (vs predict-mean MAE={base_mae:.2f})")
json.dump({"n": len(bR), "b_std": float(bR.std())}, open("data/e276_learn_offset.json", "w"))
print("\nVERDICT: if learned r ~0 and MAE >= predict-mean, b(R) is unlearnable from static pocket")
print("structure (final nail; same-receptor anchor is the only route). If r>0.3, calibration is possible.")
