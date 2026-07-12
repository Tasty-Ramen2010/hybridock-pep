"""E287 — (A) how similar are the 'similar' pockets REALLY, (B) 90%-threshold transfer, (C) double-diff n>=100.

(A) For each of the 305 held-out queries, find the best-match reference receptor by the rich pocket
    descriptor, then report the match in INTERPRETABLE units: pocket-seq % identity (difflib), |Δ net
    charge|, |Δ size|, |Δ hydrophobicity|. Answers Ram: are the 'most similar' pockets actually similar?

(B) Threshold transfer: only transfer the offset when best-match pocket identity >= T (50/70/90%); else
    fall back to absolute. Reports coverage + accuracy — does a STRICT similarity gate make it work?

(C) Double-difference at n>=100: enumerate ALL 2x2 grids (4 measured corners), predict each held-out
    corner from the other 3. Report r/MAE on >=100 predictions, and STRATIFY by R-R' pocket similarity
    (does picking a more similar R' lower the coupling/error?).
Run: OMP_NUM_THREADS=1 python experiments/e287_similarity_and_dd.py
"""
from __future__ import annotations
import json, importlib.util, difflib, numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
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
POS = set("KR"); NEG = set("DE"); AROM = set("FWY"); POL = set("STNQHY")
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v
def pchg(s):
    return sum(c in POS for c in s) - sum(c in NEG for c in s)
def phyd(s):
    return np.mean([KD.get(c, 0) for c in s]) if s else 0.0
def rich(pseq, pkf):
    s = pseq or ""; n = max(len(s), 1)
    return [sum(c in POS for c in s)/n, sum(c in NEG for c in s)/n, pchg(s)/n,
            sum(c in AROM for c in s)/n, sum(c in POL for c in s)/n, phyd(s), float(len(s))] + list(pkf)

rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")
        if json.loads(l).get("aff_type") in ("Kd", "Ki", "KD")]
recs = []
for r in rows:
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and np.isfinite(y):
        recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                     "desc": d3, "pkf": pk, "atype": r["aff_type"], "length": int(pf(r["length"])),
                     "npep": r.get("npep", r["length"]), "npocket": r.get("npocket", 0),
                     "pocket": e158.pocket_seq(r["pdb"].lower())})
ours = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
seen = set(); fresh = []
for r in sorted(recs, key=lambda x: x["pdb"]):
    if r["pdb"] in ours or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
        continue
    if abs(r["npep"] - r["length"]) > 2 or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fseq = {r["pep"] for r in fresh}; frec = {r["rec"] for r in fresh}
pool = [r for r in recs if r["pep"] not in fseq and r["rec"] not in frec]

# ----- absolute model + OOF residual offsets -----
Xtr = np.array([r["desc"] for r in pool]); ytr = np.array([r["y"] for r in pool])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
grp = np.array([hash(r["rec"]) % (10**9) for r in pool])
Sp = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(Xtr, ytr, grp):
    Sp[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=0).fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
resid = ytr - Sp
prec_idx = defaultdict(list)
for i, r in enumerate(pool):
    prec_idx[r["rec"]].append(i)
precs = list(prec_idx.keys())
RR = np.array([rich(pool[prec_idx[s][0]]["pocket"], pool[prec_idx[s][0]]["pkf"]) for s in precs])
mu, sd = RR.mean(0), RR.std(0) + 1e-9
RRz = (RR - mu) / sd
offset = {s: float(np.mean([resid[i] for i in prec_idx[s]])) for s in precs}
prec_pocket = {s: pool[prec_idx[s][0]]["pocket"] for s in precs}
Sf = om.predict(np.array([r["desc"] for r in fresh])); yf = np.array([r["y"] for r in fresh])

# ----- (A) similarity reality + (B) thresholded transfer -----
ident, dchg, dsize, dhyd, bidx = [], [], [], [], []
for r in fresh:
    qz = (np.array(rich(r["pocket"], r["pkf"])) - mu) / sd
    d = np.linalg.norm(RRz - qz, axis=1)
    j = int(np.argmin(d)); bidx.append(j)
    bp = prec_pocket[precs[j]] or ""; qp = r["pocket"] or ""
    ident.append(difflib.SequenceMatcher(None, qp, bp).ratio())
    dchg.append(abs(pchg(qp) - pchg(bp))); dsize.append(abs(len(qp) - len(bp)))
    dhyd.append(abs(phyd(qp) - phyd(bp)))
ident = np.array(ident)
print(f"=== (A) best-match pocket similarity for {len(fresh)} held-out queries ===")
print(f"  pocket-seq identity: mean={ident.mean():.2f} median={np.median(ident):.2f} max={ident.max():.2f}")
print(f"  fraction with best-match identity >=0.5: {(ident>=0.5).mean():.0%} | >=0.7: {(ident>=0.7).mean():.0%}"
      f" | >=0.9: {(ident>=0.9).mean():.0%}")
print(f"  |Δ net charge| mean={np.mean(dchg):.1f} | |Δ size| mean={np.mean(dsize):.0f} res | "
      f"|Δ hydrophobicity| mean={np.mean(dhyd):.2f}")
print("  -> even the BEST available match is typically this dissimilar (that's the deployment reality).")

def rmae(p, t, m=None):
    p, t = np.asarray(p), np.asarray(t)
    if m is not None:
        p, t = p[m], t[m]
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))))
# transfer offset only above identity threshold
print("\n=== (B) thresholded transfer (offset only when best-match identity >= T) ===")
for T in [0.0, 0.5, 0.7, 0.9]:
    preds = Sf.copy()
    for i, r in enumerate(fresh):
        if ident[i] >= T:
            preds[i] = Sf[i] + offset[precs[bidx[i]]]
    cov = (ident >= T).mean()
    ra, ma = rmae(preds, yf)
    print(f"  T={T:.1f}  coverage={cov:5.0%}  r={ra:+.3f} MAE={ma:.2f}")
print(f"  (absolute baseline r={rmae(Sf,yf)[0]:+.3f})")

# ----- (C) double-difference n>=100 -----
ymap = defaultdict(list)
for r in recs:
    ymap[(r["pep"], r["rec"])].append(r["y"])
ymap = {k: float(np.mean(v)) for k, v in ymap.items()}
pep_recs = defaultdict(set)
for (p, rr) in ymap:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]
rec_pocket = {r["rec"]: r["pocket"] for r in recs}
dd_t, dd_p, dd_sim = [], [], []
for a, b in combinations(multi, 2):
    common = list(pep_recs[a] & pep_recs[b])
    for R, Rk in combinations(common, 2):
        try:
            yPR = ymap[(a, R)]; yPRk = ymap[(a, Rk)]; yPkR = ymap[(b, R)]; yPkRk = ymap[(b, Rk)]
        except KeyError:
            continue
        dd_t.append(yPR); dd_p.append(yPRk + yPkR - yPkRk)
        pr = rec_pocket.get(R) or ""; prk = rec_pocket.get(Rk) or ""
        dd_sim.append(difflib.SequenceMatcher(None, pr, prk).ratio())
dd_t, dd_p, dd_sim = np.array(dd_t), np.array(dd_p), np.array(dd_sim)
r_all, m_all = rmae(dd_p, dd_t)
print(f"\n=== (C) DOUBLE-DIFFERENCE on n={len(dd_t)} grid corners ===")
print(f"  overall: r={r_all:+.3f} MAE={m_all:.2f}")
# stratify by R-Rk pocket similarity
hi = dd_sim >= np.median(dd_sim)
print(f"  R-R' pocket similarity HIGH (>=median): r={rmae(dd_p[hi],dd_t[hi])[0]:+.3f} MAE={rmae(dd_p[hi],dd_t[hi])[1]:.2f}")
print(f"  R-R' pocket similarity LOW  (<median):  r={rmae(dd_p[~hi],dd_t[~hi])[0]:+.3f} MAE={rmae(dd_p[~hi],dd_t[~hi])[1]:.2f}")
json.dump(dict(best_match_identity_mean=float(ident.mean()), frac_ge90=float((ident>=0.9).mean()),
               dd_n=len(dd_t), dd_r=float(r_all), dd_mae=float(m_all)),
          open("data/e287_sim_dd.json", "w"))
print("\nsaved data/e287_sim_dd.json")
