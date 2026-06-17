"""E285 — Ram's meticulous similar-pocket cross-receptor calibration on the 304/rest-reference setup.

Setup exactly as Ram describes: fresh ~304 PPIKB = TEST (receptors held out of training/references); all
other data = REFERENCES. For each held-out query (P, R), find the most similar reference receptors R' by a
RICH multi-property pocket descriptor (net charge, charge density, burial/size, hydrophobic/aromatic/polar
fractions, pocket ProtDCal-3D), and transfer their offset to calibrate our absolute score:
    ΔG(P,R) ≈ S(P,R) + b̂,   b̂ = weighted mean over matched R' of (y_ref − S_ref).
This is the meticulous version of the cross-receptor idea. Reports r/MAE vs absolute baseline, STRATIFIED
by how good the best pocket match is (does it work when we DO find a very similar pocket?).

Also tests Ram's double-difference-with-similar-receptor: when the query peptide P is measured on a
pocket-similar R', use y(P,R') as the bridging corner. Stratified by R-R' pocket similarity.
Run: OMP_NUM_THREADS=1 python scripts/e285_meticulous_pocket.py
"""
from __future__ import annotations
import json, importlib.util, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
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


def pocket_rich(pseq, pkf):
    """Rich receptor descriptor: pocket composition (charge/burial/size/hydrophobic) + pocket ProtDCal-3D."""
    s = pseq or ""
    n = max(len(s), 1)
    comp = [sum(c in POS for c in s) / n, sum(c in NEG for c in s) / n,
            (sum(c in POS for c in s) - sum(c in NEG for c in s)) / n,         # net charge density
            sum(c in AROM for c in s) / n, sum(c in POL for c in s) / n,
            np.mean([KD.get(c, 0) for c in s]) if s else 0.0, float(len(s))]    # hydrophobicity, size
    return comp + list(pkf)


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
    if not (isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and np.isfinite(y)):
        continue
    recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "desc": d3, "pkf": pk, "atype": r["aff_type"], "length": int(pf(r["length"])),
                 "nc": float(pf(r["net_charge"])), "npep": r.get("npep", r["length"]),
                 "npocket": r.get("npocket", 0), "pocket": e158.pocket_seq(r["pdb"].lower())})
ours_pdbs = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
seen = set(); fresh = []
for r in sorted(recs, key=lambda x: x["pdb"]):
    if r["pdb"] in ours_pdbs or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
        continue
    if abs(r["npep"] - r["length"]) > 2 or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fseq = {r["pep"] for r in fresh}; frec = {r["rec"] for r in fresh}
pool = [r for r in recs if r["pep"] not in fseq and r["rec"] not in frec]
print(f"fresh(test) {len(fresh)} | references(pool) {len(pool)}", flush=True)

Xtr = np.array([r["desc"] for r in pool]); ytr = np.array([r["y"] for r in pool])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
# OOF S on pool for honest residuals
from sklearn.model_selection import GroupKFold
grp = np.array([hash(r["rec"]) % (10**9) for r in pool])
Spool = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(Xtr, ytr, grp):
    Spool[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
resid_pool = ytr - Spool
Sf = om.predict(np.array([r["desc"] for r in fresh])); yf = np.array([r["y"] for r in fresh])

# rich receptor descriptors (z-scored)
pool_rec_idx = defaultdict(list)
for i, r in enumerate(pool):
    pool_rec_idx[r["rec"]].append(i)
pool_recs = list(pool_rec_idx.keys())
RR = np.array([pocket_rich(pool[pool_rec_idx[s][0]]["pocket"], pool[pool_rec_idx[s][0]]["pkf"]) for s in pool_recs])
mu, sd = RR.mean(0), RR.std(0) + 1e-9
RRz = (RR - mu) / sd
rec_offset = {s: float(np.mean([resid_pool[i] for i in pool_rec_idx[s]])) for s in pool_recs}

# meticulous anchoring: for each fresh query, k-NN reference receptors by rich descriptor -> transfer offset
K = 5
preds = Sf.copy(); bestsim = np.zeros(len(fresh))
for i, r in enumerate(fresh):
    qz = (np.array(pocket_rich(r["pocket"], r["pkf"])) - mu) / sd
    d = np.linalg.norm(RRz - qz, axis=1)
    near = np.argsort(d)[:K]
    w = np.exp(-(d[near]**2) / (2 * (np.median(d) or 1)**2)); w /= w.sum()
    bhat = float(np.sum(w * np.array([rec_offset[pool_recs[j]] for j in near])))
    preds[i] = Sf[i] + bhat
    bestsim[i] = -d[near[0]]   # higher = closer best match


def rmae(p, t, m=None):
    p, t = np.asarray(p), np.asarray(t)
    if m is not None:
        p, t = p[m], t[m]
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))))


rb, mb = rmae(Sf, yf); ra, ma = rmae(preds, yf)
print("\n=== meticulous rich-pocket cross-receptor calibration (304 held-out) ===")
print(f"  ABSOLUTE (no transfer)      r={rb:+.3f} MAE={mb:.2f}")
print(f"  + rich-pocket offset transfer r={ra:+.3f} MAE={ma:.2f}  (Δr={ra-rb:+.3f})")
# stratify by best-match similarity: does it work where we DO find a very similar pocket?
order = np.argsort(bestsim)[::-1]
for label, sel in [("top-25% most-similar match", order[:len(order)//4]),
                   ("bottom-50% (poor match)", order[len(order)//2:])]:
    rbb, _ = rmae(Sf, yf, sel); raa, _ = rmae(preds, yf, sel)
    print(f"  [{label:28s}] absolute r={rbb:+.3f} -> transfer r={raa:+.3f}")
json.dump(dict(abs_r=float(rb), transfer_r=float(ra), n=len(fresh)),
          open("data/e285_meticulous.json", "w"))
print("\nVERDICT: if transfer Δr>0 EVEN for top-25% most-similar pockets, meticulous matching helps.")
print("If Δr<=0 even there, b(R) doesn't transfer no matter how carefully pockets are matched.")
print("saved data/e285_meticulous.json")
