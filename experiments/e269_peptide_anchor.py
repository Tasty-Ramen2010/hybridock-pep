"""E269 — Ram's SECONDARY fallback: when no close-enough RECEPTOR exists, anchor on a close-enough PEPTIDE.

Tested in the ABSTAIN regime (leave-own-receptor-cluster-out, so NO same-receptor ref is available — the
exact case where the primary receptor-anchor abstains). Peptide similarity uses the descriptors Ram named:
length, net charge, mean hydrophobicity (Kyte-Doolittle), aromatic fraction, charged fraction, and a
burial/compactness proxy.

ALGEBRA (stated honestly): anchoring a similar PEPTIDE on a DIFFERENT receptor cancels c(P) (small term)
but leaves b(R)-b(R_ref) (big term) UNcancelled -- the same wall homolog transfer hit. The only way it
helps is if the weak absolute model left peptide-systematic error c(P) on the table. This measures whether
it does.

Arms (queries with NO same-receptor ref; leave-receptor-cluster-out absolute):
  ABSOLUTE       : S(P,R) cold.
  PEP_ANCHOR     : pool top-k PEPTIDE-descriptor-nearest refs across OTHER receptors; anchor.
  PEP_RESIDUAL   : S(P,R) + mean residual (y-S) of peptide-similar refs (delta-correction form).
  SHUFFLE        : random cross-receptor refs (control).
Run: OMP_NUM_THREADS=1 python experiments/e269_peptide_anchor.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
      "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
      "N": -3.5, "K": -3.9, "R": -4.5}
AROM = set("FWY"); CH = set("DEKR")


def pep_desc(seq):
    seq = "".join(c for c in seq.upper() if c in KD)
    n = max(len(seq), 1)
    nc = sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)
    return np.array([
        len(seq),                                        # length
        nc,                                              # net charge
        np.mean([KD[c] for c in seq]) if seq else 0,     # mean hydrophobicity
        sum(c in AROM for c in seq) / n,                 # aromatic fraction
        sum(c in CH for c in seq) / n,                   # charged fraction
        np.mean([KD[c] for c in seq if KD[c] > 0]) if any(KD[c] > 0 for c in seq) else 0,  # hydrophobic-core proxy
    ], float)


rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v
data = []
for r in rows:
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        yv = pf(r["y"]); d3 = pf(r.get("desc3d")); pk = pf(r.get("pocket_pkf"))
    except Exception:
        continue
    if isinstance(d3, list) and isinstance(pk, list):
        x = d3 + pk + [pf(r["length"]), pf(r["net_charge"])]
        if all(np.isfinite(x)) and np.isfinite(yv):
            data.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(yv), "x": x,
                         "pd": pep_desc(r["seq"])})
Lx = max(len(d["x"]) for d in data); data = [d for d in data if len(d["x"]) == Lx]
y = np.array([d["y"] for d in data]); X = np.array([d["x"] for d in data])
PD = np.array([d["pd"] for d in data])
# z-score peptide descriptors so distance is balanced across the named features
PDz = (PD - PD.mean(0)) / (PD.std(0) + 1e-9)
recs = [d["rec"] for d in data]
urec = sorted(set(recs)); ridx = {s: i for i, s in enumerate(urec)}
rid_of = np.array([ridx[s] for s in recs]); nR = len(urec)
print(f"rows {len(data)} | unique receptors {nR} | pep-desc dim {PD.shape[1]}", flush=True)


def km(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
UK = [km(s) for s in urec]
def jac(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0
# leakage clusters at 0.9 (single linkage) — own cluster excluded from anchors
parent = list(range(nR))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(nR):
    for j in range(i + 1, nR):
        if jac(UK[i], UK[j]) >= 0.9:
            parent[find(i)] = find(j)
clus = np.array([find(i) for i in range(nR)]); clus_of_row = clus[rid_of]

FX = np.full(len(y), np.nan)
for c in np.unique(clus_of_row):
    te = clus_of_row == c
    if not te.all():
        mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=0)
        mdl.fit(X[~te], y[~te]); FX[te] = mdl.predict(X[te])
print("absolute leave-cluster-out scores ready", flush=True)

SIGp = np.median(np.linalg.norm(PDz[::5][:, None] - PDz[::5][None], axis=2)) or 1.0
K = 5
rows_idx = np.arange(len(y))

A = defaultdict(list)  # arm -> (truth, pred)
for i in range(len(y)):
    if not np.isfinite(FX[i]):
        continue
    mask = clus_of_row != clus_of_row[i]            # different receptor cluster (no same-receptor ref)
    cand = rows_idx[mask & np.isfinite(FX)]
    if len(cand) < K:
        continue
    dp = np.linalg.norm(PDz[cand] - PDz[i], axis=1)
    near = cand[np.argsort(dp)[:K]]
    dn = np.linalg.norm(PDz[near] - PDz[i], axis=1)
    lw = -dn ** 2 / (2 * SIGp ** 2); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    pep_anchor = float(np.sum(w * (y[near] + FX[i] - FX[near])))
    pep_resid = float(FX[i] + np.sum(w * (y[near] - FX[near])))  # identical to anchor algebraically
    sh = cand[rng.choice(len(cand), size=K, replace=False)]
    shuf = float(np.mean([y[j] + FX[i] - FX[j] for j in sh]))
    A["ABSOLUTE"].append((y[i], FX[i]))
    A["PEP_ANCHOR"].append((y[i], pep_anchor))
    A["SHUFFLE"].append((y[i], shuf))
    A["PEP_simdist"].append(float(dn.mean()))


def rep(arm):
    tp = A[arm]; t = np.array([a for a, _ in tp]); p = np.array([b for _, b in tp])
    return dict(n=len(t), r=float(pearsonr(t, p)[0]), mae=float(np.mean(np.abs(t - p))),
                rmse=float(np.sqrt(np.mean((t - p) ** 2))))


print(f"\nABSTAIN regime (no same-receptor ref), n={len(A['ABSOLUTE'])} queries:")
out = {}
for arm in ["ABSOLUTE", "PEP_ANCHOR", "SHUFFLE"]:
    v = rep(arm); out[arm] = v
    print(f"  {arm:11s} r={v['r']:+.3f} RMSE={v['rmse']:.2f} MAE={v['mae']:.2f}")
json.dump(out, open("data/e269_peptide_anchor.json", "w"), indent=1)
print("\nVERDICT: if PEP_ANCHOR <= ABSOLUTE (and ~SHUFFLE), peptide anchoring across receptors does NOT")
print("beat the wall (b(R) uncancelled). If PEP_ANCHOR > ABSOLUTE, the weak absolute model left c(P) on")
print("the table and peptide anchoring is a real secondary. saved data/e269_peptide_anchor.json")
