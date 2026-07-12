"""E272 — full anchoring bake-off on fresh PPIKB (~n=304/429): every metric, pure + fallback, vs baselines.

Estimators (r + MAE on the fresh held-out set):
  BASELINES
    ML_abs       : our HGB absolute on [desc3d + pocket_pkf + len + charge]  ('ML+geometry+descriptors')
    PPI_clone_v2 : SVR(rbf)+SelectKBest on desc3d (the ProtDCal-3D class PPI-Affinity uses)
  PURE ANCHOR (anchor every query by nearest pool refs under metric M; fallback to ML only if M undefined)
    M1_nterm50   : N-term-50 sequence k-mer Jaccard           (the old 'sequence similarity')
    M2_pocketseq : pocket residue-sequence k-mer Jaccard      (pocket = interaction residues)
    M3_pocketcomp: pocket residue composition cosine
    M4_pocketpkf : pocket ProtDCal-3D descriptor similarity   (best offset-transfer metric per e271)
  ANCHOR -> FALLBACK (anchor only the top-quartile-most-confident queries by M; else ML_abs)
    M1..M4 _fb, HOMOLOG_fb (full protein_seq difflib ratio -> fallback)
Anchor pred = Σ w (y_ref + S(query) - S(ref)), w ∝ receptor-M-sim × peptide(desc3d)-closeness.
Same-receptor refs allowed (realistic deployment); S via GroupKFold OOF on pool (clean relative term).
Run: OMP_NUM_THREADS=1 python experiments/e272_bakeoff.py
"""
from __future__ import annotations
import json, importlib.util, difflib, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
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
AROM = set("FWY"); POS = set("KR"); NEG = set("DE"); POL = set("STNQHY")


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v
def comp(s):
    n = max(len(s), 1)
    return np.array([sum(c in POS for c in s) / n, sum(c in NEG for c in s) / n,
                     sum(c in AROM for c in s) / n, sum(c in POL for c in s) / n,
                     np.mean([KD.get(c, 0) for c in s]) if s else 0.0])
def km(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if s and len(s) >= k else set()
def jac(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0


ours_pdbs = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if not r.get("desc3d"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
        continue
    recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "d3": d3, "pk": pk, "len": int(pf(r["length"])), "nc": float(pf(r["net_charge"])),
                 "atype": r["aff_type"], "npocket": r.get("npocket", 0),
                 "pocket": e158.pocket_seq(r["pdb"].lower())})
Ld = max(len(r["d3"]) for r in recs); Lp = max(len(r["pk"]) for r in recs)
recs = [r for r in recs if len(r["d3"]) == Ld and len(r["pk"]) == Lp]
seen = set(); fresh = []
for r in sorted(recs, key=lambda x: x["pdb"]):
    if r["pdb"] in ours_pdbs or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["len"] <= 50) or not (-18 < r["y"] < -2) or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fresh_seqs = {r["pep"] for r in fresh}
pool = [r for r in recs if r["pep"] not in fresh_seqs]
print(f"pool {len(pool)} | fresh {len(fresh)}", flush=True)

XO = lambda rs: np.array([r["d3"] + r["pk"] + [r["len"], r["nc"]] for r in rs])
D3 = lambda rs: np.array([r["d3"] for r in rs])
ytr = np.array([r["y"] for r in pool]); yte = np.array([r["y"] for r in fresh])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(XO(pool), ytr)
po = om.predict(XO(fresh))
cm = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=min(37, Ld))),
               ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(D3(pool), ytr)
pc = cm.predict(D3(fresh))
# OOF S on pool (clean relative term) — GroupKFold by receptor
grp = np.array([hash(r["rec"]) % (10 ** 9) for r in pool])
Spool = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(XO(pool), ytr, grp):
    Spool[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0
                                              ).fit(XO(pool)[tr], ytr[tr]).predict(XO(pool)[te])

# precompute pool peptide descriptor z (desc3d) for ref weighting
D3p = D3(pool); mu, sd = D3p.mean(0), D3p.std(0) + 1e-9
D3pz = (D3p - mu) / sd; D3fz = (D3(fresh) - mu) / sd
SIGp = np.median(np.linalg.norm(D3pz[::5][:, None] - D3pz[::5][None], axis=2)) or 1.0
# pool pocket-pkf z (receptor-mean per pool complex)
PKp = np.array([r["pk"] for r in pool]); PKpz = (PKp - PKp.mean(0)) / (PKp.std(0) + 1e-9)
PKfz = (np.array([r["pk"] for r in fresh]) - PKp.mean(0)) / (PKp.std(0) + 1e-9)
# cache pool receptor metric features
p_nterm = [km(r["rec"], 4) for r in pool]
p_pseq = [km(r["pocket"]) if r["pocket"] else None for r in pool]
p_pcomp = [comp(r["pocket"]) if r["pocket"] else None for r in pool]


def recsim(metric, qi):
    """vector of receptor-M-similarity from fresh query qi to every pool complex (nan where undefined)."""
    q = fresh[qi]
    if metric == "M1":
        qk = km(q["rec"], 4); return np.array([jac(qk, pk) for pk in p_nterm])
    if metric == "M4":
        return -np.linalg.norm(PKpz - PKfz[qi], axis=1)
    if metric == "HOM":
        return np.array([difflib.SequenceMatcher(None, q["rec"], r["rec"]).ratio() for r in pool])
    qp = km(q["pocket"]) if q["pocket"] else None
    qc = comp(q["pocket"]) if q["pocket"] else None
    out = np.full(len(pool), np.nan)
    for j in range(len(pool)):
        if metric == "M2" and p_pseq[j] is not None and qp is not None:
            out[j] = jac(qp, p_pseq[j])
        elif metric == "M3" and p_pcomp[j] is not None and qc is not None:
            a, b = qc, p_pcomp[j]; out[j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return out


def anchor_pred(qi, sim, K=8):
    valid = np.where(np.isfinite(sim))[0]
    if len(valid) < 3:
        return None, -np.inf
    top = valid[np.argsort(sim[valid])[::-1][:max(K, 12)]]
    dpep = np.linalg.norm(D3pz[top] - D3fz[qi], axis=1)
    s = sim[top]; s = (s - s.min()) / (s.max() - s.min() + 1e-9)   # 0..1 receptor sim
    lw = np.log(s + 1e-3) - dpep ** 2 / (2 * SIGp ** 2)
    lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    pred = float(np.sum(w * (ytr[top] + po[qi] - Spool[top])))
    return pred, float(np.nanmax(sim))


def rmae(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))))


# compute anchored predictions + best-sim per metric
metrics = ["M1", "M2", "M3", "M4", "HOM"]
anc = {m: np.array([anchor_pred(i, recsim(m, i)) for i in range(len(fresh))], dtype=object) for m in metrics}
print("anchored preds computed", flush=True)

rows_out = []
ro, mo = rmae(po, yte); rc, mc = rmae(pc, yte)
rows_out.append(("ML_abs (geom+descriptors)", ro, mo, 1.0))
rows_out.append(("PPI_clone_v2", rc, mc, 1.0))
label = {"M1": "M1_nterm50", "M2": "M2_pocketseq", "M3": "M3_pocketcomp", "M4": "M4_pocketpkf", "HOM": "HOMOLOG_seq"}
# PURE arms (anchor where metric defined, else ML)
for m in ["M1", "M2", "M3", "M4"]:
    preds = po.copy()
    for i in range(len(fresh)):
        p, _ = anc[m][i]
        if p is not None:
            preds[i] = p
    r, ma = rmae(preds, yte); rows_out.append((label[m] + " (pure)", r, ma, 1.0))
# FALLBACK arms (anchor only top-quartile-most-confident; else ML)
for m in ["M1", "M2", "M3", "M4", "HOM"]:
    bs = np.array([anc[m][i][1] for i in range(len(fresh))], float)
    have = np.isfinite(bs) & (bs > -1e17)
    thr = np.nanpercentile(bs[have], 75) if have.any() else np.inf
    preds = po.copy(); cov = 0
    for i in range(len(fresh)):
        p, b = anc[m][i]
        if p is not None and b >= thr:
            preds[i] = p; cov += 1
    r, ma = rmae(preds, yte); rows_out.append((label[m] + " -> ML fallback", r, ma, cov / len(fresh)))

print(f"\n=== BAKE-OFF: fresh PPIKB n={len(fresh)} ===")
print(f"{'estimator':32s} {'r':>7s} {'MAE':>7s} {'anchored%':>10s}")
for name, r, ma, cov in rows_out:
    print(f"{name:32s} {r:+7.3f} {ma:7.2f} {cov:9.1%}")
json.dump([{"name": n, "r": float(r), "mae": float(m), "cov": float(c)} for n, r, m, c in rows_out],
          open("data/e272_bakeoff.json", "w"), indent=1)
print("\nsaved data/e272_bakeoff.json")
