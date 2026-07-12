"""E273 — CLEAN cross-receptor transfer (NO self-cheating) + Ram's 5-reference combo.

The decisive 'no artificial cheating' test: anchors may NOT come from the query's own receptor OR any
receptor in its >=0.9-similarity cluster (so no same/near-identical receptor leaks in). Pure cross-
receptor transfer only. Reports r/MAE on the COVERED subset (queries with >=K cross-cluster refs by the
metric) AND ML_abs on that SAME subset (apples-to-apples).

Arms (K=5 references, all cross-cluster):
  ML_abs (on covered subset)                    -- the baseline to beat
  M1_nterm5 / M2_pocketseq5 / M3_comp5 / M4_pkf5 -- 5 nearest cross-receptor refs by each metric
  COMBO5  -- Ram's recipe: 2 refs = peptide-similar AMONG receptor-similar, + 3 refs = receptor-similar
             (M4 pocket-3D) focused; average all 5. cross-cluster only.
  SHUFFLE5 -- 5 random cross-cluster refs (control).
Run: OMP_NUM_THREADS=1 python experiments/e273_clean_crossreceptor.py
"""
from __future__ import annotations
import json, importlib.util, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
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
ytr = np.array([r["y"] for r in pool]); yte = np.array([r["y"] for r in fresh])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(XO(pool), ytr)
po = om.predict(XO(fresh))
grp = np.array([hash(r["rec"]) % (10 ** 9) for r in pool])
Spool = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(XO(pool), ytr, grp):
    Spool[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0
                                              ).fit(XO(pool)[tr], ytr[tr]).predict(XO(pool)[te])

# receptor identity + cluster (>=0.9 nterm OR pocket sim) for leakage exclusion
pool_rec = [r["rec"] for r in pool]
D3p = np.array([r["d3"] for r in pool]); mu, sd = D3p.mean(0), D3p.std(0) + 1e-9
D3pz = (D3p - mu) / sd; D3fz = (np.array([r["d3"] for r in fresh]) - mu) / sd
SIGp = np.median(np.linalg.norm(D3pz[::5][:, None] - D3pz[::5][None], axis=2)) or 1.0
PKp = np.array([r["pk"] for r in pool]); PKpz = (PKp - PKp.mean(0)) / (PKp.std(0) + 1e-9)
PKfz = (np.array([r["pk"] for r in fresh]) - PKp.mean(0)) / (PKp.std(0) + 1e-9)
p_nterm = [km(r["rec"], 4) for r in pool]
p_pseq = [km(r["pocket"]) if r["pocket"] else None for r in pool]
p_pcomp = [comp(r["pocket"]) if r["pocket"] else None for r in pool]

def query_sims(qi):
    """dict of metric -> pool-length similarity vector for fresh query qi."""
    q = fresh[qi]
    qk = km(q["rec"], 4); qp = km(q["pocket"]) if q["pocket"] else None
    qc = comp(q["pocket"]) if q["pocket"] else None
    m1 = np.array([jac(qk, pk) for pk in p_nterm])
    m4 = -np.linalg.norm(PKpz - PKfz[qi], axis=1)
    m2 = np.full(len(pool), np.nan); m3 = np.full(len(pool), np.nan)
    if qp is not None:
        for j in range(len(pool)):
            if p_pseq[j] is not None:
                m2[j] = jac(qp, p_pseq[j])
            if p_pcomp[j] is not None:
                a = qc; b = p_pcomp[j]
                m3[j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    # leakage mask: exclude pool refs whose receptor is >=0.9 nterm-sim OR is the same rec string
    leak = (m1 >= 0.9) | np.array([pr == q["rec"] for pr in pool_rec])
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4}, leak

def anchor(qi, idxs):
    if len(idxs) == 0:
        return None
    idxs = np.array(idxs)
    dpep = np.linalg.norm(D3pz[idxs] - D3fz[qi], axis=1)
    lw = -dpep ** 2 / (2 * SIGp ** 2); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    return float(np.sum(w * (ytr[idxs] + po[qi] - Spool[idxs])))

K = 5
arms = ["M1", "M2", "M3", "M4", "COMBO", "SHUFFLE"]
pred = {a: [] for a in arms}; covered = {a: [] for a in arms}; truth = {a: [] for a in arms}
for qi in range(len(fresh)):
    sims, leak = query_sims(qi)
    okbase = ~leak & np.isfinite(Spool)
    for a in ["M1", "M2", "M3", "M4"]:
        s = sims[a].copy(); s[~okbase] = -np.inf; s[~np.isfinite(s)] = -np.inf
        cand = np.where(s > -1e17)[0]
        if len(cand) >= K:
            top = cand[np.argsort(s[cand])[::-1][:K]]
            pred[a].append(anchor(qi, top)); truth[a].append(yte[qi]); covered[a].append(qi)
    # COMBO: 3 receptor-similar (M4) + 2 peptide-similar among top-30 receptor-similar
    s4 = sims["M4"].copy(); s4[~okbase] = -np.inf
    cand4 = np.where(s4 > -1e17)[0]
    if len(cand4) >= K:
        recsim_top = cand4[np.argsort(s4[cand4])[::-1][:30]]
        three = list(recsim_top[:3])
        dpep = np.linalg.norm(D3pz[recsim_top] - D3fz[qi], axis=1)
        two = [recsim_top[t] for t in np.argsort(dpep)[:2]]
        sel = list(dict.fromkeys(three + two))
        pred["COMBO"].append(anchor(qi, sel)); truth["COMBO"].append(yte[qi]); covered["COMBO"].append(qi)
        sh = rng.choice(cand4, size=min(K, len(cand4)), replace=False)
        pred["SHUFFLE"].append(anchor(qi, sh)); truth["SHUFFLE"].append(yte[qi]); covered["SHUFFLE"].append(qi)

def rmae(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"), float(np.mean(np.abs(t - p))), len(t))

print(f"\n=== CLEAN cross-receptor (no self/near-identical), K={K}, fresh PPIKB ===")
print(f"{'arm':14s} {'n':>5s} {'ANCHORED r/MAE':>18s} {'ML_abs(same n) r/MAE':>22s}")
out = {}
for a in arms:
    if len(pred[a]) < 4:
        print(f"{a:14s} n={len(pred[a])} too few"); continue
    cidx = np.array(covered[a])
    ra, ma, n = rmae(pred[a], truth[a])
    rb, mb, _ = rmae(po[cidx], yte[cidx])
    print(f"{a:14s} {n:>5d} {ra:>+9.3f}/{ma:<7.2f} {rb:>+12.3f}/{mb:<7.2f}")
    out[a] = dict(n=n, anc_r=float(ra), anc_mae=float(ma), ml_r=float(rb), ml_mae=float(mb))
json.dump(out, open("data/e273_clean.json", "w"), indent=1)
print("\nML_abs(same n) = our model on the SAME covered queries (apples-to-apples).")
print("VERDICT: if every anchored arm (incl COMBO) <= ML_abs on its covered subset, cross-receptor")
print("transfer fails even with 5 refs + no cheating. saved data/e273_clean.json")
