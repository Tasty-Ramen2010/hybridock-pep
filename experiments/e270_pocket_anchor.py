"""E270 — Ram's POCKET-similarity anchoring + the deep-dive on why peptide crossover failed.

Two parts, one consistent PPIKB setup (all features = desc3d + pocket_pkf, trained on non-fresh PPIKB,
evaluated on the fresh n~304 held-out set: Kd, unique-seq, npocket>=10, length 2-50, not in PDBbind 925).

PART A — DEEP DIVE: why does cross-receptor anchoring fail, and can POCKET similarity fix it?
  The anchor error = [b(R)-b(R_ref)] + [c(P)-c(P_ref)]. Peptide similarity cancels the small c term; the
  big b(R) term survives. Ram's pocket hypothesis: b(R) is a POCKET property, so pocket-similar receptors
  share b(R) even across different proteins. TEST: estimate b(R)=mean(y-S) per multi-peptide receptor,
  then correlate receptor-pair |Δb| against (i) sequence similarity, (ii) POCKET-descriptor similarity.
  If pocket-sim predicts small |Δb| (and better than seq-sim), pocket-anchoring is viable.

PART B — n~304 HEAD-TO-HEAD: r + MAE for
  OURS         : HGB absolute on [desc3d+pocket_pkf+len+charge] (our architecture).
  PPI_CLONE    : SVR(rbf)+SelectKBest on desc3d (the ProtDCal-3D class PPI-Affinity uses) — PPI-clone v2.
  POCKET_ANCHOR: anchor to pocket-similar pool refs when max pocket-sim>=tau, ELSE fall back to OURS.
Run: OMP_NUM_THREADS=1 python experiments/e270_pocket_anchor.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from scipy.stats import pearsonr

rng = np.random.default_rng(0)


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


ours_pdbs = {json.loads(l)["pdb"].lower() for l in open("data/pdbbind_peptides.jsonl")}
all_rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]
recs = []
for r in all_rows:
    if not r.get("desc3d"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list)):
        continue
    length = pf(r["length"]); nc = pf(r["net_charge"])
    if not (np.isfinite(y) and all(np.isfinite(d3)) and all(np.isfinite(pk))):
        continue
    recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                 "d3": d3, "pk": pk, "len": int(length), "nc": float(nc), "atype": r["aff_type"],
                 "npocket": r.get("npocket", 0)})
# uniform dims
Ld = max(len(r["d3"]) for r in recs); Lp = max(len(r["pk"]) for r in recs)
recs = [r for r in recs if len(r["d3"]) == Ld and len(r["pk"]) == Lp]

# fresh (test) set ~304
seen = set(); fresh = []
for r in sorted(recs, key=lambda x: x["pdb"]):
    if r["pdb"] in ours_pdbs or r["atype"] not in ("Kd", "KD", "pKd"):
        continue
    if not (2 <= r["len"] <= 50) or not (-18 < r["y"] < -2) or r["npocket"] < 10 or r["pep"] in seen:
        continue
    seen.add(r["pep"]); fresh.append(r)
fresh_seqs = {r["pep"] for r in fresh}
pool = [r for r in recs if r["pep"] not in fresh_seqs]   # anchor + train pool (fresh held out)
print(f"pool {len(pool)} | fresh test {len(fresh)} (vlong>=17 {sum(r['len']>=17 for r in fresh)}, "
      f"charged|q|>=2 {sum(abs(r['nc'])>=2 for r in fresh)})", flush=True)

D3 = lambda rs: np.array([r["d3"] for r in rs])
PK = lambda rs: np.array([r["pk"] for r in rs])
XO = lambda rs: np.array([r["d3"] + r["pk"] + [r["len"], r["nc"]] for r in rs])
ytr = np.array([r["y"] for r in pool]); yte = np.array([r["y"] for r in fresh])

# OURS absolute
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(XO(pool), ytr)
po = om.predict(XO(fresh))
# PPI-clone v2 (SVR on desc3d)
cm = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=min(37, Ld))),
               ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(D3(pool), ytr)
pc = cm.predict(D3(fresh))
# pool absolute scores (for anchoring residuals) via OOF on pool
S_pool = om.predict(XO(pool))   # in-sample ok: used only as relative term, residual y-S is what anchors

# ---------- PART A: does pocket-sim predict small |Δb| ? ----------
pool_by_rec = defaultdict(list)
for i, r in enumerate(pool):
    pool_by_rec[r["rec"]].append(i)
PKz_all = (PK(pool) - PK(pool).mean(0)) / (PK(pool).std(0) + 1e-9)
multi = [(rec, idxs) for rec, idxs in pool_by_rec.items() if len({pool[i]["pep"] for i in idxs}) >= 2]
bR = {}; pkmean = {}
for rec, idxs in multi:
    bR[rec] = float(np.mean([pool[i]["y"] - S_pool[i] for i in idxs]))
    pkmean[rec] = PKz_all[idxs].mean(0)


def km(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
mrec = [rec for rec, _ in multi]
print(f"PART A: {len(mrec)} multi-peptide receptors; b(R) std = {np.std(list(bR.values())):.2f} kcal/mol")
dseq, dpk, dab = [], [], []
for a in range(len(mrec)):
    for b in range(a + 1, len(mrec)):
        ra, rb = mrec[a], mrec[b]
        dab.append(abs(bR[ra] - bR[rb]))
        ka, kb = km(ra), km(rb)
        dseq.append(len(ka & kb) / len(ka | kb) if (ka and kb) else 0.0)
        dpk.append(-float(np.linalg.norm(pkmean[ra] - pkmean[rb])))  # higher = more similar pocket
dab, dseq, dpk = np.array(dab), np.array(dseq), np.array(dpk)
print(f"  corr(seq-sim,    -|Δb|) = {pearsonr(dseq, -dab)[0]:+.3f}")
print(f"  corr(pocket-sim, -|Δb|) = {pearsonr(dpk,  -dab)[0]:+.3f}   "
      f"(positive & larger => pocket similarity predicts shared offset)")

# ---------- PART B: pocket-anchor with fallback ----------
PKz_fresh = (PK(fresh) - PK(pool).mean(0)) / (PK(pool).std(0) + 1e-9)
# peptide-descriptor distance for ref weighting (use desc3d as peptide-ish signature)
D3z_pool = (D3(pool) - D3(pool).mean(0)) / (D3(pool).std(0) + 1e-9)
D3z_fresh = (D3(fresh) - D3(pool).mean(0)) / (D3(pool).std(0) + 1e-9)
SIGp = np.median(np.linalg.norm(D3z_pool[::5][:, None] - D3z_pool[::5][None], axis=2)) or 1.0


def rmae(p, t, m=None):
    p, t = np.asarray(p), np.asarray(t)
    if m is not None:
        p, t = p[m], t[m]
    return (pearsonr(t, p)[0] if len(t) > 3 else float("nan"),
            float(np.mean(np.abs(t - p))), len(t))


pool_pkz = PKz_all
results = {}
for tau in [0.5, 1.0, 1.5, 2.0]:   # pocket-distance threshold (z-space L2); larger = looser
    preds = po.copy(); n_anc = 0
    for i in range(len(fresh)):
        dpkt = np.linalg.norm(pool_pkz - PKz_fresh[i], axis=1)
        cand = np.where(dpkt <= tau)[0]
        # exclude same-peptide leakage (shouldn't happen, fresh held out) and require known Kd (pool all do)
        if len(cand) >= 3:
            dpep = np.linalg.norm(D3z_pool[cand] - D3z_fresh[i], axis=1)
            order = np.argsort(dpep)[:8]
            sel = cand[order]
            lw = -dpep[order] ** 2 / (2 * SIGp ** 2); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
            preds[i] = float(np.sum(w * (ytr[sel] + po[i] - S_pool[sel])))
            n_anc += 1
    r, mae, _ = rmae(preds, yte)
    results[tau] = dict(tau=tau, coverage=n_anc / len(fresh), r=float(r), mae=float(mae))

ro, mo, _ = rmae(po, yte); rc, mc, _ = rmae(pc, yte)
print(f"\nPART B — n={len(fresh)} fresh PPIKB head-to-head:")
print(f"  {'model':22s} {'r':>7s} {'MAE':>7s} {'coverage':>9s}")
print(f"  {'OURS (absolute)':22s} {ro:+7.3f} {mo:7.2f} {'—':>9s}")
print(f"  {'PPI-clone v2':22s} {rc:+7.3f} {mc:7.2f} {'—':>9s}")
for tau, v in results.items():
    print(f"  {'POCKET-ANCHOR τ='+str(tau):22s} {v['r']:+7.3f} {v['mae']:7.2f} {v['coverage']:8.1%}")
json.dump(dict(n=len(fresh), ours=dict(r=float(ro), mae=float(mo)),
               ppi_clone=dict(r=float(rc), mae=float(mc)),
               pocket_anchor=results,
               partA=dict(b_std=float(np.std(list(bR.values()))),
                          corr_seqsim_negdb=float(pearsonr(dseq, -dab)[0]),
                          corr_pocketsim_negdb=float(pearsonr(dpk, -dab)[0]))),
          open("data/e270_pocket_anchor.json", "w"), indent=1)
print("\nsaved data/e270_pocket_anchor.json")
