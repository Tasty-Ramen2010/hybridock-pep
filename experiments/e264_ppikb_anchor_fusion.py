"""E264 — anchoring validation on PPIKB (independent of PDBbind) + FUSION with the absolute scorer.

Random-holdout / leave-receptor-out on PPIKB's STRUCTURED subset (ppikb_features.jsonl, 2229 rows with
3D descriptors). PPIKB is independent of PDBbind (PPI-Affinity's training source), so this is a fresh test.

Arms (Kd/Ki only, leave-RECEPTOR-out, homolog-clustered at receptor-seq k-mer Jaccard):
  ABSOLUTE  : GBT on PPIKB features, cold (zero-shot)  -- our "original model" analogue on PPIKB.
  ANCHORED  : bayes-weighted same/homolog-receptor anchoring (few-shot).
  SHUFFLE   : anchors from a WRONG receptor (control -- must collapse).
  FUSION    : confidence-weighted blend of ANCHORED and ABSOLUTE. Confidence rises with #refs and
              anchor similarity; when refs are close/many -> trust anchor, else -> trust absolute.
              This is "join original model + anchoring to do better."

PPI-Affinity note: PPI requires a 3D COMPLEX (ProtDCal contact descriptors); it CANNOT score seq-only
rows, and we have no PPIKB-PPI predictions. The honest PPI anchor is its published clustered-CV ceiling
~0.35 (and 0.554 on homology-redundant T100). We report our ABSOLUTE (zero-shot, the apples-to-apples
vs PPI) and ANCHORED (few-shot, the lever PPI lacks: PPI has no pose engine -> cannot anchor).
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]


def parsef(v):
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            return [float(x) for x in json.loads(v)]
        try:
            return float(v)
        except ValueError:
            return None
    return v


data = []
for r in rows:
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    y = parsef(r["y"])
    d3 = parsef(r.get("desc3d"))
    pk = parsef(r.get("pocket_pkf"))
    if y is None or not isinstance(d3, list) or not isinstance(pk, list):
        continue
    x = d3 + pk + [parsef(r["length"]) or 0.0, parsef(r["net_charge"]) or 0.0]
    if not all(np.isfinite(x)) or not np.isfinite(y):
        continue
    data.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y), "x": x})

# uniform feature length
L = max(len(d["x"]) for d in data)
data = [d for d in data if len(d["x"]) == L]
y = np.array([d["y"] for d in data])
X = np.array([d["x"] for d in data])
recs = [d["rec"] for d in data]
peps = [d["pep"] for d in data]
print(f"PPIKB Kd/Ki structured rows: {len(data)} | feature dim {L}")


def kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
KS = [kmers(s) for s in recs]


def jac(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b); return inter / (len(a) + len(b) - inter)


def cluster(th):
    reps = []; cid = []
    for ks in KS:
        best = -1; bj = th
        for rks, c in reps:
            j = jac(ks, rks)
            if j >= bj:
                bj = j; best = c
        if best < 0:
            best = len(reps); reps.append((ks, best))
        cid.append(best)
    return np.array(cid)


def fit(a, b):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    m.fit(a, b); return m


def m(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return dict(r=float(pearsonr(t, p)[0]) if len(t) > 3 else float("nan"),
                rmse=float(np.sqrt(np.mean((t - p) ** 2))), mae=float(np.mean(np.abs(t - p))),
                n=int(len(t)))


def evaluate(th, label):
    cid = cluster(th) if th <= 1 else np.arange(len(recs))
    FX = np.full(len(y), np.nan)
    for c in np.unique(cid):
        te = cid == c
        if te.all():
            continue
        FX[te] = fit(X[~te], y[~te]).predict(X[te])
    members = defaultdict(list)
    for i, c in enumerate(cid):
        members[c].append(i)
    others_clusters = list(members.keys())
    SIG = np.median([np.linalg.norm(X[i] - X[j]) for c in members for i in members[c][:6]
                     for j in members[c][:6] if i != j] or [1.0]) or 1.0
    A = defaultdict(list)  # arm -> (truth, pred)
    for c, mem in members.items():
        if len({peps[i] for i in mem}) < 2:
            continue
        for i in mem:
            others = [j for j in mem if j != i and peps[j] != peps[i]]
            if not others or not np.isfinite(FX[i]):
                continue
            d = np.array([np.linalg.norm(X[i] - X[j]) for j in others])
            logw = -(d ** 2) / (2 * SIG ** 2)
            logw -= logw.max()                       # numerical stability
            w = np.exp(logw); s = w.sum()
            w = w / s if s > 0 else np.ones_like(w) / len(w)   # fallback uniform
            anc = float(np.sum(w * (y[others] + FX[i] - FX[others])))
            absol = float(FX[i])
            # confidence: more refs + closer refs -> trust anchor
            conf = (1 - np.exp(-len(others) / 3.0)) * float(np.exp(-d.min() / SIG))
            fus = conf * anc + (1 - conf) * absol
            A["ABSOLUTE"].append((y[i], absol))
            A["ANCHORED"].append((y[i], anc))
            A["FUSION"].append((y[i], fus))
            # shuffle: anchor from a different cluster
            rc = rng.choice([cc for cc in others_clusters if cc != c])
            pool = members[rc]
            s = rng.choice(pool, size=min(3, len(pool)), replace=False)
            sh = float(np.mean([y[j] + FX[i] - FX[j] for j in s]))
            A["SHUFFLE"].append((y[i], sh))
    if not A.get("ABSOLUTE"):
        print(f"\n[{label}] covered_n=0 (no anchorable receptors at this similarity) — skipped")
        return {"label": label, "n": 0}
    res = {}
    for arm, tp in A.items():
        t = [a for a, _ in tp]; p = [b for _, b in tp]
        res[arm] = m(t, p)
    n = res["ABSOLUTE"]["n"]
    print(f"\n[{label}] covered_n={n}")
    for arm in ["ABSOLUTE", "ANCHORED", "FUSION", "SHUFFLE"]:
        v = res[arm]
        print(f"  {arm:9s} r={v['r']:+.3f} RMSE={v['rmse']:.2f} MAE={v['mae']:.2f}")
    return {"label": label, "n": n, **{a: res[a] for a in res}}


out = []
for th, lbl in [(1.01, "exact"), (0.95, "id~0.95"), (0.9, "id~0.9"), (0.7, "id~0.7"), (0.5, "id~0.5")]:
    out.append(evaluate(th, lbl))
json.dump(out, open("data/e264_ppikb_results.json", "w"), indent=1)
print("\nsaved data/e264_ppikb_results.json")
print("PPI ref: published clustered-CV ~0.35 (zero-shot); T100 0.554 (homology-redundant).")
print("apples-to-apples vs PPI = our ABSOLUTE (zero-shot). ANCHORED/FUSION = few-shot lever PPI lacks.")
