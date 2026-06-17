"""E263 — homolog-anchoring radius on REAL peptide Kd (PDBbind 925).

Tests whether receptor-SIMILARITY-gated anchoring (not exact same receptor) works, and how it degrades as
the anchor receptor gets less similar to the query. This is the coverage-vs-accuracy tradeoff that decides
how far past the 65 exact-match receptors we can reach.

For each similarity threshold T (k-mer-Jaccard PROXY for % identity):
  * cluster the 925 receptors at T (greedy single-linkage).
  * leave-CLUSTER-out: train absolute scorer S on other clusters.
  * COVERED queries = those whose cluster has >=2 distinct peptides.
  * ABSOLUTE  : S(query) cold.
  * ANCHORED  : bayes-weighted over same-cluster references: Σ w_k [y_k + S(query) - S(k)].
  * report RMSE/MAE/r on the SAME covered queries (apples-to-apples) for both arms.
T=exact is the same-receptor ceiling; looser T trades coverage for a growing b(R)-b(R') residual.
"""
from __future__ import annotations
import json, glob, os, numpy as np, importlib.util
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr

spec = importlib.util.spec_from_file_location("e261", "scripts/e261_anchor_library.py")
e261 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e261)

rows = [json.loads(l) for l in open("data/pdbbind_peptides.jsonl")]
idx = {os.path.basename(p).split("_")[0].lower(): p
       for p in glob.glob("data/drive_pull/pl/P-L/**/*_protein.pdb", recursive=True)}
FEATS = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
data = []
for r in rows:
    p = idx.get(r["pdb"].lower())
    if not p:
        continue
    s = e261.receptor_seq(p).replace("/", "")
    if len(s) < 20:
        continue
    x = [float(r[f]) for f in FEATS]
    data.append((r["pdb"], r["seq"], float(r["y"]), s, x))
pdbid = [d[0] for d in data]; pep = [d[1] for d in data]
y = np.array([d[2] for d in data]); seqs = [d[3] for d in data]
X = np.array([d[4] for d in data])
print(f"complexes: {len(data)}  features: {len(FEATS)}")


def kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}
KS = [kmers(s) for s in seqs]


def jac(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b); return inter / (len(a) + len(b) - inter)


def cluster(th):
    reps = []
    cid = []
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


def evaluate(th, label):
    cid = cluster(th) if th <= 1 else np.array([hash(s) for s in seqs])
    # leave-cluster-out absolute scores
    FX = np.full(len(y), np.nan)
    for c in np.unique(cid):
        te = cid == c
        if te.all():
            continue
        FX[te] = fit(X[~te], y[~te]).predict(X[te])
    # covered queries: cluster has >=2 distinct peptides
    members = defaultdict(list)
    for i, c in enumerate(cid):
        members[c].append(i)
    SIG = np.median([np.linalg.norm(X[i] - X[j]) for c in members for i in members[c][:6]
                     for j in members[c][:6] if i != j] or [1.0]) or 1.0
    abs_t, abs_p, anc_t, anc_p = [], [], [], []
    for c, mem in members.items():
        if len({pep[i] for i in mem}) < 2:
            continue
        for i in mem:
            others = [j for j in mem if j != i and pep[j] != pep[i]]
            if not others or not np.isfinite(FX[i]):
                continue
            d = np.array([np.linalg.norm(X[i] - X[j]) for j in others])
            w = np.exp(-(d ** 2) / (2 * SIG ** 2)); w = w / w.sum()
            pred = float(np.sum(w * (y[others] + FX[i] - FX[others])))
            abs_t.append(y[i]); abs_p.append(FX[i])
            anc_t.append(y[i]); anc_p.append(pred)
    def m(t, p):
        t, p = np.array(t), np.array(p)
        return (pearsonr(t, p)[0] if len(t) > 3 else np.nan,
                np.sqrt(np.mean((t - p) ** 2)), np.mean(np.abs(t - p)), len(t))
    ar, arm, amae, n = m(abs_t, abs_p)
    nr, nrm, nmae, _ = m(anc_t, anc_p)
    print(f"{label:10s} covered_n={n:4d} | ABSOLUTE r={ar:+.3f} RMSE={arm:.2f} MAE={amae:.2f}"
          f"  ->  ANCHORED r={nr:+.3f} RMSE={nrm:.2f} MAE={nmae:.2f}")
    return dict(threshold=label, n=n, abs_r=float(ar), abs_rmse=float(arm), abs_mae=float(amae),
                anc_r=float(nr), anc_rmse=float(nrm), anc_mae=float(nmae))


print(f"{'sim':10s} {'real peptide-Kd, leave-cluster-out, covered queries':>10s}")
out = []
out.append(evaluate(1.01, "exact"))
for th in [0.9, 0.7, 0.5]:
    out.append(evaluate(th, f"id~{th:.1f}"))
json.dump(out, open("data/e263_homolog_radius.json", "w"), indent=1)
print("\nsaved data/e263_homolog_radius.json")
print("read: ANCHORED should beat ABSOLUTE at exact/high-id; the id where the gain vanishes = the radius.")
