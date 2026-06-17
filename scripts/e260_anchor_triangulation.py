"""E260 — Ram's reference-anchoring / triangulation idea, tested on charged SKEMPI (e254_recs, n=1122).

THE CLAIM under test: charged absolute energy fails because net ΔG = small difference of large
cancelling terms (Coulomb vs desolvation). If instead we predict RELATIVE to a known-Kd reference
ON THE SAME RECEPTOR, the per-receptor offset (the FEP-bound, unpredictable part) CANCELS, and the
large terms cancel in the difference. Triangulating over K references averages out directional noise.

This script tests the STATISTICAL CORE of that idea (does anchoring remove the offset?) using static
features as the relative term. A 100ps-MD relative term would only sharpen f(x_i)-f(x_ref); if anchoring
fails even with perfect-hindsight references, MD won't save it. If it works, MD-anchoring works MORE.

HONESTY NOTE (read before interpreting):
  SKEMPI labels are ALREADY ΔΔG (WT-anchored), so the large physical b(R) is already removed from the
  LABELS. Therefore:
    * "b(R) cancels" is EXACT ALGEBRA — not something this script needs to prove.
    * NATIVE block = the RELATIVE-ENGINE test: is within-receptor anchoring better than cold cross-
      receptor absolute prediction? (the leftover after cancellation: relative-term error + eta).
    * SIMULATED-ABSOLUTE block = inject a random, feature-UNPREDICTABLE per-receptor offset O_R (faithful
      to e255: the real b(R) is unpredictable) to make the headline mechanism visible. Anchoring removes
      O_R by construction; a feature model cannot; shuffle injects the WRONG O_R and must collapse.

Arms (all leave-RECEPTOR-out CV):
  ABSOLUTE      : f(x_i) trained on OTHER receptors, predict held-out receptor cold (no anchor).
  ANCHORED k=1  : given ONE known ddg_ref on the test receptor: ddg_hat = ddg_ref + (f(x_i)-f(x_ref)).
  TRIANGULATED  : average the k=1 estimate over K nearest known references (Ram's 'triangulation').
  ANCHORED all  : triangulate over ALL same-receptor references.
  BAYES-WEIGHTED: similarity-kernel weighted triangulation over all same-receptor references.
  SHUFFLE k=1/3 : MAKE-OR-BREAK CONTROL. anchors drawn from a DIFFERENT random receptor (seeded).
                  Genuine offset cancellation => shuffle collapses to (or below) the absolute baseline.
"""
from __future__ import annotations
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr

rng = np.random.default_rng(0)
recs = json.load(open("data/e254_recs.json"))

# build feature matrix from pair+geom+pocket; label = ddg (already WT-anchored ΔΔG)   [PRESERVED]
def feats(r):
    return np.array(r["pair"] + r["geom"] + r["pocket"], float)
X = np.array([feats(r) for r in recs])
y = np.array([float(r["ddg"]) for r in recs])
pdb = np.array([r["pdb"] for r in recs])
# nan guard
ok = np.isfinite(X).all(1) & np.isfinite(y)
X, y, pdb = X[ok], y[ok], pdb[ok]

urec = np.unique(pdb)
# only receptors with >=4 muts can both anchor and be tested
counts = {u: int((pdb == u).sum()) for u in urec}
testable = [u for u in urec if counts[u] >= 4]

def fit(Xtr, ytr):                                                                   # [PRESERVED]
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    m.fit(Xtr, ytr); return m

# ---- precompute leave-receptor-out f(x) once; reuse for every arm & both blocks ----
# Only testable receptors are ever test points OR anchors, so only those rows need FX.
FX = np.full(len(y), np.nan)
for held in testable:
    te = pdb == held
    FX[te] = fit(X[~te], y[~te]).predict(X[te])

K = 3
SIGMA = None  # set per-run for the bayes kernel (median pairwise feature distance)

def metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    r = pearsonr(t, p)[0]
    rho = spearmanr(t, p).correlation
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    return dict(r=float(r), rho=float(rho), rmse=rmse, mae=mae, n=len(t))

def run_block(label_y, offset=None):
    """label_y is the target vector (native ddg, or ddg+O_R for simulated-absolute).
    f(x) is ALWAYS the native-trained FX (the scorer doesn't know the injected offset — that's the
    point: the offset is unobservable to features, just like the real b(R))."""
    out = {k: ([], []) for k in
           ["ABSOLUTE", "ANCHOR_k1", "ANCHOR_k3", "ANCHOR_all", "BAYES", "SHUF_k1", "SHUF_k3"]}
    # a flat pool of (row indices) per receptor for shuffle sampling
    rows_by_rec = {u: np.where(pdb == u)[0] for u in testable}
    all_test_rows = np.concatenate([rows_by_rec[u] for u in testable])
    # median feature distance for the bayes kernel (global scale)
    sub = X[all_test_rows]
    sig = np.median(np.linalg.norm(sub[:200, None, :] - sub[None, :200, :], axis=2)) or 1.0

    for held in testable:
        idx = rows_by_rec[held]
        other_recs = [u for u in testable if u != held]
        for i in idx:
            yi = label_y[i]
            others = idx[idx != i]
            d = np.linalg.norm(X[others] - X[i], axis=1)
            order = np.argsort(d)
            # ABSOLUTE (cold)
            out["ABSOLUTE"][0].append(yi); out["ABSOLUTE"][1].append(FX[i])
            # k=1 nearest anchor
            a = others[order[0]]
            out["ANCHOR_k1"][0].append(yi); out["ANCHOR_k1"][1].append(label_y[a] + FX[i] - FX[a])
            # k=3 nearest
            kk = others[order[:K]]
            out["ANCHOR_k3"][0].append(yi)
            out["ANCHOR_k3"][1].append(np.mean([label_y[b] + FX[i] - FX[b] for b in kk]))
            # all same-receptor references
            out["ANCHOR_all"][0].append(yi)
            out["ANCHOR_all"][1].append(np.mean([label_y[b] + FX[i] - FX[b] for b in others]))
            # bayes similarity-weighted over all references
            w = np.exp(-(d ** 2) / (2 * sig ** 2)); w = w / w.sum()
            bayes = np.sum(w * (label_y[others] + FX[i] - FX[others]))
            out["BAYES"][0].append(yi); out["BAYES"][1].append(bayes)
            # SHUFFLE controls: anchor(s) from a DIFFERENT random receptor
            rr = rng.choice(other_recs)
            pool = rows_by_rec[rr]
            s1 = rng.choice(pool)
            out["SHUF_k1"][0].append(yi); out["SHUF_k1"][1].append(label_y[s1] + FX[i] - FX[s1])
            s3 = rng.choice(pool, size=min(K, len(pool)), replace=False)
            out["SHUF_k3"][0].append(yi)
            out["SHUF_k3"][1].append(np.mean([label_y[b] + FX[i] - FX[b] for b in s3]))
    return {k: metrics(t, p) for k, (t, p) in out.items()}

# ---- variance decomposition: how much of the cold-model error is a per-receptor CONSTANT? ----
resid = FX - y
mu_by = {u: resid[pdb == u].mean() for u in testable}
btw_var = np.var([mu_by[u] for u in testable])
wth_var = np.var([resid[i] - mu_by[pdb[i]] for u in testable for i in np.where(pdb == u)[0]])
frac_between = btw_var / (btw_var + wth_var)

# ---- run native + simulated-absolute ----
native = run_block(y)

# simulated absolute: inject random per-receptor offset (feature-UNpredictable, faithful to e255)
sigma_b = float(np.std(y))  # offset scale ~ signal scale; clearly stated knob
O = {u: rng.normal(0, sigma_b) for u in urec}
y_abs = y + np.array([O[p] for p in pdb])
sim = run_block(y_abs)

results = dict(
    n_receptors=len(testable), n_records=int(sum(counts[u] for u in testable)),
    frac_error_between_receptor=float(frac_between),
    sigma_b=sigma_b, eta_ceiling_ref=0.755,
    native=native, simulated_absolute=sim,
)
json.dump(results, open("data/e260_results.json", "w"), indent=2)

def show(title, blk):
    print(f"\n=== {title} ===")
    print(f"{'arm':14s} {'r':>7s} {'rho':>7s} {'RMSE':>7s} {'MAE':>7s}")
    for k in ["ABSOLUTE", "ANCHOR_k1", "ANCHOR_k3", "ANCHOR_all", "BAYES", "SHUF_k1", "SHUF_k3"]:
        m = blk[k]
        print(f"{k:14s} {m['r']:+7.3f} {m['rho']:+7.3f} {m['rmse']:7.2f} {m['mae']:7.2f}")

print(f"charged SKEMPI | {results['n_receptors']} receptors, {results['n_records']} records")
print(f"fraction of cold-model error that is a per-receptor CONSTANT (offset-removable) = "
      f"{frac_between:.2f}")
print(f"injected offset scale sigma_b = {sigma_b:.2f} kcal/mol  (eta ceiling ref r~0.755 from E254)")
show("NATIVE (relative-engine test; labels already ΔΔG)", native)
show("SIMULATED-ABSOLUTE (offset O_R injected; mimics real b(R))", sim)

# verdict
nb, na = native["ABSOLUTE"], native["ANCHOR_all"]
sb_, sa = sim["ABSOLUTE"], sim["ANCHOR_all"]
ss = sim["SHUF_k3"]
print("\n--- VERDICT ---")
print(f"NATIVE   absolute r={nb['r']:+.3f} -> anchor_all r={na['r']:+.3f} "
      f"(RMSE {nb['rmse']:.2f} -> {na['rmse']:.2f})")
print(f"SIM-ABS  absolute r={sb_['r']:+.3f} -> anchor_all r={sa['r']:+.3f} "
      f"(RMSE {sb_['rmse']:.2f} -> {sa['rmse']:.2f})")
print(f"SIM-ABS  shuffle  r={ss['r']:+.3f}  RMSE {ss['rmse']:.2f}  "
      f"(must NOT recover; should sit at/below absolute {sb_['rmse']:.2f})")
collapse = ss["rmse"] >= sb_["rmse"] * 0.97
gain = sa["rmse"] < sb_["rmse"] * 0.8
print(f"shuffle collapses: {collapse} | true-anchor recovers >=20% RMSE: {gain} | "
      f"ENGINE VALID: {collapse and gain}")
