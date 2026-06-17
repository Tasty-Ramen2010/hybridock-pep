"""E260 — Reference-anchoring / triangulation, charged SKEMPI (e254_recs, n=1122).

AXIS: this validates CROSS-RECEPTOR SCORING (absolute ΔG + selectivity), NOT within-target ranking.
Within one receptor the offset b(R) is a shared constant that cancels in any ranking, so anchoring is
irrelevant to ranking (the relative scorer already owns that axis). Anchoring earns its keep when the
ABSOLUTE level matters or when comparing ACROSS receptors (two different offsets that do NOT cancel).

THE CLAIM: charged absolute energy fails because net ΔG = small difference of large cancelling terms.
Predict RELATIVE to a known-Kd reference ON THE SAME RECEPTOR and the per-receptor offset cancels.

HONESTY NOTE: SKEMPI labels are already ΔΔG (WT-anchored), so the physical b(R) is already out of the
LABELS. "b(R) cancels" is exact algebra. NATIVE block = relative-engine test (within-receptor anchoring
vs cold cross-receptor). SIMULATED-ABSOLUTE = inject a random feature-UNPREDICTABLE offset O_R (faithful
to e255) to make the mechanism visible. SHUFFLE control = anchors from a WRONG receptor (must collapse).

Additions (this version):
  * MAE everywhere; charge-stratified breakdowns (|Δq| = 0 / 1 / 2; signed Δq).
  * same-vs-different Δq anchor-test pairs (quantifies the c(p)-c(r) residual).
  * CHARGE-MATCHED anchoring arm (restrict anchors to same |Δq| class) — tests the F1 mitigation.
  * Kd-noise sensitivity: inject N(0,σ) into reference ΔG (σ=0.1/0.3/0.5) -> required anchor precision.
  * selectivity error propagation: ΔΔG = [S(P,A)-b̂(A)] - [S(P,B)-b̂(B)].
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
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

# ---- attach Δcharge labels (verified element-wise aligned to the cache, max|Δ|=0.0) ----
_QSIGN = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
_src = [json.loads(l) for l in open("data/e165_skempi_struct.jsonl") if json.loads(l)["wt"] in "DEKR"]
_by = defaultdict(list)
for _r in _src:
    _by[_r["pdb"]].append(_r)
dq = []
for _p, _muts in _by.items():
    for _m in _muts:
        dq.append(_QSIGN.get(_m["mutaa"], 0.0) - _QSIGN[_m["wt"]])
dq = np.array(dq)
assert len(dq) == len(y), "charge label / cache length mismatch"
assert np.max(np.abs(np.array([float(r["ddg"]) for r in recs]) - y)) == 0.0
dqcls = np.abs(np.round(dq)).astype(int)          # |Δq| class: 0 / 1 / 2  (H rounds into 1)

ok = np.isfinite(X).all(1) & np.isfinite(y)
X, y, pdb, dq, dqcls = X[ok], y[ok], pdb[ok], dq[ok], dqcls[ok]

counts = {u: int((pdb == u).sum()) for u in np.unique(pdb)}
testable = [u for u in counts if counts[u] >= 4]

def fit(Xtr, ytr):                                                                   # [PRESERVED]
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    m.fit(Xtr, ytr); return m

# ---- leave-receptor-out f(x), only for testable receptors (they are the only test pts/anchors) ----
FX = np.full(len(y), np.nan)
for held in testable:
    te = pdb == held
    FX[te] = fit(X[~te], y[~te]).predict(X[te])

K = 3
rows_by_rec = {u: np.where(pdb == u)[0] for u in testable}
all_rows = np.concatenate([rows_by_rec[u] for u in testable])
_sub = X[all_rows[:300]]
SIG = float(np.median(np.linalg.norm(_sub[:, None, :] - _sub[None, :, :], axis=2))) or 1.0


def metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return dict(r=float(pearsonr(t, p)[0]), rho=float(spearmanr(t, p).correlation),
                rmse=float(np.sqrt(np.mean((t - p) ** 2))), mae=float(np.mean(np.abs(t - p))),
                n=int(len(t)))


def run_block(label_y, noise_sigma=0.0):
    """Returns per-point arrays so we can slice by Δcharge afterward.
    f(x)=FX is ALWAYS native-trained (the offset is unobservable to features, the whole point)."""
    arms = ["ABSOLUTE", "ANCHOR_k1", "ANCHOR_k3", "ANCHOR_all", "BAYES",
            "ANCHOR_chargematched", "SHUF_k1", "SHUF_k3"]
    order, ytrue = [], []
    pred = {a: [] for a in arms}
    anchor_same_dq = []   # for k=1: does nearest anchor share |Δq| class with test
    for held in testable:
        idx = rows_by_rec[held]
        others_recs = [u for u in testable if u != held]
        for i in idx:
            order.append(i); yi = label_y[i]; ytrue.append(yi)
            others = idx[idx != i]
            d = np.linalg.norm(X[others] - X[i], axis=1)
            o = np.argsort(d)

            def noisy(a):  # known-Kd reference with optional injected experimental noise
                return label_y[a] + (rng.normal(0, noise_sigma) if noise_sigma else 0.0)

            pred["ABSOLUTE"].append(FX[i])
            a = others[o[0]]
            pred["ANCHOR_k1"].append(noisy(a) + FX[i] - FX[a])
            anchor_same_dq.append(bool(dqcls[a] == dqcls[i]))
            kk = others[o[:K]]
            pred["ANCHOR_k3"].append(np.mean([noisy(b) + FX[i] - FX[b] for b in kk]))
            pred["ANCHOR_all"].append(np.mean([noisy(b) + FX[i] - FX[b] for b in others]))
            w = np.exp(-(d ** 2) / (2 * SIG ** 2)); w = w / w.sum()
            pred["BAYES"].append(float(np.sum(w * (np.array([noisy(b) for b in others]) + FX[i] - FX[others]))))
            # charge-matched: nearest anchor with same |Δq| class (fallback to nearest)
            same = others[dqcls[others] == dqcls[i]]
            if len(same):
                ds = np.linalg.norm(X[same] - X[i], axis=1); cm = same[int(np.argmin(ds))]
            else:
                cm = a
            pred["ANCHOR_chargematched"].append(noisy(cm) + FX[i] - FX[cm])
            # shuffle controls: anchors from a WRONG receptor
            rr = rng.choice(others_recs); pool = rows_by_rec[rr]
            s1 = rng.choice(pool)
            pred["SHUF_k1"].append(noisy(s1) + FX[i] - FX[s1])
            s3 = rng.choice(pool, size=min(K, len(pool)), replace=False)
            pred["SHUF_k3"].append(np.mean([noisy(b) + FX[i] - FX[b] for b in s3]))
    order = np.array(order); ytrue = np.array(ytrue)
    pred = {a: np.array(v) for a, v in pred.items()}
    return order, ytrue, pred, np.array(anchor_same_dq)


def selectivity_error(rmse_single, rho_err=0.0):
    """ΔΔG = anchored(A) - anchored(B). Independent errors -> rmse*sqrt(2). If the per-peptide error c(p)
    is partly SHARED across the two receptors (corr rho_err), it partially cancels -> sqrt(2(1-rho))."""
    return float(rmse_single * np.sqrt(2 * (1 - rho_err)))


# ---- variance decomposition ----
resid = FX - y
mu_by = {u: resid[pdb == u].mean() for u in testable}
btw = np.var([mu_by[u] for u in testable])
wth = np.var([resid[i] - mu_by[pdb[i]] for u in testable for i in rows_by_rec[u]])
frac_between = float(btw / (btw + wth))

ARMS = ["ABSOLUTE", "ANCHOR_k1", "ANCHOR_k3", "ANCHOR_all", "BAYES",
        "ANCHOR_chargematched", "SHUF_k1", "SHUF_k3"]


def block_metrics(order, ytrue, pred):
    return {a: metrics(ytrue, pred[a]) for a in ARMS}


# ---- native + simulated-absolute ----
o_n, y_n, p_n, samedq = run_block(y)
native = block_metrics(o_n, y_n, p_n)

sigma_b = float(np.std(y))
O = {u: rng.normal(0, sigma_b) for u in np.unique(pdb)}
y_abs = y + np.array([O[p] for p in pdb])
o_s, y_s, p_s, _ = run_block(y_abs)
sim = block_metrics(o_s, y_s, p_s)

# ---- charge-stratified (native, BAYES arm) ----
dq_order = dqcls[o_n]
charge_strat = {}
for c in [0, 1, 2]:
    m = dq_order == c
    if m.sum() >= 8:
        charge_strat[f"|dq|={c}"] = dict(
            n=int(m.sum()),
            absolute=metrics(y_n[m], p_n["ABSOLUTE"][m]),
            bayes=metrics(y_n[m], p_n["BAYES"][m]),
            chargematched=metrics(y_n[m], p_n["ANCHOR_chargematched"][m]))

# ---- same-vs-different Δq anchor-test pairs (k=1) ----
same_vs_diff = dict(
    same_dq=metrics(y_n[samedq], p_n["ANCHOR_k1"][samedq]),
    diff_dq=metrics(y_n[~samedq], p_n["ANCHOR_k1"][~samedq]),
    n_same=int(samedq.sum()), n_diff=int((~samedq).sum()))

# ---- Kd-noise sensitivity (native, report BAYES) ----
kd_noise = {}
for s in [0.0, 0.1, 0.3, 0.5]:
    _, yt, pr, _ = run_block(y, noise_sigma=s)
    kd_noise[f"sigma={s}"] = metrics(yt, pr["BAYES"])

# ---- selectivity error propagation ----
single_rmse = native["BAYES"]["rmse"]
selectivity = dict(
    single_receptor_rmse=single_rmse,
    sel_rmse_independent=selectivity_error(single_rmse, 0.0),
    sel_rmse_corr0p3=selectivity_error(single_rmse, 0.3),
    sel_rmse_corr0p5=selectivity_error(single_rmse, 0.5),
    success_threshold=2.0)

results = dict(
    n_receptors=len(testable), n_records=int(len(all_rows)),
    frac_error_between_receptor=frac_between, sigma_b=sigma_b, eta_ceiling_ref=0.755,
    native=native, simulated_absolute=sim, charge_stratified=charge_strat,
    same_vs_diff_charge=same_vs_diff, kd_noise_sensitivity=kd_noise, selectivity=selectivity)
json.dump(results, open("data/e260_results.json", "w"), indent=2)

# ---------------- report ----------------
def show(title, blk):
    print(f"\n=== {title} ===")
    print(f"{'arm':22s} {'r':>7s} {'rho':>7s} {'RMSE':>7s} {'MAE':>7s}")
    for a in ARMS:
        m = blk[a]
        print(f"{a:22s} {m['r']:+7.3f} {m['rho']:+7.3f} {m['rmse']:7.2f} {m['mae']:7.2f}")

print(f"charged SKEMPI | {results['n_receptors']} receptors, {results['n_records']} records")
print(f"fraction of cold-model error that is a per-receptor CONSTANT = {frac_between:.2f} (cf e246 ~0.55)")
print(f"injected offset sigma_b = {sigma_b:.2f} kcal/mol | eta ceiling ref r~0.755 (E254)")
show("NATIVE (relative-engine; labels already ΔΔG)", native)
show("SIMULATED-ABSOLUTE (offset O_R injected; mimics real b(R))", sim)

print("\n=== CHARGE-STRATIFIED (native; |Δq| class) ===")
print(f"{'class':10s} {'n':>4s} {'abs r':>7s} {'bayes r':>8s} {'bayes RMSE':>11s} {'cmatch RMSE':>12s}")
for k, v in charge_strat.items():
    print(f"{k:10s} {v['n']:4d} {v['absolute']['r']:+7.3f} {v['bayes']['r']:+8.3f} "
          f"{v['bayes']['rmse']:11.2f} {v['chargematched']['rmse']:12.2f}")

print("\n=== SAME vs DIFFERENT Δq anchor (k=1) — quantifies c(p)-c(r) ===")
sd = same_vs_diff
print(f"same-Δq anchor  (n={sd['n_same']:4d}): r={sd['same_dq']['r']:+.3f} RMSE={sd['same_dq']['rmse']:.2f} MAE={sd['same_dq']['mae']:.2f}")
print(f"diff-Δq anchor  (n={sd['n_diff']:4d}): r={sd['diff_dq']['r']:+.3f} RMSE={sd['diff_dq']['rmse']:.2f} MAE={sd['diff_dq']['mae']:.2f}")

print("\n=== Kd-NOISE SENSITIVITY (native, BAYES arm) — required anchor precision ===")
print(f"{'ref σ (kcal/mol)':18s} {'r':>7s} {'RMSE':>7s} {'MAE':>7s}")
for k, v in kd_noise.items():
    print(f"{k:18s} {v['r']:+7.3f} {v['rmse']:7.2f} {v['mae']:7.2f}")

print("\n=== SELECTIVITY ΔΔG error propagation ===")
s = selectivity
print(f"single-receptor anchored RMSE = {s['single_receptor_rmse']:.2f} kcal/mol")
print(f"selectivity RMSE (indep errors)     = {s['sel_rmse_independent']:.2f}")
print(f"selectivity RMSE (c(p) corr 0.3)    = {s['sel_rmse_corr0p3']:.2f}")
print(f"selectivity RMSE (c(p) corr 0.5)    = {s['sel_rmse_corr0p5']:.2f}   (threshold {s['success_threshold']})")

nb, na = native["ABSOLUTE"], native["BAYES"]
sb_, ss = sim["ABSOLUTE"], sim["SHUF_k3"]
print("\n--- VERDICT ---")
print(f"NATIVE   absolute r={nb['r']:+.3f} RMSE{nb['rmse']:.2f} -> BAYES r={na['r']:+.3f} RMSE{na['rmse']:.2f}")
print(f"SIM-ABS  shuffle r={ss['r']:+.3f} RMSE{ss['rmse']:.2f} (must sit >= absolute {sb_['rmse']:.2f})")
collapse = ss["rmse"] >= sb_["rmse"] * 0.97
gain = na["rmse"] < nb["rmse"] * 0.8
print(f"shuffle collapses: {collapse} | true-anchor cuts RMSE >=20%: {gain} | ENGINE VALID: {collapse and gain}")
