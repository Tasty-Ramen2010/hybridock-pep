"""E279 — prototype C1 (analytical dynamics term) and C2 (pairwise ΔΔG anchoring).

C1: deliver MISATO's +0.066 dynamics signal WITHOUT MD, via an analytical sidechain-rotamer-entropy term
    (conformational entropy lost on binding ~ Σ n_chi(residue) × burial). Test whether it adds orthogonal
    signal on top of static — on the 758 (compare vs MISATO) AND full PPIKB (where MISATO is unavailable
    => C1 is deployable everywhere, MISATO is not).
C2: pairwise/Siamese ΔΔG — train a GBT on feature-DIFFERENCES to predict within-receptor Kd differences,
    then anchor (y_query = y_ref + ΔΔG_pred). Compare to simple subtraction anchoring (e274 r≈0.63).
Run: OMP_NUM_THREADS=1 python scripts/e279_c1_c2.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr

# sidechain rotameric DOF (number of chi angles) per amino acid
NCHI = {"G": 0, "A": 0, "S": 1, "C": 1, "T": 1, "V": 1, "P": 1, "D": 2, "N": 2, "F": 2, "H": 2,
        "I": 2, "L": 2, "W": 2, "Y": 2, "E": 3, "Q": 3, "M": 3, "K": 4, "R": 4}


def rotamer_feats(seq: str, mean_burial: float) -> list[float]:
    """Analytical conformational-entropy descriptors for a peptide (no MD needed)."""
    seq = "".join(c for c in seq.upper() if c in NCHI)
    n = max(len(seq), 1)
    chi = [NCHI[c] for c in seq]
    total_chi = sum(chi)
    # entropy lost on binding ~ flexible DOF scaled by how buried the peptide is
    burial = mean_burial / 100.0 if mean_burial > 1.5 else mean_burial
    return [total_chi, total_chi / n, total_chi * burial, float(np.std(chi)) if chi else 0.0,
            sum(c >= 3 for c in chi) / n]   # long-flexible (E/Q/M/K/R) fraction


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


# ---------- C1 part 1: on the 758 (head-to-head vs MISATO) ----------
SFEAT = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
mis = {json.loads(l)["id"]: json.loads(l) for l in open("data/e251_misato_flex.jsonl")
       if "ie_mean" in json.loads(l)}
lab = {json.loads(l)["pdb"].upper(): json.loads(l) for l in open("data/pdbbind_peptides.jsonl")}
rows = [(d, mis[k]) for k, d in lab.items() if k in mis]
Xs = np.array([[float(d[f]) for f in SFEAT] for d, _ in rows])
Xc1 = np.array([rotamer_feats(d["seq"], float(d["mean_burial"])) for d, _ in rows])
Xmd = np.array([[m["ie_mean"], m["ie_std"], m["lig_rmsf"], m["bsasa_mean"]] for _, m in rows])
y758 = np.array([float(d["y"]) for d, _ in rows])


def cv(X, y):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pred[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                 l2_regularization=1.0, random_state=0
                                                 ).fit(X[tr], y[tr]).predict(X[te])
    return pearsonr(y, pred)[0], float(np.mean(np.abs(y - pred)))


print("=== C1 analytical-dynamics vs MISATO-MD (758 peptide complexes) ===", flush=True)
for name, X in [("STATIC", Xs), ("STATIC+C1(rotamer)", np.hstack([Xs, Xc1])),
                ("STATIC+MISATO-MD", np.hstack([Xs, Xmd])),
                ("STATIC+C1+MISATO", np.hstack([Xs, Xc1, Xmd]))]:
    r, m = cv(X, y758)
    print(f"  {name:22s} r={r:+.3f} MAE={m:.2f}")

# ---------- C1 part 2: full PPIKB (MISATO unavailable -> C1 still deployable) ----------
recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD"):
        continue
    try:
        yv = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(yv)):
        continue
    recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(yv),
                 "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])],
                 "burial": float(pf(r["length"]))})
L = max(len(r["x"]) for r in recs); recs = [r for r in recs if len(r["x"]) == L]
Xp = np.array([r["x"] for r in recs])
Xpc1 = np.array([rotamer_feats(r["pep"], 40.0) for r in recs])
yp = np.array([r["y"] for r in recs])
print("\n=== C1 on full PPIKB (n=%d, MISATO N/A -> C1 deployable everywhere) ===" % len(recs))
r0, m0 = cv(Xp, yp); r1, m1 = cv(np.hstack([Xp, Xpc1]), yp)
print(f"  STATIC           r={r0:+.3f} MAE={m0:.2f}")
print(f"  STATIC+C1(rotamer) r={r1:+.3f} MAE={m1:.2f}  (Δr={r1-r0:+.3f})")

# ---------- C2: pairwise ΔΔG anchoring vs simple subtraction ----------
rec = [r["rec"] for r in recs]
grp = np.array([hash(s) % (10**9) for s in rec])
S = np.full(len(yp), np.nan)
for tr, te in GroupKFold(8).split(Xp, yp, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0
                                          ).fit(Xp[tr], yp[tr]).predict(Xp[te])
by = defaultdict(list)
for i, s in enumerate(rec):
    by[s].append(i)
panels = {s: idxs for s, idxs in by.items() if len({recs[i]["pep"] for i in idxs}) >= 2}

# build pairwise training set (feature diff -> ΔΔG) with leave-receptor-out via clusters
pair_recs = list(panels.keys())
print("\n=== C2 pairwise ΔΔG anchoring vs simple subtraction (PPIKB panels n=%d) ===" % len(panels))


def simple_anchor(idxs):
    out = []
    for i in idxs:
        others = [j for j in idxs if recs[j]["pep"] != recs[i]["pep"]]
        if others and np.isfinite(S[i]):
            d = np.linalg.norm(Xp[others] - Xp[i], axis=1)
            w = np.exp(-(d**2) / (2 * (np.median(d) or 1)**2)); w /= w.sum()
            out.append((yp[i], float(np.sum(w * (yp[others] + S[i] - S[others])))))
    return out


# leave-one-receptor-out pairwise GBT
from sklearn.model_selection import LeaveOneGroupOut
gkeys = {s: k for k, s in enumerate(pair_recs)}
simp_t, simp_p, pair_t, pair_p = [], [], [], []
for held in pair_recs:
    # train pairwise on OTHER panels
    dX, dY = [], []
    for s, idxs in panels.items():
        if s == held:
            continue
        for a in idxs:
            for b in idxs:
                if a != b:
                    dX.append(Xp[a] - Xp[b]); dY.append(yp[a] - yp[b])
    if len(dX) < 50:
        continue
    pm = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                       l2_regularization=2.0, random_state=0).fit(np.array(dX), np.array(dY))
    idxs = panels[held]
    for i in idxs:
        others = [j for j in idxs if recs[j]["pep"] != recs[i]["pep"]]
        if not others:
            continue
        # pairwise: y_i = mean_ref (y_ref + ΔΔG_pred(i,ref))
        ddg = pm.predict(np.array([Xp[i] - Xp[j] for j in others]))
        pair_t.append(yp[i]); pair_p.append(float(np.mean([yp[j] + ddg[k] for k, j in enumerate(others)])))
    for yt, yp_ in simple_anchor(idxs):
        simp_t.append(yt); simp_p.append(yp_)


def rep(t, p):
    return pearsonr(t, p)[0], float(np.mean(np.abs(np.array(t) - np.array(p))))


rs, ms = rep(simp_t, simp_p); rp, mp = rep(pair_t, pair_p)
print(f"  simple-subtraction anchor : r={rs:+.3f} MAE={ms:.2f} (n={len(simp_t)})")
print(f"  pairwise-ΔΔG (Siamese) anchor: r={rp:+.3f} MAE={mp:.2f} (n={len(pair_t)})")
json.dump(dict(c1_758_static=cv(Xs, y758)[0], c1_758_plus=cv(np.hstack([Xs, Xc1]), y758)[0],
               c1_ppikb_delta=float(r1 - r0), c2_simple_r=float(rs), c2_pairwise_r=float(rp)),
          open("data/e279_c1_c2.json", "w"))
print("\nsaved data/e279_c1_c2.json")
