"""E286 — (1) shuffle control for e285 meticulous transfer; (2) Ram's remove-the-Coulomb salt-bridge ML.

(1) e285 showed rich-pocket offset transfer Δr=+0.033 but WORSE MAE and bigger help for POOR matches than
good ones — suspicious. Decisive control: transfer offsets from RANDOM reference receptors (not similar).
If random gives the same Δr, the gain is a generic shift artifact, NOT pocket-similarity transfer.

(2) Ram: "remove the Coulomb half" — the explicit Coulomb energy is the misleading half (huge, cancels).
Test salt-bridge features WITHOUT Coulomb energy (counts + nearest distance + burial only) vs base, on
charged leave-receptor-out. Does pure geometry let ML learn the net better than including Coulomb?
Run: OMP_NUM_THREADS=1 python experiments/e286_controls.py
"""
from __future__ import annotations
import json, importlib.util, glob, os, numpy as np
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
spec2 = importlib.util.spec_from_file_location("e284", "experiments/e284_saltbridge_ml.py")
e284 = importlib.util.module_from_spec(spec2)
try:
    spec2.loader.exec_module(e284)
except SystemExit:
    pass
except Exception:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


# ---------- (1) shuffle control for meticulous transfer ----------
rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")
        if json.loads(l).get("aff_type") in ("Kd", "Ki", "KD")]
recs = []
for r in rows:
    if not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and np.isfinite(y):
        recs.append({"pdb": r["pdb"].lower(), "rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                     "desc": d3, "atype": r["aff_type"], "length": int(pf(r["length"])),
                     "nc": float(pf(r["net_charge"])), "npep": r.get("npep", r["length"]),
                     "npocket": r.get("npocket", 0)})
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
Xtr = np.array([r["desc"] for r in pool]); ytr = np.array([r["y"] for r in pool])
om = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                   l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
grp = np.array([hash(r["rec"]) % (10**9) for r in pool])
Spool = np.full(len(pool), np.nan)
for tr, te in GroupKFold(8).split(Xtr, ytr, grp):
    Spool[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
resid = ytr - Spool
Sf = om.predict(np.array([r["desc"] for r in fresh])); yf = np.array([r["y"] for r in fresh])
by = defaultdict(list)
for i, r in enumerate(pool):
    by[r["rec"]].append(i)
offsets = {s: float(np.mean([resid[i] for i in idx])) for s, idx in by.items()}
off_vals = np.array(list(offsets.values()))


def rmae(p, t):
    return (pearsonr(t, p)[0], float(np.mean(np.abs(t - p))))


# RANDOM offset transfer (shuffle): add a random reference receptor's offset
rand_preds = Sf + rng.choice(off_vals, size=len(fresh))
# mean offset transfer (global shift): add the pool mean offset
mean_preds = Sf + off_vals.mean()
print("=== (1) shuffle/sanity controls for meticulous transfer (304 held-out) ===")
print(f"  ABSOLUTE                r={rmae(Sf,yf)[0]:+.3f} MAE={rmae(Sf,yf)[1]:.2f}")
print(f"  + RANDOM offset (shuffle) r={rmae(rand_preds,yf)[0]:+.3f} MAE={rmae(rand_preds,yf)[1]:.2f}")
print(f"  + GLOBAL mean offset      r={rmae(mean_preds,yf)[0]:+.3f} MAE={rmae(mean_preds,yf)[1]:.2f}")
print("  (e285 meticulous gave r=+0.252; if random/global match that, the +0.033 is NOT similarity)")

# ---------- (2) remove-the-Coulomb salt-bridge ML ----------
print("\n=== (2) remove-the-Coulomb salt-bridge ML (charged, leave-receptor-out) ===", flush=True)
pdbrows = [json.loads(l) for l in open("data/pdbbind_peptides.jsonl")]
pidx = {os.path.basename(p).split("_")[0].lower(): p
        for p in glob.glob("data/drive_pull/pl/P-L/**/*_protein.pdb", recursive=True)}
PFEAT = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
data = []
for r in pdbrows:
    pid = r["pdb"].lower(); prot = pidx.get(pid)
    lig = glob.glob(f"data/drive_pull/pl/P-L/*/{pid}/{pid}_ligand.mol2")
    if not prot or not lig:
        continue
    q = sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"])
    try:
        sb = e284.sb_features(e284.receptor_charges(prot), e284.peptide_charges_mol2(lig[0]))
    except Exception:
        sb = [0.0] * 7
    # sb = [n4,n6,n8,nrep,coul,ebur,nearest]; NO-COULOMB drops indices 4,5 (coul, ebur)
    data.append({"pdb": pid, "x": [float(r[f]) for f in PFEAT],
                 "sb_full": sb, "sb_nocoul": [sb[0], sb[1], sb[2], sb[3], sb[6]],
                 "y": float(r["y"]), "q": abs(q)})
X = np.array([d["x"] for d in data]); y = np.array([d["y"] for d in data]); q = np.array([d["q"] for d in data])
gg = np.array([hash(d["pdb"]) % (10**9) for d in data])
ch = q >= 2
def cv(M):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, gg):
        p[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(M[tr], y[tr]).predict(M[te])
    return p
pbase = cv(X)
pfull = cv(np.hstack([X, np.array([d["sb_full"] for d in data])]))
pnoc = cv(np.hstack([X, np.array([d["sb_nocoul"] for d in data])]))
for label, p in [("base (17 feat)", pbase), ("+SB full (w/ Coulomb)", pfull),
                 ("+SB NO-Coulomb (geom only)", pnoc)]:
    rc = pearsonr(y[ch], p[ch])[0]; mc = np.mean(np.abs(y[ch] - p[ch]))
    print(f"  {label:28s} CHARGED r={rc:+.3f} MAE={mc:.2f}")
json.dump({"meticulous_random_r": float(rmae(rand_preds, yf)[0])}, open("data/e286_controls.json", "w"))
print("\nsaved data/e286_controls.json")
