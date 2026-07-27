"""E305 — prototype a charge-DESOLVATION penalty feature on the honest e298 testbed.

Builds on prior desolvation work (e33 single-pose GB, ProtDCal charged descriptors, charge_complementarity,
e75 unsatisfied-charge). The current honest-CV feature set (e298: 17 geom + 19 IFP) has raw charged-CONTACT
counts (contact_chg, sb_fav, hb_to_chg) but NO desolvation PENALTY term. Hypothesis: buried charge that is
NOT compensated by a favorable salt bridge pays a net desolvation cost the model can't see, so it overrates
charged/buried interfaces (the importin/NLS failure).

Method (same as e298): GroupKFold(8) leave-receptor-out on PDBbind crystals. First check whether each
candidate desolvation feature correlates with the OURS+IFP RESIDUAL on the charged subset (= orthogonal
signal the model is missing). Then add the best candidate(s) and re-measure ALL/charged/neutral r + MAE.
Run: OMP_NUM_THREADS=1 python scripts/e305_desolv_prototype.py
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
print(f"matched: {len(data)}", flush=True)

X = np.array([d["x"] for d in data])       # 17 geometry
IFP = np.array([d["ifp"] for d in data])   # 19 typed IFP
y = np.array([d["y"] for d in data])
q = np.array([float(d["q"]) for d in data])
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
ch = q >= 2

# --- feature indices ---
GEO = ("arom_cc","bsa_hyd","cys_frac","hb_count","length","mean_burial","mj_contact","org_density",
       "poc_eis","poc_f_arom","poc_f_hyd","poc_n","poc_net","rg_per_L","sasa_hb","sasa_sb","strength_bur")
IFPN = ("sb_fav","sb_fav_str","sb_unfav","sb_d2","sb_d3","sb_d4","hbond","hbond_str","hb_to_chg","hb_to_pol",
        "hb_to_hyd","hb_to_aro","hydrophobic","hyd_str","aromatic","contact_chg","contact_pol","contact_hyd","contact_aro")
gi = {n: i for i, n in enumerate(GEO)}
fi = {n: i for i, n in enumerate(IFPN)}
burial = X[:, gi["mean_burial"]] / 100.0
contact_chg = IFP[:, fi["contact_chg"]]
hb_to_chg = IFP[:, fi["hb_to_chg"]]
sb_fav_str = IFP[:, fi["sb_fav_str"]]
sb_fav = IFP[:, fi["sb_fav"]]
sb_unfav = IFP[:, fi["sb_unfav"]]

# --- candidate desolvation-penalty features (positive = more desolvation cost = should weaken binding) ---
cand = {
    "d1_buried_chg":        contact_chg * burial,
    "d2_netq_burial":       np.abs(q) * burial,
    "d3_uncomp_hbchg":      np.maximum(hb_to_chg - sb_fav, 0.0),
    "d4_uncomp_contactchg": np.maximum(contact_chg - 4.0 * sb_fav_str, 0.0),
    "d5_sb_unfav":          sb_unfav,
    "d6_chg_x_burial_deep": contact_chg * burial ** 2,
}


def gbt():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)


def cv(M):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, grp):
        p[te] = gbt().fit(M[tr], y[tr]).predict(M[te])
    return p


base = np.hstack([X, IFP])
p_base = cv(base)
resid = y - p_base   # >0 where model OVERpredicts strength (pred too negative)

print("\n=== step 1: does each candidate correlate with the OURS+IFP RESIDUAL? (orthogonal signal check) ===")
print(f"{'candidate':22s} {'r_resid ALL':>12s} {'r_resid CHARGED':>16s} {'r_raw vs y CHARGED':>18s}")
for name, f in cand.items():
    rr_all = pearsonr(f, resid)[0]
    rr_ch = pearsonr(f[ch], resid[ch])[0] if ch.sum() > 3 else float("nan")
    ry_ch = pearsonr(f[ch], y[ch])[0] if ch.sum() > 3 else float("nan")
    print(f"{name:22s} {rr_all:+12.3f} {rr_ch:+16.3f} {ry_ch:+18.3f}")


def rm(p, m):
    return (pearsonr(y[m], p[m])[0], float(np.mean(np.abs(y[m] - p[m]))))


print("\n=== step 2: add each candidate to OURS+IFP, re-measure honest leave-receptor-out CV ===")
print(f"{'model':26s} {'ALL r/MAE':>15s} {'CHARGED r/MAE':>16s} {'NEUTRAL r/MAE':>16s}")
a = rm(p_base, np.ones(len(y), bool)); c = rm(p_base, ch); nn = rm(p_base, q <= 1)
print(f"{'OURS + IFP (baseline)':26s} {a[0]:+.3f}/{a[1]:.2f}    {c[0]:+.3f}/{c[1]:.2f}    {nn[0]:+.3f}/{nn[1]:.2f}")
rows = {"baseline": {"all": a, "charged": c, "neutral": nn}}
for name, f in cand.items():
    p = cv(np.hstack([base, f.reshape(-1, 1)]))
    a = rm(p, np.ones(len(y), bool)); c = rm(p, ch); nn = rm(p, q <= 1)
    rows[name] = {"all": a, "charged": c, "neutral": nn}
    print(f"{'+ '+name:26s} {a[0]:+.3f}/{a[1]:.2f}    {c[0]:+.3f}/{c[1]:.2f}    {nn[0]:+.3f}/{nn[1]:.2f}")

# also all candidates together
p = cv(np.hstack([base] + [f.reshape(-1, 1) for f in cand.values()]))
a = rm(p, np.ones(len(y), bool)); c = rm(p, ch); nn = rm(p, q <= 1)
rows["+ ALL desolv"] = {"all": a, "charged": c, "neutral": nn}
print(f"{'+ ALL desolv candidates':26s} {a[0]:+.3f}/{a[1]:.2f}    {c[0]:+.3f}/{c[1]:.2f}    {nn[0]:+.3f}/{nn[1]:.2f}")

json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in rows.items()},
          open(os.path.join(ROOT, "data/e305_desolv.json"), "w"), indent=1)
print("\nsaved data/e305_desolv.json")
