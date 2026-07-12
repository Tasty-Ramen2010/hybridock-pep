"""E311 — is the charged-target failure reweighting-fixable? Diagnosis + a battery of charged ideas. REFUTED.

Ram's idea: for charged targets upweight IFP and downweight geometry, scaled by charged-residue fraction and
charge magnitude. First a diagnosis: on the charged subset the leaky (random-KFold) and honest
(leave-receptor-out) r are nearly equal (~0.42 vs ~0.40) — a tiny gap, unlike neutral — so knowing the
receptor barely helps: the charged limit is missing WITHIN-target signal, not the receptor offset b(R).
Reweighting can only redistribute signal that exists. Then a battery of charged-specific features + the two
faithful implementations of Ram's idea (a charge-gated model blend, and charge x IFP interaction features,
since the tree model ignores literal feature-weight scaling). Nothing beats the 0.401 baseline.
Run: OMP_NUM_THREADS=1 python experiments/e311_charged_ideas.py
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower(): json.loads(l)["desc"]
         for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl")) if json.loads(l).get("desc")}
data = [d for d in cache if d["pdb"] in protd and len(protd[d["pdb"]]) == 37]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
q = np.array([float(d["q"]) for d in data]); aq = np.abs(q)
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
peps = [d["pep"] for d in data]
ch = aq >= 2

POS, NEG = set("KR"), set("DE")
CHG = POS | NEG | set("H")
chg_frac = np.array([sum(c in CHG for c in p) / max(len(p), 1) for p in peps])
net_pep = np.array([sum(c in POS for c in p) - sum(c in NEG for c in p) for p in peps], float)
IFPN = ("sb_fav", "sb_fav_str", "sb_unfav", "sb_d2", "sb_d3", "sb_d4", "hbond", "hbond_str", "hb_to_chg",
        "hb_to_pol", "hb_to_hyd", "hb_to_aro", "hydrophobic", "hyd_str", "aromatic", "contact_chg",
        "contact_pol", "contact_hyd", "contact_aro")
fi = {n: i for i, n in enumerate(IFPN)}


def cv(M, groups=grp):
    p = np.full(len(y), np.nan)
    sp = GroupKFold(8).split(M, y, groups) if groups is not None else KFold(8, shuffle=True, random_state=0).split(M)
    for tr, te in sp:
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
        p[te] = m.fit(M[tr], y[tr]).predict(M[te])
    return p


def rc(p):
    return pearsonr(y[ch], p[ch])[0]


base = np.hstack([X, IFP])
print(f"charged subset n={ch.sum()} of {len(y)}")
print("\n=== DIAGNOSIS: feature-limited or receptor-offset? ===")
print(f"  charged r RANDOM KFold (leaky)      {rc(cv(base, None)):+.3f}")
pbase = cv(base)
print(f"  charged r leave-receptor-out (honest) {rc(pbase):+.3f}   (tiny gap => feature-limited, not offset)")
print(f"  neutral r leave-receptor-out          {pearsonr(y[~ch], pbase[~ch])[0]:+.3f}")

print(f"\n=== IDEA BATTERY (honest leave-receptor-out charged r; baseline geom+IFP = {rc(pbase):+.3f}) ===")
saltq = (IFP[:, fi["sb_fav_str"]] - IFP[:, fi["sb_unfav"]]).reshape(-1, 1)
unsat = np.maximum(IFP[:, fi["hb_to_chg"]] - IFP[:, fi["sb_fav"]], 0).reshape(-1, 1)
elec = (net_pep * np.sign(IFP[:, fi["sb_fav"]] - IFP[:, fi["sb_unfav"]] + 1e-9)).reshape(-1, 1)
chgfeat = np.column_stack([chg_frac, aq, net_pep])
chgxifp = chg_frac[:, None] * IFP
for name, M in [("+ charge x IFP interaction (Ram B)", np.hstack([base, chgxifp])),
                ("+ salt-bridge quality", np.hstack([base, saltq])),
                ("+ unsatisfied buried charge", np.hstack([base, unsat])),
                ("+ elec complementarity", np.hstack([base, elec])),
                ("+ raw charge descriptors", np.hstack([base, chgfeat])),
                ("+ ALL charge features", np.hstack([base, saltq, unsat, elec, chgfeat, chgxifp])),
                ("IFP-only (drop geometry)", IFP)]:
    print(f"  {name:34s}: {rc(cv(M)):+.3f}")

# Ram idea A: charge-gated blend of geom-only and IFP-only
pg, pi = cv(X), cv(IFP)
w = np.clip(chg_frac / chg_frac[ch].mean(), 0, 1)
blend = (1 - w) * pg + w * pi
print(f"  {'charge-gated geom<->IFP blend (Ram A)':34s}: {pearsonr(y[ch], blend[ch])[0]:+.3f}")

# charged specialist
psp = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(base, y, grp):
    trc = tr[ch[tr]]
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    psp[te] = m.fit(base[trc], y[trc]).predict(base[te])
print(f"  {'charged-specialist (train on charged)':34s}: {rc(psp):+.3f}")
print("\nVERDICT: nothing beats 0.401 -> charged is feature/signal-limited (FEP-bound desolvation), not reweightable.")
