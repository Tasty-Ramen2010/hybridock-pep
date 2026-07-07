"""E327 (Ram's relative-FEP-by-neutralization idea) — the charged contribution to binding as an alchemical
NEUTRALIZATION double-difference, over a real generative pose cloud.

FEP computes ΔΔG by mutating a peptide and taking bound−free, not absolute bound−free. Ram: mutate the charged
peptide → neutral, and the *cost of that mutation* (bound vs free) is the charged contribution. Thermodynamic
cycle: charged_contribution = ΔG_neutralize(bound) − ΔG_neutralize(free). ΔG_neutralize has TWO halves:
  (1) lose the favorable peptide–receptor INTERACTION  → ½⟨V_elec⟩  (this is N2, which failed alone at n=212)
  (2) un-pay the DESOLVATION penalty of burying the charge → Born ∝ q²·burial
N2 used only (1). The catastrophic cancellation is that (1) and (2) nearly cancel; the NET is the signal. This
tests, over the e93 real pose clouds (poses on disk): does the neutralization double-difference (interaction −
desolvation, ensemble-averaged) recover the charged residual where ⟨V_elec⟩ alone did not?

Variants tested (Ram's "test lots"):
  V1 ⟨V_elec⟩ interaction only (=N2)     V2 ⟨Born⟩ desolvation only
  V3 net = ½⟨V_elec⟩ + ⟨Born⟩ (LIE β=0.5 neutralization ΔΔG)   V4 learned combo   V5 per-residue max mutation cost

Run: OMP_NUM_THREADS=1 python scripts/e327_neutralization_ddg.py
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E93 = json.load(open(os.path.join(ROOT, "data/e93_realpose_results.json")))
CAMP = os.path.join(ROOT, "runs/e93_realpose_campaign")
GEOM = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
CHG = {("LYS", "NZ"): 1., ("ARG", "NH1"): .5, ("ARG", "NH2"): .5,
       ("ASP", "OD1"): -.5, ("ASP", "OD2"): -.5, ("GLU", "OE1"): -.5, ("GLU", "OE2"): -.5}
PEP_CHG = {"K": 1, "R": 1, "D": -1, "E": -1}


def parse(fn):
    """Return (charged_atoms[(q,xyz)], all_heavy_xyz)."""
    chg, heavy = [], []
    for l in open(fn):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        name = l[12:16].strip()
        el = l[76:78].strip() or (name.lstrip("0123456789")[:1] if name else "")
        try:
            xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        except ValueError:
            continue
        if el.upper() != "H":
            heavy.append(xyz)
        qc = CHG.get((l[17:20].strip(), name))
        if qc is not None:
            chg.append((qc, xyz))
    return chg, (np.array(heavy) if heavy else np.zeros((0, 3)))


def pose_terms(pep_chg, rec_chg, rec_heavy):
    """(V_elec interaction, Born desolvation, per-residue-max mutation cost) for one pose."""
    ve = 0.0
    percharge = []
    for qp, xp in pep_chg:
        vi = 0.0
        for qr, xr in rec_chg:
            r = float(np.linalg.norm(xp - xr))
            if r >= 1.0:
                vi += qp * qr / r
        ve += vi
        burial = int(np.sum(np.linalg.norm(rec_heavy - xp, axis=1) < 6.0)) if len(rec_heavy) else 0
        born_i = qp * qp * burial
        # mutation cost of THIS charged atom = interaction lost + desolvation un-paid
        percharge.append(0.5 * vi + 0.01 * born_i)
    born = sum(qp * qp * (int(np.sum(np.linalg.norm(rec_heavy - xp, axis=1) < 6.0)) if len(rec_heavy) else 0)
               for qp, xp in pep_chg)
    max_mut = max((abs(v) for v in percharge), default=0.0)
    return ve, born, max_mut


rows = []
for cid, d in E93.items():
    recf = os.path.join(CAMP, cid, "poses_raw/poses_raw/poses_raw_protein_raw.pdb")
    posef = sorted(glob.glob(os.path.join(CAMP, cid, "poses/pose_*.pdb")))
    if not os.path.exists(recf) or len(posef) < 50 or "top5" not in d:
        continue
    rec_chg, rec_heavy = parse(recf)
    if not rec_chg:
        continue
    ve_l, born_l, mm_l = [], [], []
    for pf in posef:
        pep_chg, _ = parse(pf)
        ve, born, mm = pose_terms(pep_chg, rec_chg, rec_heavy)
        ve_l.append(ve); born_l.append(born); mm_l.append(mm)
    netq = sum(PEP_CHG.get(a, 0) for a in d["seq"])
    rows.append(dict(cid=cid, y=float(d["y"]), g=[float(d["top5"][f]) for f in GEOM], netq=abs(netq),
                     ve=float(np.mean(ve_l)), born=float(np.mean(born_l)), maxmut=float(np.mean(mm_l))))

print(f"e93 clouds with electrostatics: n={len(rows)}")
y = np.array([r["y"] for r in rows]); G = np.array([r["g"] for r in rows])
ve = np.array([r["ve"] for r in rows]); born = np.array([r["born"] for r in rows])
maxmut = np.array([r["maxmut"] for r in rows]); netq = np.array([r["netq"] for r in rows])
net_ddg = 0.5 * ve + born      # neutralization ΔΔG: interaction (fav, neg) + desolvation (unfav, pos)

resid = np.full(len(y), np.nan)
for tr, te in KFold(8, shuffle=True, random_state=0).split(G):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(G[tr], y[tr])
    resid[te] = y[te] - m.predict(G[te])

for label, mask in [("ALL", np.ones(len(y), bool)), (f"CHARGED |q|>=2", netq >= 2)]:
    if mask.sum() < 8:
        continue
    print(f"\n[{label}, n={mask.sum()}]  neutralization variants vs charged residual:")
    print(f"  V1 ⟨V_elec⟩ interaction (=N2)       : r={pearsonr(ve[mask], resid[mask])[0]:+.3f}")
    print(f"  V2 ⟨Born⟩ desolvation               : r={pearsonr(born[mask], resid[mask])[0]:+.3f}")
    print(f"  V3 net ½⟨V_elec⟩+⟨Born⟩ (neut. ΔΔG)  : r={pearsonr(net_ddg[mask], resid[mask])[0]:+.3f}")
    print(f"  V5 per-residue max mutation cost     : r={pearsonr(maxmut[mask], resid[mask])[0]:+.3f}")

# V4 learned combo — does the model prefer the NET over either half? LOO improvement over geometry.
def loo(X):
    p = np.full(len(y), np.nan)
    for tr, te in KFold(8, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return pearsonr(p, y)[0]

r0 = loo(G)
r_ve = loo(np.column_stack([G, ve]))
r_full = loo(np.column_stack([G, ve, born, net_ddg]))
print(f"\nV4 LOO r: geometry={r0:+.3f}  +⟨V_elec⟩={r_ve:+.3f}  +interaction+desolv+net={r_full:+.3f}")
print("VERDICT: " + ("adding DESOLVATION rescues the charged lever the interaction-alone (N2) missed."
                     if r_full - max(r0, r_ve) > 0.03 else
                     "the neutralization double-difference does NOT beat geometry either — on e93 the ensemble "
                     "electrostatics (with or without desolvation) is not a reliable charged lever."))
