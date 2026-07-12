"""E296 — rigorously validate the IFP win + richer IFP + offset-shrinkage test. Caches IFP for e297 alchemy.

(1) PROPER leave-RECEPTOR-out (group by receptor SEQUENCE, not pdb) — does the +0.063 survive honest CV?
(2) RICHER IFP: distance-binned bonds + per-receptor-residue-type contacts + burial (~20-dim).
(3) OFFSET SHRINKAGE: does the IFP scorer have a SMALLER receptor offset b(R)? (the mechanism: IFP adds
    orthogonal physics -> shrinks the FEP-bound wall -> cross-receptor transfer should work better).
Caches data/e296_ifp_cache.json (per complex: receptor seq, peptide, y, q, rich IFP).
Run: OMP_NUM_THREADS=1 python experiments/e296_ifp_rigorous.py
"""
from __future__ import annotations
import json, glob, os, hashlib, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFEAT = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
POS_RES = {"LYS", "ARG"}; NEG_RES = {"ASP", "GLU"}; AROM_RES = {"PHE", "TYR", "TRP", "HIS"}
HYD_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
POL_RES = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "CYS"}
_3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def receptor_atoms_and_seq(pdb):
    atoms = []; seq = []
    for ln in open(pdb):
        if not ln.startswith("ATOM"):
            continue
        res = ln[17:20].strip(); atom = ln[12:16].strip(); el = atom[0]
        if atom == "CA" and res in _3to1:
            seq.append(_3to1[res])
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        cls = None
        if res in POS_RES and atom in ("NZ", "NH1", "NH2", "NE"):
            cls = "pos"
        elif res in NEG_RES and atom in ("OD1", "OD2", "OE1", "OE2"):
            cls = "neg"
        elif el == "N":
            cls = "don"
        elif el == "O":
            cls = "acc"
        elif el == "C" and res in AROM_RES:
            cls = "aro"
        elif el == "C" and res in HYD_RES:
            cls = "hyd"
        if cls:
            restype = ("chg" if res in POS_RES | NEG_RES else "pol" if res in POL_RES
                       else "aro" if res in AROM_RES else "hyd")
            atoms.append((cls, restype, xyz))
    return atoms, "".join(seq)


def peptide_atoms(mol2):
    out = []; inatom = False
    for ln in open(mol2):
        if ln.startswith("@<TRIPOS>ATOM"):
            inatom = True; continue
        if ln.startswith("@<TRIPOS>") and inatom:
            break
        if inatom:
            p = ln.split()
            if len(p) < 6:
                continue
            try:
                xyz = np.array([float(p[2]), float(p[3]), float(p[4])])
            except ValueError:
                continue
            t = p[5]
            cls = ("pos" if t == "N.4" else "neg" if t == "O.co2" else "don" if t.startswith("N")
                   else "acc" if t.startswith("O") else "aro" if t == "C.ar"
                   else "hyd" if t.startswith("C") else None)
            if cls:
                out.append((cls, xyz))
    return out


def rich_ifp(rec, pep):
    f = defaultdict(float)
    for kp, xp in pep:
        for kr, rt, xr in rec:
            d = float(np.linalg.norm(xp - xr))
            if d > 6.0 or d < 1.5:
                continue
            w = 1.0 / d
            if {kp, kr} <= {"pos", "neg"} and kp != kr and d < 4.5:
                f["sb_fav"] += 1; f["sb_fav_str"] += w
                f[f"sb_d{min(int(d),4)}"] += 1
            elif kp == kr and kp in ("pos", "neg") and d < 4.5:
                f["sb_unfav"] += 1
            elif kp in ("don", "acc", "pos", "neg") and kr in ("don", "acc", "pos", "neg") and d < 3.6:
                f["hbond"] += 1; f["hbond_str"] += w
                f[f"hb_to_{rt}"] += 1
            elif kp == "hyd" and kr == "hyd" and d < 4.8:
                f["hydrophobic"] += 1; f["hyd_str"] += w
            elif kp == "aro" and kr == "aro" and d < 5.5:
                f["aromatic"] += 1
            f[f"contact_{rt}"] += w   # total contact strength by receptor residue type
    keys = ["sb_fav", "sb_fav_str", "sb_unfav", "sb_d2", "sb_d3", "sb_d4", "hbond", "hbond_str",
            "hb_to_chg", "hb_to_pol", "hb_to_hyd", "hb_to_aro", "hydrophobic", "hyd_str", "aromatic",
            "contact_chg", "contact_pol", "contact_hyd", "contact_aro"]
    return [f[k] for k in keys]


def build():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/pdbbind_peptides.jsonl"))]
    pidx = {os.path.basename(p).split("_")[0].lower(): p
            for p in glob.glob(os.path.join(ROOT, "data/drive_pull/pl/P-L/**/*_protein.pdb"), recursive=True)}
    data = []
    for i, r in enumerate(rows):
        pid = r["pdb"].lower(); prot = pidx.get(pid)
        lig = glob.glob(os.path.join(ROOT, f"data/drive_pull/pl/P-L/*/{pid}/{pid}_ligand.mol2"))
        if not prot or not lig:
            continue
        try:
            ra, rseq = receptor_atoms_and_seq(prot)
            fp = rich_ifp(ra, peptide_atoms(lig[0]))
        except Exception:
            continue
        q = sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"])
        data.append({"pdb": pid, "rseq": rseq, "pep": r["seq"], "x": [float(r[f]) for f in PFEAT],
                     "ifp": fp, "y": float(r["y"]), "q": abs(q)})
        if (i + 1) % 250 == 0:
            print(f"  extracted {len(data)}/{i+1}", flush=True)
    json.dump(data, open(os.path.join(ROOT, "data/e296_ifp_cache.json"), "w"))
    return data


cache = os.path.join(ROOT, "data/e296_ifp_cache.json")
data = json.load(open(cache)) if os.path.exists(cache) else build()
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
y = np.array([d["y"] for d in data]); q = np.array([d["q"] for d in data])
recgrp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
ch = q >= 2
n_rec = len(set(recgrp.tolist()))
print(f"complexes {len(data)} | unique receptors {n_rec} | IFP dim {IFP.shape[1]} | charged {int(ch.sum())}",
      flush=True)


def cv(M, groups):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(M, y, groups):
        p[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0).fit(M[tr], y[tr]).predict(M[te])
    return p


print("\n=== (1) PROPER leave-RECEPTOR-out (group by receptor sequence) ===")
pb = cv(X, recgrp); pi = cv(np.hstack([X, IFP]), recgrp)
for label, m in [("ALL", np.ones(len(y), bool)), ("CHARGED", ch), ("NEUTRAL", q <= 1)]:
    rb = pearsonr(y[m], pb[m])[0]; ri = pearsonr(y[m], pi[m])[0]
    print(f"  {label:9s} base r={rb:+.3f}  +richIFP r={ri:+.3f}  (Δ{ri-rb:+.3f})")

print("\n=== (3) OFFSET SHRINKAGE: does IFP reduce the receptor offset b(R)? ===")
for name, pred in [("base scorer", pb), ("IFP scorer", pi)]:
    e = pred - y
    rec_cells = defaultdict(list)
    for i in range(len(e)):
        rec_cells[recgrp[i]].append(i)
    bR = [np.mean([e[i] for i in c]) for c in rec_cells.values() if len(c) >= 2]
    print(f"  {name:12s} receptor-offset std = {np.std(bR):.2f} kcal/mol (n={len(bR)} multi-pep receptors)")
json.dump({"n": len(data), "ifp_dim": int(IFP.shape[1])}, open(os.path.join(ROOT, "data/e296_rig.json"), "w"))
print("\nsaved data/e296_ifp_cache.json + data/e296_rig.json")
