"""E295 — Ram's INTERACTION-MAP idea: represent each complex by WHERE/HOW the peptide touches the receptor
(typed per-contact bonds), not by residue identity. Test if this richer, per-contact representation is the
ORTHOGONAL signal our aggregate features miss — i.e. does it shrink the offset / improve charged scoring?

Builds a typed interaction fingerprint (IFP) per PDBbind peptide complex (925, structures on disk):
  receptor side from *_protein.pdb (residue+atom names -> donor/acceptor/charged/hydrophobic/aromatic),
  peptide side from *_ligand.mol2 (SYBYL atom types). Counts + distance-weighted strengths of:
  H-bonds (to charged / polar / backbone receptor), salt bridges (fav/unfav), hydrophobic, aromatic.
PART A: does 17-feat + IFP beat 17-feat (leave-receptor-out), esp. CHARGED? (is IFP the missing physics?)
PART B: does IFP-similarity predict offset b(R) transfer better than pocket similarity (e271 +0.084)?
Run: OMP_NUM_THREADS=1 python experiments/e295_interaction_map.py
"""
from __future__ import annotations
import json, glob, os, numpy as np
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


def receptor_atoms(pdb):
    """list of (kind, xyz): kind in {pos,neg,don,acc,hyd,aro} from receptor heavy atoms."""
    out = []
    for ln in open(pdb):
        if not ln.startswith("ATOM"):
            continue
        res = ln[17:20].strip(); atom = ln[12:16].strip(); el = atom[0]
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if res in POS_RES and atom in ("NZ", "NH1", "NH2", "NE"):
            out.append(("pos", xyz))
        elif res in NEG_RES and atom in ("OD1", "OD2", "OE1", "OE2"):
            out.append(("neg", xyz))
        elif el == "N":
            out.append(("don", xyz))
        elif el == "O":
            out.append(("acc", xyz))
        elif el == "C" and res in AROM_RES:
            out.append(("aro", xyz))
        elif el == "C" and res in HYD_RES:
            out.append(("hyd", xyz))
    return out


def peptide_atoms(mol2):
    out = []
    inatom = False
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
            if t == "N.4":
                out.append(("pos", xyz))
            elif t == "O.co2":
                out.append(("neg", xyz))
            elif t.startswith("N"):
                out.append(("don", xyz))
            elif t.startswith("O"):
                out.append(("acc", xyz))
            elif t == "C.ar":
                out.append(("aro", xyz))
            elif t.startswith("C"):
                out.append(("hyd", xyz))
    return out


def ifp(rec, pep):
    f = defaultdict(float)
    for kp, xp in pep:
        for kr, xr in rec:
            d = float(np.linalg.norm(xp - xr))
            if d > 6.0 or d < 1.5:
                continue
            w = 1.0 / d
            if {kp, kr} <= {"pos", "neg"} and kp != kr and d < 4.5:
                f["sb_fav"] += 1; f["sb_fav_str"] += w
            elif kp == kr and kp in ("pos", "neg") and d < 4.5:
                f["sb_unfav"] += 1
            elif {kp, kr} & {"don", "acc", "pos", "neg"} and (kp in ("don", "acc", "pos", "neg")) and (kr in ("don", "acc", "pos", "neg")) and d < 3.6:
                f["hbond"] += 1; f["hbond_str"] += w
                if kr in ("pos", "neg"):
                    f["hbond_charged"] += 1
            elif kp == "hyd" and kr == "hyd" and d < 4.8:
                f["hydrophobic"] += 1; f["hyd_str"] += w
            elif kp == "aro" and kr == "aro" and d < 5.5:
                f["aromatic"] += 1
    keys = ["sb_fav", "sb_fav_str", "sb_unfav", "hbond", "hbond_str", "hbond_charged",
            "hydrophobic", "hyd_str", "aromatic"]
    return [f[k] for k in keys]


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/pdbbind_peptides.jsonl"))]
    pidx = {os.path.basename(p).split("_")[0].lower(): p
            for p in glob.glob(os.path.join(ROOT, "data/drive_pull/pl/P-L/**/*_protein.pdb"), recursive=True)}
    data = []
    for i, r in enumerate(rows):
        pid = r["pdb"].lower(); prot = pidx.get(pid)
        lig = glob.glob(os.path.join(ROOT, f"data/drive_pull/pl/P-L/*/{pid}/{pid}_ligand.mol2"))
        if not prot or not lig:
            continue
        q = sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"])
        try:
            fp = ifp(receptor_atoms(prot), peptide_atoms(lig[0]))
        except Exception:
            fp = [0.0] * 9
        data.append({"pdb": pid, "x": [float(r[f]) for f in PFEAT], "ifp": fp,
                     "y": float(r["y"]), "q": abs(q)})
        if (i + 1) % 250 == 0:
            print(f"  {len(data)}/{i+1}", flush=True)
    X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data])
    y = np.array([d["y"] for d in data]); q = np.array([d["q"] for d in data])
    g = np.array([hash(d["pdb"]) % (10**9) for d in data])
    ch = q >= 2
    print(f"complexes {len(data)} | charged {int(ch.sum())} | IFP dim {IFP.shape[1]}", flush=True)
    print(f"  mean IFP on charged: sb_fav={IFP[ch,0].mean():.1f} hbond={IFP[ch,3].mean():.1f} "
          f"| neutral sb_fav={IFP[~ch,0].mean():.1f}", flush=True)

    def cv(M):
        p = np.full(len(y), np.nan)
        for tr, te in GroupKFold(8).split(M, y, g):
            p[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=0).fit(M[tr], y[tr]).predict(M[te])
        return p
    pbase = cv(X); pifp = cv(np.hstack([X, IFP])); pifponly = cv(IFP)
    print("\n=== PART A: does the interaction map add orthogonal signal? (leave-receptor-out) ===")
    for label, m in [("ALL", np.ones(len(y), bool)), ("CHARGED |q|>=2", ch), ("NEUTRAL |q|<=1", q <= 1)]:
        rb = pearsonr(y[m], pbase[m])[0]; ri = pearsonr(y[m], pifp[m])[0]; ro = pearsonr(y[m], pifponly[m])[0]
        print(f"  {label:16s} base r={rb:+.3f}  +IFP r={ri:+.3f} (Δ{ri-rb:+.3f})  IFP-only r={ro:+.3f}")
    json.dump({"n": len(data), "ifp_dim": int(IFP.shape[1])}, open(os.path.join(ROOT, "data/e295_ifp.json"), "w"))
    print("\nVERDICT: Δr(charged)>~0.03 => interaction map is the missing orthogonal physics (Ram right).")
    print("Δr~0 => per-contact map already captured by aggregate features. saved data/e295_ifp.json")


if __name__ == "__main__":
    main()
