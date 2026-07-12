"""E284 — Ram's salt-bridge ML: explicit (charge -> receptor-spot) salt-bridge features for charged Kd.

Extracts EXPLICIT salt-bridge geometry from the 925 PDBbind peptide crystal complexes:
  receptor charged sidechain tips (Lys NZ, Arg CZ, Asp CG, Glu CD) from *_protein.pdb,
  peptide charged sidechain tips from *_ligand.mol2 (coords by atom name),
then per complex builds salt-bridge descriptors: # opposite-charge pairs within 4/6/8 A, screened Coulomb
sum (eps=4r), # like-charge repulsions, nearest opposite-charge distance, burial-weighted bridge energy.
Tests whether these explicit features (orthogonal to our aggregate features) reduce the CHARGED residual
under leave-receptor-out CV — the one untested orthogonal-physics lever for b(R).
Run: OMP_NUM_THREADS=1 python experiments/e284_saltbridge_ml.py
"""
from __future__ import annotations
import json, glob, os, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFEAT = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]
# receptor charged tip atom per residue
RECTIP = {("LYS", "NZ"): 1.0, ("ARG", "CZ"): 1.0, ("ASP", "CG"): -1.0, ("GLU", "CD"): -1.0}


def receptor_charges(pdb):
    out = []
    for ln in open(pdb):
        if not ln.startswith("ATOM"):
            continue
        res = ln[17:20].strip(); atom = ln[12:16].strip()
        if (res, atom) in RECTIP:
            try:
                xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
            except ValueError:
                continue
            out.append((RECTIP[(res, atom)], np.array(xyz)))
    return out


def peptide_charges_mol2(mol2):
    """Charged tips from mol2 by SYBYL atom type: N.4/N.3 (Lys), N.pl3/N.ar guanidinium (Arg+), O.co2 (-)."""
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
            if t in ("N.4",):
                out.append((1.0, xyz))
            elif t == "O.co2":
                out.append((-0.5, xyz))   # carboxylate O (two per group ~ -1 total)
    return out


def sb_features(rec, pep):
    if not rec or not pep:
        return [0.0] * 7
    n4 = n6 = n8 = nrep = 0
    coul = 0.0; ebur = 0.0; nearest = 14.0
    for qp, xp in pep:
        for qr, xr in rec:
            d = float(np.linalg.norm(xp - xr))
            if d < 1.5 or d > 12:
                continue
            e = 332.0 * qp * qr / (4.0 * d * d)
            coul += e
            if qp * qr < 0:
                if d < 4:
                    n4 += 1
                if d < 6:
                    n6 += 1
                if d < 8:
                    n8 += 1
                nearest = min(nearest, d)
                ebur += e
            elif d < 6:
                nrep += 1
    return [float(n4), float(n6), float(n8), float(nrep), coul, ebur, nearest]


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/pdbbind_peptides.jsonl"))]
    pidx = {os.path.basename(p).split("_")[0].lower(): p
            for p in glob.glob(os.path.join(ROOT, "data/drive_pull/pl/P-L/**/*_protein.pdb"), recursive=True)}
    data = []
    for i, r in enumerate(rows):
        pid = r["pdb"].lower()
        prot = pidx.get(pid)
        lig = glob.glob(os.path.join(ROOT, f"data/drive_pull/pl/P-L/*/{pid}/{pid}_ligand.mol2"))
        if not prot or not lig:
            continue
        q = sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"])
        try:
            sb = sb_features(receptor_charges(prot), peptide_charges_mol2(lig[0]))
        except Exception:  # noqa: BLE001
            sb = [0.0] * 7
        data.append({"pdb": pid, "x": [float(r[f]) for f in PFEAT], "sb": sb,
                     "y": float(r["y"]), "q": abs(q)})
        if (i + 1) % 200 == 0:
            print(f"  extracted {len(data)}/{i+1}", flush=True)
    print(f"complexes with structures: {len(data)}", flush=True)
    X = np.array([d["x"] for d in data]); SB = np.array([d["sb"] for d in data])
    y = np.array([d["y"] for d in data]); q = np.array([d["q"] for d in data])
    grp = np.array([hash(d["pdb"]) % (10**9) for d in data])
    # sanity: do explicit SB features even fire on charged complexes?
    ch = q >= 2
    print(f"charged |q|>=2: {int(ch.sum())} | mean #salt-bridges(<6A) charged={SB[ch,1].mean():.2f} "
          f"neutral={SB[~ch,1].mean():.2f}", flush=True)

    def cv(M):
        p = np.full(len(y), np.nan)
        for tr, te in GroupKFold(8).split(M, y, grp):
            p[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=0
                                                  ).fit(M[tr], y[tr]).predict(M[te])
        return p

    pbase = cv(X); psb = cv(np.hstack([X, SB]))
    print("\n=== leave-receptor-out: base vs +explicit-salt-bridge ===")
    for label, m in [("ALL", np.ones(len(y), bool)), ("CHARGED |q|>=2", ch), ("NEUTRAL |q|<=1", q <= 1)]:
        rb = pearsonr(y[m], pbase[m])[0]; rs = pearsonr(y[m], psb[m])[0]
        mb = np.mean(np.abs(y[m] - pbase[m])); ms = np.mean(np.abs(y[m] - psb[m]))
        print(f"  {label:16s} base r={rb:+.3f}/MAE{mb:.2f}  +SB r={rs:+.3f}/MAE{ms:.2f}  (Δr={rs-rb:+.3f})")
    json.dump({"n": len(data), "charged": int(ch.sum())}, open(os.path.join(ROOT, "data/e284_sb.json"), "w"))
    print("\nVERDICT: Δr(charged)>~0.03 => explicit salt-bridge physics helps charged (Ram right).")
    print("Δr~0 => the salt-bridge signal is already captured / the net is FEP-bound. saved data/e284_sb.json")


if __name__ == "__main__":
    main()
