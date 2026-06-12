"""E84 — the compression autopsy: is the missing variable RECEPTOR-CONTACT hydrophobic packing (Ram)?

A slope of 0.25 (the98-charged) is NOT noise — noise lowers r but keeps slope ~1. Attenuated slope =
missing variable or wrong functional form. Evidence it's a missing variable: E80 residual ~ vdw +0.32, and
the98 LOW-charge is also compressed (slope 0.66) -> the missing term is interface-wide, not charge.

Ram's hypothesis: model WHERE the receptor touches the peptide, the hydrophobicity of those contact
patches on BOTH sides, and derive interfacial-water strength NONLINEARLY (dewetting is cooperative).
Our current features are peptide-centric (bsa_hyd = how much of the PEPTIDE is buried-hydrophobic). They
miss the COMPLEMENTARITY: a hydrophobic peptide residue landing on a hydrophobic receptor patch expels
water cooperatively (strong); the same residue on a polar patch leaves frustrated water (weak). Same
buried area, very different ΔG -> compression.

Per peptide interface residue i (dSASA_i > 10):
  KD_pep_i        = Kyte-Doolittle hydrophobicity of i
  rec_patch_hyd_i = contact-weighted KD of receptor residues touching i (within 4.5 Å heavy-atom)
  pack_i          = receptor heavy atoms within 4.5 Å per peptide-i heavy atom (packing tightness)
Interface features (intensive /L), LINEAR and NONLINEAR:
  drydry_area   = Σ_i dSASA_i · [KD_pep_i>0]·[rec_patch_hyd_i>0]         (both-dry contact = dewetting)
  hyd_match     = Σ_i dSASA_i · max(0,KD_pep_i)·max(0,rec_patch_hyd_i)   (graded dry-dry product)
  hyd_mismatch  = Σ_i dSASA_i · [KD_pep_i>0]·[rec_patch_hyd_i<0]         (dry-on-wet frustration)
  packing       = mean_i pack_i                                          (shape complementarity proxy)
  drydry_sq     = drydry_area^2 / L                                       (cooperative nonlinearity)
TESTS: (1) does each recover the PROD residual OUT-OF-SAMPLE (leave-dataset-out)? (2) does adding it move
the SLOPE toward 1 (de-compress)? (3) linear vs nonlinear. A feature that lifts residual-r AND slope in
BOTH transfer directions = the missing variable Ram predicted.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from Bio.PDB.Structure import Structure  # noqa: E402
from Bio.PDB.Model import Model  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e84_contact.json")
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}


def featurize(pep_pdb, rec_pdb):
    pep = P.get_structure("p", str(pep_pdb))[0]
    rec = P.get_structure("r", str(rec_pdb))[0]
    SR.compute(pep, level="R")
    free = {(r.get_parent().id, r.id[1]): r.sasa for r in pep.get_residues() if r.id[0] == " "}
    cx = Structure("c"); m = Model(0); cx.add(m); used = set(); pep_cids = set()
    for tag, src in [("p", pep), ("r", rec)]:
        for ch in src.get_chains():
            cid = ch.id
            while cid in used:
                cid = chr((ord(cid) + 1) % 90 + 33)
            used.add(cid); c2 = ch.copy(); c2.id = cid; m.add(c2)
            if tag == "p":
                pep_cids.add(cid)
    SR.compute(cx, level="R")
    comp = {}
    for ch in cx.get_chains():
        if ch.id in pep_cids:
            for r in ch.get_residues():
                if r.id[0] == " ":
                    comp[(r.resname.upper(), r.id[1])] = r.sasa
    # receptor residues: centroid + KD + heavy atoms
    rec_res = []
    rec_heavy = []
    for ch in rec.get_chains():
        for r in ch.get_residues():
            if r.id[0] != " ":
                continue
            heavy = [a.coord.astype(float) for a in r if a.element != "H"]
            if not heavy:
                continue
            rec_res.append((KD.get(r.resname.upper(), 0.0), np.array(heavy)))
            rec_heavy.extend(heavy)
    if not rec_heavy:
        return None
    rtree = cKDTree(np.array(rec_heavy))
    # build per-rec-residue atom tree lookup: flatten with residue index
    rec_atom_xyz = []
    rec_atom_res = []
    for ri, (kd, heavy) in enumerate(rec_res):
        for h in heavy:
            rec_atom_xyz.append(h); rec_atom_res.append(ri)
    rec_atom_xyz = np.array(rec_atom_xyz); rec_atom_res = np.array(rec_atom_res)
    ratree = cKDTree(rec_atom_xyz)

    L = 0
    drydry = hydmatch = mismatch = bsa = 0.0
    pack_vals = []
    for ch in pep.get_chains():
        for r in ch.get_residues():
            if r.id[0] != " ":
                continue
            L += 1
            key = (r.get_parent().id, r.id[1])
            fs = free.get(key, 0.0); cs = comp.get((r.resname.upper(), r.id[1]), fs)
            dsasa = max(0.0, fs - cs)
            if dsasa < 10:
                continue
            bsa += dsasa
            kdp = KD.get(r.resname.upper(), 0.0)
            heavy = [a.coord.astype(float) for a in r if a.element != "H"]
            if not heavy:
                continue
            # receptor patch: residues with any atom within 4.5 Å of residue i
            near_res = set()
            n_rec_contacts = 0
            for h in heavy:
                idx = ratree.query_ball_point(h, 4.5)
                n_rec_contacts += len(idx)
                for j in idx:
                    near_res.add(rec_atom_res[j])
            if not near_res:
                continue
            patch_hyd = float(np.mean([rec_res[j][0] for j in near_res]))
            pack_vals.append(n_rec_contacts / len(heavy))
            if kdp > 0 and patch_hyd > 0:
                drydry += dsasa
                hydmatch += dsasa * kdp * patch_hyd
            elif kdp > 0 and patch_hyd < 0:
                mismatch += dsasa
    L = max(1, L)
    return dict(
        drydry_area=drydry / L,
        hyd_match=hydmatch / L,
        hyd_mismatch=mismatch / L,
        packing=float(np.mean(pack_vals)) if pack_vals else 0.0,
        drydry_sq=(drydry ** 2) / (L * 1000.0),
        bsa_chk=bsa / L,
    )


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    e49 = json.load(open("/tmp/e49b_the98.json"))
    work = Path("/tmp/ppep_work")
    for k in e49:
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            try:
                f = featurize(pep, rec)
                if f:
                    out["98_" + k] = f
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:40]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            f = featurize(r["peptide_pdb"], r["pocket_pdb"])
            if f:
                out["cr_" + r["pdb"]] = f
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:40]}")
    CACHE.write_text(json.dumps(out))
    return out


def main():
    src = open(ROOT / "scripts/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    base = ns["load"](); PROD = ns["PROD"]
    pred = ns["loo_pred"](base, PROD)
    for r, p in zip(base, pred):
        r["_pred"] = p; r["_resid"] = r["y"] - p

    cf = build()
    NEW = ["drydry_area", "hyd_match", "hyd_mismatch", "packing", "drydry_sq"]
    rows = []
    for r in base:
        k = ("cr_" + r["pdb"]) if r["ds"] == "cr65" else ("98_" + r["pdb"])
        f = cf.get(k)
        if f is None:
            continue
        rr = dict(r)
        for c in NEW:
            rr[c] = f[c]
        rows.append(rr)
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]
    print(f"=== E84 contact-hydrophobicity compression autopsy. cr65={len(c)} the98={len(n)} ===")

    # 1. residual recovery (does feature predict our ERROR?) — overall + per dataset
    print("\n1. Pearson(feature, PROD residual)  — a missing variable predicts our error:")
    print(f"{'feature':<14}{'all':>8}{'cr65':>8}{'the98':>8}")
    for f in NEW:
        def rc(rs):
            x = np.array([r[f] for r in rs]); e = np.array([r["_resid"] for r in rs])
            mk = ~(np.isnan(x) | np.isnan(e))
            return pearsonr(x[mk], e[mk])[0] if mk.sum() > 4 and np.std(x[mk]) > 0 else np.nan
        print(f"  {f:<12}{rc(rows):>+8.3f}{rc(c):>+8.3f}{rc(n):>+8.3f}")

    # 2. out-of-sample residual recovery: train resid~feat on one dataset, predict other
    print("\n2. OUT-OF-SAMPLE residual recovery (train resid~feat on A, r on B) — structured vs noise:")
    for f in NEW:
        def fit_pred(tr, te):
            x = np.array([r[f] for r in tr]); e = np.array([r["_resid"] for r in tr])
            mk = ~(np.isnan(x) | np.isnan(e)); x, e = x[mk], e[mk]
            if len(x) < 5 or np.std(x) == 0:
                return np.nan
            a, b = np.polyfit(x, e, 1)
            xe = np.array([r[f] for r in te]); ee = np.array([r["_resid"] for r in te])
            mk2 = ~(np.isnan(xe) | np.isnan(ee))
            return pearsonr(a * xe[mk2] + b, ee[mk2])[0]
        print(f"  {f:<12} cr65->the98={fit_pred(c, n):>+.3f}   the98->cr65={fit_pred(n, c):>+.3f}")
    print("  (positive in BOTH = the residual is STRUCTURED by this feature = real missing variable.)")

    # 3. slope de-compression: PROD vs PROD+best, leave-dataset-out, report r AND slope
    print("\n3. SLOPE de-compression (does the feature fix the attenuation, not just r?):")

    def ldo(tr, te, cols):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
        p = np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w
        yy = np.array([r["y"] for r in te])[oke]
        return pearsonr(p, yy)[0], np.polyfit(p, yy, 1)[0]
    for nm, cols in [("PROD", PROD), ("PROD+packing", PROD + ["packing"]),
                     ("PROD+drydry_area", PROD + ["drydry_area"]),
                     ("PROD+hyd_match", PROD + ["hyd_match"]),
                     ("PROD+hyd_match+packing", PROD + ["hyd_match", "packing"]),
                     ("PROD+all5", PROD + NEW)]:
        r1, s1 = ldo(n, c, cols); r2, s2 = ldo(c, n, cols)
        print(f"  {nm:<24} the98->cr65 r={r1:+.3f} slope={s1:.2f}   cr65->the98 r={r2:+.3f} slope={s2:.2f}")
    print("  (slope rising toward 1.0 = de-compression = the missing variable is found.)")


if __name__ == "__main__":
    main()
