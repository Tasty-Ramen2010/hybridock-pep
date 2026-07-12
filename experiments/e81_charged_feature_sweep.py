"""E81 — wide charged-binder feature sweep: does ANY context feature (density/geometry/complementarity)
separate charged strength where sums washed?

Ram's thesis: charge depends on WHERE it sits (rim vs core), whether it's PAIRED, SHIELDED, how DENSE, how
PATTERNED — not one number. We engineer ~25 features in these families and run the rigorous filter:
  GATE 1  sign-stability: Pearson(feat, ΔG) must agree in sign across charged-cr65 AND charged-the98
          (this gate has killed every single-dataset false positive in the campaign).
  GATE 2  residual: correlation with the PROD+vdw out-of-sample residual (the missing-physics direction).
  GATE 3  leave-dataset-out: does adding it to PROD+vdw improve charged transfer in BOTH directions?
Only features passing all three are real. In-sample r on 61 points is NOT trusted.
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
CACHE = Path("/tmp/e81_charged_feats.json")
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
CHG3 = POS3 | NEG3
HPHO3 = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "CYS", "PRO", "TYR"}
CHG_ATOMS = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE", "CZ"], "HIS": ["ND1", "NE2"],
             "ASP": ["OD1", "OD2", "CG"], "GLU": ["OE1", "OE2", "CD"]}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
      "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
      "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
PKA = {"D": 3.65, "E": 4.25, "H": 6.0, "C": 8.3, "Y": 10.07, "K": 10.53, "R": 12.48}


def _res_charge_sign(rn):
    return 1 if rn in POS3 else (-1 if rn in NEG3 else 0)


def scd(seq):
    q = [1 if c in "KR" else (-1 if c in "DE" else 0) for c in seq]
    N = len(seq); s = 0.0
    for m in range(1, N):
        for n in range(m):
            if q[m] and q[n]:
                s += q[m] * q[n] * np.sqrt(m - n)
    return s / max(1, N)


def pI(seq):
    def net(ph):
        nt = 1 / (1 + 10 ** (ph - 9.0)); ct = -1 / (1 + 10 ** (3.1 - ph))
        z = nt + ct
        for c in seq:
            if c in "KRH":
                z += 1 / (1 + 10 ** (ph - PKA[c]))
            elif c in "DECY":
                z += -1 / (1 + 10 ** (PKA[c] - ph))
        return z
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def featurize(pep_pdb, rec_pdb, y, ds, net_charge):
    pep = P.get_structure("p", str(pep_pdb))[0]
    rec = P.get_structure("r", str(rec_pdb))[0]
    SR.compute(pep, level="R")
    free = {(r.get_parent().id, r.id[1]): r.sasa for r in pep.get_residues() if r.id[0] == " "}
    # complex SASA
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
    # receptor charged atoms (coord, sign) + all heavy for shielding
    rec_chg = []
    for ch in rec.get_chains():
        for r in ch.get_residues():
            rn = r.resname.upper(); sgn = _res_charge_sign(rn)
            if sgn:
                for a in r:
                    if a.name in CHG_ATOMS.get(rn, []):
                        rec_chg.append((a.coord.astype(float), sgn))
    rec_heavy = np.array([a.coord for a in rec.get_atoms() if a.element != "H"], float)
    rtree = cKDTree(rec_heavy) if len(rec_heavy) else None

    pep_res = [r for r in pep.get_residues() if r.id[0] == " "]
    seq = "".join(A3.get(r.resname.upper(), "X") for r in pep_res)
    L = max(1, len(seq))
    # per residue burial + charged atom info
    chg_res = []      # (sign, dsasa, centroid, charged_atom_coords)
    bsa_total = 0.0
    pos_cen, neg_cen = [], []
    for r in pep_res:
        rn = r.resname.upper()
        key = (r.get_parent().id, r.id[1])
        fs = free.get(key, 0.0); cs = comp.get((rn, r.id[1]), fs)
        dsasa = max(0.0, fs - cs); bsa_total += dsasa
        sgn = _res_charge_sign(rn)
        if sgn:
            catoms = [a.coord.astype(float) for a in r if a.name in CHG_ATOMS.get(rn, [])]
            cen = np.mean([a.coord for a in r], axis=0).astype(float)
            chg_res.append((sgn, dsasa, cen, catoms, rn))
            (pos_cen if sgn > 0 else neg_cen).append(cen)
    nchg = len(chg_res)
    bsa = max(1.0, bsa_total)
    f = dict(ds=ds, y=y, seq=seq, net_charge=net_charge)

    # --- FAMILY A: interface charge density (Ram's idea) ---
    f["chgres_per_bsa"] = 1000.0 * nchg / bsa
    f["netq_per_bsa"] = 1000.0 * net_charge / bsa
    f["abschg_per_bsa"] = 1000.0 * sum(1 for c in seq if c in "KRDE") / bsa
    f["chg_contact_frac"] = nchg / L

    # --- FAMILY B: rim vs core ---
    buried = [c for c in chg_res if c[1] > 30]
    rim = [c for c in chg_res if c[1] < 15]
    f["buried_chg_frac"] = len(buried) / nchg if nchg else 0.0
    f["rim_chg_frac"] = len(rim) / nchg if nchg else 0.0
    f["mean_chg_burial"] = float(np.mean([c[1] for c in chg_res])) if chg_res else 0.0

    # --- FAMILY C: charge spatial geometry ---
    f["posneg_sep"] = (float(np.linalg.norm(np.mean(pos_cen, 0) - np.mean(neg_cen, 0)))
                       if pos_cen and neg_cen else 0.0)
    allc = [c[2] for c in chg_res]
    f["chg_rg"] = (float(np.sqrt(np.mean(np.sum((np.array(allc) - np.mean(allc, 0)) ** 2, 1))))
                   if len(allc) > 1 else 0.0)
    # peptide dipole magnitude (formal charges)
    if allc:
        dip = np.sum([c[0] * c[2] for c in chg_res], axis=0)
        f["dipole_mag"] = float(np.linalg.norm(dip)) / L
    else:
        f["dipole_mag"] = 0.0

    # --- FAMILY D: complementarity & satisfaction (pep charged atom vs receptor) ---
    compl_e = 0.0; n_compl = 0; n_orphan_buried = 0; n_sb_buried = 0
    for sgn, dsasa, cen, catoms, rn in chg_res:
        paired = False; bestV = 0.0
        for ca in catoms:
            # coulomb potential from receptor charges (within 12 Å)
            for rc, rs in rec_chg:
                d = np.linalg.norm(ca - rc)
                if d < 12:
                    bestV += rs / d
                if rs == -sgn and d < 4.5:
                    paired = True
        compl_e += sgn * bestV          # negative = favorable (opposite charge nearby)
        if paired:
            n_compl += 1
            if dsasa > 30:
                n_sb_buried += 1
        elif dsasa > 30:
            n_orphan_buried += 1
    f["elec_compl_energy"] = -compl_e / max(1, nchg)     # sign so higher = more complementary
    f["compl_frac"] = n_compl / nchg if nchg else 0.0
    f["orphan_buried_frac"] = n_orphan_buried / nchg if nchg else 0.0
    f["sb_buried_per_bsa"] = 1000.0 * n_sb_buried / bsa
    f["net_satisfied_frac"] = (n_compl - n_orphan_buried) / nchg if nchg else 0.0

    # --- FAMILY E: shielding / micro-environment (low dielectric around buried salt bridges) ---
    shield = 0.0
    for sgn, dsasa, cen, catoms, rn in buried:
        if rtree is not None:
            shield += len(rtree.query_ball_point(cen, 6.0))
    f["sb_shielding"] = shield / max(1, len(buried)) if buried else 0.0

    # --- FAMILY F: sequence pattern / physchem ---
    f["scd"] = scd(seq)
    f["pI"] = pI(seq)
    f["gravy"] = float(np.mean([KD.get(c, 0) for c in seq]))
    f["arg_frac_of_pos"] = (sum(c == "R" for c in seq) /
                            max(1, sum(c in "KRH" for c in seq)))
    f["frac_charged"] = sum(c in "KRDE" for c in seq) / L
    return f


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    e78 = json.load(open("/tmp/e78_dewet.json"))   # net_charge + y by id
    e49 = json.load(open("/tmp/e49b_the98.json"))
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        kk = "98_" + k
        if kk not in e78 or abs(e78[kk]["net_charge"]) < 2:
            continue
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            try:
                out[kk] = featurize(pep, rec, v["y"], "the98", e78[kk]["net_charge"])
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:40]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        kk = "cr_" + r["pdb"]
        if kk not in e78 or abs(e78[kk]["net_charge"]) < 2:
            continue
        try:
            out[kk] = featurize(r["peptide_pdb"], r["pocket_pdb"], r["dg_exp"], "cr65",
                                e78[kk]["net_charge"])
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:40]}")
    CACHE.write_text(json.dumps(out))
    return out


def main():
    # residual from PROD+vdw, keyed by seq
    src = open(ROOT / "experiments/e80_charged_gap.py").read().split("def main")[0]
    src = src.replace("Path(__file__).resolve().parents[1]", "Path('%s')" % ROOT)
    ns = {}; exec(src, ns)
    base = ns["load"]()
    PRODv = ns["PROD"]
    pred = ns["loo_pred"](base, PRODv)
    resid_by_seq = {r["seq"]: r["y"] - p for r, p in zip(base, pred)}

    feats = build()
    rows = list(feats.values())
    FEATS = [k for k in rows[0] if k not in ("ds", "y", "seq", "net_charge")]
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]
    print(f"=== E81 charged feature sweep. charged cr65={len(c)} the98={len(n)} ({len(FEATS)} feats) ===")

    def col(rs, f):
        return np.array([rs[i][f] for i in range(len(rs))], float)

    def pr(rs, f):
        x = col(rs, f); yy = np.array([r["y"] for r in rs])
        m = ~np.isnan(x)
        return pearsonr(x[m], yy[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan

    print("\nGATE 1+2: sign-stable across BOTH charged datasets, ranked by min(|r|):")
    print(f"{'feature':<20}{'cr65':>8}{'the98':>8}{'vs resid':>10}  verdict")
    results = []
    for f in FEATS:
        rc, rn = pr(c, f), pr(n, f)
        # residual corr
        xs, rs = [], []
        for r in rows:
            if r["seq"] in resid_by_seq and not np.isnan(r[f]):
                xs.append(r[f]); rs.append(resid_by_seq[r["seq"]])
        rres = pearsonr(xs, rs)[0] if len(xs) > 5 and np.std(xs) > 0 else np.nan
        stable = (not np.isnan(rc) and not np.isnan(rn) and rc * rn > 0)
        results.append((f, rc, rn, rres, stable, min(abs(rc), abs(rn)) if stable else 0))
    results.sort(key=lambda t: -t[5])
    for f, rc, rn, rres, stable, mn in results:
        v = ("STABLE" if stable and mn > 0.15 else ("stable-weak" if stable else "FLIP"))
        mark = "  <==" if stable and mn > 0.2 else ""
        print(f"  {f:<18}{rc:>+8.3f}{rn:>+8.3f}{rres:>+10.3f}  {v}{mark}")

    # GATE 3: leave-dataset-out lift over PROD+vdw, for the stable strong ones
    stables = [t[0] for t in results if t[4] and t[5] > 0.2]
    print(f"\nGATE 3: leave-dataset-out charged lift (stable feats: {stables})")
    # build merged charged matrix with PROD+vdw + candidate
    bydict = {r["seq"]: r for r in base}
    merged = []
    for r in rows:
        b = bydict.get(r["seq"])
        if b is None:
            continue
        mrow = dict(b)
        for f in FEATS:
            mrow[f] = r[f]
        merged.append(mrow)
    mc = [r for r in merged if r["ds"] == "cr65"]; mn = [r for r in merged if r["ds"] == "the98"]

    def ldo(tr, te, cols):
        X = np.array([[r[c] for c in cols] for r in tr], float); yv = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, yv = X[ok], yv[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + 1.0 * R, A.T @ yv)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
        return pearsonr(np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w,
                        np.array([r["y"] for r in te])[oke])[0]
    have_vdw = "vdw" in merged[0]
    BASE = PRODv + (["vdw"] if have_vdw else [])
    print(f"  {'model':<32}{'the98->cr65':>13}{'cr65->the98':>13}")
    print(f"  {'PROD(+vdw) base':<32}{ldo(mn, mc, BASE):>+13.3f}{ldo(mc, mn, BASE):>+13.3f}")
    for f in stables:
        print(f"  {'+ ' + f:<32}{ldo(mn, mc, BASE + [f]):>+13.3f}{ldo(mc, mn, BASE + [f]):>+13.3f}")
    print("\n  >> a feature that lifts BOTH directions over base = a real charged separator beyond shape.")


if __name__ == "__main__":
    main()
