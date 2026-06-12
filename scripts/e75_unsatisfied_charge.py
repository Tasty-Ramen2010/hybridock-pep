"""E75 — the UNSATISFIED BURIED CHARGE penalty (why net_elec washed out, and the real charged separator).

Insight: E72 showed net_elec (coul+desolv) is FLAT on charged because AVERAGING sums paired and unpaired
buried charges together — they cancel. The discriminating physics is per-charge SATISFACTION:
  * a buried charged group WITH a complementary partner (salt bridge) = favorable (desolvation repaid)
  * a buried charged group WITHOUT a partner = SEVERE penalty (unsatisfied buried charge — a textbook
    destabilizer; the desolvation is paid for nothing)
Strong charged binders bury only SATISFIED charges; weak ones bury orphans. net_elec can't see this
because it's a sum; we resolve it per charged group.

Per peptide charged residue (D/E/K/R/H), from the 3D structure:
  buried   = side-chain charged-atom ΔSASA buried (residue dSASA > 25 Å²)
  paired   = a complementary receptor charged atom within 4.5 Å of the peptide charged atom
Features: n_unsatisfied (buried & unpaired = penalty), n_satisfied, satisfaction_frac,
  unsat_per_L. Test on charged subset, sign-stability across cr65 + the98, leave-dataset-out.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser, NeighborSearch  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e75_unsat.json")
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
# charged side-chain atoms that carry the formal charge
CHG_ATOMS = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"], "HIS": ["ND1", "NE2"],
             "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
      "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
      "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def featurize(pep_pdb, rec_pdb, y, ds):
    pep = P.get_structure("p", str(pep_pdb))[0]
    SR.compute(pep, level="R")
    free = {(r.get_parent().id, r.id[1]): r.sasa for r in pep.get_residues() if r.id[0] == " "}
    # complex SASA
    rec = P.get_structure("r", str(rec_pdb))[0]
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    cx = Structure("c"); m = Model(0); cx.add(m); used = set()
    for ch in list(pep.get_chains()) + list(rec.get_chains()):
        cid = ch.id
        while cid in used:
            cid = chr((ord(cid) + 1) % 90 + 33)
        used.add(cid); c2 = ch.copy(); c2.id = cid; m.add(c2)
    SR.compute(cx, level="R")
    pep_cids = {c.id for c in pep.get_chains()}
    comp = {}
    for ch in cx.get_chains():
        for r in ch.get_residues():
            if r.id[0] == " ":
                comp.setdefault((r.resname.upper(), r.id[1]), r.sasa)
    # receptor charged atoms for pairing search
    rec_chg_atoms = []
    for ch in rec.get_chains():
        for r in ch.get_residues():
            rn = r.resname.upper()
            if rn in CHG_ATOMS:
                sign = 1 if rn in POS3 else -1
                for a in r:
                    if a.name in CHG_ATOMS[rn]:
                        rec_chg_atoms.append((a.coord, sign))
    n_sat = n_unsat = 0
    seq = ""
    pep_res = [r for r in pep.get_residues() if r.id[0] == " "]
    for r in pep_res:
        rn = r.resname.upper()
        seq += A3.get(rn, "X")
        if rn not in CHG_ATOMS:
            continue
        rfree = free.get((r.get_parent().id, r.id[1]), 0.0)
        rbound = comp.get((rn, r.id[1]), rfree)
        dsasa = max(0.0, rfree - rbound)
        if dsasa < 25:  # not buried -> still solvated, no penalty either way
            continue
        sign = 1 if rn in POS3 else -1
        # is there a complementary receptor charged atom within 4.5 Å of any charged atom?
        paired = False
        for a in r:
            if a.name in CHG_ATOMS[rn]:
                for cc, csign in rec_chg_atoms:
                    if csign == -sign and np.linalg.norm(a.coord - cc) < 4.5:
                        paired = True
                        break
            if paired:
                break
        if paired:
            n_sat += 1
        else:
            n_unsat += 1
    nc = sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)
    L = max(1, len(seq))
    nb = n_sat + n_unsat
    return dict(ds=ds, y=y, net_charge=nc, L=L, seq=seq,
                n_unsatisfied=float(n_unsat), n_satisfied=float(n_sat),
                unsat_per_L=n_unsat / L, sat_per_L=n_sat / L,
                satisfaction_frac=(n_sat / nb) if nb else 1.0,
                net_satisfied=float(n_sat - n_unsat))


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            try:
                out[f"98_{k}"] = featurize(pep, rec, v["y"], "the98")
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:30]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            out[f"cr_{r['pdb']}"] = featurize(Path(r["peptide_pdb"]), Path(r["pocket_pdb"]),
                                              r["dg_exp"], "cr65")
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:30]}")
    CACHE.write_text(json.dumps(out))
    return out


def sp(rows, f):
    x = np.array([r[f] for r in rows], float); y = np.array([r["y"] for r in rows], float)
    m = ~(np.isnan(x) | np.isnan(y))
    return spearmanr(x[m], y[m]).statistic if m.sum() > 5 else np.nan


def main():
    rows = list(build().values())
    ch = [r for r in rows if abs(r["net_charge"]) >= 2]
    chc = [r for r in ch if r["ds"] == "cr65"]; ch9 = [r for r in ch if r["ds"] == "the98"]
    print(f"=== E75 unsatisfied buried charge. charged={len(ch)} (cr65={len(chc)}, the98={len(ch9)}) ===")
    tot_u = sum(r["n_unsatisfied"] for r in ch); tot_s = sum(r["n_satisfied"] for r in ch)
    print(f"  total buried charges: satisfied={tot_s:.0f} unsatisfied={tot_u:.0f}")
    print("\nSpearman vs ΔG on CHARGED (for unsat: POSITIVE = more unsatisfied → weaker, expected):")
    print(f"{'feature':<20}{'all chg':>9}{'cr65':>9}{'the98':>9}  stable?")
    for f in ["n_unsatisfied", "unsat_per_L", "n_satisfied", "satisfaction_frac", "net_satisfied"]:
        a, c, d = sp(ch, f), sp(chc, f), sp(ch9, f)
        st = "YES" if (not np.isnan(c) and not np.isnan(d) and c * d > 0) else "flip/na"
        mark = "  <== sign-stable" if st == "YES" and min(abs(c), abs(d)) > 0.2 else ""
        print(f"  {f:<18}{a:>+9.3f}{c:>+9.3f}{d:>+9.3f}  {st}{mark}")

    # combined with mean_burial from e74 + leave-dataset-out
    e74 = json.loads(Path("/tmp/e74_charged.json").read_text())
    for r in ch:
        key = ("cr_" if r["ds"] == "cr65" else "98_")
        # match by seq
        for kk, vv in e74.items():
            if vv["seq"] == r["seq"] and vv["ds"] == r["ds"]:
                r["mean_burial"] = vv["mean_burial"]; break
        r.setdefault("mean_burial", np.nan)
    chm = [r for r in ch if not np.isnan(r.get("mean_burial", np.nan))]
    chcm = [r for r in chm if r["ds"] == "cr65"]; ch9m = [r for r in chm if r["ds"] == "the98"]

    def fp(tr, te, cols):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
        return pearsonr(np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w,
                        np.array([r["y"] for r in te])[oke])[0]
    print(f"\n=== leave-dataset-out on charged (n_burial={len(chm)}) ===")
    print(f"{'model':<30}{'cr65→the98':>12}{'the98→cr65':>12}")
    for nm, cols in [("mean_burial", ["mean_burial"]),
                     ("+ n_unsatisfied", ["mean_burial", "n_unsatisfied"]),
                     ("+ satisfaction_frac", ["mean_burial", "satisfaction_frac"]),
                     ("+ net_satisfied", ["mean_burial", "net_satisfied"])]:
        print(f"  {nm:<30}{fp(ch9m, chcm, cols):>+12.3f}{fp(chcm, ch9m, cols):>+12.3f}")
    print("\n  >> unsat penalty that helps BOTH directions = the charged separator net_elec couldn't see.")


if __name__ == "__main__":
    main()
