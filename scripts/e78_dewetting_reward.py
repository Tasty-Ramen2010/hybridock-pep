"""E78 — enclosure-weighted DEWETTING reward (Ram's wet/dry-epitope idea, grounded physics version).

Modern field route: Deep-GIST/HydraMap predict the GIST water-thermo field, but Deep-GIST is a GPL
two-model TF chain needing RISM water placement (can't ship, multi-hour install, predicts RECEPTOR
hydration only). Instead we implement the SAME physics the dewetting literature formalizes (Berne/Rossky
geometric-functional implicit water): water confined in a CONCAVE, ENCLOSED hydrophobic patch is frustrated
/ dewetted, so burying that patch on binding PAYS YOU BACK; burying a POLAR/charged group in the same dry
enclosure COSTS desolvation. The new ingredient beyond plain hydrophobic burial = ENCLOSURE (concavity):
a hydrophobic atom buried in a deep enclosed pocket dewets; the same atom in a flat contact does not.

Per peptide heavy atom i at the interface:
  dsasa_i   = atom buried area (free SASA - complex SASA)         [Shrake-Rupley, atom level]
  encl_i    = # receptor heavy atoms within 5.5 Å of atom i        (concavity/enclosure depth proxy)
  polar_i   = element in {N,O}  (polar)  vs  {C,S} (nonpolar)
Features (intensive, /L):
  hyd_dewet      = Σ_nonpolar dsasa_i * encl_i / L      (dewetting REWARD; expect NEG corr w/ ΔG)
  hyd_burial_flat= Σ_nonpolar dsasa_i / L              (plain burial, NO enclosure — control)
  polar_desolv   = Σ_polar    dsasa_i * encl_i / L      (dry-pocket desolvation COST; expect POS corr)
  net_dewet      = hyd_dewet - polar_desolv
Test: sign-stability of each across cr65 + the98; does ENCLOSURE add over flat burial; leave-dataset-out
fitted lift over mean_burial. If net_dewet is sign-stable and beats flat burial -> Ram's wet/dry idea is a
real lever; if enclosure adds nothing over flat hydrophobic burial -> it's redundant with what we ship.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from Bio.PDB.Structure import Structure  # noqa: E402
from Bio.PDB.Model import Model  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e78_dewet.json")
ENC_R = 5.5     # Å, enclosure shell
POLAR_EL = {"N", "O"}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
      "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
      "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _atom_key(at):
    r = at.get_parent()
    return (r.get_parent().id, r.id[1], at.name)


def featurize(pep_pdb, rec_pdb, y, ds):
    pep = P.get_structure("p", str(pep_pdb))[0]
    rec = P.get_structure("r", str(rec_pdb))[0]
    # free peptide atom SASA
    SR.compute(pep, level="A")
    free = {_atom_key(a): a.sasa for a in pep.get_atoms() if a.element != "H"}
    # merged complex with unique chain ids
    cx = Structure("c"); m = Model(0); cx.add(m); used = set(); pep_cids = set()
    for tag, ch_src in [("p", pep), ("r", rec)]:
        for ch in ch_src.get_chains():
            cid = ch.id
            while cid in used:
                cid = chr((ord(cid) + 1) % 90 + 33)
            used.add(cid); c2 = ch.copy(); c2.id = cid; m.add(c2)
            if tag == "p":
                pep_cids.add(cid)
    SR.compute(cx, level="A")
    bound = {}
    for ch in cx.get_chains():
        if ch.id in pep_cids:
            for a in ch.get_atoms():
                if a.element != "H":
                    bound[(ch.id, a.get_parent().id[1], a.name)] = a.sasa
    # receptor heavy-atom coords for enclosure
    rec_xyz = np.array([a.coord for a in rec.get_atoms() if a.element != "H"], float)
    rtree = cKDTree(rec_xyz)

    # per peptide heavy atom
    hyd_dewet = hyd_flat = polar_desolv = 0.0
    n_iface = 0
    seq = "".join(A3.get(r.resname.upper(), "X") for r in pep.get_residues() if r.id[0] == " ")
    # map free-key -> bound-key needs same chain id; chain ids changed in cx. Match by (resid, atomname)
    # build resid/atom -> free sasa (peptide chains assumed unique resid space)
    free_ra = {(k[1], k[2]): v for k, v in free.items()}
    bound_ra = {(k[1], k[2]): v for k, v in bound.items()}
    for ch in pep.get_chains():
        for r in ch.get_residues():
            if r.id[0] != " ":
                continue
            for a in r.get_atoms():
                if a.element == "H":
                    continue
                ra = (r.id[1], a.name)
                fs = free_ra.get(ra)
                bs = bound_ra.get(ra)
                if fs is None or bs is None:
                    continue
                dsasa = max(0.0, fs - bs)
                if dsasa < 1.0:
                    continue
                encl = len(rtree.query_ball_point(a.coord, ENC_R))
                n_iface += 1
                if a.element in POLAR_EL:
                    polar_desolv += dsasa * encl
                else:
                    hyd_dewet += dsasa * encl
                    hyd_flat += dsasa
    L = max(1, len(seq))
    nc = sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)
    # also plain mean_burial for baseline (residue-level proxy: total buried / L)
    mean_burial = sum(max(0.0, free_ra.get((r.id[1], a.name), 0) - bound_ra.get((r.id[1], a.name), 0))
                      for ch in pep.get_chains() for r in ch.get_residues() if r.id[0] == " "
                      for a in r.get_atoms() if a.element != "H") / L
    return dict(ds=ds, y=y, L=L, net_charge=nc, seq=seq,
                hyd_dewet=hyd_dewet / L, hyd_burial_flat=hyd_flat / L,
                polar_desolv=polar_desolv / L, net_dewet=(hyd_dewet - polar_desolv) / L,
                mean_burial=mean_burial)


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
                print(f"  98 {k} {str(e)[:40]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            out[f"cr_{r['pdb']}"] = featurize(r["peptide_pdb"], r["pocket_pdb"], r["dg_exp"], "cr65")
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:40]}")
    CACHE.write_text(json.dumps(out))
    return out


def sp(rows, f):
    x = np.array([r[f] for r in rows], float); y = np.array([r["y"] for r in rows], float)
    m = ~(np.isnan(x) | np.isnan(y))
    return spearmanr(x[m], y[m]).statistic if m.sum() > 5 else np.nan


def fitldo(tr, te, cols):
    X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
    ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + 0.5 * R, A.T @ y)
    Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
    pred = np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w
    return pearsonr(pred, np.array([r["y"] for r in te])[oke])[0]


def main():
    rows = list(build().values())
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]
    print(f"=== E78 dewetting reward. cr65={len(c)} the98={len(n)} ===")
    print("ΔG: lower=stronger. dewet REWARD expect NEG corr; desolv COST expect POS corr.\n")
    print(f"{'feature':<20}{'all':>9}{'cr65':>9}{'the98':>9}  stable?")
    for f in ["hyd_dewet", "hyd_burial_flat", "polar_desolv", "net_dewet", "mean_burial"]:
        a, cc, nn = sp(rows, f), sp(c, f), sp(n, f)
        st = "YES" if (not np.isnan(cc) and not np.isnan(nn) and cc * nn > 0) else "FLIP"
        mark = "  <== sign-stable" if st == "YES" and min(abs(cc), abs(nn)) > 0.2 else ""
        print(f"  {f:<18}{a:>+9.3f}{cc:>+9.3f}{nn:>+9.3f}  {st}{mark}")

    print("\n=== does ENCLOSURE add over flat hydrophobic burial? (leave-dataset-out fitted Pearson) ===")
    print(f"{'model':<34}{'cr65->the98':>13}{'the98->cr65':>13}")
    for nm, cols in [("mean_burial (baseline)", ["mean_burial"]),
                     ("hyd_burial_flat (no enclosure)", ["hyd_burial_flat"]),
                     ("hyd_dewet (enclosure-weighted)", ["hyd_dewet"]),
                     ("net_dewet", ["net_dewet"]),
                     ("mean_burial + net_dewet", ["mean_burial", "net_dewet"])]:
        print(f"  {nm:<34}{fitldo(n, c, cols):>+13.3f}{fitldo(c, n, cols):>+13.3f}")
    print("\n  >> if hyd_dewet beats hyd_burial_flat in BOTH directions, enclosure (the dewetting physics)")
    print("     is real signal beyond plain burial. If equal, wet/dry adds nothing we don't already ship.")


if __name__ == "__main__":
    main()
